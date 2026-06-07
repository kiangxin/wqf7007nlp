"""
Inference pipeline: LLM aspect extraction → DeBERTa sentiment classification.

Step 1  LLM (via OpenRouter) reads the review, reasons about which of the 12
        predefined aspect categories are mentioned, and returns a structured
        JSON with verbatim evidence spans + per-aspect reasoning + numeric
        confidence (0.0–1.0).
Step 2  Each evidence span is expanded to its enclosing sentence(s) and passed
        to a DeBERTa-v3 ACSC model for sentiment classification.
        For diffuse aspects (whole-review sentiment), the full cleaned text is used.

Usage:
  export OPENROUTER_API_KEY="sk-..."

  # Single review – interactive
  python inference.py

  # Single review – CLI
  python inference.py --text "Great service but delivery was late"

  # Batch CSV  (column: text)
  python inference.py --input reviews.csv --output predictions.csv

  # With LLM response cache (saves API cost on re-runs)
  python inference.py --input reviews.csv --output predictions.csv --cache_dir ./llm_cache
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
import warnings
from datetime import datetime
from difflib import SequenceMatcher
from html import unescape
from pathlib import Path

_HERE        = Path(__file__).parent          # → ml_dev/
_PROJECT_ROOT = _HERE.parent                  # → absa_nlp/
sys.path.insert(0, str(_PROJECT_ROOT / "utils"))  # for json_parser

import contractions
import emoji
import pandas as pd
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from openai import OpenAI
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Optional: better sentence segmentation
try:
    import nltk
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            pass
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        try:
            nltk.download("punkt", quiet=True)
        except Exception:
            pass
    from nltk.tokenize import sent_tokenize
    _NLTK_AVAILABLE = True
except Exception:
    _NLTK_AVAILABLE = False

load_dotenv()
warnings.filterwarnings("ignore")

# Prompt version — bump when you change SYSTEM_PROMPT so the cache invalidates
PROMPT_VERSION = "v3"

# ──────────────────────────────────────────────────────────────────────────────
# Aspect catalogue
# ──────────────────────────────────────────────────────────────────────────────
ASPECTS = {
    0:  "[non-aspect / irrelevant]",
    1:  "account management: account access",
    2:  "company brand: competitor",
    3:  "company brand: general satisfaction",
    4:  "company brand: reviews",
    5:  "logistics: speed",
    6:  "online experience: app or website",
    7:  "purchase or booking experience: ease of use",
    8:  "staff support: attitude of staff",
    9:  "staff support: email",
    10: "staff support: phone",
    11: "value: discounts and promotions",
    12: "value: price value for money",
}

ASPECT_SLUGS = {
    1:  "account-management.account-access",
    2:  "company-brand.competitor",
    3:  "company-brand.general-satisfaction",
    4:  "company-brand.reviews",
    5:  "logistics-rides.speed",
    6:  "online-experience.app-website",
    7:  "purchase-booking-experience.ease-of-use",
    8:  "staff-support.attitude-of-staff",
    9:  "staff-support.email",
    10: "staff-support.phone",
    11: "value.discounts-promotions",
    12: "value.price-value-for-money",
}

# ──────────────────────────────────────────────────────────────────────────────
# LLM prompt
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an aspect-based sentiment analysis (ABSA) detector for customer reviews.

Identify which of the predefined aspect categories are explicitly or clearly \
implied in the review. For each detected aspect, extract the exact verbatim \
evidence spans from the review text and explain your reasoning for THAT aspect.

Aspect categories:
[1]  account-management.account-access
[2]  company-brand.competitor
[3]  company-brand.general-satisfaction
[4]  company-brand.reviews
[5]  logistics-rides.speed  (delivery/shipping speed ONLY — NOT customer-service response time)
[6]  online-experience.app-website
[7]  purchase-booking-experience.ease-of-use
[8]  staff-support.attitude-of-staff
[9]  staff-support.email
[10] staff-support.phone
[11] value.discounts-promotions
[12] value.price-value-for-money

Output a single JSON object — no markdown fences, no extra text:
{
  "detected_aspects": [
    {
      "id": <int 1–12>,
      "category": "<slug from the list above>",
      "reasoning": "<concise explanation for THIS specific aspect>",
      "evidence_spans": ["<verbatim quote copied from the review>"],
      "confidence": <float between 0.0 and 1.0>
    }
  ],
  "irrelevant": <true | false>
}

Rules:
- reasoning: explain why THIS aspect is present, referring to the evidence. \
  Keep it specific to this aspect — do not summarise other aspects here.
- evidence_spans: copy the exact words from the review that justify this aspect. \
  Multiple spans are allowed. Spans must be verbatim substrings of the review.
- If the aspect pervades the whole review with no single extractable phrase \
  (diffuse sentiment), use an empty list and add "evidence_type": "diffuse".
- confidence: a number from 0.0 to 1.0 representing how certain you are that \
  this aspect is genuinely discussed. 1.0 = explicit and unambiguous, \
  0.7 = clearly implied, 0.5 = borderline, below 0.3 = very weak.
- Set "irrelevant": true and "detected_aspects": [] when no aspect applies.
- Do NOT infer aspects that are not expressed. Do NOT include [0] in detected_aspects.

Examples:

Review: "I was kept on hold for an hour before anyone picked up, but once I got through the agent was really helpful."
{"detected_aspects":[{"id":10,"category":"staff-support.phone","reasoning":"Being kept on hold indicates a phone-based support interaction.","evidence_spans":["kept on hold for an hour before anyone picked up"],"confidence":0.95},{"id":8,"category":"staff-support.attitude-of-staff","reasoning":"The reviewer praises the agent's behaviour, which speaks to staff attitude.","evidence_spans":["the agent was really helpful"],"confidence":0.9}],"irrelevant":false}

Review: "Delivery took 3 weeks, completely unacceptable."
{"detected_aspects":[{"id":5,"category":"logistics-rides.speed","reasoning":"The review complains specifically about how long delivery took.","evidence_spans":["Delivery took 3 weeks, completely unacceptable"],"confidence":0.98}],"irrelevant":false}

Review: "I love this brand, always my first choice."
{"detected_aspects":[{"id":3,"category":"company-brand.general-satisfaction","reasoning":"General brand satisfaction expressed across the whole review, with no single extractable phrase tying it to a specific feature.","evidence_spans":[],"evidence_type":"diffuse","confidence":0.9}],"irrelevant":false}

Review: "Got a 20% discount code which made the purchase much better value."
{"detected_aspects":[{"id":11,"category":"value.discounts-promotions","reasoning":"Reviewer mentions receiving a promotional discount code.","evidence_spans":["20% discount code"],"confidence":0.95},{"id":12,"category":"value.price-value-for-money","reasoning":"Reviewer evaluates the resulting price as better value for money.","evidence_spans":["made the purchase much better value"],"confidence":0.9}],"irrelevant":false}

Review: "The weather was nice today."
{"detected_aspects":[],"irrelevant":true}\
"""

def _build_user_prompt(text: str, context: str | None = None) -> str:
    if context:
        return f"Context: {context}\nReview: \"\"\"{text}\"\"\""
    return f'Review: """{text}"""'


# ──────────────────────────────────────────────────────────────────────────────
# Run-level statistics (tracked internally, surfaced in metadata)
# ──────────────────────────────────────────────────────────────────────────────
class RunStats:
    def __init__(self):
        self.reviews_processed = 0
        self.llm_calls = 0
        self.llm_failures = 0
        self.llm_cache_hits = 0
        self.aspects_detected = 0
        self.aspects_filtered_low_conf = 0
        self.verbatim_hits = 0          # span found verbatim
        self.verbatim_case_insensitive = 0  # found via case-insensitive search
        self.fuzzy_hits = 0             # found via fuzzy match
        self.span_misses = 0            # not locatable at all
        self.diffuse_spans = 0          # LLM returned no spans (diffuse aspect)

    def to_dict(self) -> dict:
        total_localised = (
            self.verbatim_hits + self.verbatim_case_insensitive + self.fuzzy_hits
        )
        total_attempted = total_localised + self.span_misses
        return {
            "reviews_processed":       self.reviews_processed,
            "llm_calls":               self.llm_calls,
            "llm_failures":            self.llm_failures,
            "llm_cache_hits":          self.llm_cache_hits,
            "aspects_detected":        self.aspects_detected,
            "aspects_filtered_low_conf": self.aspects_filtered_low_conf,
            "diffuse_spans":           self.diffuse_spans,
            "span_localisation": {
                "verbatim_hits":             self.verbatim_hits,
                "verbatim_case_insensitive": self.verbatim_case_insensitive,
                "fuzzy_hits":                self.fuzzy_hits,
                "span_misses":               self.span_misses,
                "verbatim_hit_rate":         round(self.verbatim_hits / total_attempted, 4) if total_attempted else None,
                "any_match_rate":            round(total_localised / total_attempted, 4) if total_attempted else None,
            },
        }


# ──────────────────────────────────────────────────────────────────────────────
# LLM client (OpenRouter — drop-in OpenAI-compatible)
# ──────────────────────────────────────────────────────────────────────────────
def build_llm_client(api_key: str) -> OpenAI:
    api_key = (api_key or "").strip()
    if not api_key:
        raise ValueError(
            "OpenRouter API key is empty. Set OPENROUTER_API_KEY in your .env file "
            "or paste it into the sidebar."
        )
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


from json_parser import (
    parse_llm_json      as _parse_llm_json,   # backward-compat alias (imported by llm_judge_eval)
    _extract_json_object,                      # kept for external callers if any
)


def extract_aspects_llm(
    client: OpenAI,
    text: str,
    model: str,
    stats: RunStats,
    max_retries: int = 3,
    timeout: int = 30,
    context: str | None = None,
) -> dict:
    """
    Ask the LLM which aspects are mentioned, with evidence spans, per-aspect
    reasoning, and numeric confidence (0.0–1.0).

    Includes exponential-backoff retry on API failures and a JSON-parse retry.
    On total failure, returns a safe fallback that marks the review irrelevant.
    """
    last_exc = None
    use_json_mode = True  # try json_object mode first; disable on 400
    for attempt in range(max_retries):
        try:
            kwargs: dict = dict(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": _build_user_prompt(text, context)},
                ],
                temperature=0,
                max_tokens=4096,
                timeout=timeout,
            )
            if use_json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**kwargs)
            stats.llm_calls += 1
            raw = response.choices[0].message.content or ""
            finish = getattr(response.choices[0], "finish_reason", None)
            if finish == "length":
                print(f"  ⚠ Response truncated (finish_reason=length, attempt {attempt+1}) — attempting repair …")
            parsed = _parse_llm_json(raw)
            if parsed is not None:
                return parsed
            last_exc = ValueError(f"Unparseable LLM JSON (attempt {attempt+1}): {raw!r}")
        except Exception as e:
            # If json_object mode caused a 400, fall back to plain text mode
            if use_json_mode and "400" in str(e):
                use_json_mode = False
            last_exc = e

        if attempt < max_retries - 1:
            time.sleep(min(2 ** attempt, 8))

    # All retries exhausted
    stats.llm_failures += 1
    return {
        "detected_aspects": [],
        "irrelevant": True,
        "_error": f"LLM call failed after {max_retries} attempts: {last_exc}",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Optional disk cache for LLM responses
# ──────────────────────────────────────────────────────────────────────────────
class LLMCache:
    """Simple disk cache keyed by (text, model, prompt_version)."""

    def __init__(self, cache_dir: str | None):
        self.enabled = bool(cache_dir)
        self.dir = Path(cache_dir) if cache_dir else None
        if self.enabled:
            self.dir.mkdir(parents=True, exist_ok=True)

    def _key(self, text: str, model: str, context: str | None = None) -> str:
        ctx = context or ""
        h = hashlib.sha256(f"{text}|{model}|{PROMPT_VERSION}|{ctx}".encode("utf-8")).hexdigest()
        return h

    def get(self, text: str, model: str, context: str | None = None) -> dict | None:
        if not self.enabled:
            return None
        f = self.dir / f"{self._key(text, model, context)}.json"
        if f.exists():
            try:
                return json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def set(self, text: str, model: str, value: dict, context: str | None = None) -> None:
        if not self.enabled:
            return
        f = self.dir / f"{self._key(text, model, context)}.json"
        try:
            f.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass


def extract_aspects_llm_cached(
    client: OpenAI,
    text: str,
    model: str,
    stats: RunStats,
    cache: LLMCache,
    context: str | None = None,
) -> dict:
    cached = cache.get(text, model, context)
    if cached is not None:
        stats.llm_cache_hits += 1
        return cached
    result = extract_aspects_llm(client, text, model, stats, context=context)
    if "_error" not in result:
        cache.set(text, model, result, context)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Evidence span expansion
# ──────────────────────────────────────────────────────────────────────────────
def _fuzzy_locate(span: str, review: str, threshold: float = 0.85) -> int:
    """
    Slide a window of similar length across the review and return the start
    index of the best fuzzy match if its ratio ≥ threshold, else -1.
    """
    if not span or not review:
        return -1
    span_l = span.lower()
    review_l = review.lower()
    window = len(span)
    # Allow window to flex ±20%
    min_w = max(5, int(window * 0.8))
    max_w = min(len(review), int(window * 1.2) + 1)

    best_ratio = 0.0
    best_idx = -1
    # Coarse scan with the original window size first (cheap)
    for i in range(0, max(1, len(review) - window + 1), max(1, window // 4 or 1)):
        ratio = SequenceMatcher(None, span_l, review_l[i:i + window]).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_idx = i

    # Refine around the best coarse match by trying nearby starts and lengths
    if best_idx >= 0:
        for w in (min_w, window, max_w):
            for delta in range(-20, 21, 2):
                start = best_idx + delta
                if start < 0 or start + w > len(review):
                    continue
                ratio = SequenceMatcher(None, span_l, review_l[start:start + w]).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_idx = start

    return best_idx if best_ratio >= threshold else -1


def _sentence_containing(review: str, char_idx: int) -> str:
    """
    Return the sentence in `review` that contains the given character index.
    Uses NLTK if available, otherwise falls back to a regex-based splitter
    that's gentler on common abbreviations than naive .!? scanning.
    """
    if _NLTK_AVAILABLE:
        try:
            sentences = sent_tokenize(review)
            cursor = 0
            for sent in sentences:
                start = review.find(sent, cursor)
                if start == -1:
                    # Should rarely happen — fall through to scan
                    cursor += len(sent)
                    continue
                end = start + len(sent)
                if start <= char_idx < end:
                    return sent.strip()
                cursor = end
        except Exception:
            pass

    # Fallback: expand outward to nearest sentence boundary.
    # Treat ".!?\n" as boundaries but skip very common abbreviations.
    start = char_idx
    while start > 0 and review[start - 1] not in ".!?\n":
        start -= 1
    end = char_idx
    while end < len(review) and review[end] not in ".!?\n":
        end += 1
    if end < len(review):
        end += 1
    return review[start:end].strip()


def expand_to_sentence(review: str, evidence_span: str, stats: RunStats) -> str:
    """
    Locate the evidence span in the review and expand to the enclosing sentence.

    Fallback chain:
      1. Verbatim case-sensitive find
      2. Verbatim case-insensitive find
      3. Fuzzy match (SequenceMatcher ≥ 0.85)
      4. Return the span as-is (no sentence context)
    """
    if not evidence_span:
        return evidence_span

    # Tier 1: case-sensitive
    idx = review.find(evidence_span)
    if idx != -1:
        stats.verbatim_hits += 1
        return _sentence_containing(review, idx)

    # Tier 2: case-insensitive
    idx = review.lower().find(evidence_span.lower())
    if idx != -1:
        stats.verbatim_case_insensitive += 1
        return _sentence_containing(review, idx)

    # Tier 3: fuzzy
    idx = _fuzzy_locate(evidence_span, review)
    if idx != -1:
        stats.fuzzy_hits += 1
        return _sentence_containing(review, idx)

    # Tier 4: give up — use the span as-is
    stats.span_misses += 1
    return evidence_span


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing  (identical to training pipeline)
# ──────────────────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = contractions.fix(text)
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = unescape(text)
    text = re.sub(r"https?://\S+|www\.\S+", "[URL]", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\S+@\S+\.\S+", "", text)
    text = re.sub(r"[^a-z0-9\s.,!?;:'\"\-_\[\]]", " ", text)
    text = re.sub(r"([.,!?;:])\1+", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ──────────────────────────────────────────────────────────────────────────────
# DeBERTa ACSC model
# ──────────────────────────────────────────────────────────────────────────────
LABEL2ID = {"positive": 0, "negative": 1, "neutral": 2}
ID2LABEL  = {v: k for k, v in LABEL2ID.items()}


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_absa_model(base_name: str, adapter_dir: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(base_name, use_fast=False)
    base = AutoModelForSequenceClassification.from_pretrained(
        base_name,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model = model.merge_and_unload()
    model.to(device)
    model.eval()
    return model, tokenizer


def predict_sentiment(absa_model, tokenizer, text: str, aspect: str,
                      device: str, max_length: int) -> dict:
    enc = tokenizer(
        text,
        text_pair=aspect,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        logits = absa_model(**enc).logits
    probs   = F.softmax(logits, dim=-1).squeeze()
    pred_id = probs.argmax().item()
    return {
        "sentiment":  ID2LABEL[pred_id],
        "confidence": round(probs[pred_id].item(), 4),
        "scores": {ID2LABEL[i]: round(p.item(), 4) for i, p in enumerate(probs)},
    }


# ──────────────────────────────────────────────────────────────────────────────
# Full pipeline: text → LLM aspects + spans → DeBERTa sentiments
# ──────────────────────────────────────────────────────────────────────────────
def run_pipeline(
    text: str,
    llm_client: OpenAI,
    llm_model: str,
    absa_model,
    tokenizer,
    device: str,
    max_length: int,
    stats: RunStats,
    cache: LLMCache,
    min_confidence: float = 0.70,
    context: str | None = None,
) -> list[dict]:
    stats.reviews_processed += 1

    clean = clean_text(text)
    llm_output = extract_aspects_llm_cached(llm_client, text, llm_model, stats, cache, context)

    if llm_output.get("irrelevant") or not llm_output.get("detected_aspects"):
        return [{
            "aspect_id": 0,
            "aspect":    ASPECTS[0],
            "category":  "",
            "reasoning": "",
            "evidence_spans": [],
            "evidence_type":  "",
            "llm_confidence": None,
            "sentiment":      "-",
            "absa_confidence": None,
            "scores":         None,
            "_llm_raw":       llm_output,
        }]

    results = []
    for asp in llm_output["detected_aspects"]:
        idx = asp.get("id", 0)
        if not (1 <= idx <= 12):
            continue

        stats.aspects_detected += 1

        # Numeric confidence (0.0–1.0). Tolerate string by coercing.
        try:
            llm_conf = float(asp.get("confidence", 0.0))
        except (TypeError, ValueError):
            llm_conf = 0.0
        llm_conf = max(0.0, min(1.0, llm_conf))

        if llm_conf < min_confidence:
            stats.aspects_filtered_low_conf += 1
            continue

        evidence_spans = asp.get("evidence_spans", []) or []
        # Determine evidence_type: explicit field wins, else infer from spans
        evidence_type = asp.get(
            "evidence_type",
            "verbatim" if evidence_spans else "diffuse",
        )
        reasoning = asp.get("reasoning", "")

        aspect_readable = ASPECTS[idx]

        # ── Per-span classification ──────────────────────────────────────
        # Classify each evidence span individually through DeBERTa, then
        # group by predicted sentiment.  This ensures mixed-polarity
        # aspects (e.g. "fast delivery" + "slow second order") produce
        # separate result entries instead of averaging into one.
        if evidence_spans:
            span_classifications = []
            for span in evidence_spans:
                expanded = expand_to_sentence(text, span, stats)
                span_text = clean_text(expanded)
                span_sent = predict_sentiment(
                    absa_model, tokenizer, span_text, aspect_readable,
                    device, max_length,
                )
                span_classifications.append({
                    "span":            span,
                    "absa_sentiment":  span_sent["sentiment"],
                    "absa_confidence": span_sent["confidence"],
                    "absa_scores":     span_sent["scores"],
                })

            # Group spans by their predicted sentiment
            groups: dict[str, list[dict]] = {}
            for sc in span_classifications:
                groups.setdefault(sc["absa_sentiment"], []).append(sc)

            for sent_label, group_spans in groups.items():
                avg_conf = round(
                    sum(s["absa_confidence"] for s in group_spans)
                    / len(group_spans), 4
                )
                avg_scores: dict[str, float] = {}
                for lbl in ("positive", "negative", "neutral"):
                    avg_scores[lbl] = round(
                        sum(s["absa_scores"].get(lbl, 0) for s in group_spans)
                        / len(group_spans), 4
                    )
                results.append({
                    "aspect_id":       idx,
                    "aspect":          aspect_readable,
                    "category":        ASPECT_SLUGS[idx],
                    "reasoning":       reasoning,
                    "evidence_spans":  [s["span"] for s in group_spans],
                    "evidence_type":   evidence_type,
                    "llm_confidence":  round(llm_conf, 4),
                    "sentiment":       sent_label,
                    "absa_confidence": avg_conf,
                    "scores":          avg_scores,
                    "span_details":    [
                        {
                            "span":            s["span"],
                            "llm_confidence":  round(llm_conf, 4),
                            "absa_sentiment":  s["absa_sentiment"],
                            "absa_confidence": s["absa_confidence"],
                            "absa_scores":     s["absa_scores"],
                        }
                        for s in group_spans
                    ],
                    "_llm_raw":        llm_output,
                })
        else:
            # Diffuse: no extractable span — classify the full review
            stats.diffuse_spans += 1
            sentiment = predict_sentiment(
                absa_model, tokenizer, clean, aspect_readable,
                device, max_length,
            )
            results.append({
                "aspect_id":       idx,
                "aspect":          aspect_readable,
                "category":        ASPECT_SLUGS[idx],
                "reasoning":       reasoning,
                "evidence_spans":  [],
                "evidence_type":   evidence_type,
                "llm_confidence":  round(llm_conf, 4),
                "sentiment":       sentiment["sentiment"],
                "absa_confidence": sentiment["confidence"],
                "scores":          sentiment["scores"],
                "span_details":    [],
                "_llm_raw":        llm_output,
            })

    if not results:
        return [{
            "aspect_id": 0,
            "aspect":    ASPECTS[0],
            "category":  "",
            "reasoning": "",
            "evidence_spans": [],
            "evidence_type":  "",
            "llm_confidence": None,
            "sentiment":      "-",
            "absa_confidence": None,
            "scores":         None,
            "_llm_raw":       llm_output,
        }]
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Metadata
# ──────────────────────────────────────────────────────────────────────────────
def save_run_metadata(args, stats: RunStats, run_started: str, run_ended: str,
                      metadata_dir: str) -> str:
    Path(metadata_dir).mkdir(parents=True, exist_ok=True)
    stamp = run_started.replace(":", "").replace("-", "").replace("T", "_")
    out_path = Path(metadata_dir) / f"run_{stamp}.json"
    payload = {
        "run_started":  run_started,
        "run_ended":    run_ended,
        "llm_model":    args.llm_model,
        "prompt_version": PROMPT_VERSION,
        "acsc_base_model":   args.base_model,
        "acsc_adapter_path": args.adapter,
        "min_confidence": args.min_confidence,
        "max_length":     args.max_length,
        "cache_dir":      args.cache_dir or None,
        "input":          args.input or None,
        "output":         args.output if args.input else None,
        "stats":          stats.to_dict(),
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(out_path)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main(args):
    device = get_device()
    run_started = datetime.now().isoformat(timespec="seconds")

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError(
            "OpenRouter API key required. Pass --api_key or set OPENROUTER_API_KEY."
        )

    print(f"Device       : {device}")
    print(f"LLM          : {args.llm_model}")
    print(f"Prompt ver.  : {PROMPT_VERSION}")
    print(f"Cache        : {args.cache_dir or '(disabled)'}")
    print(f"Loading ABSA model from {args.adapter} …\n")

    llm_client = build_llm_client(api_key)
    absa_model, tokenizer = load_absa_model(args.base_model, args.adapter, device)
    stats = RunStats()
    cache = LLMCache(args.cache_dir)

    # Sanity-check the trained model's label mapping matches our assumption
    cfg_id2label = getattr(absa_model.config, "id2label", None)
    if cfg_id2label:
        cfg_lower = {int(k): str(v).lower() for k, v in cfg_id2label.items()}
        expected  = {i: lbl for i, lbl in ID2LABEL.items()}
        if cfg_lower != expected:
            print("  ⚠️  Model config id2label differs from inference mapping:")
            print(f"      model :     {cfg_lower}")
            print(f"      inference : {expected}")
            print("      Confirm this is intentional, otherwise predictions will be mislabelled.\n")

    def fmt_results(results: list[dict]) -> None:
        for r in results:
            if r["aspect_id"] == 0:
                print("  → No relevant aspect detected. ABSA skipped.")
                continue
            if r.get("evidence_type") == "diffuse":
                spans_str = "<diffuse — full review>"
            else:
                spans_str = " | ".join(f'"{s}"' for s in r.get("evidence_spans", []))
            absa_conf = f"{r['absa_confidence']:.2%}" if r["absa_confidence"] is not None else "n/a"
            llm_conf  = f"{r['llm_confidence']:.2f}"  if r["llm_confidence"]  is not None else "n/a"
            print(
                f"  [{r['aspect_id']:>2}] {r['aspect']:<48} "
                f"LLM:{llm_conf:<5} "
                f"→ {r['sentiment']:<10} ABSA:{absa_conf}\n"
                f"       Reasoning : {r.get('reasoning','')}\n"
                f"       Evidence  : {spans_str}"
            )

    try:
        # ── Batch CSV mode ────────────────────────────────────────────────────
        if args.input:
            df = pd.read_csv(args.input)
            records = []
            for i, row in df.iterrows():
                text    = str(row["text"])
                results = run_pipeline(
                    text, llm_client, args.llm_model,
                    absa_model, tokenizer, device, args.max_length,
                    stats, cache, args.min_confidence,
                    context=args.context or None,
                )
                for r in results:
                    records.append({
                        "text":            text,
                        "aspect_id":       r["aspect_id"],
                        "aspect":          r["aspect"],
                        "category":        r.get("category", ""),
                        "reasoning":       r.get("reasoning", ""),
                        "evidence_spans":  json.dumps(r.get("evidence_spans", []), ensure_ascii=False),
                        "evidence_type":   r.get("evidence_type", ""),
                        "llm_confidence":  r["llm_confidence"],
                        "sentiment":       r["sentiment"],
                        "absa_confidence": r["absa_confidence"],
                    })
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i+1}/{len(df)} rows …")

            out_df = pd.DataFrame(records)
            out_df.to_csv(args.output, index=False)
            print(f"\nSaved {len(out_df):,} rows → {args.output}")
            return

        # ── Single CLI text ───────────────────────────────────────────────────
        if args.text:
            print(f"Review: {args.text}\n")
            results = run_pipeline(
                args.text, llm_client, args.llm_model,
                absa_model, tokenizer, device, args.max_length,
                stats, cache, args.min_confidence,
                context=args.context or None,
            )
            fmt_results(results)
            return

        # ── Interactive loop ──────────────────────────────────────────────────
        print("Interactive mode — paste a review and press Enter (empty line to quit):\n")
        while True:
            text = input("Review: ").strip()
            if not text:
                break
            results = run_pipeline(
                text, llm_client, args.llm_model,
                absa_model, tokenizer, device, args.max_length,
                stats, cache, args.min_confidence,
                context=args.context or None,
            )
            fmt_results(results)
            print()

    finally:
        run_ended = datetime.now().isoformat(timespec="seconds")
        meta_path = save_run_metadata(args, stats, run_started, run_ended, args.metadata_dir)
        print(f"\nRun metadata saved → {meta_path}")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model",  default="yangheng/deberta-v3-base-absa-v1.1")
    parser.add_argument("--adapter",     default=str(_HERE / "model" / "deberta_absa_finetuned"))
    parser.add_argument("--llm_model",   default="google/gemini-3.5-flash",
                        help="Any OpenRouter model slug")
    parser.add_argument("--api_key",     default="",
                        help="OpenRouter API key (or set OPENROUTER_API_KEY env var)")
    parser.add_argument("--text",        default="")
    parser.add_argument("--context",     default="",
                        help="Optional source context prepended to each review (e.g. 'Banking app — Trustpilot')")
    parser.add_argument("--input",       default="", help="CSV with a 'text' column")
    parser.add_argument("--output",      default="predictions.csv")
    parser.add_argument("--max_length",     type=int,   default=256)
    parser.add_argument("--min_confidence", type=float, default=0.70,
                        help="Minimum LLM aspect confidence (0.0–1.0) to run ABSA")
    parser.add_argument("--cache_dir",   default="",
                        help="Optional directory to cache LLM responses (default: disabled)")
    parser.add_argument("--metadata_dir", default=str(_HERE / "data" / "metadata"),
                        help="Directory to save per-run metadata JSON")
    args = parser.parse_args()
    main(args)
