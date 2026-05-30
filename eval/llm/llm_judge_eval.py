"""
ACD evaluation of 3 LLM models on the FABSA golden test set using:
  1. Exact match metrics (set-based precision, recall, F1, exact match)
  2. Claude judge scoring (reasoning quality 0–10, over-conservatism, hallucination)

For each of the 200 reviews, the model's detected aspect categories are compared
against the ground truth label_codes.

Run prepare_golden_test.py first if data/FABSA_golden_test_100.csv doesn't exist:
  python llm/prepare_golden_test.py

Then run evaluation:
  python llm/llm_judge_eval.py
  python llm/llm_judge_eval.py --n_reviews 3 --models "openai/gpt-4.1-mini"
  python llm/llm_judge_eval.py --skip_judge   # exact-match only, no judge API calls
"""

import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# ── Path setup ────────────────────────────────────────────────────────────────
_SCRIPT_DIR  = Path(__file__).parent          # → eval/llm/
EVAL_ROOT    = _SCRIPT_DIR.parent             # → eval/
PROJECT_ROOT = EVAL_ROOT.parent               # → absa_nlp/

sys.path.insert(0, str(PROJECT_ROOT / "dev"))      # inference.py, evaluate.py
sys.path.insert(0, str(PROJECT_ROOT / "utils"))    # json_parser.py

from inference import (
    LLMCache,
    RunStats,
    ASPECT_SLUGS,
    build_llm_client,
    extract_aspects_llm_cached,
    _parse_llm_json,
)
from evaluate import ASPECT_READABLE, parse_hf_labels

load_dotenv(PROJECT_ROOT / ".env")
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
MODELS_TO_TEST = [
    "google/gemini-3.5-flash",
    "openai/gpt-4.1-mini",
    "qwen/qwen3.6-flash",
]
OUTPUT_DIR      = str(EVAL_ROOT / "llm_judge")
CACHE_DIR       = str(EVAL_ROOT / "llm_judge" / "cache")
GOLDEN_CSV_PATH = str(PROJECT_ROOT / "dev" / "data" / "FABSA_golden_test_100.csv")

# Judge
JUDGE_MODEL          = "anthropic/claude-sonnet-4.5"
JUDGE_PROMPT_VERSION = "v1"   # bump to invalidate judge cache
JUDGE_CACHE_DIR      = str(EVAL_ROOT / "llm_judge" / "judge_cache")


# ──────────────────────────────────────────────────────────────────────────────
# Judge prompt  (placeholders filled with json.dumps values in call_judge)
# ──────────────────────────────────────────────────────────────────────────────
JUDGE_PROMPT = """\
You are evaluating an aspect detection system's reasoning quality.

REVIEW: "{review}"
GOLD ASPECTS: {gold_aspects}
PREDICTED ASPECTS: {predicted_aspects}
SYSTEM REASONING (per aspect): {reasoning_dict}
EVIDENCE SPANS: {evidence_dict}

Score the system's reasoning quality from 0-10 where:
10 = Every prediction is grounded in clear evidence with sound reasoning
7  = Most predictions are grounded, minor reasoning gaps
5  = Predictions are partially correct but reasoning is weak or generic
3  = Aspects are guessed without clear textual grounding
0  = Reasoning is fabricated or completely disconnected from the review

Return as JSON only:
{{
  "reasoning_score": <0-10>,
  "explanation": "<one sentence>"
}}\
"""


# ──────────────────────────────────────────────────────────────────────────────
# Judge cache  (independent of inference.py's PROMPT_VERSION)
# ──────────────────────────────────────────────────────────────────────────────
class JudgeCache:
    """
    Disk cache for judge LLM responses.
    Key: sha256(review | acd_model | judge_model | JUDGE_PROMPT_VERSION)
    All entries live in one flat directory — no per-model subdirs needed.
    """

    def __init__(self, cache_dir: str | None):
        self.enabled = bool(cache_dir)
        self.dir = Path(cache_dir) if cache_dir else None
        if self.enabled:
            self.dir.mkdir(parents=True, exist_ok=True)

    def _key(self, review: str, acd_model: str, judge_model: str) -> str:
        payload = f"{review}|{acd_model}|{judge_model}|{JUDGE_PROMPT_VERSION}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, review: str, acd_model: str, judge_model: str) -> dict | None:
        if not self.enabled:
            return None
        f = self.dir / f"{self._key(review, acd_model, judge_model)}.json"
        if f.exists():
            try:
                return json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def set(self, review: str, acd_model: str, judge_model: str, value: dict) -> None:
        if not self.enabled:
            return
        f = self.dir / f"{self._key(review, acd_model, judge_model)}.json"
        try:
            f.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────
def load_golden_test(csv_path: str = GOLDEN_CSV_PATH) -> pd.DataFrame:
    """Load the fixed golden test set (one row per review)."""
    print(f"Loading golden test set from {csv_path} …")
    df = pd.read_csv(csv_path)
    df["parsed_labels"] = df["label_codes"].apply(parse_hf_labels)
    df = df[df["parsed_labels"].map(len) > 0].reset_index(drop=True)
    print(f"  {len(df):,} reviews loaded\n")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# ACD output parsing
# ──────────────────────────────────────────────────────────────────────────────
_VALID_SLUGS = set(ASPECT_SLUGS.values())   # pre-built set for O(1) lookup


def get_predicted_slugs(llm_output: dict) -> set[str]:
    """
    Extract the set of aspect category slugs from a raw LLM ACD output.

    Priority order per aspect entry:
      1. 'category' field if it is a known valid slug  (normal case)
      2. Numeric 'id' field → ASPECT_SLUGS lookup      (truncated category recovery)
      3. 'category' field as-is if still truthy        (unknown format — let eval score it)
    """
    aspects = llm_output.get("detected_aspects") or []
    slugs = set()
    for asp in aspects:
        cat = asp.get("category", "").strip()
        if cat in _VALID_SLUGS:
            # Exact match against known catalogue — fast path
            slugs.add(cat)
        else:
            # Category missing, empty, or truncated (e.g. "staff-support" instead
            # of "staff-support.attitude-of-staff") — recover from numeric id
            try:
                asp_id = int(asp.get("id", 0))
                if asp_id in ASPECT_SLUGS:
                    slugs.add(ASPECT_SLUGS[asp_id])
                    continue
            except (TypeError, ValueError):
                pass
            # Last resort: use whatever category string we have
            if cat:
                slugs.add(cat)
    return slugs


# ──────────────────────────────────────────────────────────────────────────────
# Exact match metrics
# ──────────────────────────────────────────────────────────────────────────────
def compute_acd_metrics(gt_slugs: set[str], pred_slugs: set[str]) -> dict:
    """
    Set-based precision, recall, F1, and exact match for one review.

    gt_slugs   – ground truth aspect slugs from label_codes
    pred_slugs – aspect slugs detected by the LLM
    """
    tp = len(gt_slugs & pred_slugs)
    fp = len(pred_slugs - gt_slugs)
    fn = len(gt_slugs - pred_slugs)

    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall      = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1          = (2 * precision * recall / (precision + recall)
                   if (precision + recall) > 0 else 0.0)
    exact_match = int(gt_slugs == pred_slugs)

    return {
        "precision":   round(precision,   4),
        "recall":      round(recall,      4),
        "f1":          round(f1,          4),
        "exact_match": exact_match,
        "tp": tp, "fp": fp, "fn": fn,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Judge call  (no cache logic — caller handles get/set for stats tracking)
# ──────────────────────────────────────────────────────────────────────────────
_JUDGE_FAILURE = {
    "reasoning_score": None,
    "explanation":     "judge_failed",
}


def call_judge(
    client,
    review: str,
    gt_slugs: set[str],
    pred_slugs: set[str],
    llm_output: dict,
    judge_model: str,
) -> dict:
    """
    Ask the judge LLM to score the ACD system's reasoning quality.
    Returns dict with: reasoning_score (0-10 int), explanation (str).
    Returns _JUDGE_FAILURE on any error — caller must not cache failures.
    """
    detected = llm_output.get("detected_aspects") or []
    reasoning_dict = {
        asp["category"]: asp.get("reasoning", "")
        for asp in detected if asp.get("category")
    }
    evidence_dict = {
        asp["category"]: asp.get("evidence_spans", [])
        for asp in detected if asp.get("category")
    }

    gold_aspects = [ASPECT_READABLE.get(s, s) for s in sorted(gt_slugs)]
    pred_aspects = [ASPECT_READABLE.get(s, s) for s in sorted(pred_slugs)]

    prompt = JUDGE_PROMPT.format(
        review=review,
        gold_aspects=json.dumps(gold_aspects, ensure_ascii=False),
        predicted_aspects=json.dumps(pred_aspects, ensure_ascii=False),
        reasoning_dict=json.dumps(reasoning_dict, ensure_ascii=False),
        evidence_dict=json.dumps(evidence_dict, ensure_ascii=False),
    )

    use_json_mode = True
    for attempt in range(3):
        try:
            kwargs: dict = dict(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=512,
                timeout=30,
            )
            if use_json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**kwargs)
            raw = response.choices[0].message.content or ""
            parsed = _parse_llm_json(raw)
            if parsed is not None and "reasoning_score" in parsed:
                return {
                    "reasoning_score": int(parsed["reasoning_score"]) if parsed["reasoning_score"] is not None else None,
                    "explanation":     str(parsed.get("explanation", "")),
                }
        except Exception as e:
            if use_json_mode and "400" in str(e):
                use_json_mode = False

        if attempt < 2:
            time.sleep(min(2 ** attempt, 8))

    return dict(_JUDGE_FAILURE)


# ──────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ──────────────────────────────────────────────────────────────────────────────
def run_evaluation(
    client,
    sample_df: pd.DataFrame,
    models: list[str],
    cache_dir: str,
    judge_model: str,
    judge_cache_dir: str,
    skip_judge: bool = False,
    output_dir: str | None = None,   # if set, checkpoint-saves after each model
) -> list[dict]:
    rows = []
    n = len(sample_df)
    judge_cache = JudgeCache(None if skip_judge else judge_cache_dir)

    for model in models:
        model_slug = model.replace("/", "_").replace(".", "-")
        cache = LLMCache(f"{cache_dir}/{model_slug}")
        stats = RunStats()
        judge_api_calls   = 0
        judge_cache_hits  = 0
        judge_failures    = 0

        print(f"\n{'='*65}")
        print(f"  Model: {model}")
        print(f"{'='*65}")

        for i, (idx, row) in enumerate(sample_df.iterrows()):
            review    = row["clean_text"]
            gt_labels = row["parsed_labels"]                      # [(slug, sentiment), ...]
            gt_slugs  = {slug for slug, _ in gt_labels}           # only slug for ACD eval

            llm_output  = extract_aspects_llm_cached(client, review, model, stats, cache)
            pred_slugs  = get_predicted_slugs(llm_output)
            metrics     = compute_acd_metrics(gt_slugs, pred_slugs)

            # Human-readable aspect names for the CSV
            gt_readable   = [ASPECT_READABLE.get(s, s) for s in sorted(gt_slugs)]
            pred_readable = [ASPECT_READABLE.get(s, s) for s in sorted(pred_slugs)]

            # ── Judge ─────────────────────────────────────────────────────────
            if skip_judge:
                judge = {**_JUDGE_FAILURE, "explanation": "skipped"}
                j_tag = "J=–"
            else:
                cached_judge = judge_cache.get(review, model, judge_model)
                if cached_judge is not None:
                    judge = cached_judge
                    judge_cache_hits += 1
                else:
                    judge = call_judge(client, review, gt_slugs, pred_slugs,
                                       llm_output, judge_model)
                    judge_api_calls += 1
                    if judge["reasoning_score"] is None:
                        judge_failures += 1
                    else:
                        # Store context alongside the score so cache files are
                        # human-readable on their own (fields prefixed _ are
                        # ignored by compute_summary / print_summary).
                        judge_cache.set(review, model, judge_model, {
                            **judge,
                            "_acd_model":         model,
                            "_review_preview":    review[:150],
                            "_gold_aspects":      sorted(gt_slugs),
                            "_predicted_aspects": sorted(pred_slugs),
                        })

                score = judge["reasoning_score"]
                j_tag = f"J={score if score is not None else '?'}"

            # Surface ACD errors
            acd_failed = "_error" in llm_output

            print(
                f"  [{i + 1:>3}/{n}] "
                f"EM={metrics['exact_match']}  "
                f"pred={len(pred_slugs)}  gt={len(gt_slugs)}  "
                f"{j_tag}"
                + ("  ⚠ ACD FAILED" if acd_failed else "")
            )
            if acd_failed:
                print(f"       raw error : {llm_output['_error']}")

            rows.append({
                "review_idx":          idx,
                "review_text":         review,
                "ground_truth_slugs":  json.dumps(sorted(gt_slugs),  ensure_ascii=False),
                "ground_truth_names":  json.dumps(gt_readable,        ensure_ascii=False),
                "model":               model,
                "predicted_slugs":     json.dumps(sorted(pred_slugs), ensure_ascii=False),
                "predicted_names":     json.dumps(pred_readable,       ensure_ascii=False),
                "exact_match":         metrics["exact_match"],
                "judge_score":         judge["reasoning_score"],
                "judge_explanation":   judge["explanation"],
                "acd_raw":             json.dumps(llm_output, ensure_ascii=False),
            })

        print(
            f"\n  ACD   — calls: {stats.llm_calls}, "
            f"cache hits: {stats.llm_cache_hits}, "
            f"failures: {stats.llm_failures}"
        )
        if not skip_judge:
            print(
                f"  Judge — api calls: {judge_api_calls}, "
                f"cache hits: {judge_cache_hits}, "
                f"failures: {judge_failures}"
            )

        # ── Per-model checkpoint ───────────────────────────────────────────────
        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
            completed = [m for m in models if any(r["model"] == m for r in rows)]
            ckpt_summary = compute_summary(rows, completed)
            (out / "metadata.json").write_text(
                json.dumps({
                    "checkpoint":    True,
                    "models_done":   completed,
                    "models_total":  models,
                    "summary":       ckpt_summary,
                }, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"  ✓ Checkpoint saved  ({len(completed)}/{len(models)} models)  → {out}/results.csv")

    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Summary + saving
# ──────────────────────────────────────────────────────────────────────────────
def compute_summary(rows: list[dict], models: list[str]) -> list[dict]:
    summary = []
    for model in models:
        model_rows = [r for r in rows if r["model"] == model]
        if not model_rows:
            continue

        judge_scores = [r["judge_score"] for r in model_rows if r["judge_score"] is not None]

        summary.append({
            "model":               model,
            "exact_match_rate":    round(float(np.mean([r["exact_match"] for r in model_rows])), 4),
            "avg_reasoning_score": round(float(np.mean(judge_scores)), 4) if judge_scores else None,
            "n_reviews":           len(model_rows),
            "n_judge_failed":      sum(1 for r in model_rows if r["judge_score"] is None),
        })
    return summary


def save_results(rows: list[dict], summary: list[dict], args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "results.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nSaved {len(rows):,} rows → {csv_path}")

    metadata = {
        "checkpoint":    False,        # True while run is in progress; False = complete
        "run_date":      datetime.now().isoformat(timespec="seconds"),
        "golden_csv":    args.golden_csv,
        "models_tested": args.models,
        "judge_model":   args.judge_model,
        "skip_judge":    args.skip_judge,
        "metric":        "exact_match (predicted slugs == gold slugs) + judge reasoning score (0-10)",
        "summary":       summary,
    }
    meta_path = out_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved metadata → {meta_path}")


def print_summary(summary: list[dict]) -> None:
    for s in sorted(summary, key=lambda x: x["exact_match_rate"], reverse=True):
        n     = s["n_reviews"]
        model = s["model"]
        print(f"\n{'='*65}")
        print(f"  Model: {model:<40} (n={n} reviews)")
        print(f"{'='*65}")
        print(f"  Exact Match Rate             :  {s['exact_match_rate']:.2f}")
        # Judge reasoning score
        if s["avg_reasoning_score"] is not None:
            failed = s["n_judge_failed"]
            print(f"  Avg Reasoning Score (judge)  :  {s['avg_reasoning_score']:.1f} / 10  (failed={failed})")
        else:
            print(f"  Judge                        :  skipped")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main(args: argparse.Namespace) -> None:
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY", "")
    client  = build_llm_client(api_key)

    sample_df = load_golden_test(args.golden_csv)
    if args.n_reviews and args.n_reviews < len(sample_df):
        sample_df = sample_df.head(args.n_reviews).reset_index(drop=True)
        print(f"  (limited to first {args.n_reviews} reviews for quick testing)\n")

    rows    = run_evaluation(
        client, sample_df, args.models, args.cache_dir,
        judge_model=args.judge_model,
        judge_cache_dir=args.judge_cache_dir,
        skip_judge=args.skip_judge,
        output_dir=args.output_dir,   # enables per-model checkpoint saves
    )
    summary = compute_summary(rows, args.models)
    save_results(rows, summary, args)   # final save marks checkpoint:False
    print_summary(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate LLM ACD models on FABSA golden test set: exact match + Claude judge"
    )
    parser.add_argument(
        "--golden_csv",
        default=GOLDEN_CSV_PATH,
        help=f"Path to the fixed golden test set CSV (default: {GOLDEN_CSV_PATH})",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODELS_TO_TEST,
        help="OpenRouter model slugs to evaluate (default: all 3)",
    )
    parser.add_argument(
        "--output_dir",
        default=OUTPUT_DIR,
        help=f"Directory for results.csv and metadata.json (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--cache_dir",
        default=CACHE_DIR,
        help=f"Cache directory for ACD responses (default: {CACHE_DIR})",
    )
    parser.add_argument(
        "--judge_model",
        default=JUDGE_MODEL,
        help=f"OpenRouter slug for the judge LLM (default: {JUDGE_MODEL})",
    )
    parser.add_argument(
        "--judge_cache_dir",
        default=JUDGE_CACHE_DIR,
        help=f"Cache directory for judge responses (default: {JUDGE_CACHE_DIR})",
    )
    parser.add_argument(
        "--skip_judge",
        action="store_true",
        default=False,
        help="Skip judge evaluation — run exact-match metrics only",
    )
    parser.add_argument(
        "--api_key",
        default="",
        help="OpenRouter API key (or set OPENROUTER_API_KEY in .env)",
    )
    parser.add_argument(
        "--n_reviews",
        type=int,
        default=0,
        help="Limit evaluation to the first N reviews (0 = all 200, default: 0)",
    )
    args = parser.parse_args()
    main(args)
