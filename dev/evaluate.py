"""
Evaluate the fine-tuned DeBERTa-v3 ABSA model on the official FABSA test split.

Loads jordiclive/FABSA split="test" directly from HuggingFace, applies the same
preprocessing pipeline used during training, then runs inference with the saved
PEFT adapter and reports per-class and per-aspect metrics.

Usage:
  python evaluate.py
  python evaluate.py --adapter ./deberta_absa_finetuned --batch_size 16
"""

import argparse
import re
import sys
import warnings
from html import unescape
from pathlib import Path

import contractions
import emoji
import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from sklearn.metrics import classification_report, f1_score, accuracy_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

warnings.filterwarnings("ignore")

_HERE = Path(__file__).parent  # → ml_dev/

# ──────────────────────────────────────────────────────────────────────────────
# Constants  (must match train.py)
# ──────────────────────────────────────────────────────────────────────────────
BASE_MODEL    = "yangheng/deberta-v3-base-absa-v1.1"
ADAPTER_DIR   = str(_HERE / "model" / "deberta_absa_finetuned")
TEST_CSV_PATH = str(_HERE / "data" / "FABSA_test_preprocessed.csv")
MAX_LENGTH    = 256
BATCH_SIZE    = 16

LABEL2ID = {"positive": 0, "negative": 1, "neutral": 2}
ID2LABEL  = {v: k for k, v in LABEL2ID.items()}

SENTIMENT_CODE_MAP = {"1": "positive", "-1": "negative", "0": "neutral"}

ASPECT_CATEGORIES = [
    "account-management.account-access",
    "company-brand.competitor",
    "company-brand.general-satisfaction",
    "company-brand.reviews",
    "logistics-rides.speed",
    "online-experience.app-website",
    "purchase-booking-experience.ease-of-use",
    "staff-support.attitude-of-staff",
    "staff-support.email",
    "staff-support.phone",
    "value.discounts-promotions",
    "value.price-value-for-money",
]

ASPECT_READABLE = {
    "account-management.account-access":        "account management: account access",
    "company-brand.competitor":                 "company brand: competitor",
    "company-brand.general-satisfaction":       "company brand: general satisfaction",
    "company-brand.reviews":                    "company brand: reviews",
    "logistics-rides.speed":                    "logistics: speed",
    "online-experience.app-website":            "online experience: app or website",
    "purchase-booking-experience.ease-of-use":  "purchase or booking experience: ease of use",
    "staff-support.attitude-of-staff":          "staff support: attitude of staff",
    "staff-support.email":                      "staff support: email",
    "staff-support.phone":                      "staff support: phone",
    "value.discounts-promotions":               "value: discounts and promotions",
    "value.price-value-for-money":              "value: price value for money",
}


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing  (identical to preprocessing.ipynb)
# ──────────────────────────────────────────────────────────────────────────────
_LEXICAL_MAP = {
    "w/o":    "without",
    "w/":     "with",
    "b4":     "before",
    "plz":    "please",
    "pls":    "please",
    "thx":    "thanks",
    "idk":    "i don't know",
    "fyi":    "for your information",
    "approx": "approximately",
    "esp":    "especially",
    "msg":    "message",
    "amt":    "amount",
    "qty":    "quantity",
}


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = contractions.fix(text)
    for informal, standard in _LEXICAL_MAP.items():
        text = re.sub(rf"\b{re.escape(informal)}\b", standard, text, flags=re.IGNORECASE)
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = unescape(text)
    text = re.sub(r"https?://\S+|www\.\S+", "[URL]", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\S+@\S+\.\S+", "", text)
    text = re.sub(r"[^a-z0-9\s.,!?;:'\"\-_\[\]]", " ", text)
    text = re.sub(r"([.,!?;:])\1+", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_hf_labels(raw) -> list[tuple[str, str]]:
    """
    Parse the HuggingFace dataset label_codes field.
    The HF dataset stores it as a Python list of strings like
    ['staff-support.attitude-of-staff.-1', 'company-brand.reviews.-1']
    rather than the string-repr used in the local CSV.
    """
    if raw is None:
        return []
    # HF dataset gives a real list; local CSV gives a string repr
    if isinstance(raw, str):
        import ast
        try:
            raw = ast.literal_eval(raw)
        except Exception:
            return []
    result = []
    for code in raw:
        parts = str(code).rsplit(".", 1)
        if len(parts) != 2:
            continue
        aspect, sent_code = parts
        sentiment = SENTIMENT_CODE_MAP.get(sent_code)
        if sentiment and aspect in ASPECT_CATEGORIES:
            result.append((aspect, sentiment))
    return result


def build_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Expand one row per review into one row per (review, aspect) pair."""
    records = []
    for _, row in df.iterrows():
        for aspect_code, sentiment in row["parsed_labels"]:
            records.append({
                "text":        row["clean_text"],
                "aspect":      ASPECT_READABLE[aspect_code],
                "aspect_code": aspect_code,
                "sentiment":   sentiment,
                "label":       LABEL2ID[sentiment],
            })
    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────────────────────────────────────
def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────
def predict(model, tokenizer, texts: list, aspects: list,
            device: str, batch_size: int, max_length: int) -> np.ndarray:
    model.eval()
    all_preds = []
    for i in range(0, len(texts), batch_size):
        batch_texts   = texts[i: i + batch_size]
        batch_aspects = aspects[i: i + batch_size]
        enc = tokenizer(
            batch_texts,
            text_pair=batch_aspects,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        all_preds.append(logits.argmax(dim=-1).cpu().numpy())
    return np.concatenate(all_preds)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main(args):
    device = get_device()
    print(f"Device: {device}")

    # ── Load preprocessed test split ─────────────────────────────────────────
    print(f"Loading preprocessed test set from {args.test_csv} …")
    pair_df = pd.read_csv(args.test_csv)
    pair_df["label"] = pair_df["sentiment"].map(LABEL2ID)
    pair_df = pair_df.dropna(subset=["label"]).reset_index(drop=True)
    pair_df["label"] = pair_df["label"].astype(int)
    print(f"  Test pairs:    {len(pair_df):,}")
    print(f"  Sentiment dist:\n{pair_df['sentiment'].value_counts()}\n")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)

    base = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    print(f"Loading PEFT adapter: {args.adapter}")
    model = PeftModel.from_pretrained(base, args.adapter)
    model = model.merge_and_unload()   # merge LoRA weights for faster inference
    model.to(device)

    # ── Predict ───────────────────────────────────────────────────────────────
    preds = predict(
        model, tokenizer,
        pair_df["text"].tolist(),
        pair_df["aspect"].tolist(),
        device, args.batch_size, args.max_length,
    )
    labels = pair_df["label"].to_numpy()

    # ── Overall report ────────────────────────────────────────────────────────
    print("=" * 60)
    print("Overall — classification report")
    print("=" * 60)
    print(classification_report(
        labels, preds,
        target_names=list(LABEL2ID.keys()),
        zero_division=0,
    ))
    print(f"Micro F1   : {f1_score(labels, preds, average='micro', zero_division=0):.4f}")
    print(f"Macro F1   : {f1_score(labels, preds, average='macro',    zero_division=0):.4f}")
    print(f"Weighted F1: {f1_score(labels, preds, average='weighted', zero_division=0):.4f}")

    # ── Per-aspect breakdown ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Per-aspect macro F1")
    print("=" * 60)
    pair_df["pred"] = preds
    for aspect_code in ASPECT_CATEGORIES:
        sub = pair_df[pair_df["aspect_code"] == aspect_code]
        if len(sub) == 0:
            continue
        f1 = f1_score(sub["label"], sub["pred"], average="macro", zero_division=0)
        print(f"  {aspect_code:<45}  n={len(sub):>4}  macro-F1={f1:.3f}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model",  default=BASE_MODEL)
    parser.add_argument("--adapter",     default=ADAPTER_DIR,
                        help="Path to the saved PEFT adapter directory")
    parser.add_argument("--test_csv",    default=TEST_CSV_PATH,
                        help="Path to preprocessed test CSV")
    parser.add_argument("--batch_size",  type=int, default=BATCH_SIZE)
    parser.add_argument("--max_length",  type=int, default=MAX_LENGTH)
    args = parser.parse_args()
    main(args)
