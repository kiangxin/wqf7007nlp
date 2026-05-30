"""
Prepare the fixed golden test set for LLM-as-Judge evaluation.

Loads the FABSA test split directly from HuggingFace, deduplicates,
removes reviews with same-aspect conflicting sentiment labels (Section 5.4),
randomly samples N_REVIEWS unique reviews, and saves one row per review
(not aspect-level) to FABSA_golden_test_100.csv.

Run this ONCE before llm_judge_eval.py:
  python llm/prepare_golden_test.py

Output columns:
  id, text (raw), clean_text (cleaned), label_codes (ground truth)
"""

import argparse
import sys
from pathlib import Path

from datasets import load_dataset

_SCRIPT_DIR  = Path(__file__).parent          # → eval/llm/
EVAL_ROOT    = _SCRIPT_DIR.parent             # → eval/
PROJECT_ROOT = EVAL_ROOT.parent               # → absa_nlp/

sys.path.insert(0, str(PROJECT_ROOT / "dev"))      # evaluate.py
from evaluate import clean_text, parse_hf_labels

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
GOLDEN_CSV_PATH = PROJECT_ROOT / "dev" / "data" / "FABSA_golden_test_100.csv"
N_REVIEWS       = 100
SEED            = 42


def main(args: argparse.Namespace) -> None:
    print("Loading jordiclive/FABSA test split from HuggingFace …")
    ds = load_dataset("jordiclive/FABSA", split="test")
    df = ds.to_pandas()
    print(f"  Raw rows        : {len(df):,}")

    # Deduplicate by raw text
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"  After dedup     : {len(df):,}")

    # Clean text
    df["clean_text"] = df["text"].apply(clean_text)

    # Filter to reviews that have at least one valid aspect label
    df["_parsed"] = df["label_codes"].apply(parse_hf_labels)
    df = df[df["_parsed"].map(len) > 0].reset_index(drop=True)
    print(f"  With valid labels: {len(df):,}")

    # Remove reviews where the same aspect appears with conflicting sentiments
    # (Section 5.4 — contradictory ground truth makes evaluation unreliable)
    def _has_conflict(parsed) -> bool:
        seen: dict[str, set] = {}
        for aspect, sentiment in parsed:
            seen.setdefault(aspect, set()).add(sentiment)
        return any(len(s) > 1 for s in seen.values())

    before_conflict = len(df)
    df = df[~df["_parsed"].apply(_has_conflict)].reset_index(drop=True)
    print(f"  After conflict removal: {len(df):,}  (removed {before_conflict - len(df)})")

    # Sample N unique reviews
    n = min(args.n_reviews, len(df))
    if n < args.n_reviews:
        print(f"  ⚠  Requested {args.n_reviews} but only {len(df)} available — using all.")

    golden_df = df.sample(n=n, random_state=args.seed).reset_index(drop=True)

    # Save one row per review — label_codes kept for ground truth parsing in eval
    out_cols = ["id", "text", "clean_text", "label_codes"]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    golden_df[out_cols].to_csv(out_path, index=False)

    print(f"\nGolden test set  : {len(golden_df):,} unique reviews")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample N unique reviews from FABSA test split (conflicts removed) to create a fixed golden evaluation set"
    )
    parser.add_argument(
        "--output",
        default=str(GOLDEN_CSV_PATH),
        help=f"Output CSV path (default: {GOLDEN_CSV_PATH})",
    )
    parser.add_argument(
        "--n_reviews",
        type=int,
        default=N_REVIEWS,
        help=f"Number of unique reviews to sample (default: {N_REVIEWS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed (default: {SEED})",
    )
    args = parser.parse_args()
    main(args)
