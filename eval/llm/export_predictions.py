"""
Export full pipeline predictions (ACD + DeBERTa ABSA) for a chosen LLM model.
Uses the LLM response cache already populated by llm_judge_eval.py — every
LLM call is an instant cache hit, only DeBERTa runs fresh.

Run after llm_judge_eval.py:
  python llm/export_predictions.py --model "openai/gpt-4.1-mini"
  python llm/export_predictions.py --model "openai/gpt-4.1-mini" --n_reviews 100

Output: llm_judge/app_predictions_{model_slug}.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

_SCRIPT_DIR  = Path(__file__).parent          # → eval/llm/
EVAL_ROOT    = _SCRIPT_DIR.parent             # → eval/
PROJECT_ROOT = EVAL_ROOT.parent               # → absa_nlp/

sys.path.insert(0, str(PROJECT_ROOT / "dev"))      # inference.py, evaluate.py
sys.path.insert(0, str(PROJECT_ROOT / "utils"))    # json_parser.py

from inference import (
    LLMCache,
    RunStats,
    build_llm_client,
    get_device,
    load_absa_model,
    run_pipeline,
)
from evaluate import parse_hf_labels

load_dotenv(PROJECT_ROOT / ".env")

BASE_MODEL    = "yangheng/deberta-v3-base-absa-v1.1"
ADAPTER_DIR   = str(PROJECT_ROOT / "dev" / "model" / "deberta_absa_finetuned")
GOLDEN_CSV    = str(PROJECT_ROOT / "dev" / "data" / "FABSA_golden_test_100.csv")
CACHE_DIR     = str(EVAL_ROOT / "llm_judge" / "cache")
OUTPUT_DIR    = str(EVAL_ROOT / "llm_judge")
DEFAULT_MODEL = "openai/gpt-4.1-mini"


def _serialise_results(results: list[dict]) -> list[dict]:
    """Strip _llm_raw (large) and return JSON-safe dicts."""
    return [{k: v for k, v in r.items() if k != "_llm_raw"} for r in results]


def main(args: argparse.Namespace) -> None:
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY", "")
    client  = build_llm_client(api_key)

    device = get_device()
    print(f"Device  : {device}")
    print(f"Loading DeBERTa adapter from {args.adapter} …")
    absa_model, tokenizer = load_absa_model(args.base_model, args.adapter, device)
    print("Model loaded.\n")

    df = pd.read_csv(args.golden_csv)
    df["parsed_labels"] = df["label_codes"].apply(parse_hf_labels)
    df = df[df["parsed_labels"].map(len) > 0].reset_index(drop=True)

    if args.n_reviews and args.n_reviews < len(df):
        df = df.head(args.n_reviews).reset_index(drop=True)
        print(f"(limited to first {args.n_reviews} reviews)\n")

    model_slug = args.model.replace("/", "_").replace(".", "-")
    cache      = LLMCache(f"{args.cache_dir}/{model_slug}")
    stats      = RunStats()

    print(f"Model   : {args.model}")
    print(f"Reviews : {len(df)}\n")

    reviews = []
    for i, (idx, row) in enumerate(df.iterrows()):
        review  = row["clean_text"]
        results = run_pipeline(
            review, client, args.model,
            absa_model, tokenizer, device,
            max_length=256,
            stats=stats,
            cache=cache,
            min_confidence=args.min_confidence,
        )
        n_asp = sum(1 for r in results if r["aspect_id"] != 0)
        print(
            f"  [{i + 1:>3}/{len(df)}]  {n_asp:>2} aspect(s)  "
            f"cache hits: {stats.llm_cache_hits}"
        )
        reviews.append({
            "review_idx":  int(idx),
            "review_text": review,
            "results":     _serialise_results(results),
        })

    out_path = Path(args.output_dir) / f"app_predictions_{model_slug}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model":     args.model,
        "n_reviews": len(reviews),
        "reviews":   reviews,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nSaved {len(reviews)} reviews → {out_path}")
    print(f"LLM calls: {stats.llm_calls}  "
          f"Cache hits: {stats.llm_cache_hits}  "
          f"Failures: {stats.llm_failures}")
    print(f"\nTo serve:  uvicorn app.server:app --reload --port 8501")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export full pipeline predictions for the Streamlit browse tab"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="LLM model slug — must match the cache produced by llm_judge_eval.py",
    )
    parser.add_argument("--golden_csv",     default=GOLDEN_CSV)
    parser.add_argument("--adapter",        default=ADAPTER_DIR,
                        help="Path to DeBERTa PEFT adapter")
    parser.add_argument("--base_model",     default=BASE_MODEL)
    parser.add_argument("--cache_dir",      default=CACHE_DIR,
                        help="LLM cache dir written by llm_judge_eval.py")
    parser.add_argument("--output_dir",     default=OUTPUT_DIR)
    parser.add_argument("--min_confidence", type=float, default=0.70)
    parser.add_argument("--n_reviews",      type=int,   default=0,
                        help="Limit to first N reviews (0 = all)")
    parser.add_argument("--api_key",        default="")
    args = parser.parse_args()
    main(args)
