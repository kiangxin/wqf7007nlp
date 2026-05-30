"""
Export full pipeline predictions (ACD + DeBERTa ABSA) for Trustpilot scraped reviews.

Reads dev/data/trustpilot_reviews.json, filters by --domain, runs each
review through the full ACD→ACSC pipeline, and saves to:
  app/data/output/predictions_trustpilot_{domain}.json

Output format is consumed directly by app/server.py (Browse Pre-analyzed Reviews tab).
The eval/llm_judge/ folder is reserved for LLM-as-judge evaluation artifacts only.

Usage:
  # Banking (HSBC) — 50 reviews
  python eval/llm/export_trustpilot.py --domain banking

  # All 500 reviews (all domains combined)
  python eval/llm/export_trustpilot.py --domain all

  # First 5 reviews of a domain (quick smoke-test)
  python eval/llm/export_trustpilot.py --domain banking --n_reviews 5
"""

import argparse
import json
import os
import sys
from pathlib import Path

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
from evaluate import clean_text   # reuse the same cleaner

load_dotenv(PROJECT_ROOT / ".env")

BASE_MODEL         = "yangheng/deberta-v3-base-absa-v1.1"
DEFAULT_ADAPTER    = str(PROJECT_ROOT / "dev" / "model" / "deberta_absa_finetuned")
TRUSTPILOT_JSON    = str(PROJECT_ROOT / "dev" / "data" / "trustpilot_reviews.json")
CACHE_DIR          = str(EVAL_ROOT / "llm_judge" / "cache")
OUTPUT_DIR         = str(PROJECT_ROOT / "app" / "data" / "output")
DEFAULT_MODEL      = "google/gemini-3.5-flash"


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

    # ── Load and filter Trustpilot reviews ────────────────────────────────────
    raw = json.loads(Path(args.trustpilot_json).read_text(encoding="utf-8"))
    all_reviews = raw.get("reviews", [])

    if args.domain == "all":
        reviews_to_run = all_reviews
        label = "all_domains"
    else:
        reviews_to_run = [r for r in all_reviews if r["domain"] == args.domain]
        label = args.domain

    if not reviews_to_run:
        available = sorted({r["domain"] for r in all_reviews})
        print(f"ERROR: no reviews found for domain '{args.domain}'.")
        print(f"Available domains: {available}")
        sys.exit(1)

    if args.n_reviews and args.n_reviews < len(reviews_to_run):
        reviews_to_run = reviews_to_run[: args.n_reviews]
        print(f"(limited to first {args.n_reviews} reviews)\n")

    print(f"Domain  : {args.domain}  ({len(reviews_to_run)} reviews)")
    print(f"Model   : {args.model}\n")

    # ── LLM cache (shared with llm_judge_eval.py runs) ────────────────────────
    model_slug = args.model.replace("/", "_").replace(".", "-")
    cache      = LLMCache(f"{args.cache_dir}/{model_slug}")
    stats      = RunStats()

    # ── Run pipeline ──────────────────────────────────────────────────────────
    output_reviews = []
    for i, rev in enumerate(reviews_to_run):
        raw_text  = rev.get("text", "")
        clean     = clean_text(raw_text)

        results = run_pipeline(
            clean, client, args.model,
            absa_model, tokenizer, device,
            max_length=256,
            stats=stats,
            cache=cache,
            min_confidence=args.min_confidence,
        )

        n_asp = sum(1 for r in results if r["aspect_id"] != 0)
        print(
            f"  [{i + 1:>3}/{len(reviews_to_run)}]"
            f"  {n_asp:>2} aspect(s)"
            f"  cache_hits: {stats.llm_cache_hits}"
            f"  | {clean[:60]!r}"
        )

        output_reviews.append({
            "review_idx":    i,
            "review_text":   clean,
            # extra Trustpilot metadata (nice for the Browse tab)
            "domain":        rev.get("domain", ""),
            "company":       rev.get("company_display", rev.get("company", "")),
            "rating":        rev.get("rating"),
            "date":          rev.get("date", ""),
            "reviewer":      rev.get("reviewer", ""),
            "results":       _serialise_results(results),
        })

    # ── Save ──────────────────────────────────────────────────────────────────
    out_slug  = f"trustpilot_{label}"
    out_name  = f"predictions_{out_slug}.json"
    out_path  = Path(args.output_dir) / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model":     args.model,
        "adapter":   args.adapter,
        "domain":    args.domain,
        "n_reviews": len(output_reviews),
        "reviews":   output_reviews,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"Saved {len(output_reviews)} reviews → {out_path}")
    print(
        f"LLM calls: {stats.llm_calls}  "
        f"Cache hits: {stats.llm_cache_hits}  "
        f"Failures: {stats.llm_failures}"
    )
    print(f"\nTo serve:  uvicorn app.server:app --reload --port 8501")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export Trustpilot pipeline predictions for the Streamlit/FastAPI browse tab"
    )
    parser.add_argument(
        "--domain", default="banking",
        help="Domain to export (e.g. banking, it, ecommerce, all). Default: banking",
    )
    parser.add_argument("--model",           default=DEFAULT_MODEL)
    parser.add_argument("--adapter",         default=DEFAULT_ADAPTER,
                        help="Path to DeBERTa PEFT adapter (default: v3 fine-tuned)")
    parser.add_argument("--base_model",      default=BASE_MODEL)
    parser.add_argument("--trustpilot_json", default=TRUSTPILOT_JSON)
    parser.add_argument("--cache_dir",       default=CACHE_DIR)
    parser.add_argument("--output_dir",      default=OUTPUT_DIR)
    parser.add_argument("--min_confidence",  type=float, default=0.70)
    parser.add_argument("--n_reviews",       type=int,   default=0,
                        help="Limit to first N reviews (0 = all in domain)")
    parser.add_argument("--api_key",         default="")
    args = parser.parse_args()
    main(args)
