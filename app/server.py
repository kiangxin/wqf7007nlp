"""
FastAPI backend for the ABSA Analyzer.

Run (from project root):
  # Activate your virtual environment (e.g., conda activate <env-name> or source venv/bin/activate)
  uvicorn app.server:app --reload --port 8501

Then open: http://localhost:8501
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT         = Path(__file__).parent   # → app/
PROJECT_ROOT = ROOT.parent             # → absa_nlp/

sys.path.insert(0, str(PROJECT_ROOT / "dev"))    # inference.py
sys.path.insert(0, str(PROJECT_ROOT / "utils"))  # json_parser.py

# Load .env from project root
load_dotenv(PROJECT_ROOT / ".env")

# ── Constants ─────────────────────────────────────────────────────────────────
PREDICTIONS_DIR  = ROOT / "data" / "output"
BASE_MODEL       = "yangheng/deberta-v3-base-absa-v1.1"
ADAPTER_DIR      = str(PROJECT_ROOT / "dev" / "model" / "deberta_absa_finetuned")
CHECKPOINT_INFO  = PROJECT_ROOT / "dev" / "model" / "deberta_absa_finetuned" / "checkpoint_info.json"
if not CHECKPOINT_INFO.exists():
    CHECKPOINT_INFO = (
        PROJECT_ROOT / "dev" / "model"
        / "deberta_absa_finetuned" / "checkpoint-4190" / "checkpoint_info.json"
    )
PAGE_SIZE        = 20
FRONTEND_HTML    = ROOT / "frontend" / "index.html"

app = FastAPI(title="ABSA Analyzer API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ── Lazy model loading ────────────────────────────────────────────────────────
_absa_model = _tokenizer = _device = None

def _get_absa():
    global _absa_model, _tokenizer, _device
    if _absa_model is None:
        from inference import load_absa_model, get_device
        _device = get_device()
        print(f"Loading DeBERTa on {_device} …")
        _absa_model, _tokenizer = load_absa_model(BASE_MODEL, ADAPTER_DIR, _device)
        print("DeBERTa ready.")
    return _absa_model, _tokenizer, _device


# ── Per-domain predictions cache ──────────────────────────────────────────────
_domain_cache: dict[str, dict] = {}

def _load_domain(domain: str) -> dict:
    if domain not in _domain_cache:
        fname = f"predictions_trustpilot_{domain}.json"
        path  = PREDICTIONS_DIR / fname
        if not path.exists():
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No predictions for domain '{domain}'. "
                    f"Run: python eval/llm/export_trustpilot.py --domain {domain}"
                ),
            )
        _domain_cache[domain] = json.loads(path.read_text(encoding="utf-8"))
    return _domain_cache[domain]


# ══════════════════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
def root():
    return FRONTEND_HTML.read_text(encoding="utf-8")


@app.get("/api/config")
def get_config():
    """Return non-secret server configuration including ACSC evaluation metrics."""
    ckpt: dict = {}
    if CHECKPOINT_INFO.exists():
        try:
            ckpt = json.loads(CHECKPOINT_INFO.read_text())
        except Exception:
            pass
    return {
        "llm_model":     os.environ.get("LLM_MODEL", "openai/gpt-4.1-mini").strip(),
        "acsc_model":    "DeBERTa-v3 (fine-tuned)",
        "acsc_accuracy": round(ckpt["eval_accuracy"] * 100, 1) if ckpt else None,
        "acsc_macro_f1": round(ckpt["eval_macro_f1"] * 100, 1) if ckpt else None,
        "acsc_epochs":   int(ckpt["epoch"]) if ckpt else None,
    }


@app.get("/api/domains")
def get_domains():
    """List all available pre-analyzed Trustpilot domains."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(PREDICTIONS_DIR.glob("predictions_trustpilot_*.json"))
    domains = []
    for f in files:
        domain_key = f.stem.replace("predictions_trustpilot_", "")
        if domain_key == "all_domains":
            continue  # skip combined; serve individual domains only
        try:
            data    = json.loads(f.read_text(encoding="utf-8"))
            reviews = data.get("reviews", [])
            company = reviews[0].get("company", "") if reviews else ""
            domains.append({
                "domain":    data.get("domain", domain_key),
                "company":   company,
                "n_reviews": data.get("n_reviews", len(reviews)),
            })
        except Exception:
            pass
    return {"domains": domains}


@app.get("/api/reviews")
def get_reviews(page: int = 1, per_page: int = PAGE_SIZE, domain: str = "banking"):
    payload  = _load_domain(domain)
    reviews  = payload["reviews"]
    n        = len(reviews)
    n_pages  = max(1, (n + per_page - 1) // per_page)
    page     = max(1, min(page, n_pages))
    start    = (page - 1) * per_page
    end      = min(start + per_page, n)

    return {
        "model":    payload.get("model", ""),
        "domain":   payload.get("domain", domain),
        "total":    n,
        "page":     page,
        "per_page": per_page,
        "n_pages":  n_pages,
        "start":    start + 1,
        "end":      end,
        "reviews":  reviews[start:end],
    }


@app.get("/api/stats")
def get_stats(domain: str = "banking"):
    """Return sentiment breakdown and top aspects for a domain."""
    payload = _load_domain(domain)
    reviews = payload["reviews"]
    sentiment_counts: dict[str, int] = {"positive": 0, "negative": 0, "neutral": 0}
    aspect_counts: dict[str, int] = {}
    for review in reviews:
        for r in review.get("results", []):
            if r.get("aspect_id") == 0:
                continue
            sent = r.get("sentiment", "neutral")
            sentiment_counts[sent] = sentiment_counts.get(sent, 0) + 1
            aspect = r.get("aspect", "")
            if aspect:
                aspect_counts[aspect] = aspect_counts.get(aspect, 0) + 1
    top_aspects = sorted(aspect_counts.items(), key=lambda x: -x[1])[:6]
    return {
        "domain":           payload.get("domain", domain),
        "n_reviews":        len(reviews),
        "total_aspects":    sum(sentiment_counts.values()),
        "sentiment_counts": sentiment_counts,
        "top_aspects":      [{"aspect": a, "count": c} for a, c in top_aspects],
    }


@app.get("/api/review/{num}")
def get_review(num: int, domain: str = "banking"):
    payload = _load_domain(domain)
    reviews = payload["reviews"]
    if num < 1 or num > len(reviews):
        raise HTTPException(status_code=404, detail="Review not found")
    return {"num": num, **reviews[num - 1]}


class AnalyzeRequest(BaseModel):
    text:           str
    context:        str | None = None
    min_confidence: float = 0.70


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    from inference import (
        build_llm_client, run_pipeline, LLMCache, RunStats,
    )
    api_key   = os.environ.get("OPENROUTER_API_KEY", "")
    llm_model = os.environ.get("LLM_MODEL", "openai/gpt-4.1-mini").strip()
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENROUTER_API_KEY not set in server .env — contact the administrator.",
        )

    absa_model, tokenizer, device = _get_absa()
    client  = build_llm_client(api_key)
    results = run_pipeline(
        req.text, client, llm_model,
        absa_model, tokenizer, device,
        max_length=256,
        stats=RunStats(),
        cache=LLMCache(None),
        min_confidence=req.min_confidence,
        context=req.context or None,
    )
    clean = [{k: v for k, v in r.items() if k != "_llm_raw"} for r in results]
    return {"results": clean, "model": llm_model}


# ── Dev entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.server:app", host="0.0.0.0", port=8501, reload=True)
