"""One-shot batch runner — scrapes all 9 new domains in a single browser session."""
import sys, json, time, random
from pathlib import Path

_HERE        = Path(__file__).parent
PROJECT_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))

from scraper import (
    DOMAIN_COMPANIES, ALL_DOMAINS,
    scrape_domain, save_outputs,
    DEFAULT_JSON_OUT, DEFAULT_CSV_OUT,
)

BATCH = [
    ("banking",            ["www.hsbc.co.uk"],              50),
    ("it",                 ["hostinger.com"],               50),
    ("software",           ["www.shopify.com"],             50),
    ("fashion_nike",       ["www.nike.com"],                50),
    ("fashion_lululemon",  ["www.lululemon.com"],           50),
    ("ecommerce",          ["www.amazon.com"],              50),
]

# ── Load existing (banking already scraped) ───────────────────────────────────
existing: list[dict] = []
done_domains: set[str] = set()
if Path(DEFAULT_JSON_OUT).exists():
    prev = json.loads(Path(DEFAULT_JSON_OUT).read_text(encoding="utf-8"))
    existing = prev.get("reviews", [])
    for d, info in prev.get("domains", {}).items():
        if info.get("count", 0) >= 50:
            done_domains.add(d)
    print(f"Loaded {len(existing)} existing reviews. Already done: {sorted(done_domains)}\n")

# Inject slugs into DOMAIN_COMPANIES
for domain_key, slugs, _ in BATCH:
    DOMAIN_COMPANIES[domain_key] = slugs
    if domain_key not in ALL_DOMAINS:
        ALL_DOMAINS.append(domain_key)

pending = [(d, s, t) for d, s, t in BATCH if d not in done_domains]
print(f"Pending ({len(pending)}): {[d for d,_,_ in pending]}\n")

# ── Single browser session ────────────────────────────────────────────────────
from playwright.sync_api import sync_playwright

all_new: list[dict] = []

with sync_playwright() as pw:
    browser = pw.chromium.launch(headless=True)
    ctx = browser.new_context(
        user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        locale="en-US",
        viewport={"width": 1280, "height": 800},
    )
    pg = ctx.new_page()

    print("Warming up session …")
    pg.goto("https://www.trustpilot.com", wait_until="domcontentloaded", timeout=20000)
    time.sleep(random.uniform(2, 3))

    for i, (domain, slugs, target) in enumerate(pending, 1):
        print(f"\n[{i}/{len(pending)}] {domain.upper()}  (target={target})")
        print("─" * 55)
        reviews = scrape_domain(domain, target, pg, min_delay=1.8, max_delay=3.0)
        all_new.extend(reviews)
        print(f"  → {len(reviews)} reviews collected for '{domain}'")

        # Checkpoint after each domain
        combined = existing + all_new
        save_outputs(combined, 50, DEFAULT_JSON_OUT, DEFAULT_CSV_OUT)

        if i < len(pending):
            pause = random.uniform(3.0, 5.0)
            print(f"  Pausing {pause:.1f}s before next domain …")
            time.sleep(pause)

    browser.close()

# ── Final summary ─────────────────────────────────────────────────────────────
final = existing + all_new
print(f"\n{'='*55}")
print(f"Complete: {len(final)} reviews across "
      f"{len(set(r['domain'] for r in final))} domains")
save_outputs(final, 50, DEFAULT_JSON_OUT, DEFAULT_CSV_OUT)

by_d: dict[str, int] = {}
for r in final:
    by_d[r["domain"]] = by_d.get(r["domain"], 0) + 1

print("\nReviews per domain:")
for d, n in sorted(by_d.items()):
    bar  = "█" * (n // 2)
    mark = "✓" if n >= 50 else f"⚠ only {n}"
    print(f"  {mark}  {d:<28} {n:>4}  {bar}")
