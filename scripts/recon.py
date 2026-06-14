"""Reconnaissance one-shot: capture all JSON responses on the target page.

Usage:
  python scripts/recon.py                       # uses TARGET_URL from .env
  python scripts/recon.py --url https://...     # override target
  python scripts/recon.py --url https://... --out recon_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from playwright.sync_api import sync_playwright

from scraper.config import load_settings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None, help="override TARGET_URL")
    ap.add_argument("--out", default="recon_xhr.json")
    args = ap.parse_args()

    settings = load_settings()
    target_url = args.url or settings.target_url
    captured: list[dict] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=settings.headless)
        context = browser.new_context(user_agent=settings.user_agent)
        page = context.new_page()
        page.set_default_timeout(settings.page_timeout_ms)

        def on_response(response):
            try:
                ctype = (response.headers.get("content-type") or "").lower()
                if "json" not in ctype:
                    return
                body = response.body()
                payload = json.loads(body)
                sample_keys = []
                if isinstance(payload, dict):
                    sample_keys = list(payload.keys())[:10]
                elif isinstance(payload, list) and payload and isinstance(payload[0], dict):
                    sample_keys = list(payload[0].keys())[:10]
                captured.append({
                    "url": response.url,
                    "status": response.status,
                    "size": len(body),
                    "sample_keys": sample_keys,
                })
            except Exception as exc:  # noqa: BLE001
                captured.append({"url": response.url, "error": str(exc)})

        page.on("response", on_response)

        print(f"navigating to {target_url}")
        page.goto(target_url, wait_until="networkidle")
        page.wait_for_timeout(5000)

        context.close()
        browser.close()

    out = Path(args.out)
    out.write_text(json.dumps(captured, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out} entries={len(captured)}")
    # also print a compact summary
    for c in captured:
        if "error" in c:
            continue
        print(f"  [{c['status']}] {c['size']:>6}B keys={c['sample_keys']} {c['url']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
