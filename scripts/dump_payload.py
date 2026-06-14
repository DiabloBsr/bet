"""Capture the full body of an endpoint matching a URL fragment.

Usage:
  python scripts/dump_payload.py --url https://... --fragment /api/instantleagues/ --out payload.json
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
    ap.add_argument("--url", default=None)
    ap.add_argument("--fragment", required=True)
    ap.add_argument("--out", default="payload_full.json")
    args = ap.parse_args()

    settings = load_settings()
    target_url = args.url or settings.target_url
    matched: list[dict] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=settings.headless)
        context = browser.new_context(user_agent=settings.user_agent)
        page = context.new_page()
        page.set_default_timeout(settings.page_timeout_ms)

        def on_response(response):
            if args.fragment not in response.url:
                return
            try:
                payload = json.loads(response.body())
                matched.append({"url": response.url, "payload": payload})
            except Exception as exc:  # noqa: BLE001
                print(f"could not decode {response.url}: {exc}")

        page.on("response", on_response)
        page.goto(target_url, wait_until="networkidle")
        page.wait_for_timeout(5000)
        context.close()
        browser.close()

    if not matched:
        print("no matching response captured")
        return 1

    Path(args.out).write_text(
        json.dumps(matched[0]["payload"], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"saved {args.out} from {matched[0]['url']}")
    payload = matched[0]["payload"]
    if isinstance(payload, dict):
        print(f"top-level keys: {list(payload.keys())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
