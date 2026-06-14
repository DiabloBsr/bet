"""Inspecte la structure brute des XHRs Sporty-Tech pour trouver la vraie journée."""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from playwright.sync_api import sync_playwright


def main():
    captures = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context()
        page = ctx.new_page()
        page.set_default_timeout(60000)

        def on_response(response):
            if "/api/instantleagues/" in response.url:
                try:
                    captures.append((response.url, response.json()))
                except Exception:
                    pass
        page.on("response", on_response)

        try:
            page.goto("https://bet261.mg/virtual/category/instant-league/8035/matches", wait_until="networkidle")
            page.wait_for_timeout(8000)
        except Exception as e:
            print(f"navigation err: {e}")
        try:
            page.goto("https://bet261.mg/virtual/category/instant-league/8035/results", wait_until="networkidle")
            page.wait_for_timeout(8000)
        except Exception as e:
            print(f"navigation err: {e}")
        ctx.close()
        browser.close()

    print(f"\n📡 Captured {len(captures)} XHRs\n")
    for url, payload in captures:
        print(f"\n{'='*80}")
        print(f"URL : {url}")
        print(f"{'='*80}")
        # Top-level keys
        if isinstance(payload, dict):
            print(f"Top-level keys : {list(payload.keys())}")
            # Look for rounds structure
            if "rounds" in payload:
                rounds = payload["rounds"]
                print(f"Nb rounds in payload : {len(rounds)}")
                if rounds:
                    first = rounds[0]
                    print(f"\n--- First round structure ---")
                    print(f"Keys : {list(first.keys())}")
                    for k, v in first.items():
                        if k == "matches":
                            print(f"  matches : [{len(v)} items]")
                            if v:
                                print(f"    first match keys : {list(v[0].keys())[:30]}")
                        elif isinstance(v, (str, int, float, bool, type(None))):
                            print(f"  {k} : {v}")
                        elif isinstance(v, list):
                            print(f"  {k} : list len={len(v)}")
                        elif isinstance(v, dict):
                            print(f"  {k} : dict keys={list(v.keys())[:5]}")
                    # Show full first match for first round
                    if first.get("matches"):
                        print(f"\n--- First match in first round (truncated 1500 chars) ---")
                        match_str = json.dumps(first["matches"][0], indent=2)[:1500]
                        print(match_str)
                # Show all roundNumber values
                round_nums = [r.get("roundNumber") for r in rounds]
                print(f"\n📌 All roundNumber values : {round_nums}")
                # Look if matches contain matchDay/journee
                if rounds and rounds[0].get("matches"):
                    m = rounds[0]["matches"][0]
                    interesting = {k: v for k, v in m.items() if isinstance(v, (str, int, float, bool))
                                   and any(t in k.lower() for t in ["day", "round", "journ", "match", "week", "stage"])}
                    print(f"📌 Match-level day/round fields : {interesting}")
            # League-level info
            for k in ("currentRound", "currentRoundNumber", "currentMatchday", "matchday",
                     "currentStage", "league", "info", "metadata", "details"):
                if k in payload:
                    print(f"\n  📌 payload['{k}'] : {json.dumps(payload[k])[:500] if not isinstance(payload[k], (str,int,float)) else payload[k]}")


if __name__ == "__main__":
    main()
