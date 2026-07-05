"""Inspecteur réseau Aviator (Spribe) — charge l'URL du jeu et capture TOUT.

Usage : ./.venv/Scripts/python.exe tools/aviator_inspect.py "<URL_IFRAME_AVEC_TOKEN>"

Capture pendant ~40 s :
  - toutes les connexions WebSocket + leurs trames (in/out)
  - toutes les requêtes XHR/fetch + réponses (history, rounds, results…)
-> exports/aviator_capture.json  (+ résumé console)
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "exports" / "aviator_capture.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

URL = sys.argv[1] if len(sys.argv) > 1 else None
if not URL:
    print("Usage: aviator_inspect.py \"<URL>\""); sys.exit(1)
SECONDS = int(sys.argv[2]) if len(sys.argv) > 2 else 40

ws_conns, xhr, ws_frames = [], [], []


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"),
            extra_http_headers={"Referer": "https://bet261.mg/", "Origin": "https://bet261.mg"})
        page = ctx.new_page()

        def on_ws(ws):
            ws_conns.append(ws.url)
            print(f"  [WS] {ws.url}", flush=True)
            ws.on("framereceived", lambda pl: ws_frames.append(
                {"dir": "recv", "url": ws.url, "data": str(pl)[:2000]}))
            ws.on("framesent", lambda pl: ws_frames.append(
                {"dir": "sent", "url": ws.url, "data": str(pl)[:2000]}))
        page.on("websocket", on_ws)

        def on_resp(resp):
            u = resp.url
            rt = resp.request.resource_type
            if rt in ("xhr", "fetch") or any(k in u.lower() for k in
                                             ("history", "round", "result", "bet", "game", "aviator", "api")):
                rec = {"url": u, "status": resp.status, "type": rt}
                try:
                    if "json" in (resp.headers.get("content-type", "")):
                        rec["body"] = str(resp.json())[:1500]
                except Exception:
                    pass
                xhr.append(rec)
        page.on("response", on_resp)

        print(f"Chargement : {URL[:90]}…", flush=True)
        try:
            page.goto(URL, timeout=45000, wait_until="domcontentloaded")
        except Exception as exc:
            print(f"  goto: {exc}", flush=True)
        # laisse le jeu tourner et émettre des rounds
        for _ in range(SECONDS):
            time.sleep(1)
        browser.close()

    data = {"url_in": URL, "ws_connections": list(dict.fromkeys(ws_conns)),
            "n_ws_frames": len(ws_frames), "ws_frames_sample": ws_frames[:60],
            "xhr": xhr}
    OUT.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n=== RÉSUMÉ ===")
    print(f"WebSockets ({len(data['ws_connections'])}):")
    for w in data["ws_connections"]:
        print(f"   {w}")
    print(f"Trames WS capturées : {len(ws_frames)}")
    for f in ws_frames[:8]:
        print(f"   [{f['dir']}] {f['data'][:140]}")
    print(f"\nRequêtes XHR/API ({len(xhr)}):")
    for r in xhr[:20]:
        print(f"   {r['status']} {r['url'][:100]}")
        if r.get("body"):
            print(f"      -> {r['body'][:160]}")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
