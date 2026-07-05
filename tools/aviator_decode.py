"""Décode le WebSocket Aviator (MessagePack/socket.io) -> trouve le message du
multiplicateur de crash. Login + nav directe sur le jeu + décodage LIVE des trames.

Usage : BET_USER=.. BET_PASS=.. python tools/aviator_decode.py [secondes]
"""
from __future__ import annotations
import json, os, re, sys, time
from collections import Counter
from pathlib import Path
import msgpack
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "exports" / "aviator_decoded.json"
SESSION = ROOT / ".bet_session"
BET_URL = "https://bet261.mg/instant-games/llc/Aviator"
SECONDS = int(sys.argv[1]) if len(sys.argv) > 1 else 70
USER, PWD = os.environ.get("BET_USER", ""), os.environ.get("BET_PASS", "")

events = Counter()
samples = {}          # event name -> 1er payload décodé
crash_like = []       # trames contenant multiplier/crash/round


def decode(raw):
    """Trame engine.io/socket.io possiblement préfixée -> (event, payload) ou None."""
    if isinstance(raw, str):
        raw = raw.encode("latin-1", "ignore")
    if not isinstance(raw, (bytes, bytearray)):
        return None
    # engine.io peut préfixer un octet '4' (message) ; socket.io msgpack sinon direct
    for start in (0, 1):
        try:
            obj = msgpack.unpackb(raw[start:], raw=False, strict_map_key=False)
            if isinstance(obj, dict) and "data" in obj:
                d = obj["data"]
                if isinstance(d, list) and d:
                    return str(d[0]), (d[1] if len(d) > 1 else None)
            return ("_raw", obj)
        except Exception:
            continue
    return None


def click_ok(page):
    for pat in ("OK", "Oui", "Accepter"):
        try:
            el = page.get_by_role("button", name=re.compile(f"^{pat}$", re.I)).first
            if el.is_visible(timeout=1200):
                el.click(timeout=1200); return
        except Exception:
            continue


def main():
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            str(SESSION), headless=False, args=["--disable-blink-features=AutomationControlled"],
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"),
            viewport={"width": 1300, "height": 850})
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto(BET_URL, timeout=45000, wait_until="domcontentloaded"); time.sleep(4)
        # login
        try:
            if page.get_by_role("button", name=re.compile("Se connecter", re.I)).first.is_visible(timeout=2000) and USER:
                click_ok(page)
                page.locator('#Login_Id, input[name="username"]').first.fill(USER, timeout=2500)
                page.locator('#Login_Password, input[type="password"]').first.fill(PWD, timeout=2500)
                page.get_by_role("button", name=re.compile("Se connecter", re.I)).first.click(timeout=2500)
                time.sleep(10)
        except Exception:
            pass
        page.goto(BET_URL, timeout=45000, wait_until="domcontentloaded"); time.sleep(3)
        for _ in range(3):
            click_ok(page); time.sleep(2)
        game_url = next((f.url for f in page.frames if f.url and "aviator.studio" in f.url), None)
        if not game_url:
            print("⚠ jeu non trouvé"); ctx.close(); return

        def on_ws(ws):
            print(f"  [WS] {ws.url[:70]}", flush=True)
            def handle(pl):
                r = decode(pl)
                if not r:
                    return
                ev, payload = r
                events[ev] += 1
                if ev not in samples:
                    samples[ev] = payload
                txt = f"{ev}:{payload}"
                if re.search(r"crash|crush|round|multi|history|result|finish|fly|payout", txt, re.I):
                    if len(crash_like) < 60:
                        crash_like.append(txt[:400])
            ws.on("framereceived", handle)
        page.on("websocket", on_ws)

        page.goto(game_url, timeout=45000, wait_until="domcontentloaded"); time.sleep(3)
        click_ok(page)
        for t in range(SECONDS):
            if t % 20 == 0 and t:
                print(f"  … {SECONDS-t}s | events={dict(events)}", flush=True)
            time.sleep(1)

        # bonus : historique via le DOM (barre du haut) — même origine en nav directe
        dom_hist = []
        try:
            dom_hist = page.evaluate(
                """() => Array.from(document.querySelectorAll('*'))
                    .map(e=>e.textContent).filter(t=>/^\\d+\\.\\d{2}x?$/.test((t||'').trim()))
                    .slice(0,60)""")
        except Exception:
            pass
        ctx.close()

    data = {"game_url": game_url, "event_counts": dict(events),
            "samples": {k: str(v)[:600] for k, v in samples.items()},
            "crash_like": crash_like, "dom_history": dom_hist}
    OUT.write_text(json.dumps(data, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print("\n=== TYPES DE MESSAGES (WS décodé) ===")
    for ev, n in events.most_common():
        print(f"   {ev}: {n}")
    print("\n=== ÉCHANTILLONS ===")
    for ev, v in samples.items():
        print(f"   {ev} -> {str(v)[:200]}")
    print("\n=== TRAMES 'crash/round/history/result' ===")
    for c in crash_like[:20]:
        print(f"   {c}")
    print(f"\n=== HISTORIQUE DOM (barre) : {dom_hist[:30]}")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
