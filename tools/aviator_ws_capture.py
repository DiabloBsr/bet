"""Capture le WebSocket du JEU Aviator (Spribe) — charge l'iframe en page principale.

1. Session persistée (déjà connecté) -> ouvre la page Aviator bet261.
2. Récupère l'URL de l'iframe af4.crash.aviator.studio (token frais).
3. Navigue DIRECTEMENT dessus (le WS du jeu devient top-level -> capturable).
4. Capture toutes les trames WS ~90 s + CDP en secours (cross-target).

Usage : BET_USER=.. BET_PASS=.. python tools/aviator_ws_capture.py [secondes]
"""
from __future__ import annotations
import json, os, re, sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "exports" / "aviator_ws.json"
OUT.parent.mkdir(parents=True, exist_ok=True)
SESSION = ROOT / ".bet_session"
BET_URL = "https://bet261.mg/instant-games/llc/Aviator"
SECONDS = int(sys.argv[1]) if len(sys.argv) > 1 else 90

frames_ws, conns, cdp_frames = [], [], []


USER, PWD = os.environ.get("BET_USER", ""), os.environ.get("BET_PASS", "")


def click_ok(page):
    for pat in ("OK", "Oui", "Accepter"):
        try:
            el = page.get_by_role("button", name=re.compile(f"^{pat}$", re.I)).first
            if el.is_visible(timeout=1500):
                el.click(timeout=1500); return
        except Exception:
            continue


def login_if_needed(page):
    try:
        need = page.get_by_role("button", name=re.compile("Se connecter", re.I)).first.is_visible(timeout=2000)
    except Exception:
        need = False
    if not (need and USER):
        return
    print("  -> login…", flush=True)
    click_ok(page); time.sleep(1)
    try:
        page.locator('#Login_Id, input[name="username"], input[placeholder="Identifiant" i]').first.fill(USER, timeout=2500)
        page.locator('#Login_Password, input[type="password"]').first.fill(PWD, timeout=2500)
        page.get_by_role("button", name=re.compile("Se connecter", re.I)).first.click(timeout=2500)
        time.sleep(10)
        print("     ✓ tentative login effectuée", flush=True)
    except Exception as exc:
        print(f"     ⚠ login: {exc}", flush=True)


def main():
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            str(SESSION), headless=False,
            args=["--disable-blink-features=AutomationControlled"],
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"),
            viewport={"width": 1300, "height": 850})
        page = ctx.pages[0] if ctx.pages else ctx.new_page()

        # 1-2. ouvre la page bet261, connecte-toi, récupère l'URL du jeu avec token frais
        page.goto(BET_URL, timeout=45000, wait_until="domcontentloaded")
        time.sleep(4)
        login_if_needed(page)
        page.goto(BET_URL, timeout=45000, wait_until="domcontentloaded")
        time.sleep(3)
        for _ in range(3):
            click_ok(page); time.sleep(2)
        game_url = None
        for _ in range(20):
            for f in page.frames:
                if f.url and "aviator.studio" in f.url:
                    game_url = f.url; break
            if game_url:
                break
            time.sleep(1)
        if not game_url:
            print("⚠ iframe jeu non trouvée — connecte-toi/ouvre le jeu dans la fenêtre puis relance.", flush=True)
            ctx.close(); return
        print(f"  URL jeu (token frais) captée : {game_url[:80]}…", flush=True)

        # CDP en secours : capte les trames WS de TOUS les targets (iframes incluses)
        try:
            cdp = ctx.new_cdp_session(page)
            cdp.send("Network.enable")
            cdp.on("Network.webSocketFrameReceived",
                   lambda e: cdp_frames.append({"d": "recv", "x": str(e.get("response", {}).get("payloadData", ""))[:2500]}))
            cdp.on("Network.webSocketFrameSent",
                   lambda e: cdp_frames.append({"d": "sent", "x": str(e.get("response", {}).get("payloadData", ""))[:800]}))
            cdp.on("Network.webSocketCreated", lambda e: conns.append(e.get("url", "")))
        except Exception as exc:
            print(f"  (CDP indispo : {exc})", flush=True)

        def on_ws(ws):
            conns.append(ws.url); print(f"  [WS top] {ws.url[:80]}", flush=True)
            ws.on("framereceived", lambda pl: frames_ws.append({"d": "recv", "x": str(pl)[:2500]}))
            ws.on("framesent", lambda pl: frames_ws.append({"d": "sent", "x": str(pl)[:800]}))
        page.on("websocket", on_ws)

        # 3. navigue DIRECTEMENT sur le jeu (WS devient top-level)
        print("  -> navigation directe vers le jeu…", flush=True)
        try:
            page.goto(game_url, timeout=45000, wait_until="domcontentloaded")
        except Exception as exc:
            print(f"  goto jeu: {exc}", flush=True)
        time.sleep(3)
        click_ok(page)

        # 4. capture
        for t in range(SECONDS):
            if t % 20 == 0 and t:
                print(f"  … {SECONDS-t}s | conns={len(set(conns))} ws={len(frames_ws)} cdp={len(cdp_frames)}", flush=True)
            time.sleep(1)
        ctx.close()

    allframes = frames_ws + cdp_frames
    data = {"game_url": game_url, "ws_connections": list(dict.fromkeys(conns)),
            "n_frames": len(allframes), "frames": allframes[:250]}
    OUT.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    print("\n=== RÉSUMÉ ===")
    print(f"connexions WS : {list(dict.fromkeys(conns))}")
    print(f"trames totales : {len(allframes)} (playwright {len(frames_ws)} + cdp {len(cdp_frames)})")
    # cherche des trames avec des multiplicateurs (nombres décimaux type 1.53, 2.41…)
    mult = [f for f in allframes if re.search(r"\d+\.\d+", f["x"]) and len(f["x"]) < 1500]
    print(f"\ntrames candidates (contiennent des nombres) : {len(mult)}")
    for f in mult[:20]:
        print(f"   [{f['d']}] {f['x'][:220]}")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
