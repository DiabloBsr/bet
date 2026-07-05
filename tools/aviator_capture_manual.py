"""Capture Aviator — repère les champs réels, remplit Identifiant+Mot de passe, connecte, capture.

Identifiants via env BET_USER / BET_PASS (corps de requête jamais lus). Session persistée.
Usage : BET_USER=.. BET_PASS=.. python tools/aviator_capture_manual.py [secondes]
"""
from __future__ import annotations
import json, os, re, sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "exports" / "aviator_capture.json"
SHOT = ROOT / "exports" / "aviator_login.png"
OUT.parent.mkdir(parents=True, exist_ok=True)
SESSION = ROOT / ".bet_session"
URL = "https://bet261.mg/instant-games/llc/Aviator"
SECONDS = int(sys.argv[1]) if len(sys.argv) > 1 else 60
USER, PWD = os.environ.get("BET_USER", ""), os.environ.get("BET_PASS", "")
SENSITIVE = ("login", "auth", "token", "session", "signin", "customer-api", "account")

ws_conns, ws_frames, xhr, notes = [], [], [], []


def click_text(page, patterns, timeout=2000):
    for pat in patterns:
        for meth in (lambda: page.get_by_role("button", name=re.compile(pat, re.I)).first,
                     lambda: page.get_by_text(re.compile(f"^{pat}$", re.I)).first):
            try:
                el = meth()
                if el.is_visible(timeout=timeout):
                    el.click(timeout=timeout); return pat
            except Exception:
                continue
    return None


def enumerate_inputs(page):
    """Liste tous les inputs visibles avec leurs attributs (pour trouver les bons champs)."""
    js = """() => Array.from(document.querySelectorAll('input')).map(el => {
        const r = el.getBoundingClientRect();
        return {type: el.type, placeholder: el.placeholder, name: el.name, id: el.id,
                aria: el.getAttribute('aria-label'),
                visible: !!(r.width && r.height && el.offsetParent), x: Math.round(r.x), y: Math.round(r.y)};
    })"""
    try:
        return page.evaluate(js)
    except Exception:
        return []


def login(page):
    print("  -> login…", flush=True)
    click_text(page, ["OK", "Oui", "Accepter"])
    time.sleep(1)
    ins = enumerate_inputs(page)
    print(f"     {len(ins)} inputs détectés :", flush=True)
    for d in ins:
        if d["visible"]:
            print(f"        type={d['type']} ph={d['placeholder']!r} name={d['name']!r} "
                  f"id={d['id']!r} aria={d['aria']!r} @({d['x']},{d['y']})", flush=True)
    notes.append({"inputs": [d for d in ins if d["visible"]]})

    vis = [d for d in ins if d["visible"]]
    pw = next((d for d in vis if d["type"] == "password"), None)
    # identifiant = champ texte/tel/number/vide visible, le plus proche AU-DESSUS/à gauche du mdp
    cand = [d for d in vis if d["type"] in ("text", "tel", "number", "", "email")]
    ident = None
    if pw and cand:
        ident = min(cand, key=lambda d: (abs(d["y"] - pw["y"]), d["x"]))
    elif cand:
        ident = cand[0]

    def fill(d, val, label):
        if not d:
            print(f"     ⚠ {label} introuvable", flush=True); return False
        for sel in ([f'#{d["id"]}'] if d["id"] else []) + \
                   ([f'input[name="{d["name"]}"]'] if d["name"] else []) + \
                   ([f'input[placeholder="{d["placeholder"]}"]'] if d["placeholder"] else []):
            try:
                page.locator(sel).first.fill(val, timeout=2500)
                print(f"     {label} rempli via {sel}", flush=True); return True
            except Exception:
                continue
        # fallback : par coordonnées
        try:
            page.mouse.click(d["x"] + 30, d["y"] + 10); page.keyboard.type(val, delay=40)
            print(f"     {label} rempli par clic @({d['x']},{d['y']})", flush=True); return True
        except Exception as e:
            print(f"     ⚠ {label} échec : {e}", flush=True); return False

    fill(ident, USER, "Identifiant")
    fill(pw, PWD, "Mot de passe")
    try:
        page.screenshot(path=str(SHOT)); print(f"     capture -> {SHOT.name}", flush=True)
    except Exception:
        pass
    if not click_text(page, ["Se connecter", "Connexion", "Connecter"], timeout=3000):
        try: page.locator('input[type="password"]').first.press("Enter")
        except Exception: pass
    time.sleep(10)
    try:
        still = page.get_by_role("button", name=re.compile("Se connecter", re.I)).first.is_visible(timeout=1500)
    except Exception:
        still = False
    notes.append("login OK" if not still else "login échoué (bouton Se connecter encore présent)")
    print(f"     {'✓ connecté' if not still else '⚠ toujours déconnecté (SMS/CAPTCHA/mauvais champ ?)'}", flush=True)
    return not still


def main():
    ctx = None
    try:
        with sync_playwright() as p:
            ctx = p.chromium.launch_persistent_context(
                str(SESSION), headless=False,
                args=["--disable-blink-features=AutomationControlled"],
                user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                            "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"),
                viewport={"width": 1300, "height": 850})
            page = ctx.pages[0] if ctx.pages else ctx.new_page()
            page.goto(URL, timeout=45000, wait_until="domcontentloaded")
            time.sleep(4)
            try:
                need = page.get_by_role("button", name=re.compile("Se connecter", re.I)).first.is_visible(timeout=2000)
            except Exception:
                need = False
            if need and USER:
                login(page)

            def on_ws(ws):
                # capture TOUS les WS (le jeu Aviator porte un token dans l'URL)
                ws_conns.append(ws.url); print(f"  [WS] {ws.url[:90]}", flush=True)
                ws.on("framereceived", lambda pl: ws_frames.append({"d": "recv", "u": ws.url[:70], "x": str(pl)[:3000]}))
                ws.on("framesent", lambda pl: ws_frames.append({"d": "sent", "u": ws.url[:70], "x": str(pl)[:1200]}))
            page.on("websocket", on_ws)

            def on_resp(resp):
                u = resp.url.lower()
                if any(s in u for s in SENSITIVE): return
                if resp.request.resource_type in ("xhr", "fetch") or any(
                        k in u for k in ("history", "round", "result", "aviator", "spribe", "multiplier", "launch")):
                    rec = {"url": resp.url, "status": resp.status}
                    try:
                        if "json" in resp.headers.get("content-type", ""):
                            rec["body"] = str(resp.json())[:2500]
                    except Exception:
                        pass
                    xhr.append(rec)
            page.on("response", on_resp)

            print("  -> ouverture du jeu…", flush=True)
            page.goto(URL, timeout=45000, wait_until="domcontentloaded")
            time.sleep(3)
            for _ in range(3):
                if not click_text(page, ["OK", "Oui", "Accepter"]): break
                time.sleep(2)
            for t in range(SECONDS):
                if t % 20 == 0 and t:
                    print(f"  … {SECONDS-t}s | WS={len(ws_conns)} trames={len(ws_frames)} frames={len(page.frames)}", flush=True)
                    for f in page.frames:
                        if "aviator.studio" in (f.url or "") and "aviator.studio" not in str(ws_conns):
                            print("     (iframe jeu présente, en attente de son WS…)", flush=True)
                time.sleep(1)
            game_frames = [f.url for f in page.frames
                           if f.url and any(k in f.url.lower() for k in ("aviator", "spribe", "game", "launch"))]
    finally:
        try:
            if ctx: ctx.close()
        except Exception:
            pass

    data = {"notes": notes, "ws_connections": list(dict.fromkeys(ws_conns)),
            "game_iframes": list(dict.fromkeys(locals().get("game_frames", []))),
            "n_ws_frames": len(ws_frames), "ws_frames_sample": ws_frames[:150], "xhr": xhr}
    OUT.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    print("\n=== RÉSUMÉ ===")
    print(f"iframes de jeu : {data['game_iframes']}")
    print(f"WebSockets : {data['ws_connections']}")
    print(f"Trames WS : {len(ws_frames)}")
    for f in ws_frames[:16]:
        print(f"   [{f['d']}] {f['x'][:180]}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
