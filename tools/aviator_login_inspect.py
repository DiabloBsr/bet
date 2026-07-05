"""Connexion Bet261 (compte du user, machine du user) + inspection réseau Aviator.

Identifiants via env : BET_USER, BET_PASS. JAMAIS écrits sur disque.
La capture (WS + XHR) ne démarre qu'APRÈS le login -> le POST de login n'est
jamais enregistré. Session persistée dans .bet_session/ pour le collecteur futur.

Usage : BET_USER=... BET_PASS=... python tools/aviator_login_inspect.py [url] [secondes]
Fenêtre VISIBLE : gère toi-même un éventuel SMS/CAPTCHA si ça bloque.
"""
from __future__ import annotations
import json, os, re, sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "exports" / "aviator_capture.json"
OUT.parent.mkdir(parents=True, exist_ok=True)
SESSION = ROOT / ".bet_session"

URL = sys.argv[1] if len(sys.argv) > 1 else "https://bet261.mg/instant-games/llc/Aviator"
SECONDS = int(sys.argv[2]) if len(sys.argv) > 2 else 45
USER, PWD = os.environ.get("BET_USER", ""), os.environ.get("BET_PASS", "")

ws_conns, ws_frames, xhr = [], [], []


def try_login(page):
    print("  -> page d'accueil…", flush=True)
    page.goto("https://bet261.mg/", timeout=45000, wait_until="domcontentloaded")
    time.sleep(4)
    # ouvre le formulaire de connexion (bouton texte varié)
    for pat in ("Se connecter", "Connexion", "Connecter", "Log in", "Login"):
        try:
            el = page.get_by_text(re.compile(pat, re.I)).first
            if el and el.is_visible():
                el.click(timeout=3000); print(f"     clic '{pat}'", flush=True); break
        except Exception:
            continue
    time.sleep(3)
    # champ identifiant (téléphone) puis mot de passe (heuristiques)
    filled = False
    for sel in ('input[type="tel"]', 'input[name*="phone" i]', 'input[name*="login" i]',
                'input[name*="user" i]', 'input[type="number"]', 'input[type="text"]'):
        try:
            loc = page.locator(sel).first
            if loc.is_visible(timeout=1500):
                loc.fill(USER, timeout=3000); filled = True
                print(f"     identifiant -> {sel}", flush=True); break
        except Exception:
            continue
    try:
        pw = page.locator('input[type="password"]').first
        pw.fill(PWD, timeout=3000)
        print("     mot de passe rempli", flush=True)
    except Exception:
        print("     ⚠ champ mot de passe introuvable", flush=True)
    if not filled:
        print("     ⚠ champ identifiant introuvable — connecte-toi À LA MAIN dans la fenêtre", flush=True)
    # soumettre
    for pat in ("Se connecter", "Connexion", "Valider", "Connecter", "Log in"):
        try:
            btn = page.get_by_role("button", name=re.compile(pat, re.I)).first
            if btn.is_visible(timeout=1500):
                btn.click(timeout=3000); print(f"     submit '{pat}'", flush=True); break
        except Exception:
            continue
    else:
        try: page.locator('input[type="password"]').first.press("Enter")
        except Exception: pass
    # laisse le temps (redirection / SMS / CAPTCHA géré à la main)
    print("  -> attente post-login 25 s (gère un éventuel SMS/CAPTCHA dans la fenêtre)…", flush=True)
    time.sleep(25)


def main():
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            str(SESSION), headless=False,
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"),
            viewport={"width": 1280, "height": 800})
        page = ctx.pages[0] if ctx.pages else ctx.new_page()

        # déjà connecté (session persistée) ? sinon login
        try:
            page.goto(URL, timeout=30000, wait_until="domcontentloaded")
            time.sleep(3)
        except Exception:
            pass
        logged = page.locator('iframe').count() > 0 and "password" not in page.content().lower()
        if not (logged and USER == ""):
            if USER:
                try_login(page)

        # ---- capture ACTIVÉE seulement maintenant (après login) ----
        def on_ws(ws):
            ws_conns.append(ws.url); print(f"  [WS] {ws.url}", flush=True)
            ws.on("framereceived", lambda pl: ws_frames.append({"d": "recv", "u": ws.url, "x": str(pl)[:2500]}))
            ws.on("framesent", lambda pl: ws_frames.append({"d": "sent", "u": ws.url, "x": str(pl)[:2500]}))
        page.on("websocket", on_ws)

        def on_resp(resp):
            u = resp.url; rt = resp.request.resource_type
            if rt in ("xhr", "fetch") or any(k in u.lower() for k in
                                             ("history", "round", "result", "aviator", "spribe", "game", "api")):
                rec = {"url": u, "status": resp.status, "type": rt}
                try:
                    if "json" in resp.headers.get("content-type", ""):
                        rec["body"] = str(resp.json())[:2000]
                except Exception:
                    pass
                xhr.append(rec)
        page.on("response", on_resp)

        print(f"\nChargement Aviator : {URL}", flush=True)
        try:
            page.goto(URL, timeout=45000, wait_until="domcontentloaded")
        except Exception as exc:
            print(f"  goto: {exc}", flush=True)
        # entre dans l'iframe du jeu si présente (le WS y vit)
        time.sleep(5)
        for fr in page.frames:
            if any(k in (fr.url or "").lower() for k in ("aviator", "spribe", "game")):
                print(f"  [iframe jeu] {fr.url}", flush=True)
        for _ in range(SECONDS):
            time.sleep(1)
        ctx.close()

    data = {"url_in": URL, "ws_connections": list(dict.fromkeys(ws_conns)),
            "n_ws_frames": len(ws_frames), "ws_frames_sample": ws_frames[:80],
            "iframes": list(dict.fromkeys(f for f in [])), "xhr": xhr}
    OUT.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    print("\n=== RÉSUMÉ ===", flush=True)
    print(f"WebSockets ({len(data['ws_connections'])}):")
    for w in data["ws_connections"]:
        print(f"   {w}")
    print(f"Trames WS : {len(ws_frames)}")
    for f in ws_frames[:10]:
        print(f"   [{f['d']}] {f['x'][:150]}")
    print(f"Requêtes XHR/API ({len(xhr)}):")
    for r in xhr[:25]:
        print(f"   {r['status']} {r['url'][:95]}")
        if r.get("body"):
            print(f"      -> {r['body'][:150]}")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
