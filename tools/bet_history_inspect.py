"""Trouve l'endpoint de l'HISTORIQUE DE PARIS de bet261 (compte user).

Login (env BET_USER/BET_PASS), ouvre 'Mes paris', capture les appels XHR pour
repérer l'API d'historique + le token d'auth. Ne loggue jamais le mot de passe.
Usage : BET_USER=.. BET_PASS=.. python tools/bet_history_inspect.py
"""
from __future__ import annotations
import json, os, re, sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "exports" / "bet_history_probe.json"
OUT.parent.mkdir(parents=True, exist_ok=True)
SESSION = ROOT / ".bet_session"
USER, PWD = os.environ.get("BET_USER", ""), os.environ.get("BET_PASS", "")
xhr, tokens = [], set()


def click_txt(page, pats, timeout=2500):
    for p in pats:
        for meth in (lambda: page.get_by_role("button", name=re.compile(p, re.I)).first,
                     lambda: page.get_by_role("link", name=re.compile(p, re.I)).first,
                     lambda: page.get_by_text(re.compile(p, re.I)).first):
            try:
                el = meth()
                if el.is_visible(timeout=timeout):
                    el.click(timeout=timeout); return p
            except Exception:
                continue
    return None


def main():
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            str(SESSION), headless=False, args=["--disable-blink-features=AutomationControlled"],
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"),
            viewport={"width": 1300, "height": 850})
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto("https://bet261.mg/", timeout=45000, wait_until="domcontentloaded")
        time.sleep(4)
        # login si besoin
        try:
            need = page.get_by_role("button", name=re.compile("Se connecter", re.I)).first.is_visible(timeout=2500)
        except Exception:
            need = False
        if need and USER:
            click_txt(page, ["OK", "Oui"])
            try:
                page.locator('#Login_Id, input[name="username"]').first.fill(USER, timeout=2500)
                page.locator('#Login_Password, input[type="password"]').first.fill(PWD, timeout=2500)
                page.get_by_role("button", name=re.compile("Se connecter", re.I)).first.click(timeout=2500)
                time.sleep(10)
            except Exception as e:
                print("login:", e)

        # token complet -> fichier réutilisable (gitignored)
        full_tok = [None]

        def on_resp(resp):
            u = resp.url.lower()
            auth = resp.request.headers.get("authorization")
            if auth and len(auth) > 20 and not full_tok[0]:
                full_tok[0] = auth
                try: (SESSION / "token.txt").write_text(auth, encoding="utf-8")
                except Exception: pass
            if "sporty-tech" in u and resp.request.resource_type in ("xhr", "fetch"):
                if auth: tokens.add(auth[:30])
                rec = {"url": resp.url, "status": resp.status, "method": resp.request.method}
                try:
                    if "json" in resp.headers.get("content-type", ""):
                        body = resp.json()
                        rec["sample"] = str(body)[:1600]
                        rec["is_list"] = isinstance(body, list) or (
                            isinstance(body, dict) and any(isinstance(v, list) and v for v in body.values()))
                except Exception:
                    pass
                xhr.append(rec)
        page.on("response", on_resp)

        print(">>> navigation vers /myaccount/history + clic 'Tout'…", flush=True)
        page.goto("https://bet261.mg/myaccount/history", timeout=45000, wait_until="domcontentloaded")
        time.sleep(5)
        click_txt(page, ["Tout"])
        time.sleep(4)
        # scrolle pour charger plus + tente 'Simulés' (les virtuels) si présent
        click_txt(page, ["Simulés", "Simulé"])
        time.sleep(3)
        click_txt(page, ["Tout"])
        for _ in range(6):
            page.mouse.wheel(0, 3000); time.sleep(1.5)
        time.sleep(3)
        ctx.close()

    data = {"n_xhr": len(xhr), "auth_tokens_seen": list(tokens), "xhr": xhr}
    OUT.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n=== ENDPOINTS candidats (bet/coupon/history/customer) : {len(xhr)} ===")
    for r in xhr:
        print(f"  {r['status']} {r['method']} {r['url'][:110]}")
        if r.get("sample"):
            print(f"     -> {r['sample'][:180]}")
    print(f"\ntoken d'auth vu : {'oui' if tokens else 'non'}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
