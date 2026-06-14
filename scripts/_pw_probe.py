"""Sonde Playwright : capture TOUS les XHR /api/instantleagues/ quand on clique les
onglets de rounds sur la page matches 8035 — pour voir si l'UI sert les rounds FUTURS
(avec cotes) et via quel endpoint exact.
Usage: ./.venv/Scripts/python.exe scripts/_pw_probe.py
"""
import sys, json
sys.path.insert(0, ".")
from playwright.sync_api import sync_playwright
from scraper.config import load_settings

s = load_settings()
URL = "https://bet261.mg/virtual/category/instant-league/8035/matches"
caps = []  # (url, n_rounds_with_matches, n_matches_total, has_odds)

def summarize(url, payload):
    try:
        rounds = payload.get("rounds", []) if isinstance(payload, dict) else []
        rwm = [r for r in rounds if r.get("matches")]
        nmt = sum(len(r.get("matches") or []) for r in rounds)
        odds = any((m.get("eventBetTypes") for r in rwm for m in r["matches"]))
        ids = [r.get("id") for r in rwm]
        return dict(url=url[-60:], rounds_with_matches=len(rwm), ids=ids, n_matches=nmt, odds=odds)
    except Exception as ex:
        return dict(url=url[-60:], err=str(ex)[:40])

all_resp = [0]
with sync_playwright() as pw:
    b = pw.chromium.launch(headless=True, args=["--disable-blink-features=AutomationControlled"])
    ctx = b.new_context(user_agent=s.user_agent, locale="fr-FR",
                        extra_http_headers={"Accept-Language": "fr-FR,fr;q=0.9"})
    pg = ctx.new_page()
    pg.set_default_timeout(45000)

    def on_resp(r):
        all_resp[0] += 1
        if "/api/instantleagues/" not in r.url:
            return
        try:
            if "json" in (r.headers.get("content-type") or "").lower():
                caps.append(summarize(r.url, r.json()))
        except Exception:
            pass
    pg.on("response", on_resp)

    print(f"navigate {URL}", flush=True)
    try:
        pg.goto(URL, wait_until="networkidle", timeout=60000)
        pg.wait_for_timeout(8000)
    except Exception as ex:
        print("goto err:", ex, flush=True)
    # diagnostic blocage / SPA
    try:
        print(f"final_url={pg.url}", flush=True)
        print(f"title={pg.title()!r}", flush=True)
        body = pg.evaluate("document.body ? document.body.innerText.slice(0,200) : 'NO BODY'")
        print(f"body[:200]={body!r}", flush=True)
        print(f"total réponses réseau (tous domaines)={all_resp[0]}", flush=True)
    except Exception as ex:
        print("diag err:", ex, flush=True)
    print(f"XHR capturés au load : {len(caps)}", flush=True)
    for c in caps[-8:]:
        print("  load:", json.dumps(c, ensure_ascii=False), flush=True)

    # retirer overlays + cliquer chaque onglet HH:MM
    try:
        pg.evaluate("document.querySelectorAll('hg-privacy,[class*=privacy],[class*=cookie]').forEach(e=>e.remove())")
    except Exception:
        pass
    pg.wait_for_timeout(500)
    tabs = pg.query_selector_all("text=/^\\d{2}:\\d{2}$/")
    print(f"\nonglets de round trouvés : {len(tabs)}", flush=True)
    before = len(caps)
    for i, t in enumerate(tabs[:14]):
        try:
            label = t.inner_text()
            t.click(force=True, timeout=4000)
            pg.wait_for_timeout(1800)
            new = caps[before:]
            print(f"  clic onglet '{label}' -> {len(new)} nouveaux XHR; dernier: {json.dumps(new[-1],ensure_ascii=False) if new else '-'}", flush=True)
            before = len(caps)
        except Exception as ex:
            print(f"  clic onglet {i} échec: {str(ex)[:50]}", flush=True)
    ctx.close(); b.close()

# bilan : combien de rounds DISTINCTS avec cotes captés
allids = set()
for c in caps:
    for rid in c.get("ids", []) or []:
        allids.add(rid)
print(f"\n=== BILAN : {len(caps)} XHR, rounds distincts avec matchs : {sorted(allids)} ===", flush=True)
