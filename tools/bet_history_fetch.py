"""Récupère TOUT l'historique de paris virtuels via l'API reporting (token d'auth)."""
from __future__ import annotations
import gzip, json, urllib.request, urllib.error, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOK = ROOT / ".bet_session" / "token.txt"
OUT = ROOT / "exports" / "my_bets.json"
if not TOK.exists():
    print("Pas de token — relance bet_history_inspect.py."); raise SystemExit
token = TOK.read_text(encoding="utf-8").strip()
H = {"Authorization": token if token.lower().startswith("bearer") else f"Bearer {token}",
     "Accept": "application/json", "Accept-Encoding": "gzip",
     "Origin": "https://bet261.mg", "Referer": "https://bet261.mg/",
     "User-Agent": "Mozilla/5.0 Chrome/125.0"}
BASE = "https://hg-customer-api-prod.sporty-tech.net/api/reporting"


def get(url):
    try:
        r = urllib.request.urlopen(urllib.request.Request(url, headers=H), timeout=20)
        raw = r.read()
        if r.headers.get("Content-Encoding") == "gzip": raw = gzip.decompress(raw)
        return r.status, json.loads(raw.decode("utf-8", "ignore"))
    except urllib.error.HTTPError as e:
        return e.code, None
    except Exception as e:
        return None, str(e)[:50]


def as_list(body):
    if isinstance(body, list): return body
    if isinstance(body, dict):
        for k in ("items", "data", "bets", "results", "history"):
            if isinstance(body.get(k), list): return body[k]
    return []


# 1) trouver le bon jeu de params (endpoint x betState)
print("--- recherche du bon filtre ---", flush=True)
combos = []
for ep in ("simulated-bets-history", "history"):
    for st in ("", "&betState=Settled", "&betState=All", "&betState=Won", "&betState=Lost", "&betState=Cashout"):
        code, body = get(f"{BASE}/{ep}?skip=0&take=10{st}")
        lst = as_list(body) if code == 200 else []
        print(f"  {ep}{st or ' (défaut)'}: {code} -> {len(lst)} paris")
        if lst:
            combos.append((ep, st))

if not combos:
    print("\nAucun pari renvoyé. Token expiré ? relance l'inspecteur (reste connecté).")
    raise SystemExit

# 2) tout paginer
allbets = []
for ep, st in combos:
    seen = 0
    for skip in range(0, 5000, 50):
        code, body = get(f"{BASE}/{ep}?skip={skip}&take=50{st}")
        lst = as_list(body) if code == 200 else []
        if not lst: break
        allbets.extend(lst); seen += len(lst)
        time.sleep(0.2)
    print(f"  {ep}{st}: {seen} paris récupérés", flush=True)

# dédup par référence/id
def key(b): return b.get("reference") or b.get("id") or b.get("betId") or str(b)[:60]
uniq = {key(b): b for b in allbets}
OUT.write_text(json.dumps(list(uniq.values()), indent=1, ensure_ascii=False), encoding="utf-8")
print(f"\n>>> {len(uniq)} paris uniques -> {OUT}")
if uniq:
    print("champs d'un pari :", sorted(list(uniq.values())[0].keys()))
