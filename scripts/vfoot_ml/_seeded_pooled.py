"""THÉORIE DU CYCLE SEEDÉ, test DÉCISIF sur les 9 LIGUES (puissance max).
Après k unders-2.5 consécutifs dans une ligue -> le match suivant :
  taux under réel vs implicite (dévig Total de buts) + ROI au pari réel
  'Multi-Buts 0,1,2' (= under 2.5) + split OOS + bootstrap IC95.
Verdict : edge réel seulement si ROI_OOS>0 ET IC95 au-dessus de 0.
"""
import json, sqlite3
from math import sqrt

random_state = 12345
c = sqlite3.connect("data/virtual_sports.db", timeout=60)
rows = c.execute("""
    SELECT e.competition comp, e.expected_start ts, e.external_id xid, o.extra_markets xm,
           r.score_a sa, r.score_b sb
    FROM events e JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
    JOIN results r ON r.event_id=e.id
    WHERE e.competition LIKE 'InstantLeague-%' AND r.score_a IS NOT NULL""").fetchall()


def gm(xm, pref):
    for k, v in (xm or {}).items():
        if k.replace("\x82", "é").replace("\xe9", "é").startswith(pref):
            return v
    return None


# construit par ligue la séquence chrono (under?, implicite under2.5, cote under2.5 offerte)
seqs = {}
for comp, ts, xid, xm, sa, sb in rows:
    try:
        j = json.loads(xm) if isinstance(xm, str) else (xm or {})
        xi = int(xid)
    except Exception:
        continue
    tot = sa + sb
    tt = gm(j, "Total de buts")
    imp = None
    if isinstance(tt, dict):
        v = {k: 1/tt[k] for k in [str(x) for x in range(7)]
             if isinstance(tt.get(k), (int, float)) and 1 < tt[k] < 99.99}
        s = sum(v.values())
        if s and len(v) == 7:
            imp = (v["0"]+v["1"]+v["2"])/s          # implicite under 2.5
    mb = gm(j, "Multi-Buts")
    o_u25 = None
    if isinstance(mb, dict):
        for k, val in mb.items():
            if "0, 1 ou 2" in k and isinstance(val, (int, float)) and 1 < val < 99.99:
                o_u25 = val
    if imp is None:
        continue
    seqs.setdefault(comp, []).append((xi, int(tot <= 2), imp, o_u25))

# après k unders consécutifs -> le suivant
K = 5
ev = []      # (under_next, imp_next, odds_u25_next, is_test)
for comp, arr in seqs.items():
    arr.sort()
    half = arr[len(arr)//2][0]
    run = 0
    for xi, u, imp, o in arr:
        if run >= K:
            ev.append((u, imp, o, xi >= half))
        run = run + 1 if u == 1 else 0
n = len(ev)
print(f"9 ligues | {sum(len(v) for v in seqs.values())} matchs | "
      f"{n} cas 'match APRÈS {K} unders consécutifs'", flush=True)
if n < 100:
    print("pas assez de cas."); raise SystemExit

real = sum(e[0] for e in ev)/n
imp = sum(e[1] for e in ev)/n
te = [e for e in ev if e[3]]
real_te = sum(e[0] for e in te)/len(te)
imp_te = sum(e[1] for e in te)/len(te)
print(f"\n  GLOBAL : under réel {100*real:.1f}% vs implicite {100*imp:.1f}%  EDGE {100*(real-imp):+.2f}pp")
print(f"  OOS    : under réel {100*real_te:.1f}% vs implicite {100*imp_te:.1f}%  EDGE {100*(real_te-imp_te):+.2f}pp (n={len(te)})")

# ROI au pari réel Multi-Buts '0,1,2' (under 2.5), OOS
bets = [(e[0], e[2]) for e in te if e[2]]
if bets:
    pnl = [u*o - 1 for u, o in bets]
    roi = sum(pnl)/len(pnl)
    # bootstrap IC95 (déterministe, sans numpy)
    import random
    random.seed(random_state)
    boots = []
    for _ in range(2000):
        s = sum(pnl[random.randrange(len(pnl))] for _ in range(len(pnl)))/len(pnl)
        boots.append(s)
    boots.sort()
    lo, hi = boots[int(0.025*len(boots))], boots[int(0.975*len(boots))]
    print(f"\n  ROI OOS (pari under 2.5 réel, n={len(bets)}) : {100*roi:+.1f}%  "
          f"IC95 [{100*lo:+.1f}% ; {100*hi:+.1f}%]")
    print("  " + ("🚨 IC95 AU-DESSUS DE 0 -> vrai edge à vérifier !!" if lo > 0
                  else "IC95 contient 0 -> PAS un edge confirmé (bruit/marge)"))
print("="*60)
