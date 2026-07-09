"""AUTRES THÉORIES EN LIGNE, testées (pooled 9 ligues, Python pur).
A: 'ÉQUILIBRAGE' — après k OVERS consécutifs, le under devient-il plus probable ?
B: PAIR/IMPAIR — le nb de buts alterne-t-il (autocorrélation de la parité) ?
C: BTTS 'safe' — G/NG est-il bien calibré (le taux réel = l'implicite) ?
"""
import json, sqlite3
from math import sqrt

c = sqlite3.connect("data/virtual_sports.db", timeout=60)
rows = c.execute("""
    SELECT e.competition comp, e.external_id xid, o.extra_markets xm, r.score_a sa, r.score_b sb
    FROM events e JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
    JOIN results r ON r.event_id=e.id
    WHERE e.competition LIKE 'InstantLeague-%' AND r.score_a IS NOT NULL""").fetchall()


def gm(xm, pref):
    for k, v in (xm or {}).items():
        if k.replace("\x82", "é").replace("\xe9", "é").startswith(pref):
            return v
    return None


seqs = {}
for comp, xid, xm, sa, sb in rows:
    try:
        j = json.loads(xm) if isinstance(xm, str) else (xm or {})
        xi = int(xid)
    except Exception:
        continue
    tot = sa + sb
    tt = gm(j, "Total de buts")
    imp_u = None
    if isinstance(tt, dict):
        v = {k: 1/tt[k] for k in [str(x) for x in range(7)]
             if isinstance(tt.get(k), (int, float)) and 1 < tt[k] < 99.99}
        s = sum(v.values())
        if s and len(v) == 7:
            imp_u = (v["0"]+v["1"]+v["2"])/s
    gg = gm(j, "G/NG"); imp_b = None
    if isinstance(gg, dict) and all(isinstance(gg.get(x), (int, float)) for x in ("Oui", "Non")):
        imp_b = (1/gg["Oui"])/((1/gg["Oui"])+(1/gg["Non"]))
    btts = int(sa > 0 and sb > 0)
    seqs.setdefault(comp, []).append((xi, tot, int(tot <= 2), int(tot % 2 == 1), imp_u, btts, imp_b))
for comp in seqs: seqs[comp].sort()
print(f"9 ligues | {sum(len(v) for v in seqs.values())} matchs", flush=True)

# ===== A : après k OVERS -> under plus probable (équilibrage) ? =====
print("\n=== A. ÉQUILIBRAGE : après k OVERS consécutifs -> under plus probable ? ===")
print(f"  {'k overs':>8}{'n':>7}{'under réel':>12}{'implicite':>11}{'EDGE':>8}")
for K in (2, 3, 4, 5):
    n = du = di = 0
    run = 0
    for comp, arr in seqs.items():
        run = 0
        for xi, tot, u, odd, imp_u, btts, imp_b in arr:
            over = 1 - u if tot != 2 else 0        # over2.5 ? (tot>2)
            over = int(tot > 2)
            if run >= K and imp_u is not None:
                n += 1; du += u; di += imp_u
            run = run + 1 if over else 0
    if n >= 100:
        print(f"  {K:>8}{n:>7}{100*du/n:>11.1f}%{100*di/n:>10.1f}%{100*(du/n-di/n):>+7.2f}pp")
print("  EDGE≈0 -> pas d'équilibrage (le under n'est pas 'dû' après des overs).")

# ===== B : PAIR/IMPAIR — alternance de la parité des buts ? =====
print("\n=== B. PAIR/IMPAIR : la parité du total alterne-t-elle ? ===")
n = same = 0
p_odd_after_odd = [0, 0]; p_odd_after_even = [0, 0]
for comp, arr in seqs.items():
    prev = None
    for xi, tot, u, odd, imp_u, btts, imp_b in arr:
        if prev is not None:
            n += 1; same += int(odd == prev)
            if prev == 1: p_odd_after_odd[0] += 1; p_odd_after_odd[1] += odd
            else: p_odd_after_even[0] += 1; p_odd_after_even[1] += odd
        prev = odd
base_odd = sum(a[3] for a in [x for v in seqs.values() for x in v]) / sum(len(v) for v in seqs.values())
print(f"  base P(impair) = {100*base_odd:.1f}%")
print(f"  P(impair | précédent IMPAIR) = {100*p_odd_after_odd[1]/p_odd_after_odd[0]:.1f}%")
print(f"  P(impair | précédent PAIR)   = {100*p_odd_after_even[1]/p_odd_after_even[0]:.1f}%")
print(f"  taux de RÉPÉTITION de parité = {100*same/n:.1f}% (50% = aucune alternance) -> "
      f"{'ALTERNANCE/RÉPÉTITION' if abs(same/n-0.5) > 0.03 else 'aucun pattern (parité i.i.d.)'}")

# ===== C : BTTS calibré (safe ?) =====
print("\n=== C. BTTS 'safe' : G/NG est-il bien calibré ? ===")
allm = [x for v in seqs.values() for x in v if x[6] is not None]
real = sum(x[5] for x in allm)/len(allm); imp = sum(x[6] for x in allm)/len(allm)
print(f"  BTTS réel {100*real:.1f}% vs implicite {100*imp:.1f}% (écart {100*(real-imp):+.2f}pp, n={len(allm)})")
print(f"  -> BTTS est {'CALIBRÉ' if abs(real-imp) < 0.02 else 'à regarder'} : 'plus prévisible' = "
      "juste ~55-58% de base, pas un edge (marge dessus comme partout).")
print("="*60)
