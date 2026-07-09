"""VÉRIFICATION ADVERSE du 'cycle seedé' under -> under.
Le +6pp vs base global est-il un VRAI edge, ou le marché price-t-il déjà ces
matchs défensifs ? On compare le taux réel au taux IMPLICITE (dévig Total de buts)
des MÊMES matchs. + split OOS + ROI aux vraies cotes + contrôle par permutation.
"""
import json, sqlite3, random
from math import erf, sqrt

c = sqlite3.connect("data/virtual_sports.db", timeout=60)
rows = c.execute("""
    SELECT e.expected_start ts, e.external_id xid, o.extra_markets xm, r.score_a sa, r.score_b sb
    FROM events e JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
    JOIN results r ON r.event_id=e.id
    WHERE e.competition='InstantLeague-8035' AND r.score_a IS NOT NULL""").fetchall()


def gm(xm, pref):
    for k, v in (xm or {}).items():
        if k.replace("\x82", "é").replace("\xe9", "é").startswith(pref):
            return v
    return None


seq = []
for ts, xid, xm, sa, sb in rows:
    try: xi = int(xid)
    except Exception: xi = 0
    try: j = json.loads(xm) if isinstance(xm, str) else (xm or {})
    except Exception: j = {}
    tt = gm(j, "Total de buts")
    v = {k: 1/tt[k] for k in [str(x) for x in range(7)]
         if tt and isinstance(tt.get(k), (int, float)) and 1 < tt[k] < 99.99} if tt else {}
    s = sum(v.values())
    imp_u = (v["0"]+v["1"]+v["2"])/s if s and len(v) == 7 else None  # implicite under 2.5
    if imp_u is None:
        continue
    under = int(sa+sb <= 2)
    seq.append((ts, xi, under, imp_u))
seq.sort(key=lambda x: (x[0], x[1]))
N = len(seq)
under = [s[2] for s in seq]; impu = [s[3] for s in seq]
print(f"{N} matchs avec marché total", flush=True)


def rate(a): return sum(a)/len(a) if a else 0
def z_(diffs):
    n = len(diffs)
    if n < 30: return float("nan"), n
    m = sum(diffs)/n; sd = (sum((d-m)**2 for d in diffs)/n)**0.5
    return (m/(sd/sqrt(n)) if sd else 0), n
def pval(z): return 2*(1-0.5*(1+erf(abs(z)/1.4142))) if z == z else 1


cut = N//2
print(f"\n{'k':>2}{'n':>7}{'réel':>8}{'implicite':>11}{'EDGE(réel-imp)':>16}{'z':>7}{'p':>7}{'EDGE OOS':>10}")
for k in (2, 3, 4, 5):
    idx = [i for i in range(k, N) if all(under[i-j-1] for j in range(k))]
    if len(idx) < 30: continue
    real = rate([under[i] for i in idx]); imp = rate([impu[i] for i in idx])
    diffs = [under[i]-impu[i] for i in idx]
    z, n = z_(diffs)
    idx_te = [i for i in idx if i >= cut]
    edge_te = rate([under[i]-impu[i] for i in idx_te]) if len(idx_te) > 30 else float("nan")
    print(f"{k:>2}{len(idx):>7}{100*real:>7.1f}%{100*imp:>10.1f}%{100*(real-imp):>+15.2f}{z:>7.2f}{pval(z):>7.2f}{100*edge_te:>+9.2f}")

# contrôle : mêmes 'streaks' sur une séquence PERMUTÉE (aucune structure réelle)
random.seed(0); perm = under[:]; random.shuffle(perm)
print("\n  [contrôle permutation] après 3 'unders' sur séquence MÉLANGÉE :")
idx = [i for i in range(3, N) if all(perm[i-j-1] for j in range(3))]
if idx:
    print(f"    P(under) = {100*rate([perm[i] for i in idx]):.1f}% (base {100*rate(under):.1f}%) "
          f"-> {'~base = normal' if abs(rate([perm[i] for i in idx])-rate(under)) < 0.02 else 'anomalie'}")

print("\n" + "="*62)
print("  LECTURE : si EDGE(réel-imp) ≈ 0 -> le marché price DÉJÀ le under plus")
print("  élevé de ces matchs défensifs. Le +6pp vs base global était un MIRAGE")
print("  (on sélectionnait des matchs défensifs, pas un cycle). Aucun edge.")
print("  Si EDGE>0 stable + OOS>0 -> vrai signal à creuser (adverse round 2).")
print("="*62)
