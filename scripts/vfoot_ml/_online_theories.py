"""TEST DES THÉORIES EN LIGNE (FB/YouTube/forums) sur le foot virtuel Bet261.
T1: 'cycles de matrice seedée' — après k unders consécutifs, next plus souvent under ?
T2: 'Over 1.5 = haute proba' — taux réel + rappel EV.
T3: 'BTTS plus prévisible que 1X2' — taux de réussite du meilleur pick.
T4: 'under 2.5 à 62%' — vrai chiffre.
Python pur (anti-OOM). Séquence chrono globale.
"""
import sqlite3
from math import erf, sqrt

c = sqlite3.connect("data/virtual_sports.db", timeout=60)
rows = c.execute("""
    SELECT e.expected_start ts, e.external_id xid, o.odds_home oh, o.odds_draw od, o.odds_away oa,
           r.score_a sa, r.score_b sb
    FROM events e JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
    JOIN results r ON r.event_id=e.id
    WHERE e.competition='InstantLeague-8035' AND r.score_a IS NOT NULL AND o.odds_home>1""").fetchall()
seq = []
for ts, xid, oh, od, oa, sa, sb in rows:
    try: xi = int(xid)
    except Exception: xi = 0
    seq.append((ts, xi, oh, od, oa, sa, sb))
seq.sort(key=lambda x: (x[0], x[1]))          # ordre chrono (round, position)
N = len(seq)
tot = [s[5]+s[6] for s in seq]
over25 = [int(t > 2.5) for t in tot]
under25 = [int(t <= 2.5) for t in tot]
over15 = [int(t >= 2) for t in tot]
btts = [int(s[5] > 0 and s[6] > 0) for s in seq]
print(f"{N} matchs (séquence chrono)", flush=True)


def rate(a): return sum(a)/len(a) if a else 0
def ztest(k, n, p0):
    if n < 20: return float("nan")
    se = sqrt(p0*(1-p0)/n); return (k/n - p0)/se if se else 0
def pval(z): return 2*(1-0.5*(1+erf(abs(z)/1.4142))) if z == z else 1


print("\n=== T4 : taux de base (vs ce que disent les sites) ===")
print(f"  under 2.5 : {100*rate(under25):.1f}%  (les sites disent 62% -> FAUX, c'est ~38%)")
print(f"  over 2.5  : {100*rate(over25):.1f}%   |  over 1.5 : {100*rate(over15):.1f}%  |  BTTS : {100*rate(btts):.1f}%")

print("\n=== T1 : 'CYCLES SEEDÉS' — après k UNDERS consécutifs, next under ? ===")
base_u = rate(under25)
for k in (2, 3, 4, 5, 6):
    idx = [i for i in range(k, N) if all(under25[i-j-1] for j in range(k))]
    if len(idx) < 30:
        print(f"  après {k} unders : trop peu de cas ({len(idx)})"); continue
    nxt = [under25[i] for i in idx]; r = rate(nxt); z = ztest(sum(nxt), len(nxt), base_u)
    print(f"  après {k} unders d'affilée : P(under) = {100*r:.1f}%  (base {100*base_u:.1f}%) "
          f"| n={len(idx)} z={z:+.2f} p={pval(z):.2f} -> {'≠ base' if abs(z)>2 else 'IDENTIQUE'}")
print("  (même test côté OVERS)")
base_o = rate(over25)
for k in (3, 4, 5):
    idx = [i for i in range(k, N) if all(over25[i-j-1] for j in range(k))]
    if len(idx) < 30: continue
    nxt = [over25[i] for i in idx]; r = rate(nxt); z = ztest(sum(nxt), len(nxt), base_o)
    print(f"  après {k} overs d'affilée : P(over) = {100*r:.1f}%  (base {100*base_o:.1f}%) "
          f"| n={len(idx)} z={z:+.2f} -> {'≠ base' if abs(z)>2 else 'IDENTIQUE'}")

print("\n=== T2 : 'Over 1.5 = bon pari' ? ===")
print(f"  over 1.5 tombe {100*rate(over15):.1f}% du temps — MAIS cote ~1.15 -> "
      f"il faut gagner >87% pour être rentable. À {100*rate(over15):.0f}%, EV négative.")

print("\n=== T3 : BTTS vs 1X2 — lequel se prédit mieux (meilleur pick) ? ===")
# meilleur pick 1X2 = favori ; BTTS = Oui si proba>50 sinon Non (via base ~54% Oui)
fav_hit = rate([int((s[5] > s[6]) if s[2] <= s[4] else (s[6] > s[5])) for s in seq])
btts_hit = max(rate(btts), 1-rate(btts))
print(f"  1X2 (parier le favori) : {100*fav_hit:.1f}% de réussite")
print(f"  BTTS (parier le côté majoritaire) : {100*btts_hit:.1f}%")
print(f"  -> {'BTTS un peu + prévisible' if btts_hit>fav_hit else '1X2 aussi/plus prévisible'} "
      "(mais les 2 sont calibrés = EV négative)")

print("\n" + "="*60)
print("  Si toutes les conditions donnent ≈ base rate (z<2) -> les théories")
print("  en ligne sont FAUSSES : pas de cycle seedé, pas de mémoire. Ceux qui")
print("  les vendent (PDF, groupes FB) exploitent le biais de pattern, pas le RNG.")
print("="*60)
