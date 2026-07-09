"""THÉORIE 3-4-3 — la position du match dans le round change-t-elle les résultats ?
Hypothèse entendue : pos 1-3 et 8-10 = plus de NULS ; pos 4-7 (milieu) = plus de
BUTS et d'OUTSIDERS. Position déduite de l'ordre external_id dans chaque round de 10.
Python pur (pas de pandas -> évite l'OOM). Chi2 manuel + split OOS.
"""
import sqlite3
from math import sqrt

DB = "data/virtual_sports.db"
c = sqlite3.connect(DB, timeout=60)
rows = c.execute("""
    SELECT e.expected_start ts, e.external_id xid, o.odds_home oh, o.odds_draw od, o.odds_away oa,
           r.score_a sa, r.score_b sb
    FROM events e JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
    JOIN results r ON r.event_id=e.id
    WHERE e.competition='InstantLeague-8035' AND r.score_a IS NOT NULL
      AND o.odds_home>1 AND o.odds_draw>1 AND o.odds_away>1""").fetchall()

# grouper par round, garder ceux de 10 matchs, ordonner par external_id -> position 1..10
rounds = {}
for ts, xid, oh, od, oa, sa, sb in rows:
    try:
        xi = int(xid)
    except Exception:
        continue
    rounds.setdefault(ts, []).append((xi, oh, od, oa, sa, sb))
data = []  # (position, ts, oh,od,oa,sa,sb)
for ts, ms in rounds.items():
    if len(ms) != 10:
        continue
    for pos, (xi, oh, od, oa, sa, sb) in enumerate(sorted(ms), start=1):
        data.append((pos, ts, oh, od, oa, sa, sb))
print(f"{len(data)} matchs dans {len(data)//10} rounds complets de 10", flush=True)

# split chrono (par ts)
tss = sorted(set(d[1] for d in data))
cut = tss[len(tss)//2]


def stats_for(subset):
    per = {p: {"n": 0, "draw": 0, "goals": 0, "ov25": 0, "upset": 0} for p in range(1, 11)}
    for pos, ts, oh, od, oa, sa, sb in subset:
        s = per[pos]; s["n"] += 1
        s["draw"] += int(sa == sb)
        tot = sa + sb; s["goals"] += tot; s["ov25"] += int(tot > 2.5)
        fav_home = oh <= oa
        fav_won = (sa > sb) if fav_home else (sb > sa)
        s["upset"] += int(not fav_won)          # favori ne gagne pas
    return per


def show(per, title):
    print(f"\n=== {title} ===")
    print(f"  {'pos':>3}{'n':>7}{'%nul':>8}{'buts/m':>9}{'%over2.5':>10}{'%upset':>9}")
    for p in range(1, 11):
        s = per[p]; n = s["n"] or 1
        print(f"  {p:>3}{s['n']:>7}{100*s['draw']/n:>7.1f}%{s['goals']/n:>9.2f}{100*s['ov25']/n:>9.1f}%{100*s['upset']/n:>8.1f}%")
    # groupes 3-4-3
    def agg(ps):
        t = {"n": 0, "draw": 0, "goals": 0, "ov25": 0, "upset": 0}
        for p in ps:
            for k in t: t[k] += per[p][k]
        return t
    g1, g2, g3 = agg([1, 2, 3]), agg([4, 5, 6, 7]), agg([8, 9, 10])
    print("  --- groupes 3-4-3 ---")
    for lbl, g in (("1-3 (début)", g1), ("4-7 (milieu)", g2), ("8-10 (fin)", g3)):
        n = g["n"] or 1
        print(f"    {lbl:<13}: %nul {100*g['draw']/n:4.1f} | buts/m {g['goals']/n:.2f} | "
              f"over2.5 {100*g['ov25']/n:4.1f}% | upset {100*g['upset']/n:4.1f}%")
    # chi2 des nuls sur les 3 groupes
    tot_n = g1["n"]+g2["n"]+g3["n"]; tot_d = g1["draw"]+g2["draw"]+g3["draw"]
    base = tot_d/tot_n; chi = 0.0
    for g in (g1, g2, g3):
        exp = base*g["n"]
        chi += (g["draw"]-exp)**2/exp + ((g["n"]-g["draw"])-(g["n"]-exp))**2/(g["n"]-exp)
    print(f"    chi2 nuls (3 groupes, ddl=2) = {chi:.2f}  (seuil 5% = 5.99 -> "
          f"{'SIGNIFICATIF' if chi > 5.99 else 'non signif = aucun effet position'})")
    return g1, g2, g3


tr = [d for d in data if d[1] < cut]; te = [d for d in data if d[1] >= cut]
show(stats_for(tr), "TRAIN (moitié 1)")
show(stats_for(te), "TEST (moitié 2 — validation)")
print("\n" + "="*60)
print("  Théorie 3-4-3 confirmée SEULEMENT si : nuls plus hauts en 1-3 & 8-10,")
print("  buts/upsets plus hauts en 4-7, chi2>5.99, ET ça se REPRODUIT en TEST.")
print("  Sinon = superstition (position = pur affichage, résultats i.i.d.).")
print("="*60)
