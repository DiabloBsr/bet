"""MES PROPRES THÉORIES sur la structure du round (Python pur, anti-OOM).
T1: le RNG PLACE-t-il les gros favoris à certaines positions ? (force du favori/pos)
T2: positions PAIRES vs IMPAIRES.
T3: 1er match vs DERNIER match vs milieu (extrêmes).
T4: avantage DOMICILE par position.
"""
import sqlite3

c = sqlite3.connect("data/virtual_sports.db", timeout=60)
rows = c.execute("""
    SELECT e.expected_start ts, e.external_id xid, o.odds_home oh, o.odds_draw od, o.odds_away oa,
           r.score_a sa, r.score_b sb
    FROM events e JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
    JOIN results r ON r.event_id=e.id
    WHERE e.competition='InstantLeague-8035' AND r.score_a IS NOT NULL
      AND o.odds_home>1 AND o.odds_draw>1 AND o.odds_away>1""").fetchall()
rounds = {}
for ts, xid, oh, od, oa, sa, sb in rows:
    try: xi = int(xid)
    except Exception: continue
    rounds.setdefault(ts, []).append((xi, oh, od, oa, sa, sb))
data = []
for ts, ms in rounds.items():
    if len(ms) != 10: continue
    for pos, (xi, oh, od, oa, sa, sb) in enumerate(sorted(ms), 1):
        inv = 1/oh + 1/od + 1/oa
        favp = max((1/oh)/inv, (1/oa)/inv)          # proba du favori
        data.append((pos, oh, od, oa, sa, sb, favp))
print(f"{len(data)} matchs, {len(data)//10} rounds", flush=True)


def mean(xs): return sum(xs)/len(xs) if xs else 0
def sd(xs):
    m = mean(xs); return (sum((x-m)**2 for x in xs)/len(xs))**0.5 if xs else 0


# T1: force du favori par position (le RNG place-t-il les gros favoris ?)
print("\n=== T1 : force du FAVORI par position (scheduling ?) ===")
byp = {p: [] for p in range(1, 11)}
for pos, oh, od, oa, sa, sb, favp in data:
    byp[pos].append(favp)
fps = [mean(byp[p]) for p in range(1, 11)]
for p in range(1, 11):
    print(f"  pos {p:>2}: favori moyen {100*mean(byp[p]):.1f}%")
print(f"  écart max entre positions : {100*(max(fps)-min(fps)):.2f}pp | "
      f"écart-type inter-position {100*sd(fps):.2f}pp (SE~{100*sd(byp[1])/len(byp[1])**.5:.2f}) -> "
      f"{'STRUCTURE' if sd(fps) > 3*sd(byp[1])/len(byp[1])**.5 else 'PLAT = placement aléatoire'}")

# T2: pair vs impair
print("\n=== T2 : positions PAIRES vs IMPAIRES ===")
for lbl, ps in (("impaires", [1, 3, 5, 7, 9]), ("paires", [2, 4, 6, 8, 10])):
    sub = [d for d in data if d[0] in ps]
    dr = mean([int(d[4] == d[5]) for d in sub]); g = mean([d[4]+d[5] for d in sub])
    print(f"  {lbl:<9}: %nul {100*dr:.1f} | buts/m {g:.2f} | n={len(sub)}")

# T3: 1er vs dernier vs milieu
print("\n=== T3 : 1er (pos1) vs DERNIER (pos10) vs milieu (4-7) ===")
for lbl, ps in (("1er", [1]), ("dernier", [10]), ("milieu 4-7", [4, 5, 6, 7])):
    sub = [d for d in data if d[0] in ps]
    dr = mean([int(d[4] == d[5]) for d in sub]); g = mean([d[4]+d[5] for d in sub])
    up = mean([int(not ((d[4] > d[5]) if d[1] <= d[3] else (d[5] > d[4]))) for d in sub])
    print(f"  {lbl:<11}: %nul {100*dr:.1f} | buts/m {g:.2f} | %upset {100*up:.1f} | n={len(sub)}")

# T4: avantage domicile par position
print("\n=== T4 : victoire DOMICILE par position ===")
for p in range(1, 11):
    sub = [d for d in data if d[0] == p]
    hw = mean([int(d[4] > d[5]) for d in sub])
    print(f"  pos {p:>2}: domicile gagne {100*hw:.1f}%", end="  ")
    if p % 2 == 0: print()
print("\n" + "="*58)
print("  Si tout est plat (~même valeur ± bruit) sur les 4 tests -> la position")
print("  du match est PUREMENT cosmétique : aucune théorie de round n'existe.")
print("="*58)
