# -*- coding: utf-8 -*-
"""WF5 — verifications complementaires de l'audit stationnarite. READ-ONLY.
1) Couverture : combien d'events finis 8035 ont >=1 snapshot d'ouverture (et par jour) ?
2) E2 variantes : home-fav only / away-fav only / bornes [1.10,1.20] vs (1.10,1.20),
   + serie temporelle en 8 tranches egales pour voir la derive.
Sortie: exports/wf5_stationarity_checks.json
"""
import sys, json, math, collections
sys.path.insert(0, ".")
from scraper.config import load_settings
from sqlalchemy import create_engine, text

LG = "InstantLeague-8035"
e = create_engine(load_settings().db_url)
_corr = json.load(open("exports/corrupted_events.json", encoding="utf-8"))
CORRUPTED = set(int(k) for k in _corr["events"].keys())

out = {}

with e.connect() as c:
    # 1) coverage
    cov = c.execute(text("""
        SELECT substr(e.expected_start,1,10) d,
               COUNT(*) total,
               SUM(CASE WHEN EXISTS (SELECT 1 FROM odds_snapshots o WHERE o.event_id=e.id)
                   THEN 1 ELSE 0 END) with_odds
        FROM events e JOIN results r ON r.event_id = e.id
        WHERE e.competition = :lg
        GROUP BY d ORDER BY d
    """), {"lg": LG}).fetchall()
    out["coverage_by_day"] = [dict(day=r[0], finished=r[1], with_odds=r[2]) for r in cov]
    print("COVERAGE finished vs with_odds:")
    for r in cov:
        print("  %s  %5d / %5d" % (r[0], r[2], r[1]))

    # reload light 1x2 rows
    rows = c.execute(text("""
        SELECT e.id, e.expected_start, r.score_a, r.score_b,
               o.odds_home, o.odds_draw, o.odds_away
        FROM events e
        JOIN results r ON r.event_id = e.id
        JOIN odds_snapshots o ON o.event_id = e.id
        WHERE e.competition = :lg
          AND o.id = (SELECT MIN(o2.id) FROM odds_snapshots o2 WHERE o2.event_id = e.id)
    """), {"lg": LG}).fetchall()

base = []
for (eid, xs, sa, sb, oh, od, oa) in rows:
    if eid in CORRUPTED or None in (oh, od, oa, sa, sb):
        continue
    base.append(dict(xs=str(xs), sa=int(sa), sb=int(sb),
                     oh=float(oh), od=float(od), oa=float(oa)))
base.sort(key=lambda r: r["xs"])

def evaluate(bets):
    n = len(bets)
    if n == 0:
        return dict(n=0)
    wins = sum(1 for w, _ in bets if w)
    profit = sum((o - 1.0) if w else -1.0 for w, o in bets)
    odds_sum = sum(o for _, o in bets)
    return dict(n=n, wins=wins, wr=round(wins / n, 4), roi_pct=round(100 * profit / n, 2),
                avg_odds=round(odds_sum / n, 3))

def e2_sel(rows, side):
    sel = []
    for r in rows:
        fav_home = r["oh"] <= r["oa"]
        if side == "home" and not fav_home:
            continue
        if side == "away" and fav_home:
            continue
        fc = r["oh"] if fav_home else r["oa"]
        if 1.10 <= fc <= 1.20:
            won = (r["sa"] > r["sb"]) if fav_home else (r["sb"] > r["sa"])
            sel.append((won, fc))
    return sel

out["E2_variants"] = {
    "both_sides": evaluate(e2_sel(base, "both")),
    "home_fav_only": evaluate(e2_sel(base, "home")),
    "away_fav_only": evaluate(e2_sel(base, "away")),
}
print("\nE2 variants:")
for k, v in out["E2_variants"].items():
    print("  %-15s %s" % (k, v))

# E2 in 8 equal chronological slices (both sides)
allsel = []
for r in base:
    fav_home = r["oh"] <= r["oa"]
    fc = r["oh"] if fav_home else r["oa"]
    if 1.10 <= fc <= 1.20:
        won = (r["sa"] > r["sb"]) if fav_home else (r["sb"] > r["sa"])
        allsel.append((r["xs"], won, fc))
k = 8
slices = []
n = len(allsel)
for i in range(k):
    part = allsel[i * n // k:(i + 1) * n // k]
    ev = evaluate([(w, o) for _, w, o in part])
    ev["from"] = part[0][0][:16] if part else None
    ev["to"] = part[-1][0][:16] if part else None
    slices.append(ev)
out["E2_8slices"] = slices
print("\nE2 chronological slices (both sides):")
for s in slices:
    print("  %s -> %s  n=%3d wr=%.3f roi=%+7.2f%%" % (s["from"], s["to"], s["n"], s["wr"], s["roi_pct"]))

# Same for E1-population favorites <=1.30 (T1 proxy) in 8 slices
t1sel = []
for r in base:
    fav_home = r["oh"] <= r["oa"]
    fc = r["oh"] if fav_home else r["oa"]
    if fc <= 1.30:
        won = (r["sa"] > r["sb"]) if fav_home else (r["sb"] > r["sa"])
        t1sel.append((r["xs"], won, fc))
slices_t1 = []
n = len(t1sel)
for i in range(k):
    part = t1sel[i * n // k:(i + 1) * n // k]
    ev = evaluate([(w, o) for _, w, o in part])
    ev["from"] = part[0][0][:16] if part else None
    ev["to"] = part[-1][0][:16] if part else None
    slices_t1.append(ev)
out["T1proxy_8slices"] = slices_t1
print("\nT1 proxy fav<=1.30 chronological slices:")
for s in slices_t1:
    print("  %s -> %s  n=%3d wr=%.3f roi=%+7.2f%%" % (s["from"], s["to"], s["n"], s["wr"], s["roi_pct"]))

json.dump(out, open("exports/wf5_stationarity_checks.json", "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
