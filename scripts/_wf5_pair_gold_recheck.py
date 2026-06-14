# -*- coding: utf-8 -*-
"""WF5 recheck — contre-verification independante du verdict PAIR_HOME_GOLD.

Checks independants du script audite:
 1. Orientation settlement: WR home par bucket de cote (les favoris doivent gagner plus).
 2. Fair-implied (marge retiree proportionnellement) vs WR OOS des paires GOLD.
 3. Sensibilite cutoff: 2026-06-05 12:34 (mtime fichier), 2026-06-06, 2026-06-07.
 4. Verification doublons (un seul result/event, un seul opening snapshot).
Sortie: exports/wf5_pair_gold_recheck.json
"""
import sys, json, math
sys.path.insert(0, ".")
from scraper.config import load_settings
from sqlalchemy import create_engine, text
from scraper.team_gold_data import PAIR_HOME_GOLD

LEAGUE = "InstantLeague-8035"
engine = create_engine(load_settings().db_url)
corrupted = set(int(k) for k in json.load(open("exports/corrupted_events.json"))["events"].keys())

SQL = """
SELECT e.id, e.team_a, e.team_b, r.score_a, r.score_b, r.finished_at,
       os.odds_home, os.odds_draw, os.odds_away,
       (SELECT COUNT(*) FROM results r2 WHERE r2.event_id = e.id) AS n_res
FROM events e
JOIN results r ON r.event_id = e.id
JOIN (SELECT event_id, MIN(id) AS sid FROM odds_snapshots GROUP BY event_id) f
     ON f.event_id = e.id
JOIN odds_snapshots os ON os.id = f.sid
WHERE e.competition = :lg
ORDER BY r.finished_at
"""
with engine.connect() as c:
    rows = c.execute(text(SQL), {"lg": LEAGUE}).fetchall()

dup_results = sum(1 for r in rows if r[9] and r[9] > 1)
matches = [dict(id=r[0], ta=r[1], tb=r[2], sa=r[3], sb=r[4], fin=str(r[5]),
                oh=r[6], od=r[7], oa=r[8])
           for r in rows if r[0] not in corrupted
           and r[3] is not None and r[6] and r[7] and r[8]]
print(f"matchs propres: {len(matches)} | events avec >1 result: {dup_results}")

# --- 1. orientation settlement: WR home par bucket de cote home (tout l'historique)
print("\n=== Calibration globale (orientation settlement) ===")
buckets = [(1.0, 1.3), (1.3, 1.6), (1.6, 2.0), (2.0, 2.5), (2.5, 3.5), (3.5, 99)]
calib = []
for lo, hi in buckets:
    ms = [m for m in matches if lo <= m["oh"] < hi]
    if not ms:
        continue
    wr = sum(1 for m in ms if m["sa"] > m["sb"]) / len(ms)
    imp = sum(1 / m["oh"] for m in ms) / len(ms)
    calib.append(dict(bucket=f"[{lo},{hi})", n=len(ms), home_wr=round(wr, 3),
                      implied_raw=round(imp, 3)))
    print(f"  oh in [{lo},{hi}): n={len(ms):5d}  home WR={wr:.3f}  implied(brut)={imp:.3f}")

# --- 2+3. paires GOLD: WR vs fair-implied, sur 3 cutoffs
def run(cutoff):
    oos, pnls, wins = [], [], 0
    imp_raw_sum, imp_fair_sum = 0.0, 0.0
    for m in matches:
        if m["fin"] >= cutoff and (m["ta"], m["tb"]) in PAIR_HOME_GOLD:
            oos.append(m)
            win = m["sa"] > m["sb"]
            wins += win
            pnls.append((m["oh"] - 1) if win else -1.0)
            over = 1 / m["oh"] + 1 / m["od"] + 1 / m["oa"]
            imp_raw_sum += 1 / m["oh"]
            imp_fair_sum += (1 / m["oh"]) / over
    n = len(oos)
    if n == 0:
        return dict(cutoff=cutoff, n=0)
    wr = wins / n
    fair = imp_fair_sum / n
    raw = imp_raw_sum / n
    roi = sum(pnls) / n
    sd = math.sqrt(sum((x - roi) ** 2 for x in pnls) / (n - 1))
    z_roi = roi / (sd / math.sqrt(n))
    z_fair = (wr - fair) / math.sqrt(fair * (1 - fair) / n)
    overround_avg = raw / fair  # approx marge
    return dict(cutoff=cutoff, n=n, wr=round(wr, 4), implied_raw=round(raw, 4),
                implied_fair=round(fair, 4), overround=round(overround_avg, 4),
                roi=round(roi, 4), z_roi=round(z_roi, 2), z_wr_vs_fair=round(z_fair, 2))

print("\n=== PAIR_HOME_GOLD OOS, 3 cutoffs ===")
sens = []
for cut in ["2026-06-05 12:34:00", "2026-06-06", "2026-06-07"]:
    r = run(cut)
    sens.append(r)
    print(" ", r)

out = dict(dup_results=dup_results, calibration=calib, sensitivity=sens)
with open("exports/wf5_pair_gold_recheck.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("\nJSON: exports/wf5_pair_gold_recheck.json")
