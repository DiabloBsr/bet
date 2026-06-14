# -*- coding: utf-8 -*-
"""WF5 recheck independant du verdict BRACKET_GOLD_HOME = MORT.

Verifie: integrite du join (doublons), settlement, timing des snapshots d'ouverture,
filtre competition, exclusions corrupted, et recompute le ROI OOS sous 6 variantes
de sensibilite. Sortie: exports/wf5_bracket_gold_recheck.json. LECTURE SEULE.
"""
import sys, json, math
from collections import defaultdict

sys.path.insert(0, ".")
from scraper.config import load_settings
from sqlalchemy import create_engine, text
from scraper.team_gold_data import BRACKET_GOLD_HOME

LEAGUE = "InstantLeague-8035"
CUTOFF = "2026-06-06"

engine = create_engine(load_settings().db_url)
corrupted = set(int(k) for k in json.load(open("exports/corrupted_events.json"))["events"].keys())
out = {}

with engine.connect() as c:
    # --- integrite: doublons results / snapshots par event
    dup_res = c.execute(text(
        "SELECT COUNT(*) FROM (SELECT event_id FROM results GROUP BY event_id HAVING COUNT(*)>1)"
    )).scalar()
    out["dup_results_events"] = dup_res

    # --- timing: le snapshot MIN(id) est-il bien avant le coup d'envoi ?
    late = c.execute(text("""
        SELECT SUM(CASE WHEN os.captured_at >= e.expected_start THEN 1 ELSE 0 END), COUNT(*)
        FROM events e
        JOIN (SELECT event_id, MIN(id) sid FROM odds_snapshots GROUP BY event_id) f ON f.event_id=e.id
        JOIN odds_snapshots os ON os.id=f.sid
        JOIN results r ON r.event_id=e.id
        WHERE e.competition=:lg AND r.finished_at >= :co
    """), {"lg": LEAGUE, "co": CUTOFF}).fetchone()
    out["open_snapshot_after_kickoff"] = {"late": late[0], "total": late[1]}

    # --- charge les matchs (ouverture + derniere cote pour sensibilite)
    rows = c.execute(text("""
        SELECT e.id, e.team_a, e.team_b, r.score_a, r.score_b, r.finished_at,
               o1.odds_home, o2.odds_home
        FROM events e
        JOIN results r ON r.event_id=e.id
        JOIN (SELECT event_id, MIN(id) sid, MAX(id) lid FROM odds_snapshots GROUP BY event_id) f
             ON f.event_id=e.id
        JOIN odds_snapshots o1 ON o1.id=f.sid
        JOIN odds_snapshots o2 ON o2.id=f.lid
        WHERE e.competition=:lg
        ORDER BY r.finished_at
    """), {"lg": LEAGUE}).fetchall()

matches = [dict(id=r[0], ta=r[1], sa=r[3], sb=r[4], fin=str(r[5]), oh=r[6], oh_last=r[7])
           for r in rows if r[3] is not None and r[6]]
clean = [m for m in matches if m["id"] not in corrupted]
out["n_all"] = len(matches); out["n_clean"] = len(clean)
out["fin_3225th_clean"] = clean[3224]["fin"] if len(clean) > 3225 else None

def settle(ms, lo_le=True, hi_lt=True, use_last=False):
    """PnL back-1 sur les brackets GOLD HOME."""
    pnls = []
    imp = 0.0
    for m in ms:
        cote = m["oh_last"] if use_last else m["oh"]
        if not cote:
            continue
        hit = False
        for (team, (lo, hi)), _ in BRACKET_GOLD_HOME.items():
            ok_lo = (lo <= cote) if lo_le else (lo < cote)
            ok_hi = (cote < hi) if hi_lt else (cote <= hi)
            if m["ta"] == team and ok_lo and ok_hi:
                hit = True
                break
        if hit:
            pnls.append((cote - 1) if m["sa"] > m["sb"] else -1.0)
            imp += 1 / cote
    n = len(pnls)
    if not n:
        return dict(n=0)
    mean = sum(pnls) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in pnls) / max(n - 1, 1))
    z = mean / (sd / math.sqrt(n)) if sd else None
    wins = sum(1 for p in pnls if p > 0)
    return dict(n=n, roi=round(mean, 4), z=round(z, 2) if z else None,
                wr=round(wins / n, 4), implied_wr=round(imp / n, 4),
                ci95=[round(mean - 1.96 * sd / math.sqrt(n), 4),
                      round(mean + 1.96 * sd / math.sqrt(n), 4)])

oos = [m for m in clean if m["fin"] >= CUTOFF]
oos_pos = clean[3225:]  # frontiere positionnelle (sensibilite)
out["variants"] = {
    "baseline_cutoff_0606": settle(oos),
    "positional_after_3225": settle(oos_pos),
    "upper_inclusive": settle(oos, hi_lt=False),
    "without_corrupted_excl": settle([m for m in matches if m["fin"] >= CUTOFF]),
    "latest_snapshot_odds": settle(oos, use_last=True),
    "is_period_check_before_0606": settle([m for m in clean if m["fin"] < CUTOFF]),
}

# marge bookmaker moyenne sur ces matchs (1X2 ouverture) pour situer -ROI vs -marge
with engine.connect() as c:
    mg = c.execute(text("""
        SELECT AVG(1.0/os.odds_home + 1.0/os.odds_draw + 1.0/os.odds_away)
        FROM events e
        JOIN (SELECT event_id, MIN(id) sid FROM odds_snapshots GROUP BY event_id) f ON f.event_id=e.id
        JOIN odds_snapshots os ON os.id=f.sid
        JOIN results r ON r.event_id=e.id
        WHERE e.competition=:lg AND r.finished_at >= :co
          AND os.odds_home>0 AND os.odds_draw>0 AND os.odds_away>0
    """), {"lg": LEAGUE, "co": CUTOFF}).scalar()
out["avg_book_overround_oos"] = round(mg, 4) if mg else None

with open("exports/wf5_bracket_gold_recheck.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
