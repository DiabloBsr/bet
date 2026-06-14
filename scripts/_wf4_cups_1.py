# -*- coding: utf-8 -*-
"""WF4 cups - script 1: structure mapping of 8065 (CdM), 8056 (CL), 8060 (CAN).
Rounds, matches/round, teams/round, draws by round, OT/penalties detection,
goals_json minutes > 90, scores distribution by phase.
READ-ONLY.
"""
import sys, json, collections
sys.path.insert(0, ".")
from scraper.config import load_settings
from sqlalchemy import create_engine, text

e = create_engine(load_settings().db_url)

CUPS = ["InstantLeague-8065", "InstantLeague-8056", "InstantLeague-8060"]

out = {}

with e.connect() as c:
    for comp in CUPS:
        rows = c.execute(text("""
            SELECT e.id, e.round_info, e.team_a, e.team_b, e.expected_start,
                   r.score_a, r.score_b, r.ht_score_a, r.ht_score_b, r.goals_json
            FROM events e JOIN results r ON r.event_id = e.id
            WHERE e.competition = :comp
            ORDER BY e.expected_start, e.id
        """), {"comp": comp}).fetchall()

        per_round = collections.defaultdict(lambda: {"n": 0, "teams": set(), "draws": 0,
                                                     "goals": 0, "events": []})
        minutes_gt90 = 0
        max_minute = 0
        n_goals_checked = 0
        bad_goals_json = 0
        n_null_gj = 0
        all_teams = set()
        for (eid, ri, ta, tb, es, sa, sb, hta, htb, gj) in rows:
            try:
                rnum = int(ri)
            except (TypeError, ValueError):
                rnum = -1
            d = per_round[rnum]
            d["n"] += 1
            d["teams"].add(ta); d["teams"].add(tb)
            all_teams.add(ta); all_teams.add(tb)
            if sa == sb:
                d["draws"] += 1
            d["goals"] += (sa + sb)
            d["events"].append(eid)
            if gj:
                try:
                    g = json.loads(gj)
                except Exception:
                    g = None
                if g is not None:
                    n_goals_checked += 1
                    if len(g) != sa + sb:
                        bad_goals_json += 1
                    for goal in g:
                        m = goal.get("minute", 0)
                        max_minute = max(max_minute, m)
                        if m > 90:
                            minutes_gt90 += 1
            else:
                n_null_gj += 1

        rounds_sorted = sorted(per_round.keys())
        rt = []
        for rnum in rounds_sorted:
            d = per_round[rnum]
            rt.append({"round": rnum, "n_matches": d["n"], "n_teams": len(d["teams"]),
                       "draws": d["draws"], "draw_rate": round(d["draws"]/d["n"], 3),
                       "avg_goals": round(d["goals"]/d["n"], 2)})
        out[comp] = {
            "n_finished": len(rows),
            "n_teams_total": len(all_teams),
            "rounds": rt,
            "minutes_gt90": minutes_gt90,
            "max_minute": max_minute,
            "n_goals_json_parsed": n_goals_checked,
            "n_goals_json_mismatch": bad_goals_json,
            "n_goals_json_null": n_null_gj,
            "first_start": str(rows[0][4]) if rows else None,
            "last_start": str(rows[-1][4]) if rows else None,
        }

# also: distinct team names sample per cup, and check whether same team plays multiple
# times in the same round (would indicate "round" != matchday)
with e.connect() as c:
    for comp in CUPS:
        rows = c.execute(text("""
            SELECT e.round_info, e.team_a, e.team_b
            FROM events e JOIN results r ON r.event_id = e.id
            WHERE e.competition = :comp
        """), {"comp": comp}).fetchall()
        dup = 0
        seen = collections.defaultdict(collections.Counter)
        for ri, ta, tb in rows:
            seen[ri][ta] += 1
            seen[ri][tb] += 1
        for ri, cnt in seen.items():
            for t, k in cnt.items():
                if k > 1:
                    dup += 1
        out[comp]["team_round_duplicates"] = dup

print(json.dumps(out, indent=1, ensure_ascii=False))
with open("exports/wf4_cups_structure.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
