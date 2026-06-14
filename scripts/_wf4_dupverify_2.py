# -*- coding: utf-8 -*-
"""
WF4 - VERIF ADVERSARIALE 2: clones a N'IMPORTE quel gap (signature stricte:
memes cotes d'ouverture + meme goals_json non-nul + meme FT), decomposition
gap=0 vs gap>0, nb de buts des "collisions", impact du seuil 30min.
Sortie: exports/wf4_dupverify2.json. LECTURE SEULE.
"""
import sys, json
sys.path.insert(0, ".")
from collections import Counter
from datetime import datetime
from itertools import combinations
from scraper.config import load_settings
from sqlalchemy import create_engine, text

def ngoals(gj):
    if not gj:
        return 0
    try:
        return len(json.loads(gj))
    except Exception:
        return -1

def main():
    e = create_engine(load_settings().db_url)
    corrupted = set(int(k) for k in json.load(open("exports/corrupted_events.json"))["events"].keys())
    with e.connect() as c:
        res = c.execute(text("""
          SELECT e.id, e.competition, e.team_a, e.team_b, e.expected_start,
                 r.score_a, r.score_b, r.goals_json
          FROM events e JOIN results r ON r.event_id=e.id
          WHERE EXISTS (SELECT 1 FROM odds_snapshots o WHERE o.event_id=e.id)
          ORDER BY e.expected_start, e.id""")).fetchall()
        odds = {}
        for row in c.execute(text("""
          SELECT o.event_id, o.odds_home, o.odds_draw, o.odds_away
          FROM odds_snapshots o
          JOIN (SELECT event_id, MIN(id) mid FROM odds_snapshots GROUP BY event_id) m
            ON m.mid = o.id""")).fetchall():
            odds[row[0]] = (row[1], row[2], row[3])
    rows = [r for r in res if r[0] not in corrupted and r[5] is not None]
    bykey = {}
    for r in rows:
        bykey.setdefault((r[1], r[2], r[3]), []).append(r)

    # 1) scan EXHAUSTIF (toutes paires d'une cle, pas seulement consecutives)
    #    signature clone = memes cotes + meme goals_json non-nul + meme score
    clones = []
    for key, lst in bykey.items():
        if len(lst) < 2:
            continue
        for a, b in combinations(lst, 2):
            if a[7] is None or a[7] != b[7]:
                continue
            if (a[5], a[6]) != (b[5], b[6]):
                continue
            if odds.get(a[0]) != odds.get(b[0]):
                continue
            gap = abs((datetime.fromisoformat(b[4]) - datetime.fromisoformat(a[4])).total_seconds())
            clones.append((a, b, gap, ngoals(a[7])))
    gapdist = Counter("0" if g == 0 else "<30m" if g < 1800 else "30m-2h" if g < 7200
                      else ">2h" for _, _, g, _ in clones)
    goalsdist = Counter(ng for _, _, _, ng in clones)
    comp_clones = Counter(a[1] for a, _, _, _ in clones)
    day_clones = Counter(str(a[4])[:10] for a, _, _, _ in clones)
    multi_goal_far = [(a[0], b[0], a[1], a[2], a[3], str(a[4]), str(b[4]), g, ng)
                      for a, b, g, ng in clones if g >= 1800 and ng >= 2]

    # 2) paires consecutives <30m, gap>0: nb de buts quand goals_json identique
    near_nonzero_samegoals = []
    same_ft_gapgt0 = [0, 0]
    for key, lst in bykey.items():
        lst.sort(key=lambda r: (r[4], r[0]))
        for i in range(1, len(lst)):
            a, b = lst[i - 1], lst[i]
            gap = (datetime.fromisoformat(b[4]) - datetime.fromisoformat(a[4])).total_seconds()
            if 0 < gap < 1800:
                same_ft_gapgt0[0] += 1
                same_ft_gapgt0[1] += int((a[5], a[6]) == (b[5], b[6]))
                if a[7] is not None and a[7] == b[7]:
                    near_nonzero_samegoals.append((a[0], b[0], gap, ngoals(a[7])))

    # 3) controle 30m-2h goals_json identiques: nb de buts (collisions attendues = 0/1 but)
    ctrl_goals = Counter()
    for key, lst in bykey.items():
        lst.sort(key=lambda r: (r[4], r[0]))
        for i in range(1, len(lst)):
            a, b = lst[i - 1], lst[i]
            gap = (datetime.fromisoformat(b[4]) - datetime.fromisoformat(a[4])).total_seconds()
            if 1800 <= gap < 7200 and a[7] is not None and a[7] == b[7]:
                ctrl_goals[ngoals(a[7])] += 1

    # 4) cout du seuil 30min: matchs legitimes (gap>0, non clones) supprimes par la regle
    out = dict(
        n_clone_pairs_strict=len(clones),
        clone_gap_dist=dict(gapdist),
        clone_goals_dist={str(k): v for k, v in sorted(goalsdist.items())},
        clone_by_comp=dict(comp_clones.most_common()),
        clone_by_day=dict(sorted(day_clones.items())),
        clones_far_multigoal=multi_goal_far[:20],
        n_clones_far_multigoal=len(multi_goal_far),
        consec_pairs_gap_0_30m=same_ft_gapgt0[0],
        consec_pairs_gap_0_30m_same_ft=same_ft_gapgt0[1],
        near_nonzero_samegoals=[(int(x[0]), int(x[1]), x[2], x[3]) for x in near_nonzero_samegoals],
        ctrl_30m2h_samegoals_by_ngoals={str(k): v for k, v in sorted(ctrl_goals.items())},
    )
    with open("exports/wf4_dupverify2.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=str)
    print(json.dumps(out, indent=1, default=str))

if __name__ == "__main__":
    main()
