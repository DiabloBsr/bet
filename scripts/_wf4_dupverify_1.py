# -*- coding: utf-8 -*-
"""
WF4 - VERIFICATION ADVERSARIALE du finding dupaudit (_wf4_seq_5_dupaudit.py).
Autopsie des paires <30min: identite FT/HT/goals_json, cotes d'ouverture,
distribution temporelle (first_seen_at / expected_start), ligues, gaps.
Sortie: exports/wf4_dupverify.json. LECTURE SEULE.
"""
import sys, json
sys.path.insert(0, ".")
from collections import Counter
from datetime import datetime
from scraper.config import load_settings
from sqlalchemy import create_engine, text

def main():
    e = create_engine(load_settings().db_url)
    corrupted = set(int(k) for k in json.load(open("exports/corrupted_events.json"))["events"].keys())
    with e.connect() as c:
        res = c.execute(text("""
          SELECT e.id, e.competition, e.team_a, e.team_b, e.expected_start, e.round_info,
                 e.first_seen_at, r.score_a, r.score_b, r.ht_score_a, r.ht_score_b, r.goals_json
          FROM events e JOIN results r ON r.event_id=e.id
          WHERE EXISTS (SELECT 1 FROM odds_snapshots o WHERE o.event_id=e.id)
          ORDER BY e.expected_start, e.id""")).fetchall()
        # cotes d'ouverture (snapshot MIN(id) par event)
        odds = {}
        for row in c.execute(text("""
          SELECT o.event_id, o.odds_home, o.odds_draw, o.odds_away
          FROM odds_snapshots o
          JOIN (SELECT event_id, MIN(id) mid FROM odds_snapshots GROUP BY event_id) m
            ON m.mid = o.id""")).fetchall():
            odds[row[0]] = (row[1], row[2], row[3])
    rows = [r for r in res if r[0] not in corrupted and r[7] is not None]
    bykey = {}
    for r in rows:
        bykey.setdefault((r[1], r[2], r[3]), []).append(r)

    pairs = []
    for key, lst in bykey.items():
        lst.sort(key=lambda r: (r[4], r[0]))
        for i in range(1, len(lst)):
            a, b = lst[i - 1], lst[i]
            gap = (datetime.fromisoformat(b[4]) - datetime.fromisoformat(a[4])).total_seconds()
            if gap < 1800:
                pairs.append((a, b, gap))

    n = len(pairs)
    same_ft = sum(1 for a, b, _ in pairs if (a[7], a[8]) == (b[7], b[8]))
    same_ht = sum(1 for a, b, _ in pairs if (a[9], a[10]) == (b[9], b[10]))
    same_ftht = sum(1 for a, b, _ in pairs
                    if (a[7], a[8], a[9], a[10]) == (b[7], b[8], b[9], b[10]))
    same_goals = sum(1 for a, b, _ in pairs if a[11] == b[11] and a[11] is not None)
    same_round = sum(1 for a, b, _ in pairs if a[5] == b[5])
    gap0 = sum(1 for _, _, g in pairs if g == 0)
    gaps = Counter(int(g // 300) * 5 for _, _, g in pairs)  # buckets de 5 min

    # cotes d'ouverture identiques / proches
    both_odds = [(a, b) for a, b, _ in pairs if a[0] in odds and b[0] in odds]
    same_odds = sum(1 for a, b in both_odds if odds[a[0]] == odds[b[0]])
    close_odds = sum(1 for a, b in both_odds
                     if all(abs(odds[a[0]][k] - odds[b[0]][k]) <= 0.10 for k in range(3)))

    comp = Counter(a[1] for a, b, _ in pairs)
    day_start = Counter(str(a[4])[:10] for a, b, _ in pairs)
    day_seen_b = Counter(str(b[6])[:10] for a, b, _ in pairs)

    # parmi les paires au score FT identique: combien ont aussi goals_json identique
    ftpairs = [(a, b) for a, b, _ in pairs if (a[7], a[8]) == (b[7], b[8])]
    ft_and_goals = sum(1 for a, b in ftpairs if a[11] == b[11] and a[11] is not None)
    ft_and_odds = sum(1 for a, b in ftpairs
                      if a[0] in odds and b[0] in odds and odds[a[0]] == odds[b[0]])

    # controle: goals_json identiques dans les buckets 30m-2h (devrait etre ~0)
    ctrl = [0, 0]
    for key, lst in bykey.items():
        lst.sort(key=lambda r: (r[4], r[0]))
        for i in range(1, len(lst)):
            a, b = lst[i - 1], lst[i]
            gap = (datetime.fromisoformat(b[4]) - datetime.fromisoformat(a[4])).total_seconds()
            if 1800 <= gap < 7200:
                ctrl[0] += 1
                ctrl[1] += int(a[11] == b[11] and a[11] is not None)

    out = dict(
        n_pairs=n, same_ft=same_ft, same_ht=same_ht, same_ft_and_ht=same_ftht,
        same_goals_json=same_goals, same_round_info=same_round, gap_zero=gap0,
        gap_hist_5min={str(k): v for k, v in sorted(gaps.items())},
        n_both_odds=len(both_odds), same_opening_odds=same_odds, close_opening_odds=close_odds,
        ft_identical_pairs=len(ftpairs), ft_and_goals_identical=ft_and_goals,
        ft_and_odds_identical=ft_and_odds,
        by_competition=dict(comp.most_common()),
        by_day_expected_start=dict(sorted(day_start.items())),
        by_day_first_seen_dup=dict(sorted(day_seen_b.items())),
        ctrl_30m2h_pairs=ctrl[0], ctrl_30m2h_same_goals=ctrl[1],
    )
    with open("exports/wf4_dupverify.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
