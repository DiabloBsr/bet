# -*- coding: utf-8 -*-
"""
WF4 - audit doublons d'events (PIEGE anti-leakage pour tout backtest sequentiel).
Paires d'events finis+cotes de la meme ligue/equipes, consecutives dans le temps:
taux de score FT identique par bucket d'ecart de coup d'envoi.
<30min: ~41% identiques (vs ~8% partout ailleurs) => vrais doublons du meme match.
Consequence mesuree (scripts/_wf4_seq_1.py avec/sans dedup): une regle 'back
sur-regime rpts5' passait de -4.5% (train, dedup) a +14.8% ROI fantome (test, sans dedup).
Sortie: exports/wf4_dupaudit.json. LECTURE SEULE.
"""
import sys, json
sys.path.insert(0, ".")
from datetime import datetime
from scraper.config import load_settings
from sqlalchemy import create_engine, text

def main():
    e = create_engine(load_settings().db_url)
    corrupted = set(int(k) for k in json.load(open("exports/corrupted_events.json"))["events"].keys())
    with e.connect() as c:
        res = c.execute(text("""
          SELECT e.id, e.competition, e.team_a, e.team_b, e.expected_start, r.score_a, r.score_b
          FROM events e JOIN results r ON r.event_id=e.id
          WHERE EXISTS (SELECT 1 FROM odds_snapshots o WHERE o.event_id=e.id)
          ORDER BY e.expected_start, e.id""")).fetchall()
    rows = [r for r in res if r[0] not in corrupted and r[5] is not None]
    bykey = {}
    for r in rows:
        bykey.setdefault((r[1], r[2], r[3]), []).append(r)
    buckets = {"<30m": [0, 0], "30m-2h": [0, 0], "2-6h": [0, 0], "6-24h": [0, 0], ">24h": [0, 0]}
    for key, lst in bykey.items():
        lst.sort(key=lambda r: (r[4], r[0]))
        for i in range(1, len(lst)):
            a, b = lst[i - 1], lst[i]
            gap = (datetime.fromisoformat(b[4]) - datetime.fromisoformat(a[4])).total_seconds()
            same = (a[5] == b[5] and a[6] == b[6])
            k = "<30m" if gap < 1800 else "30m-2h" if gap < 7200 else "2-6h" if gap < 21600 \
                else "6-24h" if gap < 86400 else ">24h"
            buckets[k][0] += 1
            buckets[k][1] += int(same)
    out = {k: dict(n=n, same_score=s, pct=round(100 * s / max(n, 1), 1)) for k, (n, s) in buckets.items()}
    with open("exports/wf4_dupaudit.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    for k, v in out.items():
        print(f"{k:>7}: n={v['n']:>6} same_score={v['same_score']:>5} ({v['pct']}%)")

if __name__ == "__main__":
    main()
