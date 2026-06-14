# -*- coding: utf-8 -*-
"""WF5 — CONTRE-VERIFICATION independante du verdict E2 OVERFIT. READ-ONLY.
Code path independant de _wf5_stationarity_audit.py :
  1) unicite results par event (pas de double settlement)
  2) timing du snapshot d'ouverture vs expected_start (pas d'info post-kickoff)
  3) coherence settlement : score_a/score_b vs somme goals_json (echantillon E2)
  4) P&L E2 independant + IC Wilson sur WR vs break-even 1/avg_odds
  5) sensibilite aux bornes ([1.10,1.20] vs (1.09,1.21) vs [1.12,1.18])
  6) E2 sur les 8 AUTRES ligues (si data) comme replication hors-echantillon
Sortie: exports/wf5_e2_counterverify.json
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
    # 1) results uniqueness
    dup = c.execute(text("""
        SELECT COUNT(*) FROM (
          SELECT event_id FROM results GROUP BY event_id HAVING COUNT(*) > 1)
    """)).scalar()
    out["results_duplicate_event_ids"] = int(dup)

    # 2+4) E2 rows with snapshot timing
    rows = c.execute(text("""
        SELECT e.id, e.expected_start, o.captured_at,
               r.score_a, r.score_b, o.odds_home, o.odds_draw, o.odds_away
        FROM events e
        JOIN results r ON r.event_id = e.id
        JOIN odds_snapshots o ON o.event_id = e.id
        WHERE e.competition = :lg
          AND o.id = (SELECT MIN(o2.id) FROM odds_snapshots o2 WHERE o2.event_id = e.id)
          AND ((o.odds_home <= o.odds_away AND o.odds_home BETWEEN 1.05 AND 1.25)
            OR (o.odds_away <  o.odds_home AND o.odds_away BETWEEN 1.05 AND 1.25))
    """), {"lg": LG}).fetchall()

post_kickoff = 0
sel = []          # (eid, xs, won, fav_odds)
wide = []         # for boundary sensitivity (1.05-1.25 superset)
seen = set()
for (eid, xs, fa, sa, sb, oh, od, oa) in rows:
    if eid in CORRUPTED or None in (oh, od, oa, sa, sb):
        continue
    if eid in seen:
        continue
    seen.add(eid)
    xs, fa = str(xs), str(fa)
    if fa and xs and fa[:16] > xs[:16]:
        post_kickoff += 1
    oh, oa = float(oh), float(oa)
    fav_home = oh <= oa
    fc = oh if fav_home else oa
    won = (int(sa) > int(sb)) if fav_home else (int(sb) > int(sa))
    wide.append((fc, won))
    if 1.10 <= fc <= 1.20:
        sel.append((eid, xs, won, fc))
out["opening_snapshot_after_start_count"] = post_kickoff
out["n_unique_e2"] = len(sel)

# 3) settlement coherence on E2 sample: score vs goals_json
mismatch, checked, gj_null = 0, 0, 0
ids = [s[0] for s in sel]
with e.connect() as c:
    for i in range(0, len(ids), 400):
        batch = ids[i:i+400]
        q = text("SELECT event_id, score_a, score_b, goals_json FROM results "
                 "WHERE event_id IN (%s)" % ",".join(str(x) for x in batch))
        for (eid, sa, sb, gj) in c.execute(q):
            try:
                g = json.loads(gj) if gj else None
            except Exception:
                g = None
            if not isinstance(g, list):
                gj_null += 1
                continue
            checked += 1
            nh = sum(1 for x in g if x.get("team") == "Home")
            na = sum(1 for x in g if x.get("team") == "Away")
            if nh != int(sa) or na != int(sb):
                mismatch += 1
out["settlement_check"] = dict(checked=checked, goals_json_null=gj_null,
                               score_vs_goals_mismatch=mismatch)

# 4) independent P&L + Wilson CI
def pnl(bets):
    n = len(bets)
    if n == 0:
        return dict(n=0)
    wins = sum(1 for _, w in bets if w)
    profit = sum((o - 1.0) if w else -1.0 for o, w in bets)
    avg_o = sum(o for o, _ in bets) / n
    wr = wins / n
    z = 1.96
    den = 1 + z * z / n
    ctr = (wr + z * z / (2 * n)) / den
    hw = z * math.sqrt(wr * (1 - wr) / n + z * z / (4 * n * n)) / den
    return dict(n=n, wins=wins, wr=round(wr, 4),
                wr_ci95=[round(ctr - hw, 4), round(ctr + hw, 4)],
                avg_odds=round(avg_o, 4),
                breakeven_wr=round(1 / avg_o, 4),
                roi_pct=round(100 * profit / n, 2))
out["E2_pnl"] = pnl([(o, w) for _, _, w, o in sel])

# 5) boundary sensitivity
for lo, hi, tag in [(1.10, 1.20, "base"), (1.09, 1.21, "wider"),
                    (1.12, 1.18, "narrower"), (1.05, 1.25, "superset")]:
    out["E2_bounds_%s" % tag] = pnl([(o, w) for o, w in wide if lo <= o <= hi])

# 6) replication: other leagues
with e.connect() as c:
    others = c.execute(text("""
        SELECT e.competition, e.id, r.score_a, r.score_b,
               o.odds_home, o.odds_away
        FROM events e
        JOIN results r ON r.event_id = e.id
        JOIN odds_snapshots o ON o.event_id = e.id
        WHERE e.competition != :lg
          AND o.id = (SELECT MIN(o2.id) FROM odds_snapshots o2 WHERE o2.event_id = e.id)
          AND ((o.odds_home <= o.odds_away AND o.odds_home BETWEEN 1.10 AND 1.20)
            OR (o.odds_away <  o.odds_home AND o.odds_away BETWEEN 1.10 AND 1.20))
    """), {"lg": LG}).fetchall()
oth = collections.defaultdict(list)
for (comp, eid, sa, sb, oh, oa) in others:
    if eid in CORRUPTED or None in (oh, oa, sa, sb):
        continue
    oh, oa = float(oh), float(oa)
    fav_home = oh <= oa
    fc = oh if fav_home else oa
    won = (int(sa) > int(sb)) if fav_home else (int(sb) > int(sa))
    oth[comp].append((fc, won))
    oth["__ALL_OTHER__"].append((fc, won))
out["E2_other_leagues"] = {k: pnl(v) for k, v in sorted(oth.items())}

json.dump(out, open("exports/wf5_e2_counterverify.json", "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
for k, v in out.items():
    print(k, "=", json.dumps(v, ensure_ascii=False))
