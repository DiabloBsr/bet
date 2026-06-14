# -*- coding: utf-8 -*-
"""WF4 HT/FT miner v2 - step 3: fresh data build + blind calibration of the 9 outcomes.

(Re-run of the domain on grown data; previous step2 crashed on a pickle path,
this pipeline is independent and self-contained.)

- Opening odds = snapshot MIN(o.id) per event (no post-kickoff info).
- Guards: corrupted_events.json (8035), dedup, HT<=FT impossible,
  goals_json consistency when parseable (len == score_a+score_b AND final
  cumulative == FT score; null/unparseable goals_json kept).
- Output: exports/_wf4_htft3_data.pkl (cache for step 4) +
  calibration tables (pooled-9, pooled-newleagues, per-league, 8035 70/30 test).
- p-value: one-sided z-test of profit under break-even null p0=1/odds
  (Var = sum o^2 p0 (1-p0)).
READ-ONLY on DB.
"""
import sys, json, math
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from scipy import stats
from sqlalchemy import create_engine, text
from scraper.config import load_settings

eng = create_engine(load_settings().db_url)

_corr = json.load(open("exports/corrupted_events.json", encoding="utf-8"))
CORRUPTED = set(int(k) for k in _corr["events"].keys())

Q = """
SELECT e.id, e.competition, e.team_a, e.team_b, e.expected_start,
       r.score_a, r.score_b, r.ht_score_a, r.ht_score_b, r.goals_json,
       o.odds_home, o.odds_draw, o.odds_away, o.extra_markets
FROM events e
JOIN results r ON r.event_id = e.id
JOIN odds_snapshots o ON o.event_id = e.id
JOIN (SELECT event_id, MIN(id) mid FROM odds_snapshots GROUP BY event_id) m
     ON m.mid = o.id
WHERE r.score_a IS NOT NULL AND r.score_b IS NOT NULL
"""
with eng.connect() as c:
    df = pd.read_sql(text(Q), c)
print(f"raw rows: {len(df)}")

df = df[~df["id"].isin(CORRUPTED)]
df = df.drop_duplicates(subset=["competition", "team_a", "team_b", "expected_start"])
df = df.sort_values("expected_start").reset_index(drop=True)
print(f"after corrupted+dedup: {len(df)}")

OUTCOMES = ["1/1", "1/X", "1/2", "X/1", "X/X", "X/2", "2/1", "2/X", "2/2"]
def col(k): return "o_" + k.replace("/", "")
def sgn(a, b): return "1" if a > b else ("2" if b > a else "X")

rows = []
drop = {"no_em": 0, "no_htft": 0, "no_ht": 0, "ht_gt_ft": 0, "gj_mismatch": 0}
for t in df.itertuples():
    em = t.extra_markets
    if em is None:
        drop["no_em"] += 1; continue
    try:
        em = json.loads(em) if isinstance(em, str) else em
    except Exception:
        drop["no_em"] += 1; continue
    ht = em.get("HT/FT")
    if not ht:
        drop["no_htft"] += 1; continue
    ha, hb = t.ht_score_a, t.ht_score_b
    if ha is None or hb is None or (isinstance(ha, float) and math.isnan(ha)):
        drop["no_ht"] += 1; continue
    ha, hb, fa, fb = int(ha), int(hb), int(t.score_a), int(t.score_b)
    if ha > fa or hb > fb:
        drop["ht_gt_ft"] += 1; continue
    # goals_json consistency guard (maison, new leagues not covered by audit)
    gj = t.goals_json
    if gj:
        try:
            g = json.loads(gj) if isinstance(gj, str) else gj
        except Exception:
            g = None
        if isinstance(g, list) and len(g) > 0:
            if len(g) != fa + fb:
                drop["gj_mismatch"] += 1; continue
            last = g[-1]
            if int(last.get("homeScore", -1)) != fa or int(last.get("awayScore", -1)) != fb:
                drop["gj_mismatch"] += 1; continue
        elif isinstance(g, list) and len(g) == 0 and fa + fb != 0:
            drop["gj_mismatch"] += 1; continue
    row = {"id": t.id, "lg": t.competition, "ts": t.expected_start,
           "oh": t.odds_home, "od": t.odds_draw, "oa": t.odds_away,
           "res": f"{sgn(ha, hb)}/{sgn(fa, fb)}"}
    ok = True
    for k in OUTCOMES:
        v = ht.get(k)
        if v is None:
            ok = False; break
        row[col(k)] = float(v)
    if not ok:
        drop["no_htft"] += 1; continue
    rows.append(row)

d = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
print(f"settled: {len(d)}  drops: {drop}")
print(d.groupby("lg").size())
d.to_pickle("exports/_wf4_htft3_data.pkl")

def calib(sub, scope):
    out = []
    for k in OUTCOMES:
        c_ = col(k)
        s = sub[sub[c_] < 99.99]
        n = len(s)
        if n == 0:
            out.append({"scope": scope, "outcome": k, "n": 0}); continue
        w = (s["res"] == k).astype(float).values
        o = s[c_].values
        wins = w.sum()
        profit = (w * o - 1.0).sum()
        p0 = 1.0 / o
        var = (o * o * p0 * (1 - p0)).sum()
        z = profit / math.sqrt(var) if var > 0 else 0.0
        p = float(1 - stats.norm.cdf(z))  # one-sided: profit > 0
        out.append({"scope": scope, "outcome": k, "n": n, "wins": int(wins),
                    "freq": round(wins / n, 4), "implied": round(p0.mean(), 4),
                    "ratio": round((wins / n) / p0.mean(), 3),
                    "avg_odds": round(float(o.mean()), 3),
                    "roi": round(profit / n, 4), "z": round(z, 2),
                    "p_onesided": round(p, 5)})
    return out

e35 = d[d["lg"] == "InstantLeague-8035"]
cut = int(len(e35) * 0.7)
te35 = e35.iloc[cut:]
new = d[d["lg"] != "InstantLeague-8035"]

cal = []
cal += calib(d, "pooled-9")
cal += calib(new, "pooled-newleagues")
cal += calib(te35, "8035-test30")
for lg, sub in d.groupby("lg"):
    cal += calib(sub, lg)

fmt = "{:<22} {:<4} {:>6} {:>6} {:>7} {:>8} {:>7} {:>8} {:>8} {:>8}"
print(fmt.format("scope", "out", "n", "wins", "freq", "implied", "ratio", "odds", "ROI", "p1s"))
for r in cal:
    if r["n"] == 0: continue
    print(fmt.format(r["scope"].replace("InstantLeague-", ""), r["outcome"], r["n"], r["wins"],
                     r["freq"], r["implied"], r["ratio"], r["avg_odds"],
                     f"{r['roi']:+.4f}", r["p_onesided"]))

json.dump({"meta": {"settled": int(len(d)), "drops": drop,
                    "per_league": {k: int(v) for k, v in d.groupby("lg").size().items()},
                    "split_8035": {"train": cut, "test": int(len(e35) - cut)}},
           "calibration": cal},
          open("exports/wf4_htft.json", "w"), indent=1)
print("saved exports/wf4_htft.json (calibration) + exports/_wf4_htft3_data.pkl")
