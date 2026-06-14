# -*- coding: utf-8 -*-
"""WF4 HT/FT miner - step 1: load + settle + calibration of the 9 outcomes.

- Opening odds = snapshot MIN(o.id) per event (no post-kickoff info).
- 9 leagues pooled + per-league + 8035 walk-forward 70/30 (test only).
- Guards: corrupted_events.json (8035), ht>ft impossible, dedup, odds cap 100.
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
SELECT e.id, e.competition, e.team_a, e.team_b, e.expected_start, e.round_info,
       r.score_a, r.score_b, r.ht_score_a, r.ht_score_b,
       o.odds_home, o.odds_draw, o.odds_away, o.extra_markets
FROM events e
JOIN results r ON r.event_id = e.id
JOIN odds_snapshots o ON o.event_id = e.id
JOIN (SELECT event_id, MIN(id) mid FROM odds_snapshots GROUP BY event_id) m
     ON m.mid = o.id
WHERE r.score_a IS NOT NULL
"""
with eng.connect() as c:
    df = pd.read_sql(text(Q), c)
print(f"raw rows: {len(df)}", file=sys.stderr)

df = df[~df["id"].isin(CORRUPTED)]
df = df.drop_duplicates(subset=["competition", "team_a", "team_b", "expected_start"])
df = df.sort_values("expected_start").reset_index(drop=True)

OUTCOMES = ["1/1", "1/X", "1/2", "X/1", "X/X", "X/2", "2/1", "2/X", "2/2"]

def sgn(a, b):
    return "1" if a > b else ("2" if b > a else "X")

rows, n_no_em, n_no_htft, n_no_ht, n_impossible = [], 0, 0, 0, 0
for t in df.itertuples():
    em = t.extra_markets
    if em is None:
        n_no_em += 1; continue
    em = json.loads(em) if isinstance(em, str) else em
    ht = em.get("HT/FT")
    if not ht:
        n_no_htft += 1; continue
    ha, hb = t.ht_score_a, t.ht_score_b
    if ha is None or hb is None or (isinstance(ha, float) and math.isnan(ha)):
        n_no_ht += 1; continue
    ha, hb, fa, fb = int(ha), int(hb), int(t.score_a), int(t.score_b)
    if ha > fa or hb > fb:           # impossible -> corrupted result
        n_impossible += 1; continue
    res = f"{sgn(ha, hb)}/{sgn(fa, fb)}"
    r = {"id": t.id, "lg": t.competition, "ts": t.expected_start,
         "round": t.round_info, "res": res,
         "oh": t.odds_home, "od": t.odds_draw, "oa": t.odds_away}
    ok = True
    for k in OUTCOMES:
        v = ht.get(k)
        if v is None:
            ok = False; break
        r["o_" + k.replace("/", "")] = float(v)
    if ok:
        rows.append(r)

d = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
print(f"settled: {len(d)} | no_em={n_no_em} no_htft={n_no_htft} "
      f"no_ht={n_no_ht} impossible={n_impossible}", file=sys.stderr)
print(d.groupby("lg").size(), file=sys.stderr)
d.to_pickle("exports/_wf4_htft_data.pkl")

def col(k): return "o_" + k.replace("/", "")

def calib(sub, label):
    out = []
    for k in OUTCOMES:
        c_ = col(k)
        s = sub[sub[c_] < 99.99]
        n = len(s)
        if n == 0: continue
        w = (s["res"] == k).astype(int)
        freq = w.mean()
        imp = (1.0 / s[c_]).mean()
        profit = w * s[c_] - 1.0
        roi = profit.mean()
        # t-test on per-bet profit
        tt = stats.ttest_1samp(profit, 0.0) if n > 5 else None
        p = float(tt.pvalue) if tt else 1.0
        out.append({"scope": label, "outcome": k, "n": int(n),
                    "wins": int(w.sum()), "freq": round(float(freq), 4),
                    "implied": round(float(imp), 4),
                    "ratio": round(float(freq / imp), 3) if imp > 0 else None,
                    "avg_odds": round(float(s[c_].mean()), 3),
                    "roi": round(float(roi), 4), "p": round(p, 5)})
    return out

report = {"meta": {"settled": len(d), "no_em": n_no_em, "no_htft": n_no_htft,
                   "no_ht": n_no_ht, "impossible": n_impossible,
                   "per_league": d.groupby("lg").size().to_dict()}}

res = []
res += calib(d, "pooled-9")
new = d[d["lg"] != "InstantLeague-8035"]
res += calib(new, "pooled-newleagues")
for lg, sub in d.groupby("lg"):
    res += calib(sub, lg)

# 8035 walk-forward
e35 = d[d["lg"] == "InstantLeague-8035"].sort_values("ts")
cut = int(len(e35) * 0.7)
res += calib(e35.iloc[:cut], "8035-train")
res += calib(e35.iloc[cut:], "8035-test")
# robustness: 8035 without round 0
e35n0 = e35[e35["round"] != "0"]
cut2 = int(len(e35n0) * 0.7)
res += calib(e35n0.iloc[cut2:], "8035-test-noJ0")

report["calibration"] = res
json.dump(report, open("exports/wf4_htft_step1.json", "w"), indent=1)

fmt = "{:<22} {:>4} {:>5} {:>5} {:>7} {:>7} {:>6} {:>7} {:>8} {:>8}"
print(fmt.format("scope", "out", "n", "wins", "freq", "impl", "ratio", "odds", "ROI", "p"))
for r in res:
    print(fmt.format(r["scope"].replace("InstantLeague-", ""), r["outcome"],
                     r["n"], r["wins"], r["freq"], r["implied"], r["ratio"],
                     r["avg_odds"], f"{r['roi']:+.4f}", r["p"]))
