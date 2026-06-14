# WF4 jitter/drift - step 4: jitter autocorrelation structure + executable pair-value strategy
# jitter_sel = log(open_odds_sel) - LOO/train pair mean of log(odds_sel)
# Output: exports/wf4_jitter_structure.json
import sys, json, pickle
import numpy as np
import pandas as pd
from scipy import stats

with open("scripts/_wf4_jitter_data.pkl", "rb") as f:
    df = pickle.load(f)

df["res"] = np.where(df.score_a > df.score_b, "home", np.where(df.score_a < df.score_b, "away", "draw"))
df["pair"] = df.competition + "|" + df.team_a + "|" + df.team_b
df["lh"] = np.log(df.open_home)
df["ld"] = np.log(df.open_draw)
df["la"] = np.log(df.open_away)

results = {}

# ===================== (2) AUTOCORRELATION STRUCTURE (8035, LOO jitter) ==============
d35 = df[df.competition == "InstantLeague-8035"].copy()
# LOO pair mean of log home odds
g = d35.groupby("pair")["lh"]
d35["pair_n"] = g.transform("count")
d35["pair_sum"] = g.transform("sum")
d35 = d35[d35.pair_n >= 3].copy()
d35["jit"] = d35.lh - (d35.pair_sum - d35.lh) / (d35.pair_n - 1)
print(f"8035 events with pair_n>=3: {len(d35)}, jitter std={d35.jit.std():.4f} (log odds)")
results["jitter_std_8035"] = round(float(d35.jit.std()), 4)

# (a) lag-1 autocorrelation in publication order (open snapshot id order)
d35s = d35.sort_values("open_snap_id")
j = d35s.jit.values
r1 = np.corrcoef(j[:-1], j[1:])[0, 1]
n = len(j) - 1
z1 = r1 * np.sqrt(n)
print(f"lag-1 autocorr (publication order): r={r1:+.4f} (n={n}, z={z1:.2f}, p={2*(1-stats.norm.cdf(abs(z1))):.3g})")
results["lag1_autocorr_puborder"] = {"r": round(float(r1), 4), "n": int(n),
                                     "p": float(2 * (1 - stats.norm.cdf(abs(z1))))}

# (b) ICC by scrape_run_id of opening snapshot (same fetch batch)
def icc_oneway(values, groups):
    dfm = pd.DataFrame({"v": values, "g": groups})
    cnt = dfm.groupby("g")["v"].agg(["count", "mean"])
    cnt = cnt[cnt["count"] >= 2]
    dfm = dfm[dfm.g.isin(cnt.index)]
    k = len(cnt)
    N = len(dfm)
    if k < 2:
        return None
    grand = dfm.v.mean()
    ssb = (cnt["count"] * (cnt["mean"] - grand) ** 2).sum()
    ssw = ((dfm.v - dfm.groupby("g")["v"].transform("mean")) ** 2).sum()
    msb = ssb / (k - 1)
    msw = ssw / (N - k)
    F = msb / msw
    p = 1 - stats.f.cdf(F, k - 1, N - k)
    nbar = N / k
    icc = (msb - msw) / (msb + (nbar - 1) * msw)
    return {"icc": round(float(icc), 4), "F": round(float(F), 3), "p": float(p),
            "k_groups": int(k), "N": int(N)}

r_run = icc_oneway(d35.jit.values, d35.open_scrape_run_id.values)
print("ICC by scrape_run (open):", r_run)
results["icc_scrape_run"] = r_run

# (c) ICC by simultaneous publication batch: same expected_start (round batch)
d35["start_key"] = d35.expected_start.astype(str)
r_batch = icc_oneway(d35.jit.values, d35.start_key.values)
print("ICC by exact expected_start (round batch):", r_batch)
results["icc_same_kickoff"] = r_batch

# (d) ICC by (round_info, season-window): round + expected_start day
d35["round_day"] = d35.round_info.astype(str) + "|" + d35.expected_start.dt.strftime("%Y-%m-%d")
r_round = icc_oneway(d35.jit.values, d35.round_day.values)
print("ICC by round+day:", r_round)
results["icc_round_day"] = r_round

# (e) does open jitter predict the open->close drift? (multi-snap subset)
m = d35[d35.n_snaps >= 2].copy()
m["drift_h"] = np.log(m.open_home / m.close_home)
m2 = m[m.drift_h.abs() > 1e-9]
if len(m2) > 10:
    rho, p = stats.pearsonr(m2.jit, m2.drift_h)
    print(f"corr(open jitter, open->close drift home): r={rho:+.4f} (n={len(m2)}, p={p:.3g})")
    results["corr_jitter_drift"] = {"r": round(float(rho), 4), "n": int(len(m2)), "p": float(p)}

# ===================== (3) EXECUTABLE pair-value strategy =====================
# walk-forward 8035: pair means from TRAIN, bets on TEST at open odds
def pair_value_eval(dtrain, dtest, sels=("home", "draw", "away"), thr_list=(0.02, 0.04, 0.06),
                    odds_min=1.6, min_pair_n=3):
    out = []
    means = {}
    for sel in sels:
        means[sel] = dtrain.groupby("pair")["l" + sel[0]].agg(["mean", "count"])
    for thr in thr_list:
        bets = []
        for sel in sels:
            mm = means[sel]
            te = dtest.join(mm, on="pair", rsuffix="_m")
            te = te[te["count"] >= min_pair_n]
            jit = np.log(te["open_" + sel]) - te["mean"]
            pick = te[(jit >= thr) & (te["open_" + sel] >= odds_min)]
            pnl = (pick.res == sel).astype(int) * pick["open_" + sel] - 1
            bets.append(pd.DataFrame({"pnl": pnl, "win": (pick.res == sel).astype(int),
                                      "odds": pick["open_" + sel], "sel": sel}))
        B = pd.concat(bets, ignore_index=True)
        if len(B) < 5:
            out.append({"thr": thr, "n": int(len(B))})
            continue
        t, p = stats.ttest_1samp(B.pnl, 0)
        out.append({"thr": thr, "n": int(len(B)), "roi_pct": round(100 * B.pnl.mean(), 2),
                    "wr": round(float(B.win.mean()), 4), "avg_odds": round(float(B.odds.mean()), 3),
                    "p": float(p),
                    "by_sel": B.groupby("sel")["pnl"].agg(["count", "mean"]).round(3).to_dict()})
    return out

d35f = df[(df.competition == "InstantLeague-8035")].copy()
cut = d35f.expected_start.quantile(0.7)
tr, te = d35f[d35f.expected_start <= cut], d35f[d35f.expected_start > cut]
print(f"\n8035 walk-forward: train {len(tr)} / test {len(te)}")
res_tr = pair_value_eval(tr.iloc[: int(len(tr) * 0.7)], tr.iloc[int(len(tr) * 0.7):])
print("inner-train scan (train70/train30):")
for r in res_tr:
    print("  ", r)
res_te = pair_value_eval(tr, te)
print("TEST (pair means from full train):")
for r in res_te:
    print("  ", r)
results["pair_value_8035_innertrain"] = res_tr
results["pair_value_8035_test"] = res_te

# pooled new leagues with LOO means (small data) -- report as pooled-newleagues
dn = df[df.competition != "InstantLeague-8035"].copy()
out_new = []
for sel in ["home", "draw", "away"]:
    col = "l" + sel[0]
    g = dn.groupby("pair")[col]
    dn["pn"] = g.transform("count")
    dn["ps"] = g.transform("sum")
    sub = dn[dn.pn >= 3].copy()
    jit = sub[col] - (sub.ps - sub[col]) / (sub.pn - 1)
    for thr in (0.02, 0.04, 0.06):
        pick = sub[(jit >= thr) & (sub["open_" + sel] >= 1.6)]
        if len(pick) < 5:
            continue
        pnl = (pick.res == sel).astype(int) * pick["open_" + sel] - 1
        t, p = stats.ttest_1samp(pnl, 0)
        out_new.append({"sel": sel, "thr": thr, "n": int(len(pick)),
                        "roi_pct": round(100 * pnl.mean(), 2),
                        "wr": round(float((pick.res == sel).mean()), 4),
                        "avg_odds": round(float(pick["open_" + sel].mean()), 3), "p": float(p)})
print("pooled new leagues (LOO jitter, caution in-sample):")
for r in out_new:
    print("  ", r)
results["pair_value_newleagues_loo"] = out_new

with open("exports/wf4_jitter_structure.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=1, default=str)
print("saved exports/wf4_jitter_structure.json")
