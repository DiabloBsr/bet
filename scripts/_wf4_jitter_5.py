# WF4 jitter/drift - step 5: mirror tests
# (a) NEGATIVE pair-jitter strategy (odds below pair mean) at open odds, walk-forward 8035
# (b) repricing component: close below pair mean (jit_close<=-thr) bet at close odds
# (c) follow_drift breakdown by selection
# Output: exports/wf4_jitter_mirror.json
import sys, json, pickle
import numpy as np
import pandas as pd
from scipy import stats

with open("scripts/_wf4_jitter_data.pkl", "rb") as f:
    df = pickle.load(f)

df = df[df.competition == "InstantLeague-8035"].copy()
df["res"] = np.where(df.score_a > df.score_b, "home", np.where(df.score_a < df.score_b, "away", "draw"))
df["pair"] = df.team_a + "|" + df.team_b
cut = df.expected_start.quantile(0.7)
tr, te = df[df.expected_start <= cut], df[df.expected_start > cut]
results = {}

SELS = ["home", "draw", "away"]
means = {s: tr.groupby("pair")["open_" + s].apply(lambda x: np.log(x).mean()) for s in SELS}
cnts = tr.groupby("pair").size()

def eval_bets(B):
    if len(B) < 5:
        return {"n": int(len(B))}
    t, p = stats.ttest_1samp(B.pnl, 0)
    return {"n": int(len(B)), "roi_pct": round(100 * B.pnl.mean(), 2),
            "wr": round(float(B.win.mean()), 4), "avg_odds": round(float(B.odds.mean()), 3),
            "p": float(p)}

# ---- (a) negative jitter at open odds (test set) ----
print("== (a) negative pair-jitter (odds below train pair mean), TEST, open odds ==")
out_a = []
for thr in (0.02, 0.04, 0.06):
    for sel_set, name in [(SELS, "all"), (["home", "away"], "h+a")]:
        bets = []
        for s in sel_set:
            t2 = te.join(means[s].rename("pm"), on="pair").join(cnts.rename("pc"), on="pair")
            t2 = t2[t2.pc >= 3]
            jit = np.log(t2["open_" + s]) - t2.pm
            pick = t2[(jit <= -thr) & (t2["open_" + s] >= 1.6)]
            bets.append(pd.DataFrame({"pnl": (pick.res == s).astype(int) * pick["open_" + s] - 1,
                                      "win": (pick.res == s).astype(int), "odds": pick["open_" + s]}))
        r = eval_bets(pd.concat(bets, ignore_index=True))
        out_a.append({"thr": -thr, "sels": name, **r})
        print(f"  thr<=-{thr} [{name}]: {r}")
results["neg_jitter_open_test"] = out_a

# ---- (b) repricing: close below pair mean, bet at close odds (multi-snap, moved) ----
print("\n== (b) close below pair mean (repricing), bet at CLOSE odds, walk-forward ==")
m = df[df.n_snaps >= 2].copy()
m["moved"] = (np.log(m.open_home / m.close_home).abs() > 1e-9) | \
             (np.log(m.open_draw / m.close_draw).abs() > 1e-9) | \
             (np.log(m.open_away / m.close_away).abs() > 1e-9)
m = m[m.moved]
mtr, mte = m[m.expected_start <= cut], m[m.expected_start > cut]
out_b = []
for thr in (0.01, 0.02, 0.04):
    for split, name in [(mtr, "train"), (mte, "test")]:
        bets = []
        for s in SELS:
            t2 = split.join(means[s].rename("pm"), on="pair").join(cnts.rename("pc"), on="pair")
            t2 = t2[t2.pc >= 3]
            jitc = np.log(t2["close_" + s]) - t2.pm
            drift = np.log(t2["open_" + s] / t2["close_" + s])
            pick = t2[(jitc <= -thr) & (drift > 1e-9) & (t2["close_" + s] >= 1.6)]
            bets.append(pd.DataFrame({"pnl": (pick.res == s).astype(int) * pick["close_" + s] - 1,
                                      "win": (pick.res == s).astype(int), "odds": pick["close_" + s]}))
        r = eval_bets(pd.concat(bets, ignore_index=True))
        out_b.append({"thr": -thr, "split": name, **r})
        print(f"  jit_close<=-{thr}, drift>0 [{name}]: {r}")
results["repricing_close_below_mean"] = out_b

# ---- (c) follow_drift >=3% breakdown by selection, full 8035 + train/test ----
print("\n== (c) follow_drift>=3% by selection ==")
out_c = {}
for split, name in [(mtr, "train"), (mte, "test"), (m, "pooled")]:
    rows = []
    for s in SELS:
        drift = np.log(split["open_" + s] / split["close_" + s])
        pick = split[drift >= 0.03]
        rows.append(pd.DataFrame({"pnl": (pick.res == s).astype(int) * pick["open_" + s] - 1,
                                  "win": (pick.res == s).astype(int), "odds": pick["open_" + s],
                                  "sel": s,
                                  "pnl_close": (pick.res == s).astype(int) * pick["close_" + s] - 1}))
    B = pd.concat(rows, ignore_index=True)
    bysel = B.groupby("sel").agg(n=("pnl", "size"), roi_open=("pnl", "mean"),
                                 roi_close=("pnl_close", "mean"), wr=("win", "mean"),
                                 avg_odds=("odds", "mean")).round(4)
    print(f" [{name}]")
    print(bysel)
    ha = B[B.sel != "draw"]
    r_ha = eval_bets(ha)
    t, pcl = stats.ttest_1samp(ha.pnl_close, 0) if len(ha) > 5 else (np.nan, np.nan)
    print(f"  home+away only: open {r_ha} | close roi={100*ha.pnl_close.mean():.2f}% p={pcl:.3g}")
    out_c[name] = {"by_sel": bysel.reset_index().to_dict("records"),
                   "home_away_open": r_ha,
                   "home_away_close": {"roi_pct": round(100 * float(ha.pnl_close.mean()), 2),
                                       "p": float(pcl), "n": int(len(ha))}}
results["follow_drift_by_sel"] = out_c

with open("exports/wf4_jitter_mirror.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=1, default=str)
print("\nsaved exports/wf4_jitter_mirror.json")
