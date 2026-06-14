# WF4 jitter/drift - step 2: drift open->close predictivity + follow_drift ROI + CLV
# Input: scripts/_wf4_jitter_data.pkl ; output: exports/wf4_jitter_drift.json
import sys, json, pickle, math
import numpy as np
import pandas as pd
from scipy import stats

with open("scripts/_wf4_jitter_data.pkl", "rb") as f:
    df = pickle.load(f)

df = df[df.n_snaps >= 2].copy()
df["dt_min"] = (df["close_captured_at"] - df["open_captured_at"]).dt.total_seconds() / 60
print("multi-snap events:", len(df))
print(df.groupby("competition").size())
print("dt open->close (min):", df.dt_min.describe()[["mean", "50%", "max"]].to_dict())

# outcome
df["res"] = np.where(df.score_a > df.score_b, "home", np.where(df.score_a < df.score_b, "away", "draw"))

# long format: one row per selection
rows = []
for sel in ["home", "draw", "away"]:
    sub = pd.DataFrame({
        "event_id": df.event_id, "competition": df.competition,
        "expected_start": df.expected_start, "sel": sel,
        "open_odds": df["open_" + sel], "close_odds": df["close_" + sel],
        "win": (df.res == sel).astype(int), "n_snaps": df.n_snaps,
    })
    # normalized open implied prob
    inv = 1 / df[["open_home", "open_draw", "open_away"]]
    sub["p_open"] = (1 / df["open_" + sel]) / inv.sum(axis=1)
    invc = 1 / df[["close_home", "close_draw", "close_away"]]
    sub["p_close"] = (1 / df["close_" + sel]) / invc.sum(axis=1)
    rows.append(sub)
L = pd.concat(rows, ignore_index=True)
L["drift"] = np.log(L.open_odds / L.close_odds)  # >0 = odds dropped (steam in)

print("\ndrift distribution (log odds-ratio):")
print(L.drift.describe([.05, .25, .5, .75, .95]))
print("share of selections with any move:", (L.drift.abs() > 1e-9).mean())

results = {"n_multi_events": int(len(df))}

# ---- (A) Does close beat open in log-loss? (is the close 'truer'?) ----
for comp_grp, name in [(L.competition == "InstantLeague-8035", "8035"),
                       (L.competition != "InstantLeague-8035", "newleagues")]:
    sub = L[comp_grp & (L.drift.abs() > 1e-9).groupby(L.event_id).transform("max").astype(bool)]
    # per event log-loss
    ev = sub[sub.win == 1]
    ll_open = -np.log(ev.p_open).mean()
    ll_close = -np.log(ev.p_close).mean()
    # paired t-test on per-event logloss diff
    d = (-np.log(ev.p_open)) - (-np.log(ev.p_close))
    t, p = stats.ttest_1samp(d, 0) if len(d) > 5 else (np.nan, np.nan)
    results[f"logloss_{name}"] = {"n_events": int(len(ev)), "ll_open": round(ll_open, 5),
                                  "ll_close": round(ll_close, 5),
                                  "diff_mean": round(float(d.mean()), 5),
                                  "t": round(float(t), 3), "p": float(p)}
    print(f"\n[{name}] log-loss open={ll_open:.5f} close={ll_close:.5f} "
          f"diff={d.mean():+.5f} (t={t:.2f}, p={p:.2e}, n={len(ev)})")

# ---- (B) calibration by drift bucket: actual - p_open ----
moved = L[L.drift.abs() > 1e-9].copy()
moved["bucket"] = pd.cut(moved.drift, [-np.inf, -0.05, -0.01, -1e-9, 1e-9, 0.01, 0.05, np.inf],
                         labels=["<-5%", "-5..-1%", "-1..0%", "0", "0..1%", "1..5%", ">5%"])
cal = moved.groupby("bucket", observed=True).agg(
    n=("win", "size"), actual=("win", "mean"), implied=("p_open", "mean"),
    avg_open=("open_odds", "mean"))
cal["edge"] = cal.actual - cal.implied
print("\ncalibration by drift bucket (all leagues, all selections):")
print(cal.round(4))
results["calib_by_drift"] = cal.reset_index().astype(str).to_dict("records")

# ---- (C) strategy follow_drift: back selection with drift >= thr ----
# walk-forward on 8035: 70/30 by expected_start; pooled on new leagues
def run_strategy(sub, thr, odds_col, odds_min=1.0, odds_max=100.0):
    bets = sub[(sub.drift >= thr) & (sub[odds_col] >= odds_min) & (sub[odds_col] <= odds_max)]
    if len(bets) == 0:
        return None
    pnl = bets.win * bets[odds_col] - 1
    roi = pnl.mean()
    t, p = stats.ttest_1samp(pnl, 0) if len(pnl) > 5 else (np.nan, 1.0)
    return {"n": int(len(bets)), "roi_pct": round(100 * roi, 2), "wr": round(float(bets.win.mean()), 4),
            "avg_odds": round(float(bets[odds_col].mean()), 3), "p": float(p)}

m35 = moved[moved.competition == "InstantLeague-8035"].copy()
cut = m35.expected_start.quantile(0.7)
train, test = m35[m35.expected_start <= cut], m35[m35.expected_start > cut]
mnew = moved[moved.competition != "InstantLeague-8035"]

n_tests = 0
scan = []
for thr in [0.0001, 0.01, 0.02, 0.03, 0.05]:
    for odds_min in [1.0, 1.6]:
        for odds_col in ["open_odds", "close_odds"]:
            n_tests += 1
            r_tr = run_strategy(train, thr, odds_col, odds_min)
            scan.append({"thr": thr, "odds_min": odds_min, "odds_col": odds_col, "train": r_tr})
print(f"\nscan train 8035 ({n_tests} variants):")
for s in scan:
    if s["train"]:
        print(f"  thr={s['thr']} omin={s['odds_min']} {s['odds_col']}: {s['train']}")

results["scan_train_8035"] = scan
results["n_tests_scanned_strategyC"] = n_tests

# evaluate ALL variants on test too (transparency) + pooled new leagues
test_eval = []
for s in scan:
    r_te = run_strategy(test, s["thr"], s["odds_col"], s["odds_min"])
    r_nw = run_strategy(mnew, s["thr"], s["odds_col"], s["odds_min"])
    test_eval.append({**{k: s[k] for k in ["thr", "odds_min", "odds_col"]},
                      "test_8035": r_te, "pooled_new": r_nw})
print("\ntest 8035 + pooled new leagues:")
for s in test_eval:
    print(f"  thr={s['thr']} omin={s['odds_min']} {s['odds_col']}:")
    print(f"     test={s['test_8035']}")
    print(f"     new ={s['pooled_new']}")
results["test_eval"] = test_eval

# ---- (D) CLV capturable: for picks drift>=thr, open/close odds ratio ----
for name, sub in [("8035_test", test), ("newleagues", mnew)]:
    picks = sub[sub.drift >= 0.01]
    clv = (picks.open_odds / picks.close_odds - 1)
    results[f"clv_{name}"] = {"n": int(len(picks)), "mean_clv_pct": round(100 * clv.mean(), 2),
                              "median_clv_pct": round(100 * clv.median(), 2)}
    print(f"\nCLV {name}: n={len(picks)} mean={100*clv.mean():.2f}% median={100*clv.median():.2f}%")

with open("exports/wf4_jitter_drift.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=1, default=str)
print("\nsaved exports/wf4_jitter_drift.json")
