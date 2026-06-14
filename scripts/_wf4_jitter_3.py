# WF4 jitter/drift - step 3: rigorous drift predictivity (GLM offset) + who is right for movers
# 8035 only (new leagues have zero odds re-publication). Output: exports/wf4_jitter_glm.json
import sys, json, pickle
import numpy as np
import pandas as pd
from scipy import stats


def logit_offset_fit(y, x, offset):
    """Logistic regression y ~ const + x with offset. Newton-Raphson.
    Returns (params[const, beta], std_errs)."""
    X = np.column_stack([np.ones(len(x)), x])
    b = np.zeros(2)
    for _ in range(50):
        eta = offset + X @ b
        p = 1 / (1 + np.exp(-eta))
        W = p * (1 - p)
        g = X.T @ (y - p)
        H = (X * W[:, None]).T @ X
        step = np.linalg.solve(H, g)
        b = b + step
        if np.abs(step).max() < 1e-10:
            break
    cov = np.linalg.inv(H)
    return b, np.sqrt(np.diag(cov))

with open("scripts/_wf4_jitter_data.pkl", "rb") as f:
    df = pickle.load(f)

df = df[(df.competition == "InstantLeague-8035") & (df.n_snaps >= 2)].copy()
df["res"] = np.where(df.score_a > df.score_b, "home", np.where(df.score_a < df.score_b, "away", "draw"))

rows = []
for sel in ["home", "draw", "away"]:
    inv = 1 / df[["open_home", "open_draw", "open_away"]]
    invc = 1 / df[["close_home", "close_draw", "close_away"]]
    rows.append(pd.DataFrame({
        "event_id": df.event_id, "expected_start": df.expected_start, "sel": sel,
        "open_odds": df["open_" + sel], "close_odds": df["close_" + sel],
        "win": (df.res == sel).astype(int),
        "p_open": (1 / df["open_" + sel]) / inv.sum(axis=1),
        "p_close": (1 / df["close_" + sel]) / invc.sum(axis=1),
    }))
L = pd.concat(rows, ignore_index=True)
L["drift"] = np.log(L.open_odds / L.close_odds)
moved_ev = L.groupby("event_id")["drift"].transform(lambda s: s.abs().max() > 1e-9)
L = L[moved_ev].copy()
print("selections (events with movement):", len(L), "events:", L.event_id.nunique())

results = {}

# ---- GLM: win ~ drift with offset logit(p_open) ----
off = np.log(L.p_open / (1 - L.p_open)).values
b, se = logit_offset_fit(L.win.values.astype(float), L.drift.values, off)
z_drift = b[1] / se[1]
p_drift = 2 * (1 - stats.norm.cdf(abs(z_drift)))
print(f"GLM offset: const={b[0]:+.4f} (se {se[0]:.4f})  drift={b[1]:+.3f} (se {se[1]:.3f}, z={z_drift:.2f}, p={p_drift:.3g})")
results["glm_all"] = {"coef_drift": round(float(b[1]), 3),
                      "p_drift": float(p_drift),
                      "coef_const": round(float(b[0]), 4),
                      "n": int(len(L))}

# robustness: cluster by event via bootstrap over events (drift coef)
rng = np.random.default_rng(42)
Lix = L.set_index("event_id")
eids = L.event_id.unique()
boots = []
for _ in range(400):
    samp = rng.choice(eids, size=len(eids), replace=True)
    bl = Lix.loc[samp].reset_index()
    offb = np.log(bl.p_open / (1 - bl.p_open)).values
    try:
        bb, _ = logit_offset_fit(bl.win.values.astype(float), bl.drift.values, offb)
        boots.append(bb[1])
    except Exception:
        pass
boots = np.array(boots)
p_boot = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
p_boot = max(p_boot, 1 / len(boots))
print(f"bootstrap (cluster event): coef={boots.mean():.3f} sd={boots.std():.3f} p~{p_boot:.4f}")
results["glm_bootstrap_cluster"] = {"coef_mean": round(float(boots.mean()), 3),
                                    "coef_sd": round(float(boots.std()), 3),
                                    "p_two_sided": float(p_boot), "n_boot": len(boots)}

# ---- who is right for big movers: actual vs p_open vs p_close ----
for lo, hi, name in [(0.03, np.inf, "drift>=3%"), (0.05, np.inf, "drift>=5%"),
                     (-np.inf, -0.03, "drift<=-3%"), (-np.inf, -0.05, "drift<=-5%")]:
    sub = L[(L.drift >= lo) & (L.drift <= hi)] if lo > -np.inf else L[L.drift <= hi]
    if lo > -np.inf and hi == np.inf:
        sub = L[L.drift >= lo]
    n = len(sub)
    act, po, pc = sub.win.mean(), sub.p_open.mean(), sub.p_close.mean()
    # binomial tests actual vs open-implied and vs close-implied
    pv_open = stats.binomtest(int(sub.win.sum()), n, po).pvalue if n > 0 else None
    pv_close = stats.binomtest(int(sub.win.sum()), n, pc).pvalue if n > 0 else None
    print(f"{name}: n={n} actual={act:.4f} p_open={po:.4f} (p={pv_open:.3g}) p_close={pc:.4f} (p={pv_close:.3g})")
    results[f"movers_{name}"] = {"n": int(n), "actual": round(float(act), 4),
                                 "p_open": round(float(po), 4), "pv_vs_open": float(pv_open),
                                 "p_close": round(float(pc), 4), "pv_vs_close": float(pv_close)}

# ---- one-sided pooled test: all positive-drift selections vs all negative ----
pos, neg = L[L.drift > 1e-9], L[L.drift < -1e-9]
e_pos = pos.win.mean() - pos.p_open.mean()
e_neg = neg.win.mean() - neg.p_open.mean()
se = np.sqrt(pos.win.var() / len(pos) + neg.win.var() / len(neg))
z = (e_pos - e_neg) / se
print(f"\npooled: edge(pos)={e_pos:+.4f} (n={len(pos)}) edge(neg)={e_neg:+.4f} (n={len(neg)}) z={z:.2f}")
results["pooled_pos_vs_neg"] = {"edge_pos": round(float(e_pos), 4), "n_pos": int(len(pos)),
                                "edge_neg": round(float(e_neg), 4), "n_neg": int(len(neg)),
                                "z": round(float(z), 2),
                                "p": float(2 * (1 - stats.norm.cdf(abs(z))))}

with open("exports/wf4_jitter_glm.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=1, default=str)
print("saved exports/wf4_jitter_glm.json")
