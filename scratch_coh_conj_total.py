# -*- coding: utf-8 -*-
"""LOT coh_conj_total : 1X2 & Total vs reconstruction Poisson (grille 7x7, correlation incluse).
Discovery TRAIN -> validation OOS TEST, binomtest + bootstrap + BH-FDR, ROI_test>0 requis.
"""
import sys, json
import numpy as np, pandas as pd
from scipy.stats import poisson, binomtest

PATH = r"D:\AGENTOVA\SAMY\virtual-sports-scraper\data\vfoot_ml\conjunctive_wide.csv"
SELS = ["1 / > 3.5", "X / > 3.5", "2 / > 3.5", "1 / < 3.5", "X / < 3.5", "2 / < 3.5"]
PCOLS = [f"p|1X2 & Total|{s}" for s in SELS]
OCOLS = [f"o|1X2 & Total|{s}" for s in SELS]
usecols = ["ts", "sa", "sb", "phase", "p|1X2|1", "p|1X2|X", "p|1X2|2", "p|+/-|> 3.5"] + PCOLS + OCOLS

df = pd.read_csv(PATH, usecols=usecols)
n_total = len(df)
# rows needing complete data for reconstruction + comparison
need = ["sa", "sb", "phase", "p|1X2|1", "p|1X2|2"] + PCOLS
df = df.dropna(subset=need).reset_index(drop=True)
n_used = len(df)

p1 = df["p|1X2|1"].to_numpy(float)
px = df["p|1X2|X"].to_numpy(float)
p2 = df["p|1X2|2"].to_numpy(float)

# ---------- Poisson grid 7x7 (0..5 exact, 6 = tail >=6) ----------
K = 7
ks = np.arange(K)
I = np.repeat(ks, K).reshape(K, K).astype(float)   # home goals
J = I.T.copy()                                      # away goals
maskH = I > J
maskA = I < J
maskD = I == J
maskOver = (I + J) >= 4   # O/U 3.5 ; tail cell 6=6+ always >=4 when paired with >=0? 6+0=6>=4 yes.

def marg(lam):
    lam = np.asarray(lam, float)[:, None]
    pm = poisson.pmf(ks[None, :], lam)
    pm[:, K - 1] = poisson.sf(K - 2, lam[:, 0])  # P(X >= 6)
    return pm

def p12(lh, la):
    ph = marg(lh); pa = marg(la)
    Jt = np.einsum("ni,nj->nij", ph, pa)
    return Jt[:, maskH].sum(1), Jt[:, maskA].sum(1)

# ---------- vectorized Newton on (log lh, log la) matching (p1, p2) ----------
n = len(df)
d = p1 - p2
x = np.stack([np.log(np.clip(1.35 + d, 0.05, 8.0)), np.log(np.clip(1.35 - d, 0.05, 8.0))], 1)
eps = 1e-5
for it in range(60):
    lh = np.exp(x[:, 0]); la = np.exp(x[:, 1])
    P1, P2 = p12(lh, la)
    r1 = P1 - p1; r2 = P2 - p2
    mres = max(np.abs(r1).max(), np.abs(r2).max())
    if mres < 1e-9:
        break
    P1a, P2a = p12(np.exp(x[:, 0] + eps), la)
    P1b, P2b = p12(lh, np.exp(x[:, 1] + eps))
    J11 = (P1a - P1) / eps; J21 = (P2a - P2) / eps
    J12 = (P1b - P1) / eps; J22 = (P2b - P2) / eps
    det = J11 * J22 - J12 * J21
    det = np.where(np.abs(det) < 1e-14, 1e-14, det)
    dx1 = (J22 * r1 - J12 * r2) / det
    dx2 = (-J21 * r1 + J11 * r2) / det
    x[:, 0] -= np.clip(dx1, -0.7, 0.7)
    x[:, 1] -= np.clip(dx2, -0.7, 0.7)
    x = np.clip(x, np.log(0.01), np.log(9.0))

lh = np.exp(x[:, 0]); la = np.exp(x[:, 1])
P1, P2 = p12(lh, la)
res = np.maximum(np.abs(P1 - p1), np.abs(P2 - p2))
conv = res < 1e-6
frac_conv = float(conv.mean())

# ---------- reconstruction of the 6 joint selections (correlation included) ----------
ph = marg(lh); pa = marg(la)
Jt = np.einsum("ni,nj->nij", ph, pa)
recon = {
    "1 / > 3.5": Jt[:, maskH & maskOver].sum(1),
    "X / > 3.5": Jt[:, maskD & maskOver].sum(1),
    "2 / > 3.5": Jt[:, maskA & maskOver].sum(1),
    "1 / < 3.5": Jt[:, maskH & ~maskOver].sum(1),
    "X / < 3.5": Jt[:, maskD & ~maskOver].sum(1),
    "2 / < 3.5": Jt[:, maskA & ~maskOver].sum(1),
}
pmkt = {s: df[f"p|1X2 & Total|{s}"].to_numpy(float) for s in SELS}
gap = {s: recon[s] - pmkt[s] for s in SELS}  # >0 => market underprices => value at offered odds

# per-match incoherence: total variation distance across the 6 cells
G = np.stack([np.abs(gap[s]) for s in SELS], 1)
tvd = 0.5 * G.sum(1)
sum6 = np.stack([pmkt[s] for s in SELS], 1).sum(1)

# sanity: recon P(Over3.5) vs market devig p|+/-|>3.5
pov_mkt = df["p|+/-|> 3.5"].to_numpy(float)
pov_rec = recon["1 / > 3.5"] + recon["X / > 3.5"] + recon["2 / > 3.5"]
ok = np.isfinite(pov_mkt)
sanity_over_mae = float(np.abs(pov_rec[ok] - pov_mkt[ok]).mean())
sanity_over_mean = float((pov_rec[ok] - pov_mkt[ok]).mean())

# ---------- settlement ----------
sa = df["sa"].to_numpy(int); sb = df["sb"].to_numpy(int)
over = (sa + sb) >= 4
w1 = sa > sb; wx = sa == sb; w2 = sa < sb
win = {
    "1 / > 3.5": w1 & over, "X / > 3.5": wx & over, "2 / > 3.5": w2 & over,
    "1 / < 3.5": w1 & ~over, "X / < 3.5": wx & ~over, "2 / < 3.5": w2 & ~over,
}
odds = {s: df[f"o|1X2 & Total|{s}"].to_numpy(float) for s in SELS}
is_train = (df["phase"] == "train").to_numpy()
is_test = (df["phase"] == "test").to_numpy()

rng = np.random.default_rng(261)
def roi_stats(mask, s):
    m = mask & np.isfinite(odds[s]) & conv
    nb = int(m.sum())
    if nb == 0:
        return dict(n=0, roi=None, wr=None, be=None)
    o = odds[s][m]; w = win[s][m].astype(float)
    pnl = w * o - 1.0
    roi = float(pnl.mean())
    return dict(n=nb, roi=roi, wr=float(w.mean()), be=float((1.0 / o).mean()),
                k=int(w.sum()), pnl=pnl, o=o, w=w)

cells = []
for s in SELS:
    for thr in (0.02, 0.04, 0.06):
        sel_mask = gap[s] > thr
        tr = roi_stats(sel_mask & is_train, s)
        te = roi_stats(sel_mask & is_test, s)
        cells.append(dict(sel=s, thr=thr, tr=tr, te=te))

# discovery on TRAIN: ROI_train > 0 with n_train >= 30
tested = [c for c in cells if c["tr"]["n"] >= 30]
qual = [c for c in tested if c["tr"]["roi"] is not None and c["tr"]["roi"] > 0]

# validation OOS: binomtest vs breakeven + bootstrap P(ROI<=0), BH-FDR over qualified cells
for c in qual:
    te = c["te"]
    if te["n"] < 10:
        c["p_bin"] = 1.0; c["p_boot"] = 1.0; c["p"] = 1.0
        continue
    pb = binomtest(te["k"], te["n"], te["be"], alternative="greater").pvalue
    pnl = te["pnl"]
    idx = rng.integers(0, len(pnl), size=(10000, len(pnl)))
    rois = pnl[idx].mean(1)
    pboot = float((rois <= 0).mean())
    c["p_bin"] = float(pb); c["p_boot"] = pboot; c["p"] = max(float(pb), pboot)

# BH-FDR (alpha=0.05) over qualified cells
mq = len(qual)
survivors = []
if mq:
    order = sorted(range(mq), key=lambda i: qual[i]["p"])
    passed = -1
    for rank, i in enumerate(order, 1):
        if qual[i]["p"] <= 0.05 * rank / mq:
            passed = rank
    if passed > 0:
        survivors = [qual[i] for i in order[:passed]]
    survivors = [c for c in survivors if c["te"]["roi"] is not None and c["te"]["roi"] > 0]

def fmt(c):
    return dict(sel=c["sel"], thr_pp=int(c["thr"] * 100),
                n_train=c["tr"]["n"], roi_train_pct=None if c["tr"]["roi"] is None else round(c["tr"]["roi"] * 100, 2),
                n_test=c["te"]["n"], roi_test_pct=None if c["te"]["roi"] is None else round(c["te"]["roi"] * 100, 2),
                wr_test=None if c["te"].get("wr") is None else round(c["te"]["wr"], 4),
                be_test=None if c["te"].get("be") is None else round(c["te"]["be"], 4),
                p_bin=c.get("p_bin"), p_boot=c.get("p_boot"))

out = dict(
    n_rows_file=n_total, n_rows_used=n_used, frac_newton_converged=round(frac_conv, 5),
    lambda_home_mean=round(float(lh[conv].mean()), 4), lambda_away_mean=round(float(la[conv].mean()), 4),
    sanity_recon_over_vs_market=dict(mae=round(sanity_over_mae, 5), mean_bias=round(sanity_over_mean, 5)),
    sum_of_6_market_probs=dict(mean=round(float(sum6.mean()), 5), sd=round(float(sum6.std()), 5)),
    tvd_per_match=dict(mean=round(float(tvd[conv].mean()), 5), median=round(float(np.median(tvd[conv])), 5),
                       p95=round(float(np.percentile(tvd[conv], 95)), 5), max=round(float(tvd[conv].max()), 5)),
    gap_mean_pp={s: round(float(gap[s][conv].mean()) * 100, 3) for s in SELS},
    gap_mae_pp={s: round(float(np.abs(gap[s][conv]).mean()) * 100, 3) for s in SELS},
    n_cells_tested=len(tested),
    n_qualified_train=len(qual),
    qualified=[fmt(c) for c in sorted(qual, key=lambda c: c.get("p", 1.0))],
    n_survivors_oos_fdr=len(survivors),
    survivors=[fmt(c) for c in survivors],
    all_cells=[fmt(c) for c in cells],
)
print(json.dumps(out, indent=1))
