"""Chasse aux edges EXHAUSTIVE pour accuracy score-exact + total-buts.
Protocole anti-overfit : split chrono 70/30, découverte TRAIN, validation TEST,
réplication OOS obligatoire, correction multi-tests (Bonferroni + BH-FDR).
Inclut signaux normaux, géométrie de cotes, internes-marché, et ABSURDES."""
from __future__ import annotations
import sys, math, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd

CSV = Path(__file__).resolve().parents[1] / "exports" / "combokeys_features.csv"
SCORE_COLS = ['1-1','2-1','1-2','1-0','0-1','2-0','0-2','0-0','2-2','3-0','0-3','3-1','1-3','3-2','2-3']
MIN_TR, MIN_TE = 150, 60

df = pd.read_csv(CSV)
df = df.sort_values("expected_start").reset_index(drop=True)
n = len(df)
cut = int(n * 0.70)
tr, te = df.iloc[:cut].copy(), df.iloc[cut:].copy()
print(f"n={n}  TRAIN={len(tr)} ({tr.expected_start.min()[:10]}→{tr.expected_start.max()[:10]})  "
      f"TEST={len(te)} ({te.expected_start.min()[:10]}→{te.expected_start.max()[:10]})\n")

# baselines globaux (sur TRAIN) -------------------------------------------------
def rate(s, val): return (s == val).mean()
g_score = tr.exact_score.value_counts(normalize=True).to_dict()
g_total = tr.total_goals.value_counts(normalize=True).to_dict()
g_o25 = (tr.total_goals >= 3).mean(); g_u25 = (tr.total_goals <= 2).mean()
g_o35 = (tr.total_goals >= 4).mean()
print(f"Baseline TRAIN : modal score 1-1={g_score.get('1-1',0)*100:.1f}%  2-1={g_score.get('2-1',0)*100:.1f}%  "
      f"3-0={g_score.get('3-0',0)*100:.1f}% | total modal 3={g_total.get(3,0)*100:.1f}% | "
      f"Over2.5={g_o25*100:.0f}% Under2.5={g_u25*100:.0f}% Over3.5={g_o35*100:.0f}%\n")

# helpers de binning ------------------------------------------------------------
def qbins(col, k=5):
    """retourne une Series de labels de quantiles (sur TRAIN) appliqués à tout df."""
    try:
        edges = pd.qcut(tr[col].dropna(), k, duplicates="drop").cat.categories
    except Exception:
        return None
    bins = [edges[0].left] + [c.right for c in edges]
    return bins
def apply_bins(s, bins):
    return pd.cut(s, bins=bins, include_lowest=True)

# construire les colonnes de signaux (normaux + absurdes) -----------------------
def add_signals(d):
    d = d.copy()
    d["abs_diff"] = d.lam_diff.abs()
    d["home_fav"] = (d.oh < d.oa).map({True: "home", False: "away"})
    d["odds_sum"] = d.oh + d.oa + d.od
    d["oh_last"] = (d.oh * 100).round().astype("Int64") % 10
    d["oa_last"] = (d.oa * 100).round().astype("Int64") % 10
    d["od_last"] = (d.od * 100).round().astype("Int64") % 10
    d["oh_round"] = ((d.oh * 100).round().astype("Int64") % 10 == 0).map({True:"rond",False:"non"})
    d["mirror"] = ((d.oh - d.oa).abs() < 0.15).map({True:"sym",False:"asym"})
    d["a_initial"] = d.team_a.str[0]
    d["a_before_b"] = (d.team_a < d.team_b).map({True:"A<B",False:"A>=B"})
    d["namelen_diff"] = d.team_a.str.len() - d.team_b.str.len()
    # internes marché : score le moins cher offert (= modal selon le book)
    sc_cols = [c for c in d.columns if c.startswith("off_s_")]
    d["book_modal_score"] = d[sc_cols].idxmin(axis=1).str.replace("off_s_", "", regex=False)
    t_cols = [f"off_t{k}" for k in range(7) if f"off_t{k}" in d.columns]
    d["book_modal_total"] = d[t_cols].idxmin(axis=1).str.replace("off_t", "", regex=False)
    # peakedness du book : ratio 2e score le moins cher / 1er
    arr = np.sort(d[sc_cols].values, axis=1)
    d["ladder_ratio"] = arr[:, 1] / np.where(arr[:, 0] > 0, arr[:, 0], np.nan)
    return d

tr, te = add_signals(tr), add_signals(te)

# catalogue de signaux : (nom, type, k_bins_or_None) ----------------------------
NUMERIC = ["fav","dog","odds_ratio","od","lam_tot","lam_diff","abs_diff","lam_h","lam_a",
           "p_btts","p_total_eq3","p_total_le2","p_total_ge4","dc_X2","residual",
           "score_gap","total_gap","odds_sum","ladder_ratio","namelen_diff"]
CATEG   = ["home_fav","fit_quality","oh_last","oa_last","od_last","oh_round","mirror",
           "a_initial","a_before_b","book_modal_score","book_modal_total"]

candidates = []  # (signal_label, mask_tr, mask_te)
def gen_oneway():
    for col in NUMERIC:
        bins = qbins(col, 5)
        if bins is None: continue
        cats = apply_bins(tr[col], bins)
        cats_te = apply_bins(te[col], bins)
        for b in cats.cat.categories:
            yield (f"{col}∈{b}", (cats == b).values, (cats_te == b).values)
    for col in CATEG:
        vals = tr[col].value_counts()
        for v in vals[vals >= MIN_TR].index:
            yield (f"{col}={v}", (tr[col] == v).values, (te[col] == v).values)

def gen_twoway():
    pairs = [("abs_diff","lam_tot"),("home_fav","lam_tot"),("od","odds_ratio"),
             ("p_btts","lam_tot"),("book_modal_score","lam_tot"),("lam_diff","p_btts"),
             ("fav","lam_tot"),("odds_ratio","p_total_eq3")]
    for c1, c2 in pairs:
        b1 = qbins(c1, 4) if c1 in NUMERIC else None
        b2 = qbins(c2, 4) if c2 in NUMERIC else None
        s1tr = apply_bins(tr[c1], b1) if b1 else tr[c1]
        s2tr = apply_bins(tr[c2], b2) if b2 else tr[c2]
        s1te = apply_bins(te[c1], b1) if b1 else te[c1]
        s2te = apply_bins(te[c2], b2) if b2 else te[c2]
        combo_tr = s1tr.astype(str) + " & " + s2tr.astype(str)
        combo_te = s1te.astype(str) + " & " + s2te.astype(str)
        for v in combo_tr.value_counts().index:
            mtr = (combo_tr == v).values
            if mtr.sum() < MIN_TR: continue
            yield (f"[{c1}&{c2}] {v}", mtr, (combo_te == v).values)

def zscore(p_obs, p0, n):
    if n <= 0 or p0 <= 0 or p0 >= 1: return 0.0
    return (p_obs - p0) / math.sqrt(p0 * (1 - p0) / n)

# évaluation -------------------------------------------------------------------
rows_score, rows_total, rows_ou = [], [], []
for label, mtr, mte in list(gen_oneway()) + list(gen_twoway()):
    ntr, nte = int(mtr.sum()), int(mte.sum())
    if ntr < MIN_TR or nte < MIN_TE: continue
    sub_tr, sub_te = tr[mtr], te[mte]

    # --- SCORE EXACT : modal du bin (train) ---
    vc = sub_tr.exact_score.value_counts(normalize=True)
    m = vc.index[0]; r_tr = vc.iloc[0]; g = g_score.get(m, 1e-9)
    top3 = vc.head(3); top3_cum_tr = top3.sum()
    r_te = rate(sub_te.exact_score, m)
    top3_cum_te = sub_te.exact_score.isin(top3.index).mean()
    rows_score.append(dict(signal=label, n_tr=ntr, n_te=nte, modal=m,
                           tr=r_tr, te=r_te, base=g, lift=r_tr/g,
                           z=zscore(r_tr, g, ntr),
                           top3=" ".join(top3.index), top3_tr=top3_cum_tr, top3_te=top3_cum_te))

    # --- TOTAL EXACT : modal du bin ---
    vt = sub_tr.total_goals.value_counts(normalize=True)
    mt = vt.index[0]; rt_tr = vt.iloc[0]; gt = g_total.get(mt, 1e-9)
    rt_te = rate(sub_te.total_goals, mt)
    rows_total.append(dict(signal=label, n_tr=ntr, n_te=nte, modal=int(mt),
                           tr=rt_tr, te=rt_te, base=gt, lift=rt_tr/gt, z=zscore(rt_tr, gt, ntr)))

    # --- OVER/UNDER : meilleure direction du bin ---
    o25 = (sub_tr.total_goals >= 3).mean(); u25 = (sub_tr.total_goals <= 2).mean()
    o35 = (sub_tr.total_goals >= 4).mean(); u35 = (sub_tr.total_goals <= 3).mean()
    cand = [("Over2.5", o25, g_o25, (sub_te.total_goals>=3).mean()),
            ("Under2.5", u25, g_u25, (sub_te.total_goals<=2).mean()),
            ("Over3.5", o35, g_o35, (sub_te.total_goals>=4).mean()),
            ("Under3.5", u35, 1-g_o35, (sub_te.total_goals<=3).mean())]
    best = max(cand, key=lambda c: c[1])
    nm, p_tr, p0, p_te = best
    rows_ou.append(dict(signal=label, n_tr=ntr, n_te=nte, market=nm,
                        tr=p_tr, te=p_te, base=p0, lift=p_tr/p0, z=zscore(p_tr, p0, ntr)))

H = len(rows_score)  # nb hypothèses par target
z_bonf = abs(__import__("scipy.stats", fromlist=["norm"]).norm.ppf(0.05/(2*H)))
print(f"Hypothèses testées/target : {H}  → seuil Bonferroni z*={z_bonf:.2f}\n")

def bh_fdr(rows, q=0.10):
    """marque survivor_fdr=True selon Benjamini-Hochberg sur p-values unilatérales."""
    from scipy.stats import norm
    for r in rows: r["p"] = 1 - norm.cdf(r["z"])
    srt = sorted(rows, key=lambda r: r["p"])
    m = len(srt); thr = 0
    for i, r in enumerate(srt, 1):
        if r["p"] <= (i/m)*q: thr = i
    crit = srt[thr-1]["p"] if thr > 0 else -1
    for r in rows: r["fdr"] = r["p"] <= crit

for rows in (rows_score, rows_total, rows_ou): bh_fdr(rows)

def show(title, rows, key, extra=None, topn=18):
    print("="*120); print(title); print("="*120)
    # garder : réplication OOS (te >= base) ET volume, trier par te
    keep = [r for r in rows if r["te"] >= r["base"] and r["n_te"] >= MIN_TE]
    keep.sort(key=lambda r: r[key], reverse=True)
    for r in keep[:topn]:
        flag = "✅FDR" if r.get("fdr") else ("•bonf" if r["z"]>=z_bonf else "     ")
        line = (f"{flag} {r['signal'][:42]:<42} n={r['n_tr']:>4}/{r['n_te']:>4}  "
                f"TR={r['tr']*100:>4.0f}% TE={r['te']*100:>4.0f}% base={r['base']*100:>4.0f}% "
                f"lift×{r['lift']:.2f} z={r['z']:>4.1f}")
        if extra: line += "  " + extra(r)
        print(line)
    print()

show("🎯 SCORE EXACT — bins où un score est le + sur-représenté (tri par accuracy OOS)",
     rows_score, "te", extra=lambda r: f"→ {r['modal']}  (Top3 {r['top3']} = TE {r['top3_te']*100:.0f}%)")
show("🎯 SCORE EXACT — bins où le TOP-3 couvre le + (tri par Top3 OOS)",
     [r for r in rows_score], "top3_te", extra=lambda r: f"Top3={r['top3']}")
show("⚽ TOTAL EXACT — bins où un total est le + sur-représenté (tri par accuracy OOS)",
     rows_total, "te", extra=lambda r: f"→ {r['modal']} buts")
show("⚽ OVER/UNDER — bins les + tranchés (tri par accuracy OOS)",
     rows_ou, "te", extra=lambda r: f"→ {r['market']}")

# section absurdes : ont-ils survécu ?
print("="*120); print("🤡 SIGNAUX ABSURDES — ont-ils porté de l'info OOS ?"); print("="*120)
absurd_tags = ["_last","oh_round","mirror","a_initial","a_before_b","namelen","odds_sum"]
for tag, rows, tgt in [("score",rows_score,"te"),("total",rows_total,"te"),("OU",rows_ou,"te")]:
    surv = [r for r in rows if any(a in r["signal"] for a in absurd_tags) and r.get("fdr")]
    print(f"  [{tag}] absurdes survivants FDR : {len(surv)}" + ("" if not surv else
          " → " + ", ".join(f"{r['signal']}({r['te']*100:.0f}%)" for r in surv[:5])))
print("\n(0 survivant = confirme que ces indices sont du bruit, comme attendu.)")
