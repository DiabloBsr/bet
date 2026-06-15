"""ÉTUDE PAR TOTAL DE BUTS (0→6) — pour chaque total exact : les conditions qui le
font pencher (signaux, OOS+FDR) + BACKTEST LIVE (hit-rate réel + EV à la cote offerte
off_t{k}). Répond : peut-on TAPER chaque total en live, et est-ce rentable ?
Split chrono 70/30. Usage: ./.venv/Scripts/python.exe scripts/_per_total.py [total]"""
from __future__ import annotations
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy.stats import norm

CSV = Path(__file__).resolve().parents[1] / "exports" / "combokeys_features.csv"
MIN_TR, MIN_TE = 120, 40
ONLY = int(sys.argv[1]) if len(sys.argv) > 1 else None

df = pd.read_csv(CSV).sort_values("expected_start").reset_index(drop=True)
df["abs_diff"] = df.lam_diff.abs()
df["spread"] = (df.oh - df.oa).abs()
df["odds_sum"] = df.oh + df.oa + df.od
df["home_fav"] = (df.oh < df.oa).map({True: "home", False: "away"})
df["fit_quality"] = df.fit_quality.fillna("?")
# total book modal (échelle off_t devig)
oc = [f"off_t{k}" for k in range(7)]
prob = (1/df[oc]).div((1/df[oc]).sum(axis=1), axis=0)
df["book_modal_total"] = prob.idxmax(axis=1).str.replace("off_t", "", regex=False)

cut = int(len(df)*0.70); tr, te = df.iloc[:cut], df.iloc[cut:]
NUMERIC = ["fav","dog","odds_ratio","od","lam_tot","lam_diff","abs_diff","lam_h","lam_a",
           "p_btts","p_total_eq3","p_total_le2","p_total_ge4","dc_X2","odds_sum","spread"]
CATEG = ["home_fav","fit_quality","book_modal_total"]

def qbins(col, k=5):
    try:
        ed = pd.qcut(tr[col].dropna(), k, duplicates="drop").cat.categories
        return [ed[0].left] + [c.right for c in ed]
    except Exception:
        return None

def gen():
    for col in NUMERIC:
        b = qbins(col)
        if not b: continue
        ctr = pd.cut(tr[col], b, include_lowest=True); cte = pd.cut(te[col], b, include_lowest=True)
        for v in ctr.cat.categories:
            yield f"{col}∈{v}", (ctr==v).values, (cte==v).values
    for col in CATEG:
        for v in tr[col].value_counts().pipe(lambda s: s[s>=MIN_TR]).index:
            yield f"{col}={v}", (tr[col]==v).values, (te[col]==v).values
    for c1,c2 in [("lam_tot","abs_diff"),("lam_tot","p_btts"),("fav","lam_tot"),("lam_tot","od"),("lam_diff","p_btts")]:
        b1,b2 = qbins(c1,4), qbins(c2,4)
        if not (b1 and b2): continue
        s1=pd.cut(tr[c1],b1,include_lowest=True).astype(str)+" & "+pd.cut(tr[c2],b2,include_lowest=True).astype(str)
        s1te=pd.cut(te[c1],b1,include_lowest=True).astype(str)+" & "+pd.cut(te[c2],b2,include_lowest=True).astype(str)
        for v in s1.value_counts().index:
            m=(s1==v).values
            if m.sum()<MIN_TR: continue
            yield f"[{c1}&{c2}] {v}", m, (s1te==v).values

BINS = list(gen())
def z(p_obs, p0, n):
    if n<=0 or p0<=0 or p0>=1: return 0.0
    return (p_obs-p0)/math.sqrt(p0*(1-p0)/n)

targets = [ONLY] if ONLY is not None else list(range(7))
print(f"{'='*120}")
print(f"ÉTUDE PAR TOTAL (TR n={len(tr)} / TE n={len(te)}) — hit OOS + EV à la cote offerte (off_t)")
print(f"{'='*120}")
summary = []
for t in targets:
    base = (tr.total_goals==t).mean()
    cote_med = df[f"off_t{t}"].median()
    cand = []
    for label, mtr, mte in BINS:
        ntr, nte = int(mtr.sum()), int(mte.sum())
        if ntr<MIN_TR or nte<MIN_TE: continue
        p_tr = (tr.total_goals[mtr]==t).mean(); p_te = (te.total_goals[mte]==t).mean()
        if p_tr <= base: continue
        # EV au prix offert sur le TEST (cote = off_t du sous-ensemble test)
        sub_te = te[mte].dropna(subset=[f"off_t{t}"])
        ev_te = ((sub_te.total_goals==t)*sub_te[f"off_t{t}"]-1).mean() if len(sub_te)>20 else float('nan')
        cand.append(dict(cond=label, n_tr=ntr, n_te=nte, tr=p_tr, te=p_te, base=base,
                         lift=p_tr/base, z=z(p_tr,base,ntr), ev_te=ev_te,
                         cote=sub_te[f"off_t{t}"].mean() if len(sub_te)>0 else np.nan))
    if cand:
        for c in cand: c["p"]=1-norm.cdf(c["z"])
        srt=sorted(cand,key=lambda c:c["p"]); m=len(srt); thr=0
        for i,c in enumerate(srt,1):
            if c["p"]<=(i/m)*0.10: thr=i
        crit=srt[thr-1]["p"] if thr>0 else -1
        for c in cand: c["fdr"]=c["p"]<=crit
    keep=[c for c in cand if c["te"]>c["base"]]
    keep.sort(key=lambda c:c["te"], reverse=True)
    tl = "6+" if t==6 else str(t)
    print(f"\n■ TOTAL = {tl} buts  (base {base*100:.1f}%, cote médiane {cote_med:.1f}, {len(keep)} conditions répliquées)")
    if not keep:
        print("   aucune condition ne pousse ce total OOS."); summary.append((tl,base,None)); continue
    for c in keep[:5]:
        flag="✅FDR" if c.get("fdr") else "     "
        evs = f"EV {c['ev_te']*100:+.0f}%@{c['cote']:.1f}" if c['ev_te']==c['ev_te'] else "EV n/d"
        print(f"  {flag} {c['cond'][:46]:<46} n={c['n_tr']:>4}/{c['n_te']:<4} TR{c['tr']*100:>4.0f}% TE{c['te']*100:>4.0f}% ×{c['lift']:.2f} z={c['z']:.1f}  {evs}")
    best=keep[0]; summary.append((tl, base, best))

print(f"\n{'='*120}")
print("VERDICT PAR TOTAL — tapable en accuracy ? rentable (EV) en live ?")
print(f"{'='*120}")
print(f"{'total':>6}{'base':>7}{'meilleur hit OOS':>18}{'lift':>7}{'EV live':>9}  verdict")
for tl, base, best in summary:
    if best is None:
        print(f"{tl:>6}{base*100:>6.0f}%{'—':>18}{'—':>7}{'—':>9}  pas d'edge"); continue
    ev = best['ev_te']; evs = f"{ev*100:+.0f}%" if ev==ev else "n/d"
    tap = "🎯 tapable" if best['te']>base*1.25 else "~ faible"
    rent = " +RENTABLE" if (ev==ev and ev>0.03) else ""
    print(f"{tl:>6}{base*100:>6.0f}%{best['te']*100:>16.0f}%{best['lift']:>7.2f}{evs:>9}  {tap}{rent}")
print("\nNote: 'tapable' = hit OOS > 1.25× base (le total se penche vraiment). 'rentable' = EV>+3% à la cote offerte (rare, marché efficient).")
