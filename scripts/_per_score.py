"""Étude PAR SCORE : pour chaque score exact, trouve les conditions (signaux,
y compris absurdes) qui maximisent P(ce score). Split chrono 70/30, validation
OOS, FDR. Sortie : la 'recette' de chaque score + JSON exports/per_score_edges.json.
Usage: ./.venv/Scripts/python.exe scripts/_per_score.py [score_cible]"""
from __future__ import annotations
import sys, json, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "exports" / "combokeys_features.csv"
SCORES = ['0-0','1-0','0-1','1-1','2-0','0-2','2-1','1-2','2-2','3-0','0-3','3-1','1-3','3-2','2-3']
MIN_TR, MIN_TE = 120, 40
ONLY = sys.argv[1] if len(sys.argv) > 1 else None

df = pd.read_csv(CSV).sort_values("expected_start").reset_index(drop=True)
# --- signaux (normaux + géométrie + internes-marché + absurdes) ---
df["abs_diff"] = df.lam_diff.abs()
df["home_fav"] = (df.oh < df.oa).map({True:"home",False:"away"})
df["odds_sum"] = df.oh + df.oa + df.od
df["spread"] = (df.oh - df.oa).abs()           # fourchette home/away
df["draw_short"] = (df.od < df[["oh","oa"]].min(axis=1)).map({True:"nul_court",False:"non"})
df["oh_last"] = ((df.oh*100).round().astype(int) % 10).astype(str)
df["oa_last"] = ((df.oa*100).round().astype(int) % 10).astype(str)
df["a_initial"] = df.team_a.str[0]
df["mirror"] = ((df.oh-df.oa).abs()<0.15).map({True:"sym",False:"asym"})
sc_cols = [c for c in df.columns if c.startswith("off_s_")]
df["book_modal_score"] = df[sc_cols].idxmin(axis=1).str.replace("off_s_","",regex=False)
t_cols = [f"off_t{k}" for k in range(7) if f"off_t{k}" in df.columns]
df["book_modal_total"] = df[t_cols].idxmin(axis=1).str.replace("off_t","",regex=False)

cut = int(len(df)*0.70); tr, te = df.iloc[:cut], df.iloc[cut:]

NUMERIC = ["fav","dog","odds_ratio","od","lam_tot","lam_diff","abs_diff","lam_h","lam_a",
           "p_btts","p_total_eq3","p_total_le2","p_total_ge4","dc_X2","residual","odds_sum","spread"]
CATEG = ["home_fav","fit_quality","oh_last","oa_last","a_initial","mirror","draw_short",
         "book_modal_score","book_modal_total"]

def qbins(col, k=5):
    try:
        edges = pd.qcut(tr[col].dropna(), k, duplicates="drop").cat.categories
        return [edges[0].left] + [c.right for c in edges]
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
    # 2-way ciblés bas-score / dominance
    pairs = [("lam_tot","p_btts"),("lam_tot","abs_diff"),("lam_diff","p_btts"),
             ("fav","lam_tot"),("lam_tot","od"),("p_btts","spread")]
    for c1,c2 in pairs:
        b1,b2 = qbins(c1,4), qbins(c2,4)
        if not (b1 and b2): continue
        s1=pd.cut(tr[c1],b1,include_lowest=True).astype(str)+" & "+pd.cut(tr[c2],b2,include_lowest=True).astype(str)
        s1te=pd.cut(te[c1],b1,include_lowest=True).astype(str)+" & "+pd.cut(te[c2],b2,include_lowest=True).astype(str)
        for v in s1.value_counts().index:
            m=(s1==v).values
            if m.sum()<MIN_TR: continue
            yield f"[{c1}&{c2}] {v}", m, (s1te==v).values

# pré-calc des bins (une fois)
BINS = list(gen())
out = {}
targets = [ONLY] if ONLY else SCORES
for score in targets:
    base = (tr.exact_score==score).mean()
    cand = []
    for label, mtr, mte in BINS:
        ntr, nte = int(mtr.sum()), int(mte.sum())
        if ntr<MIN_TR or nte<MIN_TE: continue
        p_tr = (tr.exact_score[mtr]==score).mean()
        p_te = (te.exact_score[mte]==score).mean()
        if p_tr <= base: continue
        z = (p_tr-base)/math.sqrt(base*(1-base)/ntr) if base>0 else 0
        cand.append(dict(cond=label, n_tr=ntr, n_te=nte, tr=p_tr, te=p_te, base=base, lift=p_tr/base, z=z))
    # FDR
    if cand:
        for c in cand: c["p"]=1-norm.cdf(c["z"])
        srt=sorted(cand,key=lambda c:c["p"]); m=len(srt); thr=0
        for i,c in enumerate(srt,1):
            if c["p"]<=(i/m)*0.10: thr=i
        crit=srt[thr-1]["p"] if thr>0 else -1
        for c in cand: c["fdr"]=c["p"]<=crit
    # garder ceux qui répliquent OOS (te > base) trié par te
    keep=[c for c in cand if c["te"]>c["base"]]
    keep.sort(key=lambda c:c["te"], reverse=True)
    out[score]=dict(base=round(base,4), n_edges=len(keep), edges=keep[:6])

if not ONLY:
    json.dump(out, open(ROOT/"exports"/"per_score_edges.json","w",encoding="utf-8"), ensure_ascii=False, indent=1,
              default=lambda o: o.item() if hasattr(o,"item") else str(o))

print(f"{'='*108}")
print(f"ÉTUDE PAR SCORE — base globale + meilleures conditions OOS (TR n={len(tr)}, TE n={len(te)})")
print(f"{'='*108}")
for score in targets:
    d = out[score]
    print(f"\n■ SCORE {score}  (base {d['base']*100:.1f}%, {d['n_edges']} conditions répliquées)")
    for c in d["edges"]:
        flag = "✅FDR" if c.get("fdr") else "     "
        print(f"  {flag} {c['cond'][:50]:<50} n={c['n_tr']:>4}/{c['n_te']:<4} "
              f"TR {c['tr']*100:>4.1f}% TE {c['te']*100:>4.1f}% base {c['base']*100:>3.0f}% ×{c['lift']:.2f} z={c['z']:.1f}")
