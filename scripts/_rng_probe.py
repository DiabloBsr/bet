"""MOTEUR DE MESURE RNG — compare la réalité au Poisson PUR (sans déviations sim)
pour révéler l'empreinte du moteur : où la réalité dévie de Poisson(λh)×Poisson(λa).
λh,λa sont inversés EXACTEMENT depuis le 1X2 (le pricing est Poisson pur).
Sortie : résidu (empirique − Poisson) par cellule/marché, z-score, n, par région.
Split chrono 70/30 pour fiabilité OOS.
Usage: ./.venv/Scripts/python.exe scripts/_rng_probe.py [mode]
  mode = global (def) | grid (carte λh×λa) | <fav_lo> <fav_hi> (région custom)
"""
from __future__ import annotations
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scraper.market_inversion import exact_invert_1x2, _fast_grid, CAP

CSV = Path(__file__).resolve().parents[1] / "exports" / "combokeys_features.csv"
SCORES_SHOW = ['0-0','1-0','0-1','1-1','2-0','0-2','2-1','1-2','2-2','3-0','0-3','3-1','1-3','2-2','3-2','2-3','4-0','0-4']
MAXG = 8

def load():
    df = pd.read_csv(CSV).sort_values("expected_start").reset_index(drop=True)
    return df

def build_grids(df):
    """Pour chaque match : grille Poisson pure depuis (λh,λa) inversés du 1X2."""
    grids = []; realized_s = []; realized_t = []
    for r in df.itertuples():
        lh, la = exact_invert_1x2(float(r.oh), float(r.od), float(r.oa))
        g = _fast_grid(lh, la, 0.0)  # POISSON PUR, rho=0
        grids.append(g)
        sa, sb = [int(x) for x in r.exact_score.split("-")]
        realized_s.append((min(sa, MAXG), min(sb, MAXG)))
        realized_t.append(min(sa + sb, CAP))
    return grids, realized_s, realized_t

def probe(df, grids, rs, rt, label):
    """Résidu empirique − Poisson pour scores, totaux, 1X2, BTTS."""
    n = len(df)
    if n < 60:
        print(f"  [{label}] n={n} trop peu"); return
    # --- scores exacts ---
    print(f"\n■ RÉGION {label}  (n={n})")
    # matrice cumulée des proba Poisson + comptage empirique
    score_imp = {}; score_emp = {}
    tot_imp = np.zeros(CAP + 1); tot_emp = np.zeros(CAP + 1)
    p1 = pd = p2 = 0.0; e1 = ed = e2 = 0
    btts_imp = 0.0; btts_emp = 0
    for g, (sa, sb), tg in zip(grids, rs, rt):
        gg = g[:MAXG+1, :MAXG+1]
        # scores
        for h in range(min(5, gg.shape[0])):
            for a in range(min(5, gg.shape[1])):
                k = f"{h}-{a}"; score_imp[k] = score_imp.get(k, 0) + gg[h, a]
        kk = f"{min(sa,4)}-{min(sb,4)}" if sa<5 and sb<5 else None
        if kk: score_emp[kk] = score_emp.get(kk, 0) + 1
        # totaux
        td = np.array([g[i, :].sum() if False else 0 for i in range(0)])  # placeholder
        # total via convolution simple
        tg_imp = np.zeros(CAP+1)
        for h in range(g.shape[0]):
            for a in range(g.shape[1]):
                tg_imp[min(h+a, CAP)] += g[h, a]
        tot_imp += tg_imp; tot_emp[tg] += 1
        # 1X2
        tri_l = np.tril(g, -1).sum(); diag = np.trace(g); tri_u = np.triu(g, 1).sum()
        p1 += tri_l; pd += diag; p2 += tri_u
        if sa > sb: e1 += 1
        elif sa == sb: ed += 1
        else: e2 += 1
        # BTTS
        btts_imp += (1 - g[0, :].sum()) * 1  # approx; better below
    # recompute BTTS implied properly
    btts_imp = 0.0; btts_emp = 0
    for g, (sa, sb), tg in zip(grids, rs, rt):
        p_no = g[0, :].sum() + g[:, 0].sum() - g[0, 0]
        btts_imp += (1 - p_no)
        if sa >= 1 and sb >= 1: btts_emp += 1

    def z(emp, imp, n):
        p0 = imp / n
        if p0 <= 0 or p0 >= 1: return 0.0
        return (emp/n - p0) / math.sqrt(p0*(1-p0)/n)

    # 1X2
    print(f"  1X2  : 1 emp {e1/n*100:4.1f}% vs Poisson {p1/n*100:4.1f}% (z {z(e1,p1,n):+.1f}) | "
          f"X emp {ed/n*100:4.1f}% vs {pd/n*100:4.1f}% (z {z(ed,pd,n):+.1f}) | "
          f"2 emp {e2/n*100:4.1f}% vs {p2/n*100:4.1f}% (z {z(e2,p2,n):+.1f})")
    print(f"  BTTS : emp {btts_emp/n*100:4.1f}% vs Poisson {btts_imp/n*100:4.1f}% (z {z(btts_emp,btts_imp,n):+.1f})")
    # totaux : résidu par total
    devt = []
    for t in range(CAP+1):
        devt.append((t, tot_emp[t], tot_imp[t], z(tot_emp[t], tot_imp[t], n)))
    sig_t = [d for d in devt if abs(d[3]) >= 2.5]
    print("  TOTAUX déviants (|z|>=2.5):", " ".join(f"{t}b:emp{int(e)}/{imp:.0f}(z{zz:+.1f})" for t,e,imp,zz in sig_t) or "aucun")
    # scores : top déviations
    devs = []
    for k in set(list(score_imp)+list(score_emp)):
        imp = score_imp.get(k, 0); emp = score_emp.get(k, 0)
        devs.append((k, emp, imp, emp-imp, z(emp, imp, n)))
    devs.sort(key=lambda d: -abs(d[4]))
    print("  SCORES déviants (top 6 |z|):")
    for k,emp,imp,res,zz in devs[:6]:
        tag = "↑sous-pricé" if res>0 else "↓sur-pricé"
        print(f"     {k}: emp {emp} vs Poisson {imp:.1f}  résidu {res:+.1f} (z {zz:+.1f}) {tag if abs(zz)>=2 else ''}")

def main():
    df = load()
    mode = sys.argv[1] if len(sys.argv) > 1 else "global"
    cut = int(len(df)*0.70)
    if mode == "global":
        for lbl, sub in [("GLOBAL TRAIN", df.iloc[:cut]), ("GLOBAL TEST", df.iloc[cut:])]:
            g, rs, rt = build_grids(sub); probe(sub, g, rs, rt, lbl)
    elif mode == "grid":
        # carte fine : bandes de λ_tot × λ_diff
        tot_edges = [0, 2.0, 2.45, 2.8, 3.13, 3.5, 9]
        diff_edges = [-9, -1.0, -0.4, 0.0, 0.4, 1.0, 9]
        for i in range(len(tot_edges)-1):
            for j in range(len(diff_edges)-1):
                sub = df[(df.lam_tot>=tot_edges[i])&(df.lam_tot<tot_edges[i+1])&(df.lam_diff>=diff_edges[j])&(df.lam_diff<diff_edges[j+1])]
                if len(sub) < 120: continue
                g, rs, rt = build_grids(sub)
                probe(sub, g, rs, rt, f"λtot[{tot_edges[i]}-{tot_edges[i+1]}] λdiff[{diff_edges[j]},{diff_edges[j+1]}]")
    else:
        lo, hi = float(sys.argv[1]), float(sys.argv[2])
        sub = df[(df.fav>=lo)&(df.fav<hi)]
        g, rs, rt = build_grids(sub); probe(sub, g, rs, rt, f"fav[{lo},{hi}] n={len(sub)}")

if __name__ == "__main__":
    main()
