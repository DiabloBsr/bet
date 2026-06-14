# WF4 TOTALS - step 3: verifier truncation totale a 6 + comment le book price le +/- 3.5
import sys, json, pickle, math
sys.path.insert(0, ".")
import numpy as np
from scipy.stats import poisson
from scraper.config import load_settings
from sqlalchemy import create_engine, text

e = create_engine(load_settings().db_url)
with e.connect() as conn:
    mx = conn.execute(text(
        "SELECT MAX(score_a+score_b), MAX(score_a), MAX(score_b), COUNT(*) FROM results"
    )).fetchone()
    print(f"ALL results table: max_total={mx[0]} max_a={mx[1]} max_b={mx[2]} n={mx[3]}")
    dist = conn.execute(text(
        "SELECT score_a+score_b AS t, COUNT(*) FROM results GROUP BY t ORDER BY t"
    )).fetchall()
    print("dist totaux (toutes results, brut):", dict(dist))

with open("exports/wf4_totals_data.pkl", "rb") as f:
    D = pickle.load(f)
GMAX = 13
ar = np.arange(GMAX + 1)

# Comment le +/- est-il price ? compare implied (1/odds) aux probas grille:
# h0: grille brute * (1+m_side)  -> ratio_over ~ ratio_under ~ 1.06
# h1: marge chargee sur l'over (favourite-longshot)
# h2: price = grille TRONQUEE a 6 (renormalisee)
ratios_o, ratios_u, ratios_o_trunc, ratios_u_trunc = [], [], [], []
for r in D:
    if not r["ou_o"] or not r["ou_u"] or r["ou_o"] >= 100 or r["ou_u"] >= 100:
        continue
    g = np.outer(poisson.pmf(ar, r["lh"]), poisson.pmf(ar, r["la"]))
    tot = np.add.outer(ar, ar)
    po = g[tot >= 4].sum(); pu = 1 - po
    io, iu = 1 / r["ou_o"], 1 / r["ou_u"]
    ratios_o.append(io / po); ratios_u.append(iu / pu)
    # tronque a 6
    gt = g.copy(); gt[tot > 6] = 0; gt /= gt.sum()
    pot = gt[tot >= 4].sum(); put = 1 - pot
    ratios_o_trunc.append(io / pot); ratios_u_trunc.append(iu / put)

print(f"\nimplied/grille_brute: over {np.mean(ratios_o):.4f} (std {np.std(ratios_o):.4f}) | under {np.mean(ratios_u):.4f} (std {np.std(ratios_u):.4f})")
print(f"implied/grille_tronq6: over {np.mean(ratios_o_trunc):.4f} (std {np.std(ratios_o_trunc):.4f}) | under {np.mean(ratios_u_trunc):.4f} (std {np.std(ratios_u_trunc):.4f})")

# ratio par bucket lambda pour voir si la marge over est fonction de lambda
print("\npar bucket lambda_total (implied/grille brute):")
edges = [0, 2.2, 2.5, 2.8, 3.1, 3.4, 99]
for i in range(len(edges) - 1):
    sel = [(ro, ru) for r, ro, ru in zip(
        [r for r in D if r["ou_o"] and r["ou_u"] and r["ou_o"] < 100 and r["ou_u"] < 100],
        ratios_o, ratios_u) if edges[i] <= r["lh"] + r["la"] < edges[i + 1]]
    if sel:
        print(f"  lam[{edges[i]}-{edges[i+1]}) n={len(sel):5d} over_ratio={np.mean([s[0] for s in sel]):.4f} under_ratio={np.mean([s[1] for s in sel]):.4f}")

# le "Total de buts" exact est-il price sur la grille brute ou tronquee ?
print("\nTotal de buts exact: implied/grille par selection (brute vs tronquee):")
for sel_s in ["0", "1", "2", "3", "4", "5", "6"]:
    rb, rt = [], []
    for r in D[:4000]:
        o = r["totx"].get(sel_s)
        if not o or o <= 1 or o >= 100:
            continue
        g = np.outer(poisson.pmf(ar, r["lh"]), poisson.pmf(ar, r["la"]))
        tot = np.add.outer(ar, ar)
        t = int(sel_s)
        pb = g[tot == t].sum() if t < 6 else g[tot >= 6].sum()
        gt = g.copy(); gt[tot > 6] = 0; gt /= gt.sum()
        pt = gt[tot == t].sum() if t < 6 else gt[tot >= 6].sum()
        if pb > 1e-4:
            rb.append((1 / o) / pb); rt.append((1 / o) / pt)
    print(f"  sel={sel_s} n={len(rb):4d} impl/brute={np.mean(rb):.4f} impl/tronq={np.mean(rt):.4f}")
