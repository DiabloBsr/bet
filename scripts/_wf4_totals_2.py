# WF4 TOTALS - step 2: calibration O/U 3.5 + totals exacts vs grille pricee
# Empirique d'abord: distribution reelle des totaux vs grille Poisson pricee, marges, ROI blanket.
import sys, json, pickle, math
sys.path.insert(0, ".")
import numpy as np
from scipy.stats import poisson, norm

with open("exports/wf4_totals_data.pkl", "rb") as f:
    D = pickle.load(f)

GMAX = 13
ar = np.arange(GMAX + 1)
NEW = {"InstantLeague-8036", "InstantLeague-8037", "InstantLeague-8042",
       "InstantLeague-8043", "InstantLeague-8044"}
CUPS = {"InstantLeague-8056", "InstantLeague-8060", "InstantLeague-8065"}

def p_over35(lh, la, boost=False):
    if boost:
        lh, la = lh * 1.700 / 1.635, la * 1.254 / 1.196
    g = np.outer(poisson.pmf(ar, lh), poisson.pmf(ar, la))
    tot = np.add.outer(ar, ar)
    return g[tot >= 4].sum()

def roi_stats(bets):
    """bets = list of (won_bool, odds). returns n, wr, roi, avg_odds, pvalue (t-test vs 0)"""
    if not bets:
        return 0, 0, 0, 0, 1.0
    r = np.array([(o - 1) if w else -1.0 for w, o in bets])
    n = len(r)
    roi = r.mean()
    wr = np.mean([w for w, _ in bets])
    ao = np.mean([o for _, o in bets])
    se = r.std(ddof=1) / math.sqrt(n) if n > 1 else 1e9
    p = 2 * (1 - norm.cdf(abs(roi) / se)) if se > 0 else 1.0
    return n, wr, roi, ao, p

# ---------- 0. margins sanity ----------
m_ou = [1 / r["ou_u"] + 1 / r["ou_o"] - 1 for r in D if r["ou_u"] and r["ou_o"]]
print(f"marge +/- 3.5: mean {np.mean(m_ou)*100:.2f}% std {np.std(m_ou)*100:.3f}% n={len(m_ou)}")
m_th = [1 / r["th_u"] + 1 / r["th_o"] - 1 for r in D
        if r["th_u"] and r["th_o"] and r["th_u"] < 100 and r["th_o"] < 100]
print(f"marge Total dom 3.5: mean {np.mean(m_th)*100:.2f}% n={len(m_th)}")
m_tx = []
for r in D:
    if r["totx"] and len(r["totx"]) == 7:
        s = sum(1 / v for v in r["totx"].values() if v and v > 1)
        m_tx.append(s - 1)
print(f"marge Total de buts (exact): mean {np.mean(m_tx)*100:.2f}% n={len(m_tx)}")

# ---------- 1. distribution reelle des totaux vs grille pricee (pooled 9) ----------
print("\n=== TOTAUX REELS vs GRILLE PRICEE (pooled 9 ligues) ===")
exp_tot = np.zeros(15); obs_tot = np.zeros(15)
exp_o35 = 0.0
n_all = len(D)
for r in D:
    g = np.outer(poisson.pmf(ar, r["lh"]), poisson.pmf(ar, r["la"]))
    tot = np.add.outer(ar, ar)
    for t in range(15):
        exp_tot[t] += g[tot == t].sum() if t < 14 else g[tot >= 14].sum()
    obs_tot[min(r["tot"], 14)] += 1
    exp_o35 += g[tot >= 4].sum()
print(" tot | obs    | exp(grille) | ratio  | z")
for t in range(10):
    e_, o_ = exp_tot[t], obs_tot[t]
    z = (o_ - e_) / math.sqrt(max(e_, 1e-9))
    print(f"  {t:2d} | {int(o_):6d} | {e_:10.1f} | {o_/max(e_,1e-9):.3f} | {z:+.2f}")
obs_o35 = sum(1 for r in D if r["tot"] >= 4)
z = (obs_o35 - exp_o35) / math.sqrt(exp_o35 * (1 - exp_o35 / n_all))
print(f"P(>3.5): obs {obs_o35/n_all:.4f} vs grille {exp_o35/n_all:.4f} (z={z:+.2f}, n={n_all})")
mu_price = np.mean([r['lh'] + r['la'] for r in D])
print(f"buts moyens: reel {np.mean([r['tot'] for r in D]):.3f} vs price {mu_price:.3f} (delta {np.mean([r['tot'] for r in D])-mu_price:+.3f})")

# par groupe de ligues
for name, grp in [("8035", lambda c: c == "InstantLeague-8035"),
                  ("domestiques-new", lambda c: c in NEW),
                  ("coupes", lambda c: c in CUPS)]:
    sub = [r for r in D if grp(r["comp"])]
    mu_p = np.mean([r['lh'] + r['la'] for r in sub])
    mu_r = np.mean([r['tot'] for r in sub])
    o35r = np.mean([r['tot'] >= 4 for r in sub])
    o35p = np.mean([p_over35(r['lh'], r['la']) for r in sub])
    print(f"  {name:16s} n={len(sub):5d} buts reel {mu_r:.3f} vs price {mu_p:.3f} ({mu_r-mu_p:+.3f}) | P(o3.5) reel {o35r:.4f} vs price {o35p:.4f}")

# ---------- 2. ROI blanket Over / Under 3.5 ----------
print("\n=== ROI BLANKET +/- 3.5 (cote offerte reelle) ===")
def blanket(sub, side):
    bets = []
    for r in sub:
        o = r["ou_o"] if side == "over" else r["ou_u"]
        if not o or o <= 1 or o >= 100:
            continue
        won = (r["tot"] >= 4) if side == "over" else (r["tot"] <= 3)
        bets.append((won, o))
    return roi_stats(bets)

for scope, sub in [("pooled-9", D),
                   ("8035", [r for r in D if r["comp"] == "InstantLeague-8035"]),
                   ("pooled-newleagues", [r for r in D if r["comp"] != "InstantLeague-8035"])]:
    for side in ("over", "under"):
        n, wr, roi, ao, p = blanket(sub, side)
        print(f"{scope:18s} {side:5s} n={n:5d} WR={wr:.4f} ROI={roi*100:+.2f}% avg_odds={ao:.3f} p={p:.4f}")

# ---------- 3. par bucket lambda_total ----------
print("\n=== ROI +/- 3.5 par bucket lambda_total (pooled 9) ===")
edges = [0, 2.2, 2.5, 2.8, 3.1, 3.4, 99]
for i in range(len(edges) - 1):
    sub = [r for r in D if edges[i] <= r["lh"] + r["la"] < edges[i + 1]]
    for side in ("over", "under"):
        n, wr, roi, ao, p = blanket(sub, side)
        if n:
            print(f"lam[{edges[i]:.1f}-{edges[i+1]:.1f}) {side:5s} n={n:5d} WR={wr:.4f} ROI={roi*100:+.2f}% odds={ao:.3f} p={p:.4f}")

# ---------- 4. par force du favori ----------
print("\n=== ROI +/- 3.5 par cote du favori (pooled 9) ===")
fedges = [1.0, 1.3, 1.6, 2.0, 2.6, 99]
for i in range(len(fedges) - 1):
    sub = [r for r in D if fedges[i] <= min(r["oh"], r["oa"]) < fedges[i + 1]]
    for side in ("over", "under"):
        n, wr, roi, ao, p = blanket(sub, side)
        if n:
            print(f"fav[{fedges[i]:.1f}-{fedges[i+1]:.1f}) {side:5s} n={n:5d} WR={wr:.4f} ROI={roi*100:+.2f}% odds={ao:.3f} p={p:.4f}")

# ---------- 5. Totals exacts: obs vs implied par selection ----------
print("\n=== Total de buts (exact) : freq reelle vs implicite, ROI par selection (pooled 9) ===")
for sel in ["0", "1", "2", "3", "4", "5", "6"]:
    bets = []
    for r in D:
        o = r["totx"].get(sel)
        if not o or o <= 1 or o >= 100:
            continue
        won = (r["tot"] == int(sel)) if sel != "6" else (r["tot"] >= 6)
        bets.append((won, o))
    n, wr, roi, ao, p = roi_stats(bets)
    imp = np.mean([1 / o for _, o in bets]) if bets else 0
    print(f"sel={sel} n={n:5d} freq={wr:.4f} impl={imp:.4f} ROI={roi*100:+.2f}% odds={ao:.2f} p={p:.4f}")
