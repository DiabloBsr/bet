# WF4 TOTALS - step 4: scan residuel
# (a) calibration implied vs reel par decile, par groupe de ligues
# (b) ROI under/over par groupe x bucket lambda
# (c) team totals 3.5
# (d) 1X2 & Total (6 selections) par force favori
import sys, pickle, math
sys.path.insert(0, ".")
import numpy as np
from scipy.stats import norm

with open("exports/wf4_totals_data.pkl", "rb") as f:
    D = pickle.load(f)

NEW = {"InstantLeague-8036", "InstantLeague-8037", "InstantLeague-8042",
       "InstantLeague-8043", "InstantLeague-8044"}
CUPS = {"InstantLeague-8056", "InstantLeague-8060", "InstantLeague-8065"}
def grp(c):
    return "8035" if c == "InstantLeague-8035" else ("dom-new" if c in NEW else "coupes")

def roi_stats(bets):
    if not bets:
        return 0, 0, 0, 0, 1.0
    r = np.array([(o - 1) if w else -1.0 for w, o in bets])
    n = len(r); roi = r.mean()
    wr = float(np.mean([w for w, _ in bets])); ao = float(np.mean([o for _, o in bets]))
    se = r.std(ddof=1) / math.sqrt(n) if n > 1 else 1e9
    p = 2 * (1 - norm.cdf(abs(roi) / se)) if se > 0 else 1.0
    return n, wr, float(roi), ao, float(p)

n_tests = 0

# ---------- (a) calibration par decile d'implied fair p_over ----------
print("=== CALIBRATION +/- 3.5 : implied fair vs reel, par groupe ===")
print("groupe  bucket_implied      n    impl_fair  reel    edge_over(reel-impl)")
for g in ["8035", "dom-new", "coupes"]:
    sub = [r for r in D if grp(r["comp"]) == g and r["ou_o"] and r["ou_u"]
           and r["ou_o"] < 100 and r["ou_u"] < 100]
    arrs = []
    for r in sub:
        io, iu = 1 / r["ou_o"], 1 / r["ou_u"]
        pf = io / (io + iu)
        arrs.append((pf, r["tot"] >= 4))
    arrs.sort()
    K = 8
    for k in range(K):
        chunk = arrs[k * len(arrs) // K:(k + 1) * len(arrs) // K]
        pf = np.mean([a[0] for a in chunk]); re = np.mean([a[1] for a in chunk])
        lo = min(a[0] for a in chunk); hi = max(a[0] for a in chunk)
        se = math.sqrt(re * (1 - re) / len(chunk))
        flag = "*" if abs(re - pf) > 2 * se else " "
        print(f"{g:7s} [{lo:.3f}-{hi:.3f}] {len(chunk):6d}  {pf:.4f}  {re:.4f}  {re-pf:+.4f} {flag}")
        n_tests += 1

# ---------- (b) ROI par groupe x lambda bucket ----------
print("\n=== ROI +/- 3.5 par groupe x lambda bucket ===")
edges = [0, 2.2, 2.6, 3.0, 3.4, 99]
best = []
for g in ["8035", "dom-new", "coupes"]:
    for i in range(len(edges) - 1):
        sub = [r for r in D if grp(r["comp"]) == g and edges[i] <= r["lh"] + r["la"] < edges[i + 1]]
        for side in ("over", "under"):
            bets = []
            for r in sub:
                o = r["ou_o"] if side == "over" else r["ou_u"]
                if not o or o <= 1 or o >= 100:
                    continue
                won = (r["tot"] >= 4) if side == "over" else (r["tot"] <= 3)
                bets.append((won, o))
            n, wr, roi, ao, p = roi_stats(bets)
            n_tests += 1
            if n >= 150:
                tag = " <<<" if roi > 0.02 and p < 0.05 else ""
                print(f"{g:7s} lam[{edges[i]}-{edges[i+1]}) {side:5s} n={n:5d} WR={wr:.4f} ROI={roi*100:+.2f}% odds={ao:.3f} p={p:.4f}{tag}")

# ---------- (c) team totals 3.5 ----------
print("\n=== TEAM TOTALS 3.5 (pooled 9) ===")
for side_key, lam_key, score_key, label in [
        ("th_o", "lh", "sa", "home>3.5"), ("th_u", "lh", "sa", "home<3.5"),
        ("ta_o", "la", "sb", "away>3.5"), ("ta_u", "la", "sb", "away<3.5")]:
    over = side_key.endswith("_o")
    bets = [( (r[score_key] >= 4) if over else (r[score_key] <= 3), r[side_key])
            for r in D if r[side_key] and 1 < r[side_key] < 100]
    n, wr, roi, ao, p = roi_stats(bets)
    n_tests += 1
    print(f"{label:10s} n={n:6d} WR={wr:.4f} ROI={roi*100:+.2f}% odds={ao:.3f} p={p:.4f}")
# par bucket lambda equipe
print("team-over par lambda equipe:")
tedges = [0, 1.2, 1.6, 2.0, 99]
for i in range(len(tedges) - 1):
    for side_key, lam_key, score_key, label in [("th_o", "lh", "sa", "home>3.5"),
                                                ("ta_o", "la", "sb", "away>3.5")]:
        bets = [((r[score_key] >= 4), r[side_key]) for r in D
                if r[side_key] and 1 < r[side_key] < 100 and tedges[i] <= r[lam_key] < tedges[i + 1]]
        n, wr, roi, ao, p = roi_stats(bets)
        n_tests += 1
        if n >= 150:
            print(f"  lam_eq[{tedges[i]}-{tedges[i+1]}) {label} n={n:5d} WR={wr:.4f} ROI={roi*100:+.2f}% odds={ao:.2f} p={p:.4f}")

# ---------- (d) 1X2 & Total ----------
print("\n=== 1X2 & TOTAL (pooled 9) ===")
SELS = ["1 / < 3.5", "1 / > 3.5", "X / < 3.5", "X / > 3.5", "2 / < 3.5", "2 / > 3.5"]
def settle_x2t(r, sel):
    res = "1" if r["sa"] > r["sb"] else ("2" if r["sb"] > r["sa"] else "X")
    part, ou = sel.split(" / ")
    okres = (res == part)
    okou = (r["tot"] <= 3) if ou == "< 3.5" else (r["tot"] >= 4)
    return okres and okou
for sel in SELS:
    bets = []
    for r in D:
        o = r["x2t"].get(sel)
        if not o or o <= 1 or o >= 100:
            continue
        bets.append((settle_x2t(r, sel), o))
    n, wr, roi, ao, p = roi_stats(bets)
    n_tests += 1
    print(f"{sel:10s} n={n:6d} WR={wr:.4f} ROI={roi*100:+.2f}% odds={ao:.3f} p={p:.4f}")
# par force favori home
print("1/>3.5 et 1/<3.5 par cote home:")
fedges = [1.0, 1.25, 1.5, 1.8, 99]
for i in range(len(fedges) - 1):
    for sel in ["1 / > 3.5", "1 / < 3.5"]:
        bets = []
        for r in D:
            if not (fedges[i] <= r["oh"] < fedges[i + 1]):
                continue
            o = r["x2t"].get(sel)
            if not o or o <= 1 or o >= 100:
                continue
            bets.append((settle_x2t(r, sel), o))
        n, wr, roi, ao, p = roi_stats(bets)
        n_tests += 1
        if n >= 100:
            tag = " <<<" if roi > 0.02 and p < 0.05 else ""
            print(f"  oh[{fedges[i]}-{fedges[i+1]}) {sel} n={n:5d} WR={wr:.4f} ROI={roi*100:+.2f}% odds={ao:.2f} p={p:.4f}{tag}")

print(f"\nn_tests so far in this script: {n_tests}")
