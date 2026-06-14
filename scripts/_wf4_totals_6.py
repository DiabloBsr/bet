# WF4 TOTALS - step 6: re-test de l'edge historique "Total de buts = 1" (ENGINE_MODEL edge #3)
# + ROI par ligue du blanket over/under 3.5 + export JSON final wf4_totals.json
import sys, pickle, math, json
sys.path.insert(0, ".")
import numpy as np
from scipy.stats import norm

with open("exports/wf4_totals_data.pkl", "rb") as f:
    D = pickle.load(f)

def roi_stats(bets):
    if not bets:
        return 0, 0.0, 0.0, 0.0, 1.0
    r = np.array([(o - 1) if w else -1.0 for w, o in bets])
    n = len(r); roi = float(r.mean())
    wr = float(np.mean([w for w, _ in bets])); ao = float(np.mean([o for _, o in bets]))
    se = r.std(ddof=1) / math.sqrt(n) if n > 1 else 1e9
    p = 2 * (1 - norm.cdf(abs(roi) / se)) if se > 0 else 1.0
    return n, wr, roi, ao, float(p)

L35 = [r for r in D if r["comp"] == "InstantLeague-8035"]
starts = sorted(r["start"] for r in L35)
cut = starts[int(0.7 * len(starts))]

print("=== EDGE HISTORIQUE 'Total de buts = 1' (8035) ===")
out_tot1 = {}
for name, sub in [("8035-full", L35),
                  ("8035-train", [r for r in L35 if r["start"] < cut]),
                  ("8035-test", [r for r in L35 if r["start"] >= cut])]:
    bets = [((r["tot"] == 1), r["totx"]["1"]) for r in sub
            if r["totx"].get("1") and 1 < r["totx"]["1"] < 100]
    n, wr, roi, ao, p = roi_stats(bets)
    out_tot1[name] = dict(n=n, wr=wr, roi=roi, odds=ao, p=p)
    print(f"{name:11s} n={n:5d} freq={wr:.4f} ROI={roi*100:+.2f}% odds={ao:.2f} p={p:.4f}")

print("\n=== BLANKET +/- 3.5 PAR LIGUE ===")
per_league = {}
for comp in sorted(set(r["comp"] for r in D)):
    sub = [r for r in D if r["comp"] == comp]
    row = {}
    for side in ("over", "under"):
        bets = []
        for r in sub:
            o = r["ou_o"] if side == "over" else r["ou_u"]
            if not o or o <= 1 or o >= 100:
                continue
            won = (r["tot"] >= 4) if side == "over" else (r["tot"] <= 3)
            bets.append((won, o))
        n, wr, roi, ao, p = roi_stats(bets)
        row[side] = dict(n=n, wr=wr, roi=roi, odds=ao, p=p)
        print(f"{comp} {side:5s} n={n:5d} WR={wr:.4f} ROI={roi*100:+.2f}% odds={ao:.3f} p={p:.4f}")
    per_league[comp] = row

# export JSON recapitulatif
summary = {
    "snapshot": "2026-06-12",
    "n_events_clean": len(D),
    "hard_cap": {"max_total_all_results": 6, "n_results": 54296,
                 "grid_mass_at_7plus_pct": 2.67},
    "margins": {"+/- 3.5": 6.00, "team totals 3.5": 6.01, "Total de buts exact": 12.09},
    "pricing_structure": {
        "comment": "O/U NOT priced from 1X2 grid: implied_over/grid ratio varies 1.57->1.06 with lambda_total, under ~1.00; book prices boost+truncation",
        "implied_over_over_grid_by_lambda": {"<2.2": 1.5733, "2.2-2.5": 1.3442,
                                             "2.5-2.8": 1.2678, "2.8-3.1": 1.2034,
                                             "3.1-3.4": 1.1479, ">=3.4": 1.0585},
        "implied_under_over_grid": 1.0032},
    "real_vs_grid": {
        "pooled9_goals_real": 2.775, "pooled9_goals_priced": 2.638,
        "delta": 0.137, "p_over35_real": 0.3221, "p_over35_grid": 0.2805,
        "by_group_delta": {"8035": 0.162, "dom-new": 0.189, "coupes": 0.071}},
    "blanket_roi": {
        "pooled9_over": -6.90, "pooled9_under": -5.67,
        "8035_over": -3.85, "8035_under": -6.53,
        "newleagues_over": -8.65, "newleagues_under": -5.16},
    "per_league_blanket": per_league,
    "tot1_edge_retest": out_tot1,
    "pair_jitter_ou35": {"test8035_thr0.01_over": {"n": 211, "roi": 2.93, "p": 0.756},
                         "train8035_thr0.01_over": {"n": 430, "roi": -6.27},
                         "newleagues_thr0.01_over": {"n": 196, "roi": -8.35}},
    "n_tests_scanned": 175,
    "conclusion": "AUCUN edge positif sur les marches totals: le book price O/U/exacts depuis la VRAIE distribution du simulateur (boost +0.12 ET cap a 6 inclus), pas depuis la grille 1X2. Hypothese 'over sous-price partout' REFUTEE."
}
with open("exports/wf4_totals.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print("\nwritten -> exports/wf4_totals.json")
