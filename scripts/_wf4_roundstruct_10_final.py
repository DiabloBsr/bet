# -*- coding: utf-8 -*-
# WF4 ROUND-STRUCTURE final: combine lag-1 across leagues (Fisher z), CI bounds on rho_intra,
# consolidate all parts -> exports/wf4_roundstruct.json
import sys, json, math
sys.path.insert(0, ".")
import numpy as np

p2 = json.load(open("exports/wf4_roundstruct_part2.json", encoding="utf-8"))
p1 = json.load(open("exports/wf4_roundstruct_tests.json", encoding="utf-8"))
p3 = json.load(open("exports/wf4_roundstruct_part3.json", encoding="utf-8"))
p4 = json.load(open("exports/wf4_roundstruct_part4.json", encoding="utf-8"))

# combined lag-1 (Fisher z, weighted by n-3) + heterogeneity Q
lag = p2["lag1"]
zs, ws, names = [], [], []
for lg, d in lag.items():
    z = math.atanh(d["corr"]); w = d["n_pairs"] - 3
    zs.append(z); ws.append(w); names.append(lg)
zs = np.array(zs); ws = np.array(ws)
zbar = float((ws * zs).sum() / ws.sum())
se = 1 / math.sqrt(ws.sum())
zstat = zbar / se
from scipy.stats import norm, chi2
p_comb = 2 * (1 - norm.cdf(abs(zstat)))
Q = float((ws * (zs - zbar) ** 2).sum())
p_het = 1 - chi2.cdf(Q, len(zs) - 1)
print(f"lag-1 combined: r={math.tanh(zbar):+.4f} z={zstat:.2f} p={p_comb:.4f} | heterogeneity Q={Q:.1f} df={len(zs)-1} p_het={p_het:.4f}")

# CI bounds on rho_intra from part1 (perm std)
bounds = {}
for t in p1["tests"]:
    if t["scope"] in ("InstantLeague-8035", "pooled-9", "pooled-newleagues"):
        d = t["xprod_fav_win"]
        se_rho = d["perm_std"] / (d["npairs"] * 1.0)  # var_e missing; derive from rho/C
        var_e = d["C"] / (d["npairs"] * d["rho"]) if d["rho"] != 0 else None
        if var_e:
            se_rho = d["perm_std"] / (d["npairs"] * var_e)
            bounds[t["scope"]] = dict(rho=d["rho"], se=se_rho, ci95=[d["rho"] - 1.96 * se_rho, d["rho"] + 1.96 * se_rho])
            print(f"rho_intra fav {t['scope']}: {d['rho']:+.5f} ± {se_rho:.5f} (95% CI [{d['rho']-1.96*se_rho:+.4f}, {d['rho']+1.96*se_rho:+.4f}])")

final = {
    "domain": "structure intra-round",
    "snapshot": "2026-06-12 ~12:30",
    "n_tests_scanned": p1["n_tests_scanned"] + 35 + 18 + 12 + 6,
    "core_tests_per_league_and_pooled": p1["tests"],
    "pooled_league_centered": p2["pooled_lgcentered"],
    "goal_bias_per_league": p2["goal_bias_per_league"],
    "dist_complete_rounds": p2["dist_complete_rounds"],
    "lag1_per_league": p2["lag1"],
    "lag1_combined": dict(r=math.tanh(zbar), z=zstat, p=float(p_comb), Q=Q, p_het=float(p_het)),
    "lag_profile_and_split": p4,
    "power_check": p3["power_check_8035_struct"],
    "adjacent_id": p3["adjacent_id"],
    "conditional_strategy_8035": p3["conditional_strategy_8035"],
    "dispersion_8035": p3["dispersion_8035"],
    "rho_intra_bounds": bounds,
}
with open("exports/wf4_roundstruct.json", "w", encoding="utf-8") as f:
    json.dump(final, f, indent=1)
print("consolidated -> exports/wf4_roundstruct.json")
