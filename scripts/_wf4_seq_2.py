# -*- coding: utf-8 -*-
"""
WF4 - cycle saisonnier (J1-J5 vs fin de saison) + effet derniere journee,
pooled sur les 6 championnats (8035 En, 8036 It, 8037 Es, 8042 Fr, 8043 De, 8044 Pt)
+ exploratoire coupes (8056/8060/8065, rounds sequentiels).
Question: les residus de calibration (resultat - proba implicite open) ou le total
de buts (vs mu = lh+la) varient-ils selon la position dans la saison ?
Sortie brute: exports/wf4_seq_cycle.json. LECTURE SEULE.
Reutilise load_data/invert_lambdas de scripts/_wf4_seq_1.py (dedup anti-doublons inclus).
"""
import sys, json, math, importlib.util, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
from scipy import stats

spec = importlib.util.spec_from_file_location("wf4seq1", "scripts/_wf4_seq_1.py")
m1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m1)

MAXR = {"InstantLeague-8035": 38, "InstantLeague-8036": 38, "InstantLeague-8037": 38,
        "InstantLeague-8042": 34, "InstantLeague-8043": 34, "InstantLeague-8044": 34}
CUPS = {"InstantLeague-8056", "InstantLeague-8060", "InstantLeague-8065"}

def main():
    rows = m1.load_data()
    oh = np.array([r["oh"] for r in rows]); od = np.array([r["od"] for r in rows]); oa = np.array([r["oa"] for r in rows])
    inv = 1 / oh + 1 / od + 1 / oa
    ph, pd, pa = (1 / oh) / inv, (1 / od) / inv, (1 / oa) / inv
    lh, la = m1.invert_lambdas(ph, pd)
    for i, r in enumerate(rows):
        r.update(ph=float(ph[i]), pd=float(pd[i]), pa=float(pa[i]), lh=float(lh[i]), la=float(la[i]))

    out = {"bins": {}, "tests": [], "last_round": {}}
    n_tests = 0

    def metrics(sub):
        hw = np.array([1.0 if r["sa"] > r["sb"] else 0.0 for r in sub])
        dr = np.array([1.0 if r["sa"] == r["sb"] else 0.0 for r in sub])
        fav_is_home = np.array([r["ph"] >= r["pa"] for r in sub])
        favwin = np.where(fav_is_home, hw, np.array([1.0 if r["sb"] > r["sa"] else 0.0 for r in sub]))
        pfav = np.array([max(r["ph"], r["pa"]) for r in sub])
        tot = np.array([float(r["sa"] + r["sb"]) for r in sub])
        mu = np.array([r["lh"] + r["la"] for r in sub])
        return {
            "res_home": hw - np.array([r["ph"] for r in sub]),
            "res_draw": dr - np.array([r["pd"] for r in sub]),
            "res_fav": favwin - pfav,
            "res_total": tot - mu,
        }

    def summarize(res):
        return {k: dict(mean=float(v.mean()), se=float(v.std(ddof=1) / math.sqrt(len(v))), n=len(v))
                for k, v in res.items()}

    def compare(name, scope, A, B):
        nonlocal n_tests
        ra, rb = metrics(A), metrics(B)
        for k in ra:
            t, p = stats.ttest_ind(ra[k], rb[k], equal_var=False)
            out["tests"].append(dict(name=name, scope=scope, metric=k,
                                     nA=len(A), nB=len(B),
                                     meanA=float(ra[k].mean()), meanB=float(rb[k].mean()),
                                     t=float(t), p=float(p)))
            n_tests += 1

    # ---- championnats: bins de position saisonniere ----
    champs = [r for r in rows if r["comp"] in MAXR and 1 <= r["rnd"] <= MAXR[r["comp"]]]
    for r in champs:
        r["posfrac"] = (r["rnd"] - 1) / (MAXR[r["comp"]] - 1)
    scopes = {
        "8035": [r for r in champs if r["comp"] == "InstantLeague-8035"],
        "pooled-newchamps": [r for r in champs if r["comp"] != "InstantLeague-8035"],
        "pooled-6champs": champs,
    }
    for scope, sub in scopes.items():
        # courbe par quintile de saison (diagnostic)
        curve = {}
        for b in range(5):
            seg = [r for r in sub if b / 5 <= r["posfrac"] < (b + 1) / 5 or (b == 4 and r["posfrac"] == 1.0)]
            if seg:
                curve[f"Q{b+1}"] = summarize(metrics(seg))
        out["bins"][scope] = curve
        # test principal: debut (J1-J5) vs fin (5 dernieres journees)
        early = [r for r in sub if r["rnd"] <= 5]
        late = [r for r in sub if r["rnd"] >= MAXR[r["comp"]] - 4]
        if len(early) > 100 and len(late) > 100:
            compare("early_J1-5_vs_late_last5", scope, early, late)
        # derniere journee vs tout le reste
        lastr = [r for r in sub if r["rnd"] == MAXR[r["comp"]]]
        rest = [r for r in sub if r["rnd"] < MAXR[r["comp"]]]
        if len(lastr) > 80:
            compare("lastround_vs_rest", scope, lastr, rest)
            out["last_round"][scope] = dict(n_last=len(lastr), summary=summarize(metrics(lastr)))
        # 1ere journee vs reste (reset de saison)
        firstr = [r for r in sub if r["rnd"] == 1]
        if len(firstr) > 80:
            compare("round1_vs_rest", scope, firstr, [r for r in sub if r["rnd"] > 1])

    # ---- coupes (exploratoire): tiers de rounds ----
    cups = [r for r in rows if r["comp"] in CUPS and r["rnd"] >= 1]
    if cups:
        mx = {c: max(r["rnd"] for r in cups if r["comp"] == c) for c in CUPS}
        early = [r for r in cups if r["rnd"] <= mx[r["comp"]] / 3]
        late = [r for r in cups if r["rnd"] > 2 * mx[r["comp"]] / 3]
        compare("cups_earlythird_vs_latethird", "pooled-cups", early, late)

    out["n_tests_scanned"] = n_tests
    with open("exports/wf4_seq_cycle.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    print("\n=== tests tries par p ===")
    for t in sorted(out["tests"], key=lambda x: x["p"]):
        print(f"{t['name']:<28} {t['scope']:<17} {t['metric']:<9} nA={t['nA']:>5} nB={t['nB']:>6} "
              f"mA={t['meanA']:+.4f} mB={t['meanB']:+.4f} p={t['p']:.4g}")
    print("\nn_tests_scanned =", n_tests)

if __name__ == "__main__":
    main()
