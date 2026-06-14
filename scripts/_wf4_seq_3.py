# -*- coding: utf-8 -*-
"""
WF4 - validation approfondie du candidat "cycle saisonnier du favori":
favori sur-performe sa proba implicite en debut de saison (J1-5) et sous-performe
en fin de saison (5 dernieres journees), miroir sur les nuls.
Checks anti-artefact:
 1. composition p_fav par bin (confound E2 favoris extremes)
 2. residus stratifies par bucket de p_fav
 3. replication par ligue ET par instance de saison (clusters)
 4. regression logistique favwin ~ logit(pfav) + posfrac (LRT)
 5. ROI aux cotes reelles offertes + walk-forward 8035 (train70/test30 temporel)
Sortie: exports/wf4_seq_fav_cycle.json. LECTURE SEULE.
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
RNG = np.random.default_rng(7)

def main():
    rows = m1.load_data()
    oh = np.array([r["oh"] for r in rows]); od = np.array([r["od"] for r in rows]); oa = np.array([r["oa"] for r in rows])
    inv = 1 / oh + 1 / od + 1 / oa
    ph, pd, pa = (1 / oh) / inv, (1 / od) / inv, (1 / oa) / inv
    for i, r in enumerate(rows):
        r.update(ph=float(ph[i]), pd=float(pd[i]), pa=float(pa[i]))

    champs = [r for r in rows if r["comp"] in MAXR and 1 <= r["rnd"] <= MAXR[r["comp"]]]
    for r in champs:
        r["pfav"] = max(r["ph"], r["pa"])
        r["ofav"] = r["oh"] if r["ph"] >= r["pa"] else r["oa"]
        r["favwin"] = 1.0 if ((r["sa"] > r["sb"]) if r["ph"] >= r["pa"] else (r["sb"] > r["sa"])) else 0.0
        r["draw"] = 1.0 if r["sa"] == r["sb"] else 0.0
        r["posfrac"] = (r["rnd"] - 1) / (MAXR[r["comp"]] - 1)
        r["early"] = r["rnd"] <= 5
        r["late"] = r["rnd"] >= MAXR[r["comp"]] - 4

    out = {}
    n_tests = 0

    # ---- 1. composition p_fav early vs late ----
    comp = {}
    for tag in ("early", "late"):
        sub = [r for r in champs if r[tag]]
        comp[tag] = dict(n=len(sub), mean_pfav=float(np.mean([r["pfav"] for r in sub])),
                         mean_ofav=float(np.mean([r["ofav"] for r in sub])),
                         mean_pd=float(np.mean([r["pd"] for r in sub])))
    out["composition"] = comp

    # ---- 2. residus stratifies par p_fav ----
    strata = [(0.30, 0.45), (0.45, 0.55), (0.55, 0.65), (0.65, 0.75), (0.75, 1.0)]
    strat_res = []
    for lo, hi in strata:
        row = dict(stratum=f"{lo}-{hi}")
        for tag in ("early", "late"):
            sub = [r for r in champs if r[tag] and lo <= r["pfav"] < hi]
            if len(sub) > 30:
                res = np.array([r["favwin"] - r["pfav"] for r in sub])
                row[tag] = dict(n=len(sub), mean=float(res.mean()),
                                se=float(res.std(ddof=1) / math.sqrt(len(sub))))
            else:
                row[tag] = dict(n=len(sub))
        strat_res.append(row)
        n_tests += 1
    out["stratified_resfav"] = strat_res

    # ---- 3. replication par ligue et par instance de saison ----
    # detection saisons: rounds ordonnes par mediane d'expected_start; nouvelle saison quand le round redescend
    per_league = {}
    season_contrasts = []
    for lg in MAXR:
        sub = [r for r in champs if r["comp"] == lg]
        # contraste early vs late par ligue
        e = np.array([r["favwin"] - r["pfav"] for r in sub if r["early"]])
        l = np.array([r["favwin"] - r["pfav"] for r in sub if r["late"]])
        t, p = stats.ttest_ind(e, l, equal_var=False) if len(e) > 30 and len(l) > 30 else (None, None)
        per_league[lg] = dict(n_early=len(e), n_late=len(l),
                              resfav_early=float(e.mean()) if len(e) else None,
                              resfav_late=float(l.mean()) if len(l) else None,
                              p=float(p) if p is not None else None)
        n_tests += 1
        # saisons: groupes par round avec mediane temporelle
        byround = {}
        for r in sub:
            byround.setdefault(r["rnd"], []).append(r)
        rounds_sorted = sorted(byround,
                               key=lambda rd: sorted(rr["start"] for rr in byround[rd])[len(byround[rd]) // 2])
        season_id, prev = 0, None
        round2season = {}
        for rd in rounds_sorted:
            if prev is not None and rd < prev:
                season_id += 1
            round2season[rd] = season_id  # NB: approximation, un round par saison max dans la fenetre
            prev = rd
        # contraste par saison (early et late presentes)
        bysea = {}
        for r in sub:
            bysea.setdefault(round2season[r["rnd"]], []).append(r)
        for sid, lst in bysea.items():
            e = [r["favwin"] - r["pfav"] for r in lst if r["early"]]
            l = [r["favwin"] - r["pfav"] for r in lst if r["late"]]
            if len(e) >= 20 and len(l) >= 20:
                season_contrasts.append(dict(league=lg, season=sid, n_e=len(e), n_l=len(l),
                                             contrast=float(np.mean(e) - np.mean(l))))
    out["per_league"] = per_league
    pos = sum(1 for s in season_contrasts if s["contrast"] > 0)
    out["per_season_signtest"] = dict(n_seasons=len(season_contrasts), n_positive=pos,
                                      p_sign=float(stats.binomtest(pos, len(season_contrasts), 0.5).pvalue)
                                      if season_contrasts else None,
                                      contrasts=season_contrasts)
    n_tests += 1

    # ---- 4. LRT favwin ~ logit(pfav) [+ posfrac] ----
    X0 = np.array([[math.log(r["pfav"] / (1 - r["pfav"]))] for r in champs])
    Xe = np.array([[r["posfrac"]] for r in champs])
    y = np.array([r["favwin"] for r in champs])
    t = m1.lrt_logistic(X0, Xe, y)
    out["lrt_posfrac_pooled6"] = dict(n=len(y), **t)
    n_tests += 1
    # idem 8035 seul et newchamps seuls
    for scope, flt in (("8035", lambda r: r["comp"] == "InstantLeague-8035"),
                       ("newchamps", lambda r: r["comp"] != "InstantLeague-8035")):
        subi = [i for i, r in enumerate(champs) if flt(r)]
        t = m1.lrt_logistic(X0[subi], Xe[subi], y[subi])
        out[f"lrt_posfrac_{scope}"] = dict(n=len(subi), **t)
        n_tests += 1

    # ---- 5. ROI aux cotes reelles ----
    def roi(bets):  # bets = list of (odds, won)
        if not bets:
            return dict(n=0)
        pnl = np.array([(o - 1) if w else -1.0 for o, w in bets])
        n = len(pnl)
        boot = RNG.choice(pnl, size=(4000, n), replace=True).mean(axis=1)
        return dict(n=n, roi_pct=float(100 * pnl.mean()),
                    se_roi_pct=float(100 * pnl.std(ddof=1) / math.sqrt(n)),
                    p_boot_le_0=float((boot <= 0).mean()),
                    wr=float(np.mean([w for _, w in bets])),
                    avg_odds=float(np.mean([o for o, _ in bets])))

    rules = {}
    for scope, flt in (("pooled-6champs", lambda r: True),
                       ("8035", lambda r: r["comp"] == "InstantLeague-8035"),
                       ("newchamps", lambda r: r["comp"] != "InstantLeague-8035")):
        sub = [r for r in champs if flt(r)]
        rules[scope] = {
            "backfav_J1-5": roi([(r["ofav"], r["favwin"] == 1.0) for r in sub if r["early"]]),
            "backfav_last5": roi([(r["ofav"], r["favwin"] == 1.0) for r in sub if r["late"]]),
            "backdraw_last5": roi([(r["od"], r["draw"] == 1.0) for r in sub if r["late"]]),
            "backdraw_J1-5": roi([(r["od"], r["draw"] == 1.0) for r in sub if r["early"]]),
            "backdog_last5": roi([(r["oa"] if r["ph"] >= r["pa"] else r["oh"],
                                   (r["sb"] > r["sa"]) if r["ph"] >= r["pa"] else (r["sa"] > r["sb"]))
                                  for r in sub if r["late"]]),
        }
        n_tests += 5
    # walk-forward 8035: regle sans parametre fitte -> evaluation directe sur test30 temporel
    sub8 = sorted([r for r in champs if r["comp"] == "InstantLeague-8035"], key=lambda r: (r["start"], r["id"]))
    cut = int(len(sub8) * 0.7)
    test = sub8[cut:]
    rules["8035-test30"] = {
        "backfav_J1-5": roi([(r["ofav"], r["favwin"] == 1.0) for r in test if r["early"]]),
        "backfav_last5": roi([(r["ofav"], r["favwin"] == 1.0) for r in test if r["late"]]),
        "backdraw_last5": roi([(r["od"], r["draw"] == 1.0) for r in test if r["late"]]),
    }
    n_tests += 3
    out["roi_rules"] = rules
    out["n_tests_scanned"] = n_tests

    with open("exports/wf4_seq_fav_cycle.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    print(json.dumps({k: v for k, v in out.items() if k != "per_season_signtest"}, indent=1))
    pst = out["per_season_signtest"]
    print("per_season_signtest:", dict(n=pst["n_seasons"], pos=pst["n_positive"], p=pst["p_sign"]))

if __name__ == "__main__":
    main()
