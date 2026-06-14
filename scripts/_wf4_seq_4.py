# -*- coding: utf-8 -*-
"""
WF4 - finition du finding "cycle saisonnier du favori":
 1. sign test par instance de saison (detection sequentielle CORRIGEE: scan temporel,
    nouvelle saison quand le round chute) — robustesse aux clusters.
 2. regle executable: Double Chance CONTRE le favori (X2 si favori domicile, 1X sinon)
    sur les 5 dernieres journees — cotes DC depuis extra_markets du snapshot d'ouverture.
 3. variantes: dog/draw last5 restreints a pfav [0.55,0.75] (strates les plus touchees).
Sortie: exports/wf4_seq_final.json. LECTURE SEULE.
"""
import sys, json, math, importlib.util, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
from scipy import stats
from sqlalchemy import create_engine, text

spec = importlib.util.spec_from_file_location("wf4seq1", "scripts/_wf4_seq_1.py")
m1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m1)
from scraper.config import load_settings

MAXR = {"InstantLeague-8035": 38, "InstantLeague-8036": 38, "InstantLeague-8037": 38,
        "InstantLeague-8042": 34, "InstantLeague-8043": 34, "InstantLeague-8044": 34}
RNG = np.random.default_rng(7)

def roi(bets):
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
        r["fav_home"] = r["ph"] >= r["pa"]
        r["favwin"] = 1.0 if ((r["sa"] > r["sb"]) if r["fav_home"] else (r["sb"] > r["sa"])) else 0.0
        r["draw"] = 1.0 if r["sa"] == r["sb"] else 0.0
        r["dogwin"] = 1.0 - r["favwin"] - r["draw"]
        r["early"] = r["rnd"] <= 5
        r["late"] = r["rnd"] >= MAXR[r["comp"]] - 4

    out = {}
    n_tests = 0

    # ---- 1. sign test par instance de saison (scan sequentiel) ----
    contrasts = []
    for lg in MAXR:
        sub = sorted([r for r in champs if r["comp"] == lg], key=lambda r: (r["start"], r["id"]))
        season, prev_rnd = 0, None
        for r in sub:
            if prev_rnd is not None and r["rnd"] < prev_rnd - 10:  # chute franche => nouvelle saison
                season += 1
            r["season"] = season
            prev_rnd = max(prev_rnd, r["rnd"]) if prev_rnd is not None and r["rnd"] >= prev_rnd - 10 else r["rnd"]
        bysea = {}
        for r in sub:
            bysea.setdefault(r["season"], []).append(r)
        for sid, lst in bysea.items():
            e = [x["favwin"] - x["pfav"] for x in lst if x["early"]]
            l = [x["favwin"] - x["pfav"] for x in lst if x["late"]]
            if len(e) >= 10 and len(l) >= 10:
                contrasts.append(dict(league=lg, season=sid, n_e=len(e), n_l=len(l),
                                      contrast=float(np.mean(e) - np.mean(l))))
    pos = sum(1 for c in contrasts if c["contrast"] > 0)
    out["per_season_signtest"] = dict(
        n_seasons=len(contrasts), n_positive=pos,
        p_sign=float(stats.binomtest(pos, len(contrasts), 0.5).pvalue) if contrasts else None,
        detail=[(c["league"][-4:], c["season"], round(c["contrast"], 3)) for c in contrasts])
    n_tests += 1

    # ---- 2. cotes Double Chance du snapshot d'ouverture ----
    eng = create_engine(load_settings().db_url)
    ids = [r["id"] for r in champs]
    dc = {}
    with eng.connect() as c:
        res = c.execute(text("""
            SELECT o.event_id, o.extra_markets FROM odds_snapshots o
            JOIN (SELECT event_id, MIN(id) mid FROM odds_snapshots GROUP BY event_id) m
              ON m.mid = o.id
        """)).fetchall()
    idset = set(ids)
    for eid, xm in res:
        if eid not in idset or not xm:
            continue
        try:
            d = json.loads(xm) if isinstance(xm, str) else xm
            mkt = d.get("Double Chance")
            if mkt:
                dc[eid] = {k: float(v) for k, v in mkt.items()}
        except Exception:
            pass
    out["dc_coverage"] = dict(n_champs=len(champs), n_with_dc=len(dc))
    # marge DC estimee (verif structure): 1/1X + 1/X2 + 1/12 - 2 n'a pas de sens;
    # marge pairwise: (1/o_1X)/(ph+pd) moyenne
    margs = []
    for r in champs[:2000]:
        m = dc.get(r["id"])
        if m and "1X" in m:
            margs.append((1 / m["1X"]) / (r["ph"] + r["pd"]))
    out["dc_margin_est"] = float(np.mean(margs)) if margs else None

    def dc_bet(r):
        m = dc.get(r["id"])
        if not m:
            return None
        sel = "X2" if r["fav_home"] else "1X"
        o = m.get(sel)
        if not o or o >= 90:
            return None
        won = (r["favwin"] == 0.0)
        return (o, won)

    rules = {}
    scopes = (("pooled-6champs", lambda r: True),
              ("8035", lambda r: r["comp"] == "InstantLeague-8035"),
              ("newchamps", lambda r: r["comp"] != "InstantLeague-8035"))
    for scope, flt in scopes:
        sub = [r for r in champs if flt(r)]
        bets_late = [b for r in sub if r["late"] and (b := dc_bet(r))]
        bets_early = [b for r in sub if r["early"] and (b := dc_bet(r))]
        bets_mid = [b for r in sub if not r["late"] and not r["early"] and (b := dc_bet(r))]
        rules[scope] = {"dc_antifav_last5": roi(bets_late),
                        "dc_antifav_J1-5_control": roi(bets_early),
                        "dc_antifav_mid_control": roi(bets_mid)}
        n_tests += 3
        # variantes strates pfav 0.55-0.75
        sub2 = [r for r in sub if 0.55 <= r["pfav"] < 0.75]
        rules[scope]["dc_antifav_last5_pfav55-75"] = roi([b for r in sub2 if r["late"] and (b := dc_bet(r))])
        rules[scope]["dog_last5_pfav55-75"] = roi(
            [(r["oa"] if r["fav_home"] else r["oh"], r["dogwin"] == 1.0) for r in sub2 if r["late"]])
        rules[scope]["draw_last5_pfav55-75"] = roi(
            [(r["od"], r["draw"] == 1.0) for r in sub2 if r["late"]])
        n_tests += 3
    # walk-forward 8035 test30 temporel
    sub8 = sorted([r for r in champs if r["comp"] == "InstantLeague-8035"], key=lambda r: (r["start"], r["id"]))
    test = sub8[int(len(sub8) * 0.7):]
    rules["8035-test30"] = {
        "dc_antifav_last5": roi([b for r in test if r["late"] and (b := dc_bet(r))]),
        "dc_antifav_last5_pfav55-75": roi([b for r in test if r["late"] and 0.55 <= r["pfav"] < 0.75 and (b := dc_bet(r))]),
    }
    n_tests += 2
    out["rules"] = rules
    out["n_tests_scanned"] = n_tests

    with open("exports/wf4_seq_final.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
