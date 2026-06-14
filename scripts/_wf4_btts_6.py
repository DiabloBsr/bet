# WF4 - BTTS - etape 6: verifications finales
#  (a) calibration logistique du G/NG principal: logit(real) ~ logit(p_implied_norm)
#  (b) calibration par ligue (9) : real vs implied, ROI Oui/Non
#  (c) zone explicite mission: o_yes in [1.8, 2.2] x buckets gap (pooled9 + newleagues)
#  (d) scan direct newleagues pooled (gap x odds x side) avec comptage des tests
#  (e) deviation de calibration (real - implied_norm) zone gap [0.6,1.3) avec z binomial
# Sortie: exports/wf4_btts.json (synthese finale du domaine)
import json
import math
import numpy as np

GAP_EDGES = [0.0, 0.15, 0.35, 0.60, 0.90, 1.30, 99.0]
GAP_NAMES = ["g0_015", "g015_035", "g035_060", "g060_090", "g090_130", "g130p"]
ODDS_EDGES = [1.0, 1.5, 1.7, 1.9, 2.1, 2.4, 99.0]
ODDS_NAMES = ["o<1.5", "o1.5-1.7", "o1.7-1.9", "o1.9-2.1", "o2.1-2.4", "o>=2.4"]


def tstat(r):
    if len(r) < 2:
        return 0.0, 1.0
    se = r.std(ddof=1) / math.sqrt(len(r))
    if se == 0:
        return 0.0, 1.0
    t = r.mean() / se
    return float(t), float(2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2)))))


def main():
    with open("exports/wf4_btts_data.json", encoding="utf-8") as f:
        rows = json.load(f)["rows"]
    lz = np.load("exports/wf4_btts_lambdas.npz")
    lh, la = lz["lh"], lz["la"]
    assert list(lz["ids"]) == [r["id"] for r in rows]
    n = len(rows)
    o_yes = np.array([r["o_yes"] for r in rows], float)
    o_no = np.array([r["o_no"] for r in rows], float)
    comp = np.array([r["comp"] for r in rows])
    sa = np.array([r["sa"] for r in rows])
    sb = np.array([r["sb"] for r in rows])
    win = (sa > 0) & (sb > 0)
    ret_y = o_yes * win - 1
    ret_n = o_no * (~win) - 1
    gap = np.abs(lh - la)
    s2 = 1 / o_yes + 1 / o_no
    p_imp = (1 / o_yes) / s2
    is35 = comp == "InstantLeague-8035"

    out = {}

    # (a) regression logistique 1-var (Newton IRLS simple)
    def logistic_calib(mask):
        x = np.log(p_imp[mask] / (1 - p_imp[mask]))
        y = win[mask].astype(float)
        b0, b1 = 0.0, 1.0
        for _ in range(50):
            z = b0 + b1 * x
            p = 1 / (1 + np.exp(-z))
            w = p * (1 - p)
            g0 = (y - p).sum()
            g1 = ((y - p) * x).sum()
            h00 = w.sum(); h01 = (w * x).sum(); h11 = (w * x * x).sum()
            det = h00 * h11 - h01 * h01
            if abs(det) < 1e-12:
                break
            d0 = (g0 * h11 - g1 * h01) / det
            d1 = (g1 * h00 - g0 * h01) / det
            b0 += d0; b1 += d1
            if abs(d0) < 1e-10 and abs(d1) < 1e-10:
                break
        # erreurs std
        z = b0 + b1 * x
        p = 1 / (1 + np.exp(-z))
        w = p * (1 - p)
        h00 = w.sum(); h01 = (w * x).sum(); h11 = (w * x * x).sum()
        det = h00 * h11 - h01 * h01
        se0 = math.sqrt(h11 / det); se1 = math.sqrt(h00 / det)
        return {"intercept": round(b0, 4), "se_int": round(se0, 4),
                "slope": round(b1, 4), "se_slope": round(se1, 4),
                "z_slope_vs_1": round((b1 - 1) / se1, 2),
                "z_int_vs_0": round(b0 / se0, 2), "n": int(mask.sum())}

    out["logistic_calibration"] = {
        "pooled9": logistic_calib(np.ones(n, bool)),
        "8035": logistic_calib(is35),
        "newleagues": logistic_calib(~is35),
    }
    print("(a) calibration logistique (slope=1, int=0 si marche parfaitement calibre):")
    for k, v in out["logistic_calibration"].items():
        print("  %-10s n=%5d slope=%.3f+-%.3f (z vs 1: %+0.2f)  int=%+.4f+-%.4f (z: %+0.2f)" % (
            k, v["n"], v["slope"], v["se_slope"], v["z_slope_vs_1"],
            v["intercept"], v["se_int"], v["z_int_vs_0"]))

    # (b) par ligue
    print("\n(b) calibration par ligue:")
    out["per_league"] = {}
    for lg in sorted(set(comp)):
        m = comp == lg
        ty, py = tstat(ret_y[m]); tn, pn = tstat(ret_n[m])
        d = {"n": int(m.sum()), "real": round(float(win[m].mean()), 4),
             "implied_norm": round(float(p_imp[m].mean()), 4),
             "roi_yes_pct": round(float(ret_y[m].mean()) * 100, 2), "p_yes": round(py, 5),
             "roi_no_pct": round(float(ret_n[m].mean()) * 100, 2), "p_no": round(pn, 5)}
        out["per_league"][lg] = d
        print("  %-22s n=%5d real=%.4f imp=%.4f | ROIyes=%+6.2f%% p=%.4f | ROIno=%+6.2f%% p=%.4f" % (
            lg, d["n"], d["real"], d["implied_norm"], d["roi_yes_pct"], d["p_yes"],
            d["roi_no_pct"], d["p_no"]))

    # (c) zone mission o_yes in [1.8, 2.2]
    print("\n(c) zone mission o_yes [1.8,2.2] x gap:")
    out["mission_zone"] = {}
    zone = (o_yes >= 1.8) & (o_yes <= 2.2)
    for scope, sm in (("pooled9", np.ones(n, bool)), ("newleagues", ~is35)):
        tab = {}
        for i, gn in enumerate(GAP_NAMES):
            m = zone & sm & (gap >= GAP_EDGES[i]) & (gap < GAP_EDGES[i + 1])
            if m.sum() < 50:
                continue
            ty, py = tstat(ret_y[m]); tn, pn = tstat(ret_n[m])
            tab[gn] = {"n": int(m.sum()), "real": round(float(win[m].mean()), 4),
                       "implied_norm": round(float(p_imp[m].mean()), 4),
                       "avg_o_yes": round(float(o_yes[m].mean()), 3),
                       "roi_yes_pct": round(float(ret_y[m].mean()) * 100, 2), "p_yes": round(py, 5),
                       "roi_no_pct": round(float(ret_n[m].mean()) * 100, 2), "p_no": round(pn, 5)}
        out["mission_zone"][scope] = tab
        for gn, v in tab.items():
            print("  %-10s %-9s n=%4d real=%.3f imp=%.3f ROIyes=%+6.2f%% (p=%.4f) ROIno=%+6.2f%% (p=%.4f)" % (
                scope, gn, v["n"], v["real"], v["implied_norm"],
                v["roi_yes_pct"], v["p_yes"], v["roi_no_pct"], v["p_no"]))

    # (d) scan direct newleagues pooled: gap x odds x side
    print("\n(d) scan newleagues pooled (gap x bande de cote x side):")
    hits = []
    n_scanned = 0
    newl = ~is35
    for side, (ret, osd) in (("yes", (ret_y, o_yes)), ("no", (ret_n, o_no))):
        for i, gn in enumerate(GAP_NAMES):
            gm = (gap >= GAP_EDGES[i]) & (gap < GAP_EDGES[i + 1])
            for j, on in enumerate(ODDS_NAMES):
                m = newl & gm & (osd >= ODDS_EDGES[j]) & (osd < ODDS_EDGES[j + 1])
                n_scanned += 1
                if m.sum() < 150:
                    continue
                t, p = tstat(ret[m])
                if ret[m].mean() * 100 >= 4.0 and p <= 0.01:
                    hits.append({"cell": "%s|%s" % (gn, on), "side": side,
                                 "n": int(m.sum()), "roi_pct": round(float(ret[m].mean()) * 100, 2),
                                 "p": round(p, 6), "avg_odds": round(float(osd[m].mean()), 3),
                                 "wr": round(float((ret[m] > 0).mean()), 4)})
    out["newleagues_scan"] = {"n_tests_scanned": n_scanned, "hits": hits}
    print("  n_tests=%d hits(ROI>=4%%, p<=0.01, n>=150)=%d" % (n_scanned, len(hits)))
    for h in hits:
        print("   ", h)

    # (e) deviation de calibration zone gap [0.6,1.3) pooled
    m = (gap >= 0.6) & (gap < 1.3)
    real = float(win[m].mean()); imp = float(p_imp[m].mean())
    se = math.sqrt(real * (1 - real) / m.sum())
    z = (real - imp) / se
    pz = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    tn, pn = tstat(ret_n[m])
    out["midgap_deviation"] = {
        "n": int(m.sum()), "real": round(real, 4), "implied_norm": round(imp, 4),
        "z": round(z, 2), "p": round(pz, 5),
        "roi_no_pct": round(float(ret_n[m].mean()) * 100, 2), "p_no": round(pn, 6),
        "avg_o_no": round(float(o_no[m].mean()), 3),
        "wr_no": round(float((~win[m]).mean()), 4),
        "roi_yes_pct": round(float(ret_y[m].mean()) * 100, 2),
    }
    print("\n(e) gap [0.6,1.3) pooled9: n=%d real=%.4f imp=%.4f z=%+.2f p=%.5f | ROIno=%.2f%% (p=%.5f) ROIyes=%.2f%%" % (
        m.sum(), real, imp, z, pz, out["midgap_deviation"]["roi_no_pct"],
        out["midgap_deviation"]["p_no"], out["midgap_deviation"]["roi_yes_pct"]))
    # idem 8035 test-only (confirmation temporelle de la deviation, pas du ROI)
    start = np.array([r["start"] for r in rows])
    s35 = np.sort(start[is35])
    cut = s35[int(0.70 * len(s35))]
    for scope, sm in (("8035_train", is35 & (start < cut)), ("8035_test", is35 & (start >= cut)),
                      ("newleagues", newl)):
        mm = m & sm
        r2 = float(win[mm].mean()); i2 = float(p_imp[mm].mean())
        se2 = math.sqrt(max(r2 * (1 - r2), 1e-9) / mm.sum())
        z2 = (r2 - i2) / se2
        tn2, pn2 = tstat(ret_n[mm])
        print("   %-10s n=%5d real=%.4f imp=%.4f z=%+.2f | ROIno=%+6.2f%% p=%.4f" % (
            scope, mm.sum(), r2, i2, z2, float(ret_n[mm].mean()) * 100, pn2))
        out["midgap_deviation"][scope] = {"n": int(mm.sum()), "real": round(r2, 4),
                                          "implied": round(i2, 4), "z": round(z2, 2),
                                          "roi_no_pct": round(float(ret_n[mm].mean()) * 100, 2),
                                          "p_no": round(pn2, 5)}

    with open("exports/wf4_btts.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
