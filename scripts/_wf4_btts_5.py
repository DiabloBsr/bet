# WF4 - BTTS - etape 5: calibration + walk-forward des marches freres BTTS
#   G/NG domicile (Oui/Non), G/NG exterieur (Oui/Non), BTTS 1ere MT (Oui/Non),
#   1X2 & G/NG (6 selections)
# Cellules: global + 6 buckets de gap=|lh-la| -> 12 sel x 7 = 84 tests
# Selection train 8035 (70%): n>=150, ROI>=+4% -> test 8035 + newleagues pooled
# Sortie: exports/wf4_btts_family_scan.json
import json
import math
import numpy as np

GMAX = 16
GAP_EDGES = [0.0, 0.15, 0.35, 0.60, 0.90, 1.30, 99.0]
GAP_NAMES = ["g0_015", "g015_035", "g035_060", "g060_090", "g090_130", "g130p"]


def invert_lambdas(ph, pd):
    n = len(ph)
    lh = np.full(n, 1.6)
    la = np.full(n, 1.2)
    ks = np.arange(GMAX)
    logfact = np.array([math.lgamma(k + 1) for k in ks])

    def probs(lh, la):
        ph_ = np.exp(-lh[:, None] + ks[None, :] * np.log(lh[:, None]) - logfact[None, :])
        pa_ = np.exp(-la[:, None] + ks[None, :] * np.log(la[:, None]) - logfact[None, :])
        grid = ph_[:, :, None] * pa_[:, None, :]
        i = np.arange(GMAX)
        home = np.where(i[:, None] > i[None, :], 1.0, 0.0)
        return (grid * home).sum((1, 2)), (grid * np.eye(GMAX)).sum((1, 2))

    eps = 1e-6
    for _ in range(60):
        f1, f2 = probs(lh, la)
        r1, r2 = f1 - ph, f2 - pd
        if max(np.abs(r1).max(), np.abs(r2).max()) < 1e-12:
            break
        a1, a2 = probs(lh + eps, la)
        b1, b2 = probs(lh, la + eps)
        j11, j21 = (a1 - f1) / eps, (a2 - f2) / eps
        j12, j22 = (b1 - f1) / eps, (b2 - f2) / eps
        det = j11 * j22 - j12 * j21
        det = np.where(np.abs(det) < 1e-14, 1e-14, det)
        dlh = (r1 * j22 - r2 * j12) / det
        dla = (r2 * j11 - r1 * j21) / det
        step = np.clip(np.maximum(np.abs(dlh), np.abs(dla)), 0, 1)
        damp = np.where(step > 0.5, 0.5 / np.maximum(step, 1e-9), 1.0)
        lh = np.clip(lh - dlh * damp, 0.05, 8.0)
        la = np.clip(la - dla * damp, 0.05, 8.0)
    return lh, la


def tstat(r):
    if len(r) < 2:
        return 0.0, 1.0
    se = r.std(ddof=1) / math.sqrt(len(r))
    if se == 0:
        return 0.0, 1.0
    t = r.mean() / se
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return float(t), float(p)


def main():
    with open("exports/wf4_btts_family_data.json", encoding="utf-8") as f:
        rows = json.load(f)["rows"]
    n = len(rows)
    oh = np.array([r["oh"] for r in rows], float)
    od = np.array([r["od"] for r in rows], float)
    oa = np.array([r["oa"] for r in rows], float)
    s = 1 / oh + 1 / od + 1 / oa
    lh, la = invert_lambdas((1 / oh) / s, (1 / od) / s)
    gap = np.abs(lh - la)

    comp = np.array([r["comp"] for r in rows])
    start = np.array([r["start"] for r in rows])
    sa = np.array([r["sa"] for r in rows])
    sb = np.array([r["sb"] for r in rows])
    hta = np.array([(-1 if r["hta"] is None else r["hta"]) for r in rows])
    htb = np.array([(-1 if r["htb"] is None else r["htb"]) for r in rows])

    def odds_of(key, sel):
        return np.array([(r[key] or {}).get(sel) or np.nan for r in rows], float)

    # (nom, cotes, gagne, valide)
    ht_ok = (hta >= 0) & (htb >= 0)
    selections = [
        ("home_Oui", odds_of("home", "Oui"), sa > 0, None),
        ("home_Non", odds_of("home", "Non"), sa == 0, None),
        ("away_Oui", odds_of("away", "Oui"), sb > 0, None),
        ("away_Non", odds_of("away", "Non"), sb == 0, None),
        ("htBTTS_Oui", odds_of("ht", "Oui"), (hta > 0) & (htb > 0), ht_ok),
        ("htBTTS_Non", odds_of("ht", "Non"), ~((hta > 0) & (htb > 0)), ht_ok),
        ("1&BTTS", odds_of("combo", "1 gagne et les deux équipes marquent"),
         (sa > sb) & (sb > 0), None),
        ("1&only1", odds_of("combo", "1 gagne et seulement  1  marque"),
         (sa > sb) & (sb == 0), None),
        ("2&BTTS", odds_of("combo", "2 gagne et les deux équipes marquent"),
         (sb > sa) & (sa > 0), None),
        ("2&only2", odds_of("combo", "2 gagne et seulement 2 marque"),
         (sb > sa) & (sa == 0), None),
        ("X&0but", odds_of("combo", "X et aucun but"), (sa == 0) & (sb == 0), None),
        ("X&BTTS", odds_of("combo", "X et les deux équipes marquent"),
         (sa == sb) & (sa > 0), None),
    ]

    is35 = comp == "InstantLeague-8035"
    s35 = np.sort(start[is35])
    cut = s35[int(0.70 * len(s35))]
    train = is35 & (start < cut)
    test = is35 & (start >= cut)
    newl = ~is35

    def ev(m, ret, odds):
        nn = int(m.sum())
        if nn == 0:
            return {"n": 0, "roi_pct": 0.0, "p": 1.0, "avg_odds": None, "wr": None}
        t, p = tstat(ret[m])
        return {"n": nn, "roi_pct": round(float(ret[m].mean()) * 100, 2),
                "t": round(t, 2), "p": round(p, 6),
                "avg_odds": round(float(odds[m].mean()), 3),
                "wr": round(float((ret[m] > 0).mean()), 4)}

    results = []
    n_scanned = 0
    print("%-12s %-9s | POOLED9 n, roi, p | marge implicite" % ("selection", "bucket"))
    for name, odds, win, valid in selections:
        base = ~np.isnan(odds) & (odds < 99.5)
        if valid is not None:
            base &= valid
        ret = odds * win - 1
        cells = {"ALL": np.ones(n, bool)}
        for i, gn in enumerate(GAP_NAMES):
            cells[gn] = (gap >= GAP_EDGES[i]) & (gap < GAP_EDGES[i + 1])
        for cname, cm in cells.items():
            m = base & cm
            n_scanned += 1
            tr = ev(m & train, ret, odds)
            entry = {"sel": name, "cell": cname, "train": tr,
                     "pooled9": ev(m, ret, odds)}
            if tr["n"] >= 150 and tr["roi_pct"] >= 4.0:
                entry["test"] = ev(m & test, ret, odds)
                entry["newleagues"] = ev(m & newl, ret, odds)
                entry["selected"] = True
            results.append(entry)
        # print global pooled calibration per selection
        g = [r for r in results if r["sel"] == name and r["cell"] == "ALL"][0]["pooled9"]
        print("%-12s %-9s   n=%5d roi=%+6.2f%% p=%.5f avg_odds=%.2f wr=%.3f" % (
            name, "ALL", g["n"], g["roi_pct"], g["p"], g["avg_odds"], g["wr"]))

    sel = [r for r in results if r.get("selected")]
    print("\nn_tests_scanned=%d ; selectionnees train: %d" % (n_scanned, len(sel)))
    for r in sorted(sel, key=lambda x: -x["train"]["roi_pct"]):
        print("%-12s %-9s TRAIN n=%4d roi=%+7.2f%% p=%.4f | TEST n=%4d roi=%+7.2f%% p=%.4f | NEW n=%5d roi=%+7.2f%% p=%.4f" % (
            r["sel"], r["cell"], r["train"]["n"], r["train"]["roi_pct"], r["train"]["p"],
            r["test"]["n"], r["test"]["roi_pct"], r["test"]["p"],
            r["newleagues"]["n"], r["newleagues"]["roi_pct"], r["newleagues"]["p"]))

    with open("exports/wf4_btts_family_scan.json", "w", encoding="utf-8") as f:
        json.dump({"n_tests_scanned": n_scanned, "cut": str(cut), "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
