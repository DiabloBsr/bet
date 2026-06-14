# WF4 - BTTS pooled 9 ligues - etape 2: calibration par bucket de |lh-la|
# - inverse (lh, la) depuis le 1X2 d'ouverture (grille Poisson independante, marge plate)
# - p_grid(BTTS) = (1-e^-lh)(1-e^-la) ; compare implied marche (normalise) / grille / realise
# - calibration + ROI Oui/Non par bucket de gap=|lh-la|, pooled-9 / 8035 / newleagues
# Sortie: exports/wf4_btts_calib.json + lambdas exports/wf4_btts_lambdas.npz
import json
import math
import numpy as np

GMAX = 16

def invert_lambdas(ph, pd):
    """Newton vectorise: trouve (lh, la) tq grille Poisson -> P(home)=ph, P(draw)=pd."""
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
        draw = np.eye(GMAX)
        return (grid * home).sum((1, 2)), (grid * draw).sum((1, 2))

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
    f1, f2 = probs(lh, la)
    return lh, la, np.abs(f1 - ph), np.abs(f2 - pd)


GAP_EDGES = [0.0, 0.15, 0.35, 0.60, 0.90, 1.30, 99.0]
GAP_NAMES = ["g0_015", "g015_035", "g035_060", "g060_090", "g090_130", "g130p"]


def bucket_table(mask, gap, o_yes, o_no, win_yes, p_imp_norm, p_grid):
    """calibration + ROI Oui/Non par bucket de gap, sur le sous-ensemble mask."""
    tab = {}
    for bi in range(len(GAP_NAMES)):
        m = mask & (gap >= GAP_EDGES[bi]) & (gap < GAP_EDGES[bi + 1])
        n = int(m.sum())
        if n == 0:
            continue
        w = win_yes[m]
        real = float(w.mean())
        ret_y = o_yes[m] * w - 1
        ret_n = o_no[m] * (~w) - 1
        def tstat(r):
            if len(r) < 2:
                return 0.0, 1.0
            se = r.std(ddof=1) / math.sqrt(len(r))
            if se == 0:
                return 0.0, 1.0
            t = r.mean() / se
            # p-value bilaterale approx normale
            from math import erf
            p = 2 * (1 - 0.5 * (1 + erf(abs(t) / math.sqrt(2))))
            return float(t), float(p)
        ty, py = tstat(ret_y)
        tn, pn = tstat(ret_n)
        tab[GAP_NAMES[bi]] = {
            "n": n,
            "avg_o_yes": round(float(o_yes[m].mean()), 3),
            "avg_o_no": round(float(o_no[m].mean()), 3),
            "p_yes_implied_raw": round(float((1 / o_yes[m]).mean()), 4),
            "p_yes_implied_norm": round(float(p_imp_norm[m].mean()), 4),
            "p_yes_grid": round(float(p_grid[m].mean()), 4),
            "p_yes_real": round(real, 4),
            "real_minus_norm": round(real - float(p_imp_norm[m].mean()), 4),
            "roi_yes_pct": round(float(ret_y.mean()) * 100, 2),
            "t_yes": round(ty, 2), "p_yes": round(py, 5),
            "roi_no_pct": round(float(ret_n.mean()) * 100, 2),
            "t_no": round(tn, 2), "p_no": round(pn, 5),
        }
    return tab


def main():
    with open("exports/wf4_btts_data.json", encoding="utf-8") as f:
        data = json.load(f)
    rows = data["rows"]
    n = len(rows)
    oh = np.array([r["oh"] for r in rows], float)
    od = np.array([r["od"] for r in rows], float)
    oa = np.array([r["oa"] for r in rows], float)
    o_yes = np.array([r["o_yes"] for r in rows], float)
    o_no = np.array([r["o_no"] for r in rows], float)
    comp = np.array([r["comp"] for r in rows])
    start = np.array([r["start"] for r in rows])
    sa = np.array([r["sa"] for r in rows])
    sb = np.array([r["sb"] for r in rows])
    win_yes = (sa > 0) & (sb > 0)

    s = 1 / oh + 1 / od + 1 / oa
    ph, pd = (1 / oh) / s, (1 / od) / s
    lh, la, res_h, res_d = invert_lambdas(ph, pd)
    print("inversion: residu max=%.1e ; marge 1X2 moy=%.4f" % (
        max(res_h.max(), res_d.max()), (s - 1).mean()))

    gap = np.abs(lh - la)
    p_grid = (1 - np.exp(-lh)) * (1 - np.exp(-la))
    s2 = 1 / o_yes + 1 / o_no
    p_imp_norm = (1 / o_yes) / s2
    print("marge G/NG: moy=%.4f std=%.4f" % ((s2 - 1).mean(), (s2 - 1).std()))
    print("coherence grille vs marche: corr(p_grid, p_imp_norm)=%.4f  ecart moy=%.4f  |ecart| max=%.4f"
          % (np.corrcoef(p_grid, p_imp_norm)[0, 1], (p_grid - p_imp_norm).mean(),
             np.abs(p_grid - p_imp_norm).max()))
    print("BTTS global: real=%.4f  implied_norm=%.4f  grid=%.4f  (n=%d)" % (
        win_yes.mean(), p_imp_norm.mean(), p_grid.mean(), n))

    is35 = comp == "InstantLeague-8035"
    newl = ~is35
    groups = {"pooled9": np.ones(n, bool), "8035": is35, "newleagues": newl}
    out = {"n_events": n, "gap_edges": GAP_EDGES, "groups": {}}
    for gname, gm in groups.items():
        out["groups"][gname] = bucket_table(gm, gap, o_yes, o_no, win_yes, p_imp_norm, p_grid)

    with open("exports/wf4_btts_calib.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    np.savez("exports/wf4_btts_lambdas.npz", lh=lh, la=la,
             ids=np.array([r["id"] for r in rows]))

    for gname in groups:
        print("\n=== %s ===" % gname)
        print("%-9s %5s %6s %6s %7s %7s %7s %8s | %7s %7s | %7s %7s" % (
            "bucket", "n", "o_yes", "o_no", "imp_nrm", "grid", "real", "d(re-im)",
            "ROIyes", "p", "ROIno", "p"))
        for bn, v in out["groups"][gname].items():
            print("%-9s %5d %6.2f %6.2f %7.4f %7.4f %7.4f %+8.4f | %+6.2f%% %7.5f | %+6.2f%% %7.5f" % (
                bn, v["n"], v["avg_o_yes"], v["avg_o_no"], v["p_yes_implied_norm"],
                v["p_yes_grid"], v["p_yes_real"], v["real_minus_norm"],
                v["roi_yes_pct"], v["p_yes"], v["roi_no_pct"], v["p_no"]))


if __name__ == "__main__":
    main()
