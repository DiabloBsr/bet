# WF4 - score exact - etape 2: inversion lambda + analyse par cellule
# - inverse (lh, la) depuis le 1X2 d'ouverture (grille Poisson independante, marge 6% supposee plate)
# - verifie que le pricing des 8 NOUVELLES ligues est aussi exactement Poisson (residu draw)
# - par cellule de score: marge de cellule (cote offerte vs grille), deviation simulateur
#   (freq reelle vs grille), edge brut (freq reelle x cote offerte - 1)
# - par groupe de ligues: 8035 / champs (8036,8037,8042,8043,8044) / coupes (8056,8060,8065)
# Sortie: exports/wf4_scoreexact_cells.json
import json
import numpy as np

GMAX = 16  # grille 0..15

def invert_lambdas(ph, pd):
    """Newton vectorise: trouve (lh, la) tq grille Poisson -> P(home)=ph, P(draw)=pd."""
    n = len(ph)
    lh = np.full(n, 1.6)
    la = np.full(n, 1.2)
    ks = np.arange(GMAX)
    import math
    logfact = np.array([math.lgamma(k + 1) for k in ks])

    def probs(lh, la):
        # pois[n, k]
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
        # damping
        step = np.clip(np.maximum(np.abs(dlh), np.abs(dla)), 0, 1)
        damp = np.where(step > 0.5, 0.5 / np.maximum(step, 1e-9), 1.0)
        lh = np.clip(lh - dlh * damp, 0.05, 8.0)
        la = np.clip(la - dla * damp, 0.05, 8.0)
    f1, f2 = probs(lh, la)
    return lh, la, np.abs(f1 - ph), np.abs(f2 - pd)


def cell_probs(lh, la):
    """P(i-j) pour i,j in 0..6 (cellules du marche)."""
    import math
    ks = np.arange(GMAX)
    logfact = np.array([math.lgamma(k + 1) for k in ks])
    ph_ = np.exp(-lh[:, None] + ks[None, :] * np.log(lh[:, None]) - logfact[None, :])
    pa_ = np.exp(-la[:, None] + ks[None, :] * np.log(la[:, None]) - logfact[None, :])
    return ph_[:, :7, None] * pa_[:, None, :7]  # [n, 7, 7]


def main():
    with open("exports/wf4_scoreexact_data.json", encoding="utf-8") as f:
        data = json.load(f)
    rows = data["rows"]
    n = len(rows)
    oh = np.array([r["oh"] for r in rows], float)
    od = np.array([r["od"] for r in rows], float)
    oa = np.array([r["oa"] for r in rows], float)
    comp = np.array([r["comp"] for r in rows])
    sa = np.array([r["sa"] for r in rows])
    sb = np.array([r["sb"] for r in rows])

    s = 1 / oh + 1 / od + 1 / oa
    ph, pd = (1 / oh) / s, (1 / od) / s
    lh, la, res_h, res_d = invert_lambdas(ph, pd)
    print("inversion: residu max home=%.2e draw=%.2e ; frac<1e-8: %.4f" % (
        res_h.max(), res_d.max(), float(((res_h < 1e-8) & (res_d < 1e-8)).mean())))
    # verif Poisson-exact par ligue (residu apres inversion = 0 si la grille fitte exactement
    # ph ET pd; le test reel d'exactitude: la marge 1X2 doit etre plate -> verif marge
    for grp in sorted(set(comp)):
        m = comp == grp
        print("  %s: margin 1X2 mean=%.4f std=%.4f resmax=%.1e" % (
            grp, (s[m] - 1).mean(), (s[m] - 1).std(), max(res_h[m].max(), res_d[m].max())))

    cp = cell_probs(lh, la)  # [n,7,7]

    cells = ["%d-%d" % (i, j) for i in range(7) for j in range(7) if i + j <= 6]
    se_odds = np.full((n, len(cells)), np.nan)
    for k, r in enumerate(rows):
        for ci, cname in enumerate(cells):
            v = r["se"].get(cname)
            if v is not None:
                se_odds[k, ci] = v

    groups = {
        "8035": comp == "InstantLeague-8035",
        "champs5": np.isin(comp, ["InstantLeague-8036", "InstantLeague-8037",
                                  "InstantLeague-8042", "InstantLeague-8043", "InstantLeague-8044"]),
        "cups3": np.isin(comp, ["InstantLeague-8056", "InstantLeague-8060", "InstantLeague-8065"]),
        "pooled9": np.ones(n, bool),
    }

    out = {"n_events": n, "groups": {}}
    for gname, gm in groups.items():
        gtab = {}
        for ci, cname in enumerate(cells):
            i, j = int(cname[0]), int(cname[2])
            offered = gm & ~np.isnan(se_odds[:, ci]) & (se_odds[:, ci] < 99.5)
            no = int(offered.sum())
            if no < 30:
                continue
            odds_c = se_odds[offered, ci]
            gp = cp[offered, i, j]
            win = (sa[offered] == i) & (sb[offered] == j)
            nw = int(win.sum())
            exp_w = float(gp.sum())
            # marge cellule: mediane de 1/(cote*grille) - 1
            marg = float(np.median(1 / (odds_c * gp) - 1))
            ret = odds_c * win - 1
            roi = float(ret.mean())
            roi_se = float(ret.std(ddof=1) / np.sqrt(no)) if no > 1 else 0.0
            ratio = nw / exp_w if exp_w > 0 else 0.0
            z_dev = (nw - exp_w) / np.sqrt(exp_w) if exp_w > 0 else 0.0
            gtab[cname] = {
                "n_offered": no, "avg_odds": float(odds_c.mean()),
                "wins": nw, "exp_wins_grid": round(exp_w, 1),
                "real_over_grid": round(ratio, 3), "z_dev": round(float(z_dev), 2),
                "cell_margin_med": round(marg, 4),
                "roi_pct": round(roi * 100, 2), "roi_se_pct": round(roi_se * 100, 2),
            }
        out["groups"][gname] = gtab

    with open("exports/wf4_scoreexact_cells.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    # affichage condense pooled9 trie par ROI
    print("\nPOOLED-9 par cellule (cote<99.5):")
    print("%-5s %6s %7s %5s %7s %9s %7s %8s %7s" % (
        "cell", "n", "avgodd", "wins", "exp", "real/grid", "z", "margin", "ROI%"))
    g = out["groups"]["pooled9"]
    for cname, v in sorted(g.items(), key=lambda kv: -kv[1]["roi_pct"]):
        print("%-5s %6d %7.2f %5d %7.1f %9.3f %7.2f %8.3f %7.2f +-%.1f" % (
            cname, v["n_offered"], v["avg_odds"], v["wins"], v["exp_wins_grid"],
            v["real_over_grid"], v["z_dev"], v["cell_margin_med"], v["roi_pct"], v["roi_se_pct"]))

    # sauvegarde lambdas pour l'etape 3
    np.savez("exports/wf4_scoreexact_lambdas.npz", lh=lh, la=la,
             ids=np.array([r["id"] for r in rows]))


if __name__ == "__main__":
    main()
