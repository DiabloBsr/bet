# -*- coding: utf-8 -*-
# WF4 - score exact - etape 3 (reprise apres interruption de l'etape 2):
#   - inverse (lh, la) depuis le 1X2 d'ouverture (grille Poisson independante)
#   - verifie que le pricing score-exact des 9 ligues = grille + marge par cellule
#     (std de la marge de cellule par ligue ~ 0 => pricing exact-grid partout)
#   - table par cellule: n offert, cote moyenne, marge cellule (med/std),
#     ratio reel/grille (deviation simulateur), z, ROI a la cote offerte
#   - groupes: 8035 / champs5 / cups3 / pooled9
# Entree: exports/wf4_scoreexact_data.json (produit par scripts/_wf4_scoreexact_1.py)
# Sortie: exports/wf4_scoreexact_cells.json + exports/wf4_scoreexact_lambdas.npz
import json, math
import numpy as np

GMAX = 16


def poisson_pmf(lam, ks, logfact):
    return np.exp(-lam[:, None] + ks[None, :] * np.log(lam[:, None]) - logfact[None, :])


def invert_lambdas(ph, pd):
    """Newton 2D vectorise: (lh, la) tq grille Poisson independante -> P(home)=ph, P(draw)=pd."""
    n = len(ph)
    lh = np.full(n, 1.6)
    la = np.full(n, 1.2)
    ks = np.arange(GMAX)
    logfact = np.array([math.lgamma(k + 1) for k in ks])
    iu = np.arange(GMAX)
    Mhome = (iu[:, None] > iu[None, :]).astype(float)
    Mdraw = np.eye(GMAX)

    def probs(lh, la):
        a = poisson_pmf(lh, ks, logfact)
        b = poisson_pmf(la, ks, logfact)
        grid = a[:, :, None] * b[:, None, :]
        return (grid * Mhome).sum((1, 2)), (grid * Mdraw).sum((1, 2))

    eps = 1e-6
    for _ in range(80):
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
        step = np.maximum(np.abs(dlh), np.abs(dla))
        damp = np.where(step > 0.5, 0.5 / np.maximum(step, 1e-9), 1.0)
        lh = np.clip(lh - dlh * damp, 0.05, 8.0)
        la = np.clip(la - dla * damp, 0.05, 8.0)
    f1, f2 = probs(lh, la)
    return lh, la, np.abs(f1 - ph), np.abs(f2 - pd)


def main():
    with open("exports/wf4_scoreexact_data.json", encoding="utf-8") as f:
        data = json.load(f)
    rows = [r for r in data["rows"]
            if r["oh"] and r["od"] and r["oa"] and r["oh"] > 1 and r["od"] > 1 and r["oa"] > 1]
    n = len(rows)
    print("events analysables:", n)
    oh = np.array([r["oh"] for r in rows])
    od = np.array([r["od"] for r in rows])
    oa = np.array([r["oa"] for r in rows])
    comp = np.array([r["comp"] for r in rows])
    sa = np.array([r["sa"] for r in rows])
    sb = np.array([r["sb"] for r in rows])

    s = 1 / oh + 1 / od + 1 / oa
    ph, pd = (1 / oh) / s, (1 / od) / s
    lh, la, res_h, res_d = invert_lambdas(ph, pd)
    ok = (res_h < 1e-8) & (res_d < 1e-8)
    print("inversion: residu<1e-8 pour %.4f des events (max %.1e)" % (
        ok.mean(), max(res_h.max(), res_d.max())))

    ks = np.arange(GMAX)
    logfact = np.array([math.lgamma(k + 1) for k in ks])
    a = poisson_pmf(lh, ks, logfact)
    b = poisson_pmf(la, ks, logfact)
    cp = a[:, :7, None] * b[:, None, :7]   # P(i-j) i,j in 0..6

    cells = ["%d-%d" % (i, j) for i in range(7) for j in range(7) if i + j <= 6]
    se_odds = np.full((n, len(cells)), np.nan)
    for k, r in enumerate(rows):
        for ci, cname in enumerate(cells):
            v = r["se"].get(cname)
            if v is not None:
                se_odds[k, ci] = float(v)

    groups = {
        "8035": comp == "InstantLeague-8035",
        "champs5": np.isin(comp, ["InstantLeague-8036", "InstantLeague-8037",
                                  "InstantLeague-8042", "InstantLeague-8043",
                                  "InstantLeague-8044"]),
        "cups3": np.isin(comp, ["InstantLeague-8056", "InstantLeague-8060",
                                "InstantLeague-8065"]),
        "pooled9": np.ones(n, bool),
    }

    # --- verif pricing exact-grid par ligue: marge de cellule (med/std) ---
    print("\nmarge de cellule par ligue (med, std), cote<99.5:")
    lg_margins = {}
    for lg in sorted(set(comp)):
        m = comp == lg
        ent = {}
        for cname in ("2-1", "1-2", "1-1", "2-0", "1-0", "2-2"):
            ci = cells.index(cname)
            i, j = int(cname[0]), int(cname[2])
            mm = m & ~np.isnan(se_odds[:, ci]) & (se_odds[:, ci] < 99.5)
            if mm.sum() < 20:
                continue
            marg = 1 / (se_odds[mm, ci] * cp[mm, i, j]) - 1
            ent[cname] = {"n": int(mm.sum()), "med": round(float(np.median(marg)), 4),
                          "std": round(float(np.std(marg)), 4)}
        lg_margins[lg] = ent
        print(" %s: %s" % (lg, {k: (v["med"], v["std"]) for k, v in ent.items()}))

    out = {"n_events": n, "cell_margins_by_league": lg_margins, "groups": {}}
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
            marg = float(np.median(1 / (odds_c * gp) - 1))
            ret = odds_c * win - 1
            roi = float(ret.mean())
            # p-value H0: EV=0 a la cote offerte (q=1/cote)
            q0 = 1 / odds_c
            var0 = (q0 * (odds_c - 1) ** 2 + (1 - q0)).sum()
            z_roi = ret.sum() / math.sqrt(var0)
            ratio = nw / exp_w if exp_w > 0 else 0.0
            z_dev = (nw - exp_w) / math.sqrt(exp_w) if exp_w > 0 else 0.0
            gtab[cname] = {
                "n_offered": no, "avg_odds": round(float(odds_c.mean()), 2),
                "wins": nw, "exp_wins_grid": round(exp_w, 1),
                "real_over_grid": round(ratio, 3), "z_dev": round(z_dev, 2),
                "cell_margin_med": round(marg, 4),
                "roi_pct": round(roi * 100, 2), "z_roi": round(float(z_roi), 2),
            }
        out["groups"][gname] = gtab

    with open("exports/wf4_scoreexact_cells.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    for gname in ("pooled9", "8035", "champs5", "cups3"):
        g = out["groups"][gname]
        print("\n%s par cellule (trie ROI desc):" % gname)
        print("%-5s %6s %7s %5s %7s %9s %6s %8s %8s %6s" % (
            "cell", "n", "avgodd", "wins", "expW", "real/grid", "z_dev", "margin", "ROI%", "z_roi"))
        for cname, v in sorted(g.items(), key=lambda kv: -kv[1]["roi_pct"])[:14]:
            print("%-5s %6d %7.2f %5d %7.1f %9.3f %6.2f %8.3f %8.2f %6.2f" % (
                cname, v["n_offered"], v["avg_odds"], v["wins"], v["exp_wins_grid"],
                v["real_over_grid"], v["z_dev"], v["cell_margin_med"], v["roi_pct"], v["z_roi"]))

    np.savez("exports/wf4_scoreexact_lambdas.npz",
             lh=lh, la=la, ids=np.array([r["id"] for r in rows]))
    print("\necrit: exports/wf4_scoreexact_cells.json + lambdas.npz")


if __name__ == "__main__":
    main()
