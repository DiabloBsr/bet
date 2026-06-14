# WF4 - BTTS - etape 3: scan walk-forward de cellules 1D/2D
# - dimensions: gap=|lh-la| (6 buckets), bande de cote o_yes, min(lh,la), mu=lh+la
# - cellules scannees: chaque (cellule, cote Oui/Non) sur 8035-TRAIN (70% temporel)
# - selection train: n>=150, ROI>=+4% -> validation sur 8035-TEST puis newleagues pooled
# - rapporte n_tests_scanned et p-values brutes (t-test des retours)
# Sortie: exports/wf4_btts_scan.json
import json
import math
import numpy as np

GAP_EDGES = [0.0, 0.15, 0.35, 0.60, 0.90, 1.30, 99.0]
GAP_NAMES = ["g0_015", "g015_035", "g035_060", "g060_090", "g090_130", "g130p"]
ODDS_EDGES = [1.0, 1.5, 1.7, 1.9, 2.1, 2.4, 99.0]
ODDS_NAMES = ["o<1.5", "o1.5-1.7", "o1.7-1.9", "o1.9-2.1", "o2.1-2.4", "o>=2.4"]
MIN_EDGES = [0.0, 0.8, 1.0, 1.2, 1.5, 99.0]
MIN_NAMES = ["mn<0.8", "mn0.8-1.0", "mn1.0-1.2", "mn1.2-1.5", "mn>=1.5"]
MU_EDGES = [0.0, 2.4, 2.8, 3.2, 3.6, 99.0]
MU_NAMES = ["mu<2.4", "mu2.4-2.8", "mu2.8-3.2", "mu3.2-3.6", "mu>=3.6"]


def tstat(r):
    if len(r) < 2:
        return 0.0, 1.0
    se = r.std(ddof=1) / math.sqrt(len(r))
    if se == 0:
        return 0.0, 1.0
    t = r.mean() / se
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return float(t), float(p)


def cell_masks(gap, o_side, mn, mu):
    """genere {nom_cellule: mask} pour les familles 1D et 2D."""
    cells = {}
    def add_dim(name, edges, names, x):
        for i, nm in enumerate(names):
            cells["%s:%s" % (name, nm)] = (x >= edges[i]) & (x < edges[i + 1])
    add_dim("gap", GAP_EDGES, GAP_NAMES, gap)
    add_dim("odds", ODDS_EDGES, ODDS_NAMES, o_side)
    add_dim("minlam", MIN_EDGES, MIN_NAMES, mn)
    add_dim("mu", MU_EDGES, MU_NAMES, mu)
    # 2D gap x odds
    for i, gn in enumerate(GAP_NAMES):
        gm = (gap >= GAP_EDGES[i]) & (gap < GAP_EDGES[i + 1])
        for j, on in enumerate(ODDS_NAMES):
            cells["gapXodds:%s|%s" % (gn, on)] = gm & (o_side >= ODDS_EDGES[j]) & (o_side < ODDS_EDGES[j + 1])
        for j, mnm in enumerate(MIN_NAMES):
            cells["gapXminlam:%s|%s" % (gn, mnm)] = gm & (mn >= MIN_EDGES[j]) & (mn < MIN_EDGES[j + 1])
        for j, mun in enumerate(MU_NAMES):
            cells["gapXmu:%s|%s" % (gn, mun)] = gm & (mu >= MU_EDGES[j]) & (mu < MU_EDGES[j + 1])
    return cells


def eval_mask(m, ret):
    n = int(m.sum())
    if n == 0:
        return {"n": 0, "roi_pct": 0.0, "p": 1.0}
    r = ret[m]
    t, p = tstat(r)
    return {"n": n, "roi_pct": round(float(r.mean()) * 100, 2), "t": round(t, 2), "p": round(p, 6)}


def main():
    with open("exports/wf4_btts_data.json", encoding="utf-8") as f:
        rows = json.load(f)["rows"]
    lz = np.load("exports/wf4_btts_lambdas.npz")
    lh, la = lz["lh"], lz["la"]
    ids = lz["ids"]
    assert list(ids) == [r["id"] for r in rows]

    o_yes = np.array([r["o_yes"] for r in rows], float)
    o_no = np.array([r["o_no"] for r in rows], float)
    comp = np.array([r["comp"] for r in rows])
    start = np.array([r["start"] for r in rows])
    sa = np.array([r["sa"] for r in rows])
    sb = np.array([r["sb"] for r in rows])
    win_yes = (sa > 0) & (sb > 0)
    ret_yes = o_yes * win_yes - 1
    ret_no = o_no * (~win_yes) - 1

    gap = np.abs(lh - la)
    mn = np.minimum(lh, la)
    mu = lh + la

    is35 = comp == "InstantLeague-8035"
    s35 = np.sort(start[is35])
    cut = s35[int(0.70 * len(s35))]
    train = is35 & (start < cut)
    test = is35 & (start >= cut)
    newl = ~is35
    print("8035 train=%d test=%d cut=%s | newleagues=%d" % (train.sum(), test.sum(), cut, newl.sum()))

    sides = {"yes": (ret_yes, o_yes), "no": (ret_no, o_no)}
    results = []
    n_scanned = 0
    for side, (ret, odd_side) in sides.items():
        cells = cell_masks(gap, odd_side, mn, mu)
        for cname, cm in cells.items():
            n_scanned += 1
            tr = eval_mask(cm & train, ret)
            entry = {"cell": cname, "side": side, "train": tr}
            if tr["n"] >= 150 and tr["roi_pct"] >= 4.0:
                entry["test"] = eval_mask(cm & test, ret)
                entry["newleagues"] = eval_mask(cm & newl, ret)
                entry["avg_odds_test"] = round(float(odd_side[cm & test].mean()), 3) if entry["test"]["n"] else None
                entry["selected"] = True
            results.append(entry)

    sel = [r for r in results if r.get("selected")]
    print("n_tests_scanned=%d ; selectionnees train (n>=150, ROI>=4%%): %d" % (n_scanned, len(sel)))
    sel.sort(key=lambda r: -r["train"]["roi_pct"])
    for r in sel:
        print("%-38s %-3s TRAIN n=%4d roi=%+6.2f%% p=%.4f | TEST n=%4d roi=%+6.2f%% p=%.4f | NEW n=%4d roi=%+6.2f%% p=%.4f" % (
            r["cell"], r["side"], r["train"]["n"], r["train"]["roi_pct"], r["train"]["p"],
            r["test"]["n"], r["test"]["roi_pct"], r["test"]["p"],
            r["newleagues"]["n"], r["newleagues"]["roi_pct"], r["newleagues"]["p"]))

    # vue descriptive: top/bottom cellules train (toutes, pour contexte)
    allr = sorted([r for r in results if r["train"]["n"] >= 150], key=lambda r: -r["train"]["roi_pct"])
    print("\nTop 12 train (contexte):")
    for r in allr[:12]:
        print("  %-38s %-3s n=%4d roi=%+6.2f%% p=%.4f" % (
            r["cell"], r["side"], r["train"]["n"], r["train"]["roi_pct"], r["train"]["p"]))

    with open("exports/wf4_btts_scan.json", "w", encoding="utf-8") as f:
        json.dump({"n_tests_scanned": n_scanned, "cut": str(cut),
                   "train_n": int(train.sum()), "test_n": int(test.sum()),
                   "selected": sel, "all": results}, f, indent=1)


if __name__ == "__main__":
    main()
