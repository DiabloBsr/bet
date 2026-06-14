# WF4 - bttsrefute - etape 1: refutation adversariale du finding
#   "Combos 1X2&G/NG toxiques (X&0but -19.3%)" (script _wf4_btts_5.py)
# Checks (DB en LECTURE SEULE):
#  A. look-ahead: snapshots d'ouverture captures apres finished_at ?
#  B. doublons exacts (comp, team_a, team_b, expected_start) DANS le dataset
#     -> dedup MIN(id) et recompute
#  C. recompute independant des 6 selections (settlement re-ecrit from scratch):
#     pooled9, dedup, 8035 test70 (meme cut), test50 (split alternatif), newleagues,
#     par ligue, quintiles temporels pooled
#  D. bootstrap 95% CI (X&0but, 1&BTTS) pooled + dedup
#  E. marge implicite du combo (somme 1/cote des 6) -> verifie "marge 12%"
#  F. calibration implied(no-margin) vs observe par selection
#  G. scan adversarial de poches positives (deciles de cote), concordance
#     8035-test ET newleagues
# Sortie: exports/wf4_bttsrefute.json
import sys
import json
import math

sys.path.insert(0, ".")
import numpy as np
from scraper.config import load_settings
from sqlalchemy import create_engine, text

COMBO_SELS = [
    ("1&BTTS", "1 gagne et les deux équipes marquent", lambda sa, sb: (sa > sb) & (sb > 0)),
    ("1&only1", "1 gagne et seulement  1  marque", lambda sa, sb: (sa > sb) & (sb == 0)),
    ("2&BTTS", "2 gagne et les deux équipes marquent", lambda sa, sb: (sb > sa) & (sa > 0)),
    ("2&only2", "2 gagne et seulement 2 marque", lambda sa, sb: (sb > sa) & (sa == 0)),
    ("X&0but", "X et aucun but", lambda sa, sb: (sa == 0) & (sb == 0)),
    ("X&BTTS", "X et les deux équipes marquent", lambda sa, sb: (sa == sb) & (sa > 0)),
]

rng = np.random.default_rng(42)


def tstat(r):
    if len(r) < 2:
        return 0.0, 1.0
    se = r.std(ddof=1) / math.sqrt(len(r))
    if se == 0:
        return 0.0, 1.0
    t = r.mean() / se
    return float(t), float(2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2)))))


def ev(ret, odds, m):
    nn = int(m.sum())
    if nn == 0:
        return {"n": 0}
    t, p = tstat(ret[m])
    return {"n": nn, "roi_pct": round(float(ret[m].mean()) * 100, 2),
            "t": round(t, 2), "p": float("%.3g" % p),
            "avg_odds": round(float(odds[m].mean()), 3),
            "wr": round(float((ret[m] > 0).mean()), 4)}


def main():
    out = {}
    with open("exports/wf4_btts_family_data.json", encoding="utf-8") as f:
        rows = json.load(f)["rows"]
    n = len(rows)
    ids = np.array([r["id"] for r in rows])
    comp = np.array([r["comp"] for r in rows])
    start = np.array([r["start"] for r in rows])
    sa = np.array([r["sa"] for r in rows])
    sb = np.array([r["sb"] for r in rows])
    out["n_rows"] = n
    out["n_unique_ids"] = int(len(set(ids.tolist())))

    # corrupted vraiment exclus ?
    corrupted = set(int(k) for k in json.load(
        open("exports/corrupted_events.json", encoding="utf-8"))["events"].keys())
    out["corrupted_in_dataset"] = int(sum(1 for i in ids.tolist() if int(i) in corrupted))

    # ---------- A + B : DB lecture seule ----------
    e = create_engine(load_settings().db_url)
    id_list = ",".join(str(int(i)) for i in ids.tolist())
    with e.connect() as c:
        db = c.execute(text("""
          SELECT e.id, e.team_a, e.team_b, e.expected_start, e.competition,
                 o.captured_at, res.finished_at, res.score_a, res.score_b, o.status
          FROM events e JOIN results res ON res.event_id = e.id
          JOIN odds_snapshots o ON o.id = (SELECT MIN(id) FROM odds_snapshots WHERE event_id = e.id)
          WHERE e.id IN (%s)
        """ % id_list)).fetchall()
    n_after_finish = sum(1 for r0 in db if r0[5] is not None and r0[6] is not None
                         and str(r0[5]) >= str(r0[6]))
    n_not_upcoming = sum(1 for r0 in db if r0[9] != "upcoming")
    out["A_look_ahead"] = {"n_db": len(db),
                           "openings_captured_after_finished_at": n_after_finish,
                           "openings_status_not_upcoming": n_not_upcoming}

    groups = {}
    for r0 in db:
        groups.setdefault((r0[4], r0[1], r0[2], str(r0[3])), []).append(r0)
    dup_groups = {k: v for k, v in groups.items() if len(v) > 1}
    same_score = sum(1 for v in dup_groups.values()
                     if len(set((x[7], x[8]) for x in v)) == 1)
    out["B_dups_in_dataset"] = {
        "n_groups": len(groups), "n_dup_groups": len(dup_groups),
        "rows_in_dup_groups": sum(len(v) for v in dup_groups.values()),
        "dup_groups_same_score": same_score,
        "dup_groups_diff_score": len(dup_groups) - same_score}
    keep_ids = set(min(x[0] for x in v) for v in groups.values())
    dedup = np.array([int(i) in keep_ids for i in ids.tolist()])
    out["B_dedup_kept"] = int(dedup.sum())

    # ---------- C-G ----------
    is35 = comp == "InstantLeague-8035"
    s35 = np.sort(start[is35])
    cut70 = s35[int(0.70 * len(s35))]
    cut50 = s35[int(0.50 * len(s35))]
    test70 = is35 & (start >= cut70)
    test50 = is35 & (start >= cut50)
    newl = ~is35
    leagues = sorted(set(comp.tolist()))

    allodds = {}
    for name, key, _ in COMBO_SELS:
        allodds[name] = np.array(
            [(r0["combo"] or {}).get(key) or np.nan for r0 in rows], float)
    mat = np.vstack([allodds[n0] for n0, _, _ in COMBO_SELS])
    full = ~np.isnan(mat).any(0) & (mat < 99.5).all(0)
    overround = (1 / mat[:, full]).sum(0)
    out["E_overround_combo"] = {
        "n_full6": int(full.sum()),
        "mean": round(float(overround.mean()), 4),
        "p5": round(float(np.quantile(overround, 0.05)), 4),
        "p95": round(float(np.quantile(overround, 0.95)), 4)}

    sel_out = {}
    for i, (name, key, winf) in enumerate(COMBO_SELS):
        odds = allodds[name]
        base = ~np.isnan(odds) & (odds < 99.5)
        win = winf(sa, sb)
        ret = odds * win - 1
        d = {"pooled9": ev(ret, odds, base),
             "pooled9_dedup": ev(ret, odds, base & dedup),
             "test70_8035": ev(ret, odds, base & test70),
             "test50_8035": ev(ret, odds, base & test50),
             "newleagues": ev(ret, odds, base & newl),
             "by_league": {lg: ev(ret, odds, base & (comp == lg)) for lg in leagues}}
        # quintiles temporels pooled (rang temporel global)
        bidx = np.where(base)[0]
        ranks = np.argsort(np.argsort(start[bidx]))
        nq = len(bidx)
        quint = {}
        for q in range(5):
            mq = np.zeros(n, bool)
            lo, hi = int(q * nq / 5), int((q + 1) * nq / 5)
            mq[bidx[(ranks >= lo) & (ranks < hi)]] = True
            quint["Q%d" % (q + 1)] = ev(ret, odds, mq)
        d["pooled_time_quintiles"] = quint
        # calibration sur events avec les 6 cotes
        imp = (1 / mat[i, full]) / overround
        obs = winf(sa[full], sb[full])
        d["calib"] = {"implied_nomargin": round(float(imp.mean()), 4),
                      "observed": round(float(obs.mean()), 4)}
        sel_out[name] = d

    # bootstrap CI
    for name in ("X&0but", "1&BTTS"):
        key = [k for n0, k, _ in COMBO_SELS if n0 == name][0]
        winf = [w for n0, _, w in COMBO_SELS if n0 == name][0]
        odds = allodds[name]
        base = ~np.isnan(odds) & (odds < 99.5)
        ret = odds * winf(sa, sb) - 1
        for tag, m in (("pooled", base), ("dedup", base & dedup)):
            x = ret[m]
            boots = np.array([rng.choice(x, len(x)).mean() for _ in range(3000)])
            sel_out[name]["bootstrap_" + tag] = {
                "roi_pct": round(float(x.mean()) * 100, 2),
                "ci95_lo": round(float(np.quantile(boots, 0.025)) * 100, 2),
                "ci95_hi": round(float(np.quantile(boots, 0.975)) * 100, 2)}

    # G. poches positives concordantes
    pockets = []
    n_scanned = 0
    for name, key, winf in COMBO_SELS:
        odds = allodds[name]
        base = ~np.isnan(odds) & (odds < 99.5)
        ret = odds * winf(sa, sb) - 1
        qs = np.nanquantile(odds[base], np.linspace(0, 1, 11))
        for q in range(10):
            hi = qs[q + 1] + (1e-9 if q == 9 else 0)
            m = base & (odds >= qs[q]) & (odds < hi)
            n_scanned += 1
            a = ev(ret, odds, m & test70)
            b = ev(ret, odds, m & newl)
            if a.get("n", 0) >= 150 and b.get("n", 0) >= 150 \
               and a.get("roi_pct", -99) > 0 and b.get("roi_pct", -99) > 0:
                pockets.append({"sel": name, "decile": q,
                                "odds_lo": round(float(qs[q]), 2),
                                "odds_hi": round(float(qs[q + 1]), 2),
                                "test70": a, "newleagues": b})
    out["G_positive_pockets_concordant"] = {"n_scanned": n_scanned,
                                            "pockets": pockets}
    out["selections"] = sel_out
    out["cut70"] = str(cut70)
    out["cut50"] = str(cut50)

    with open("exports/wf4_bttsrefute.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    print("ids:", out["n_rows"], "unique:", out["n_unique_ids"],
          "corrupted incl:", out["corrupted_in_dataset"])
    print("A:", out["A_look_ahead"])
    print("B:", out["B_dups_in_dataset"], "-> dedup kept:", out["B_dedup_kept"])
    print("E overround combo:", out["E_overround_combo"])
    print()

    def fmt(x):
        if x.get("n", 0) == 0:
            return "n=0"
        return "n=%5d roi=%+7.2f%% t=%+5.2f" % (x["n"], x["roi_pct"], x["t"])

    print("%-8s | %-30s | %-30s | %-30s | %-30s | %-30s" % (
        "sel", "pooled9", "pooled_dedup", "test70_8035", "test50_8035", "newleagues"))
    for name, _, _ in COMBO_SELS:
        d = sel_out[name]
        print("%-8s | %-30s | %-30s | %-30s | %-30s | %-30s" % (
            name, fmt(d["pooled9"]), fmt(d["pooled9_dedup"]),
            fmt(d["test70_8035"]), fmt(d["test50_8035"]), fmt(d["newleagues"])))
    print()
    for name, _, _ in COMBO_SELS:
        d = sel_out[name]
        print("%-8s calib implied=%.4f obs=%.4f | quintiles roi: %s" % (
            name, d["calib"]["implied_nomargin"], d["calib"]["observed"],
            " ".join("%+.1f" % d["pooled_time_quintiles"]["Q%d" % q].get("roi_pct", 0)
                     for q in range(1, 6))))
    for name in ("X&0but", "1&BTTS"):
        print(name, "bootstrap pooled:", sel_out[name]["bootstrap_pooled"],
              "| dedup:", sel_out[name]["bootstrap_dedup"])
    print()
    print("poches positives concordantes:", len(pockets), "/", n_scanned, "cellules")
    for p0 in pockets:
        print(" ", p0["sel"], "decile", p0["decile"],
              "[%.2f-%.2f]" % (p0["odds_lo"], p0["odds_hi"]),
              "test70:", fmt(p0["test70"]), "new:", fmt(p0["newleagues"]))


if __name__ == "__main__":
    main()
