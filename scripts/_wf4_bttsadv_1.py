# WF4 - BTTS ADVERSARIAL - refutation du finding "BTTS Non gap [0.6,1.3) ROI -9.15%"
# (script _wf4_btts_6.py / dataset _wf4_btts_1.py)
# Verifications:
#  A. integrite dataset vs DB: ids uniques, cross-check cotes/scores/snapshot d'ouverture (sample)
#  B. look-ahead: fetched_at du snapshot d'ouverture vs expected_start / finished_at
#  C. doublons: memes equipes + meme ligue a <30min -> dedup et impact sur le ROI zone
#  D. decomposition marge vs miscalibration: ROI attendu si marche parfaitement calibre (marge seule)
#  E. test zone vs complement (2-sample), deviation z par scope, Stouffer OOS
#  F. sous-periodes (quartiles temporels) pooled + par scope
#  G. bootstrap 10k: CI du ROI Non zone (pooled, OOS-only)
#  H. split alternatif 50/50 sur 8035
#  I. ROI corrige conservateur = OOS only (8035_test 70/30 + newleagues)
# Sortie: exports/wf4_bttsadv.json   (LECTURE SEULE sur la DB)
import sys
import json
import math
from datetime import datetime, timedelta

sys.path.insert(0, ".")
import numpy as np
from scraper.config import load_settings
from sqlalchemy import create_engine, text

LEAGUES = [
    "InstantLeague-8035", "InstantLeague-8036", "InstantLeague-8037",
    "InstantLeague-8042", "InstantLeague-8043", "InstantLeague-8044",
    "InstantLeague-8056", "InstantLeague-8060", "InstantLeague-8065",
]

ZLO, ZHI = 0.6, 1.3


def tstat(r):
    if len(r) < 2:
        return 0.0, 1.0
    se = r.std(ddof=1) / math.sqrt(len(r))
    if se == 0:
        return 0.0, 1.0
    t = r.mean() / se
    return float(t), float(2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2)))))


def two_sample(a, b):
    sea = a.std(ddof=1) / math.sqrt(len(a))
    seb = b.std(ddof=1) / math.sqrt(len(b))
    se = math.sqrt(sea ** 2 + seb ** 2)
    z = (a.mean() - b.mean()) / se
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return float(z), float(p)


def parse_dt(s):
    if s is None:
        return None
    s = str(s).replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S%z",
                "%Y-%m-%d %H:%M:%S.%f%z"):
        try:
            return datetime.strptime(s.split("+")[0].strip(), fmt.replace("%z", ""))
        except ValueError:
            continue
    return None


def main():
    out = {}
    with open("exports/wf4_btts_data.json", encoding="utf-8") as f:
        data = json.load(f)
    rows = data["rows"]
    lz = np.load("exports/wf4_btts_lambdas.npz")
    lh, la = lz["lh"], lz["la"]
    assert list(lz["ids"]) == [r["id"] for r in rows]
    n = len(rows)
    ids = np.array([r["id"] for r in rows])
    o_yes = np.array([r["o_yes"] for r in rows], float)
    o_no = np.array([r["o_no"] for r in rows], float)
    oh = np.array([r["oh"] for r in rows], float)
    od = np.array([r["od"] for r in rows], float)
    oa = np.array([r["oa"] for r in rows], float)
    comp = np.array([r["comp"] for r in rows])
    start = np.array([r["start"] for r in rows])
    sa = np.array([r["sa"] for r in rows])
    sb = np.array([r["sb"] for r in rows])
    win = (sa > 0) & (sb > 0)
    ret_n = o_no * (~win) - 1
    gap = np.abs(lh - la)
    s2 = 1 / o_yes + 1 / o_no
    p_imp = (1 / o_yes) / s2          # proba Oui normalisee (sans marge)
    is35 = comp == "InstantLeague-8035"
    zone = (gap >= ZLO) & (gap < ZHI)

    # --- A. integrite ---
    out["A_unique_ids"] = {"n": n, "n_unique": int(len(set(ids.tolist())))}
    corrupted = set(int(k) for k in json.load(
        open("exports/corrupted_events.json", encoding="utf-8"))["events"].keys())
    out["A_corrupted_in_dataset"] = int(sum(1 for i in ids.tolist() if i in corrupted))

    eng = create_engine(load_settings().db_url)
    in_list = ",".join("'%s'" % l for l in LEAGUES)
    with eng.connect() as c:
        db = c.execute(text("""
            SELECT e.id, e.competition, e.team_a, e.team_b, e.expected_start,
                   o.id, o.odds_home, o.odds_draw, o.odds_away, o.extra_markets, o.captured_at,
                   r.score_a, r.score_b, r.finished_at
            FROM events e
            JOIN results r ON r.event_id = e.id
            JOIN odds_snapshots o ON o.event_id = e.id
            WHERE o.id = (SELECT MIN(o2.id) FROM odds_snapshots o2 WHERE o2.event_id = e.id)
              AND e.competition IN (%s)
        """ % in_list)).fetchall()
    dbmap = {r[0]: r for r in db}
    missing = [int(i) for i in ids.tolist() if i not in dbmap]
    mismatch = []
    rng = np.random.default_rng(7)
    sample_idx = rng.choice(n, size=min(400, n), replace=False)
    for k in sample_idx:
        r = rows[k]
        d = dbmap.get(r["id"])
        if d is None:
            continue
        gng = json.loads(d[9]).get("G/NG") if d[9] else None
        ok = (abs(d[6] - r["oh"]) < 1e-9 and abs(d[7] - r["od"]) < 1e-9
              and abs(d[8] - r["oa"]) < 1e-9 and int(d[11]) == r["sa"]
              and int(d[12]) == r["sb"] and gng
              and abs(float(gng["Oui"]) - r["o_yes"]) < 1e-9
              and abs(float(gng["Non"]) - r["o_no"]) < 1e-9)
        if not ok:
            mismatch.append(int(r["id"]))
    out["A_db_crosscheck"] = {"sampled": int(len(sample_idx)),
                              "missing_in_db": len(missing), "mismatch": mismatch[:10],
                              "n_mismatch": len(mismatch)}

    # --- B. timing du snapshot d'ouverture ---
    after_start, after_finish, no_parse = 0, 0, 0
    deltas = []
    for i in ids.tolist():
        d = dbmap.get(int(i))
        if d is None:
            continue
        est, fat, fin = parse_dt(d[4]), parse_dt(d[10]), parse_dt(d[13])
        if est is None or fat is None:
            no_parse += 1
            continue
        deltas.append((fat - est).total_seconds())
        if fat >= est:
            after_start += 1
        if fin is not None and fat >= fin:
            after_finish += 1
    deltas = np.array(deltas)
    out["B_opening_timing"] = {
        "n": int(len(deltas)), "no_parse": no_parse,
        "fetched_after_start": after_start,
        "fetched_after_finish": after_finish,
        "median_lead_min": round(float(np.median(-deltas)) / 60, 1),
        "p95_lead_min": round(float(np.percentile(-deltas, 5)) / 60, 1),
    }

    # --- C. doublons memes equipes <30min meme ligue ---
    ev = {}
    for i in ids.tolist():
        d = dbmap.get(int(i))
        if d is not None:
            ev[int(i)] = (d[1], d[2], d[3], parse_dt(d[4]))
    bykey = {}
    for i, (cp, ta, tb, est) in ev.items():
        bykey.setdefault((cp, ta, tb), []).append((est, i))
    dup_ids = set()
    pairs = 0
    for k, lst in bykey.items():
        lst.sort(key=lambda x: (x[0] or datetime(2000, 1, 1)))
        for j in range(1, len(lst)):
            if lst[j][0] and lst[j - 1][0] and lst[j][0] - lst[j - 1][0] <= timedelta(minutes=30):
                pairs += 1
                dup_ids.add(lst[j][1])  # garde le premier, drop le second
    keep = np.array([int(i) not in dup_ids for i in ids.tolist()])
    out["C_duplicates"] = {"pairs_within_30min": pairs, "dropped": int((~keep).sum())}

    # --- D. decomposition marge vs miscalibration (zone) ---
    def zone_stats(mask, label):
        m = zone & mask
        nn = int(m.sum())
        if nn == 0:
            return None
        roi = float(ret_n[m].mean()) * 100
        t, p = tstat(ret_n[m])
        # ROI attendu si le marche etait parfaitement calibre (cout = marge seule)
        roi_margin_only = float((o_no[m] * (1 - p_imp[m])).mean() - 1) * 100
        real = float(win[m].mean()); imp = float(p_imp[m].mean())
        se = math.sqrt(max(real * (1 - real), 1e-9) / nn)
        z = (real - imp) / se
        return {"label": label, "n": nn, "roi_no_pct": round(roi, 2), "p_vs_0": round(p, 6),
                "roi_margin_only_pct": round(roi_margin_only, 2),
                "excess_pct": round(roi - roi_margin_only, 2),
                "real_yes": round(real, 4), "implied_yes": round(imp, 4),
                "z_deviation": round(z, 2),
                "avg_o_no": round(float(o_no[m].mean()), 3),
                "wr_no": round(float((~win[m]).mean()), 4)}

    allm = np.ones(n, bool)
    s35 = np.sort(start[is35]); cut70 = s35[int(0.70 * len(s35))]
    scopes = {
        "pooled9": allm,
        "pooled9_dedup": keep,
        "8035_train70": is35 & (start < cut70),
        "8035_test30": is35 & (start >= cut70),
        "newleagues": ~is35,
    }
    out["D_zone"] = {k: zone_stats(m, k) for k, m in scopes.items()}

    # baseline: tout BTTS Non hors zone et global
    t, p = tstat(ret_n)
    out["D_baseline_all"] = {"n": n, "roi_no_pct": round(float(ret_n.mean()) * 100, 2),
                             "roi_margin_only_pct": round(float((o_no * (1 - p_imp)).mean() - 1) * 100, 2)}
    mout = ~zone
    out["D_baseline_outzone"] = {"n": int(mout.sum()),
                                 "roi_no_pct": round(float(ret_n[mout].mean()) * 100, 2)}

    # --- E. zone vs complement + Stouffer OOS ---
    z2, p2 = two_sample(ret_n[zone], ret_n[~zone])
    out["E_zone_vs_complement"] = {"diff_pct": round(float(ret_n[zone].mean() - ret_n[~zone].mean()) * 100, 2),
                                   "z": round(z2, 2), "p": round(p2, 5)}

    def dev_z(mask):
        m = zone & mask
        real = float(win[m].mean()); imp = float(p_imp[m].mean())
        se = math.sqrt(max(real * (1 - real), 1e-9) / m.sum())
        return (real - imp) / se

    z_test = dev_z(scopes["8035_test30"]); z_new = dev_z(scopes["newleagues"])
    z_oos = (z_test + z_new) / math.sqrt(2)
    out["E_stouffer_oos_deviation"] = {
        "z_8035_test": round(z_test, 2), "z_newleagues": round(z_new, 2),
        "z_combined": round(z_oos, 2),
        "p_two_sided": round(2 * (1 - 0.5 * (1 + math.erf(abs(z_oos) / math.sqrt(2)))), 4)}

    # --- F. sous-periodes ---
    out["F_subperiods"] = {}
    order = np.argsort(start)
    zr = zone[order]
    rn = ret_n[order]
    qs = np.array_split(np.arange(n), 4)
    for qi, idx in enumerate(qs):
        m = zr[idx]
        r = rn[idx][m]
        t, p = tstat(r)
        out["F_subperiods"]["Q%d" % (qi + 1)] = {
            "n": int(m.sum()), "roi_no_pct": round(float(r.mean()) * 100, 2), "p": round(p, 4),
            "start_range": [str(start[order][idx][0])[:16], str(start[order][idx][-1])[:16]]}

    # --- G. bootstrap ---
    def boot(r, B=10000):
        bs = rng.choice(r, size=(B, len(r)), replace=True).mean(axis=1) * 100
        return [round(float(np.percentile(bs, q)), 2) for q in (2.5, 50, 97.5)]

    out["G_bootstrap_pooled_zone"] = boot(ret_n[zone])
    oos = zone & (scopes["8035_test30"] | scopes["newleagues"])
    out["G_bootstrap_oos_zone"] = boot(ret_n[oos])
    out["G_oos_zone_roi"] = {"n": int(oos.sum()),
                             "roi_no_pct": round(float(ret_n[oos].mean()) * 100, 2)}

    # --- H. split alternatif 50/50 sur 8035 ---
    cut50 = s35[int(0.50 * len(s35))]
    out["H_8035_test50"] = zone_stats(is35 & (start >= cut50), "8035_test50")

    # --- I. par ligue dans la zone ---
    out["I_zone_per_league"] = {}
    for lg in sorted(set(comp.tolist())):
        m = zone & (comp == lg)
        if m.sum() < 30:
            continue
        t, p = tstat(ret_n[m])
        out["I_zone_per_league"][lg] = {
            "n": int(m.sum()), "roi_no_pct": round(float(ret_n[m].mean()) * 100, 2),
            "p": round(p, 4),
            "dev": round(float(win[m].mean() - p_imp[m].mean()), 4)}

    with open("exports/wf4_bttsadv.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
