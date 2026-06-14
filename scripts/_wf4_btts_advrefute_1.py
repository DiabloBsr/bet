# WF4 - BTTS - ADVERSARIAL REFUTE du finding "midgap [0.6,1.3) BTTS Oui sous-price"
# Attaques:
#  (1) duplicats exacts de fixtures (meme comp/equipes/expected_start) dans l'echantillon
#  (2) OOS pur: events finis APRES l'extraction (ids absents du fichier), memes filtres
#  (3) controle d'offset de ligue: deviation in-zone MINUS deviation out-zone, par ligue
#      (le pooled melange l'offset global de 8035 avec l'effet de zone)
#  (4) Monte-Carlo post-selection: sous H0 (win ~ Bern(p_imp)), proba que le MEILLEUR
#      merge de 2 buckets adjacents (ou 1 bucket) atteigne z>=2.74 -> p-value corrigee
#  (5) bootstrap du ROI Oui en zone + sensibilite au modele de marge (proportionnel vs additif)
#  (6) split temporel alternatif 50/50 sur 8035 + serie par tranches temporelles pooled
# Sortie: exports/wf4_btts_advrefute.json   (LECTURE SEULE sur la DB)
import sys, json, math
sys.path.insert(0, ".")
import numpy as np

GMAX = 16
GAP_EDGES = [0.0, 0.15, 0.35, 0.60, 0.90, 1.30, 99.0]
GAP_NAMES = ["g0_015", "g015_035", "g035_060", "g060_090", "g090_130", "g130p"]
LEAGUES = [
    "InstantLeague-8035", "InstantLeague-8036", "InstantLeague-8037",
    "InstantLeague-8042", "InstantLeague-8043", "InstantLeague-8044",
    "InstantLeague-8056", "InstantLeague-8060", "InstantLeague-8065",
]
rng = np.random.default_rng(20260612)


def invert_lambdas(ph, pd):
    n = len(ph)
    lh = np.full(n, 1.6); la = np.full(n, 1.2)
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


def main():
    out = {}
    with open("exports/wf4_btts_data.json", encoding="utf-8") as f:
        rows = json.load(f)["rows"]
    lz = np.load("exports/wf4_btts_lambdas.npz")
    lh, la = lz["lh"], lz["la"]
    assert list(lz["ids"]) == [r["id"] for r in rows]
    ids_orig = set(r["id"] for r in rows)
    n = len(rows)
    o_yes = np.array([r["o_yes"] for r in rows], float)
    o_no = np.array([r["o_no"] for r in rows], float)
    comp = np.array([r["comp"] for r in rows])
    start = np.array([r["start"] for r in rows])
    sa = np.array([r["sa"] for r in rows]); sb = np.array([r["sb"] for r in rows])
    win = (sa > 0) & (sb > 0)
    gap = np.abs(lh - la)
    s2 = 1 / o_yes + 1 / o_no
    p_imp = (1 / o_yes) / s2
    zone = (gap >= 0.6) & (gap < 1.3)
    ret_y = o_yes * win - 1

    # ---------- (1) duplicats de fixtures ----------
    from scraper.config import load_settings
    from sqlalchemy import create_engine, text
    e = create_engine(load_settings().db_url)
    in_list = ",".join("'%s'" % l for l in LEAGUES)
    with e.connect() as c:
        tr = c.execute(text(
            "SELECT id, competition, team_a, team_b, expected_start FROM events "
            "WHERE competition IN (%s)" % in_list)).fetchall()
    teams = {r[0]: (r[1], r[2], r[3], str(r[4])) for r in tr}
    seen = {}
    dup_pairs = []
    for r in rows:
        k = teams.get(r["id"])
        if k is None:
            continue
        if k in seen:
            dup_pairs.append((seen[k], r["id"], k))
        else:
            seen[k] = r["id"]
    out["dup_exact_fixtures_in_sample"] = len(dup_pairs)
    print("(1) duplicats exacts (comp,teamA,teamB,expected_start) dans l'echantillon: %d" % len(dup_pairs))
    for d in dup_pairs[:10]:
        print("   ", d)
    # nb de paires distinctes dans la zone (clustering)
    zpairs = set()
    for i, r in enumerate(rows):
        if zone[i] and r["id"] in teams:
            t = teams[r["id"]]
            zpairs.add((t[0], t[1], t[2]))
    out["zone_n_matches"] = int(zone.sum())
    out["zone_n_distinct_pairs"] = len(zpairs)
    print("    zone: %d matchs, %d paires distinctes (ratio %.1f matchs/paire)" % (
        zone.sum(), len(zpairs), zone.sum() / max(len(zpairs), 1)))

    # ---------- (2) OOS pur: nouveaux events finis depuis l'extraction ----------
    corrupted = set()
    with open("exports/corrupted_events.json", encoding="utf-8") as f:
        corrupted = set(int(k) for k in json.load(f)["events"].keys())
    with e.connect() as c:
        fresh = c.execute(text("""
            SELECT e.id, e.competition, e.expected_start,
                   o.odds_home, o.odds_draw, o.odds_away, o.extra_markets,
                   r.score_a, r.score_b, r.ht_score_a, r.ht_score_b, r.goals_json
            FROM events e
            JOIN results r ON r.event_id = e.id
            JOIN odds_snapshots o ON o.event_id = e.id
            WHERE o.id = (SELECT MIN(o2.id) FROM odds_snapshots o2 WHERE o2.event_id = e.id)
              AND e.competition IN (%s)
        """ % in_list)).fetchall()
    new_rows = []
    for (eid, cp, est, oh, od, oa, em, a, b, hta, htb, gj) in fresh:
        if eid in ids_orig or eid in corrupted:
            continue
        if a is None or b is None or oh is None or od is None or oa is None or not em:
            continue
        if hta is not None and htb is not None and (hta > a or htb > b):
            continue
        if gj:
            try:
                gl = json.loads(gj)
                if isinstance(gl, list) and len(gl) > 0 and len(gl) != int(a) + int(b):
                    continue
            except Exception:
                pass
        try:
            gng = json.loads(em).get("G/NG")
        except Exception:
            continue
        if not gng or "Oui" not in gng or "Non" not in gng:
            continue
        new_rows.append({"comp": cp, "oh": oh, "od": od, "oa": oa,
                         "o_yes": float(gng["Oui"]), "o_no": float(gng["Non"]),
                         "sa": int(a), "sb": int(b)})
    out["oos_new_n_total"] = len(new_rows)
    if new_rows:
        noh = np.array([r["oh"] for r in new_rows], float)
        nod = np.array([r["od"] for r in new_rows], float)
        noa = np.array([r["oa"] for r in new_rows], float)
        ns = 1 / noh + 1 / nod + 1 / noa
        nlh, nla = invert_lambdas((1 / noh) / ns, (1 / nod) / ns)
        ngap = np.abs(nlh - nla)
        nzone = (ngap >= 0.6) & (ngap < 1.3)
        noy = np.array([r["o_yes"] for r in new_rows], float)
        non_ = np.array([r["o_no"] for r in new_rows], float)
        npimp = (1 / noy) / (1 / noy + 1 / non_)
        nwin = (np.array([r["sa"] for r in new_rows]) > 0) & (np.array([r["sb"] for r in new_rows]) > 0)
        m = nzone
        if m.sum() >= 30:
            real = float(nwin[m].mean()); imp = float(npimp[m].mean())
            se = math.sqrt(max(real * (1 - real), 1e-9) / m.sum())
            roi = float((noy[m] * nwin[m] - 1).mean()) * 100
            out["oos_zone"] = {"n": int(m.sum()), "real": round(real, 4),
                               "implied": round(imp, 4), "z": round((real - imp) / se, 2),
                               "roi_yes_pct": round(roi, 2)}
            print("(2) OOS pur (events finis apres extraction): n_total=%d, zone n=%d real=%.4f imp=%.4f z=%+.2f ROIyes=%+.2f%%"
                  % (len(new_rows), m.sum(), real, imp, (real - imp) / se, roi))
        else:
            print("(2) OOS pur: n_total=%d, zone n=%d -> trop petit" % (len(new_rows), int(m.sum())))
            out["oos_zone"] = {"n": int(m.sum())}

    # ---------- (3) controle d'offset de ligue ----------
    print("(3) deviation in-zone vs out-zone PAR LIGUE (l'effet zone net d'offset):")
    per = []
    for lg in LEAGUES:
        ml = comp == lg
        for nm, mm in (("in", ml & zone), ("out", ml & ~zone)):
            pass
        mi, mo = ml & zone, ml & ~zone
        if mi.sum() < 50 or mo.sum() < 50:
            continue
        di = float(win[mi].mean() - p_imp[mi].mean())
        do = float(win[mo].mean() - p_imp[mo].mean())
        sei = math.sqrt(win[mi].mean() * (1 - win[mi].mean()) / mi.sum())
        seo = math.sqrt(win[mo].mean() * (1 - win[mo].mean()) / mo.sum())
        diff = di - do
        sed = math.sqrt(sei ** 2 + seo ** 2)
        per.append({"league": lg, "n_in": int(mi.sum()), "dev_in": round(di, 4),
                    "n_out": int(mo.sum()), "dev_out": round(do, 4),
                    "zone_excess": round(diff, 4), "z_excess": round(diff / sed, 2)})
        print("   %-22s in n=%4d dev=%+.4f | out n=%5d dev=%+.4f | excess=%+.4f (z=%+.2f)" % (
            lg, mi.sum(), di, mo.sum(), do, diff, diff / sed))
    out["per_league_zone_excess"] = per
    w = np.array([1 / ((p["zone_excess"] / p["z_excess"]) ** 2) if p["z_excess"] != 0 else 0 for p in per])
    x = np.array([p["zone_excess"] for p in per])
    ivw = float((w * x).sum() / w.sum()); ivse = float(1 / math.sqrt(w.sum()))
    out["zone_excess_ivw"] = {"excess": round(ivw, 4), "se": round(ivse, 4),
                              "z": round(ivw / ivse, 2),
                              "n_pos": int((x > 0).sum()), "n_leagues": len(x)}
    print("   IVW zone-excess (net des offsets de ligue): %+.4f +- %.4f (z=%+.2f) ; signes + : %d/%d"
          % (ivw, ivse, ivw / ivse, (x > 0).sum(), len(x)))

    # ---------- (4) Monte-Carlo post-selection ----------
    # candidats du type de selection reellement opere: 6 buckets isoles + 5 merges adjacents de 2
    bmasks = []
    for i in range(6):
        bmasks.append((gap >= GAP_EDGES[i]) & (gap < GAP_EDGES[i + 1]))
    cands = [bmasks[i] for i in range(6)] + [bmasks[i] | bmasks[i + 1] for i in range(5)]
    obs_z = []
    for cm in cands:
        r = float(win[cm].mean()); ip = float(p_imp[cm].mean())
        se = math.sqrt(r * (1 - r) / cm.sum())
        obs_z.append((r - ip) / se)
    obs_max = max(obs_z)
    NSIM = 4000
    cnt = 0
    u = rng.random((NSIM, n))
    for s_ in range(NSIM):
        wsim = u[s_] < p_imp
        zmax = -99
        for cm in cands:
            r = wsim[cm].mean()
            se = math.sqrt(max(r * (1 - r), 1e-9) / cm.sum())
            z = (r - p_imp[cm].mean()) / se
            zmax = max(zmax, z)
        if zmax >= 2.74:
            cnt += 1
    out["mc_post_selection"] = {
        "obs_z_by_candidate": [round(z, 2) for z in obs_z],
        "obs_max_z": round(obs_max, 2), "nsim": NSIM,
        "p_corrected_max_merge2": round(cnt / NSIM, 4),
        "note": "ne corrige QUE la selection parmi 11 candidats gap; le scan reel = 470 tests"}
    print("(4) MC post-selection (11 candidats bucket/merge2, H0 win~Bern(p_imp)):")
    print("    z observes: %s ; max=%.2f" % (["%.2f" % z for z in obs_z], obs_max))
    print("    P(max z >= 2.74 | H0) = %.4f  (p brute rapportee: 0.0061)" % (cnt / NSIM))

    # ---------- (5) bootstrap ROI + sensibilite marge ----------
    idx = np.where(zone)[0]
    boots = []
    for _ in range(4000):
        bs = rng.choice(idx, size=len(idx), replace=True)
        boots.append(ret_y[bs].mean() * 100)
    boots = np.sort(np.array(boots))
    out["roi_yes_zone_bootstrap"] = {
        "mean": round(float(ret_y[zone].mean()) * 100, 2),
        "ci2.5": round(float(boots[int(0.025 * len(boots))]), 2),
        "ci97.5": round(float(boots[int(0.975 * len(boots))]), 2),
        "p_roi_pos": round(float((boots > 0).mean()), 4)}
    print("(5) ROI Oui zone pooled bootstrap: %.2f%% [%.2f%%, %.2f%%]  P(ROI>0)=%.3f" % (
        out["roi_yes_zone_bootstrap"]["mean"], out["roi_yes_zone_bootstrap"]["ci2.5"],
        out["roi_yes_zone_bootstrap"]["ci97.5"], out["roi_yes_zone_bootstrap"]["p_roi_pos"]))
    # marge additive au lieu de proportionnelle
    p_imp_add = 1 / o_yes - (s2 - 1) / 2
    dev_add = float(win[zone].mean() - p_imp_add[zone].mean())
    dev_pro = float(win[zone].mean() - p_imp[zone].mean())
    # et hors zone, pour voir si le modele additif resterait calibre ailleurs
    dev_add_out = float(win[~zone].mean() - p_imp_add[~zone].mean())
    dev_pro_out = float(win[~zone].mean() - p_imp[~zone].mean())
    out["margin_model_sensitivity"] = {
        "dev_zone_proportional": round(dev_pro, 4), "dev_zone_additive": round(dev_add, 4),
        "dev_out_proportional": round(dev_pro_out, 4), "dev_out_additive": round(dev_add_out, 4)}
    print("    sensibilite marge: dev zone prop=%+.4f / additif=%+.4f ; hors zone prop=%+.4f / additif=%+.4f"
          % (dev_pro, dev_add, dev_pro_out, dev_add_out))

    # ---------- (6) splits temporels alternatifs ----------
    is35 = comp == "InstantLeague-8035"
    s35 = np.sort(start[is35]); cut50 = s35[int(0.50 * len(s35))]
    print("(6) splits alternatifs:")
    for nm, mm in (("8035_h1_50", is35 & (start < cut50)), ("8035_h2_50", is35 & (start >= cut50))):
        m = mm & zone
        r = float(win[m].mean()); ip = float(p_imp[m].mean())
        se = math.sqrt(r * (1 - r) / m.sum())
        print("    %-10s n=%5d real=%.4f imp=%.4f z=%+.2f ROIyes=%+.2f%%" % (
            nm, m.sum(), r, ip, (r - ip) / se, float(ret_y[m].mean()) * 100))
        out["split_" + nm] = {"n": int(m.sum()), "real": round(r, 4), "imp": round(ip, 4),
                              "z": round((r - ip) / se, 2),
                              "roi_yes_pct": round(float(ret_y[m].mean()) * 100, 2)}
    # tranches temporelles pooled (6 quantiles de start, toutes ligues)
    qs = np.quantile(np.sort(start), np.linspace(0, 1, 7), method="nearest") if False else None
    order = np.argsort(start)
    slices = np.array_split(order, 6)
    devs = []
    for k, sl in enumerate(slices):
        msl = np.zeros(n, bool); msl[sl] = True
        m = msl & zone
        if m.sum() < 100:
            continue
        r = float(win[m].mean()); ip = float(p_imp[m].mean())
        se = math.sqrt(r * (1 - r) / m.sum())
        devs.append({"slice": k, "n": int(m.sum()), "dev": round(r - ip, 4),
                     "z": round((r - ip) / se, 2),
                     "roi_yes_pct": round(float(ret_y[m].mean()) * 100, 2)})
        print("    tranche %d: n=%4d dev=%+.4f z=%+.2f ROIyes=%+.2f%%" % (
            k, m.sum(), r - ip, (r - ip) / se, float(ret_y[m].mean()) * 100))
    out["time_slices"] = devs

    with open("exports/wf4_btts_advrefute.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
