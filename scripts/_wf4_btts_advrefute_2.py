# WF4 - BTTS - ADVERSARIAL REFUTE etape 2: durcissement
#  (a) impact des 180 duplicats exacts: stats zone apres drop du second id
#  (b) OOS re-pull (quelques minutes plus tard): residu d'inversion, breakdown par ligue,
#      et estimation combinee (echantillon original + OOS)
# Sortie: exports/wf4_btts_advrefute2.json   (LECTURE SEULE sur la DB)
import sys, json, math
sys.path.insert(0, ".")
import numpy as np

GMAX = 16
LEAGUES = [
    "InstantLeague-8035", "InstantLeague-8036", "InstantLeague-8037",
    "InstantLeague-8042", "InstantLeague-8043", "InstantLeague-8044",
    "InstantLeague-8056", "InstantLeague-8060", "InstantLeague-8065",
]


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
    f1, f2 = probs(lh, la)
    return lh, la, float(np.abs(f1 - ph).max()), float(np.abs(f2 - pd).max())


def zstat(win_m, imp_m):
    r = float(win_m.mean()); ip = float(imp_m.mean())
    se = math.sqrt(max(r * (1 - r), 1e-9) / len(win_m))
    return r, ip, (r - ip) / se


def main():
    out = {}
    with open("exports/wf4_btts_data.json", encoding="utf-8") as f:
        rows = json.load(f)["rows"]
    lz = np.load("exports/wf4_btts_lambdas.npz")
    lh, la = lz["lh"], lz["la"]
    ids_orig = set(r["id"] for r in rows)
    n = len(rows)
    o_yes = np.array([r["o_yes"] for r in rows], float)
    o_no = np.array([r["o_no"] for r in rows], float)
    comp = np.array([r["comp"] for r in rows])
    sa = np.array([r["sa"] for r in rows]); sb = np.array([r["sb"] for r in rows])
    win = (sa > 0) & (sb > 0)
    gap = np.abs(lh - la)
    p_imp = (1 / o_yes) / (1 / o_yes + 1 / o_no)
    zone = (gap >= 0.6) & (gap < 1.3)
    ret_y = o_yes * win - 1
    ids_arr = np.array([r["id"] for r in rows])

    from scraper.config import load_settings
    from sqlalchemy import create_engine, text
    e = create_engine(load_settings().db_url)
    in_list = ",".join("'%s'" % l for l in LEAGUES)
    with e.connect() as c:
        tr = c.execute(text(
            "SELECT id, competition, team_a, team_b, expected_start FROM events "
            "WHERE competition IN (%s)" % in_list)).fetchall()
    meta = {r[0]: (r[1], r[2], r[3], str(r[4])) for r in tr}

    # (a) drop des duplicats (garde le premier id de chaque fixture)
    seen = {}
    keep = np.ones(n, bool)
    dup_same_score = 0; dup_diff_score = 0
    by_id = {r["id"]: r for r in rows}
    for i, r in enumerate(rows):
        k = meta.get(r["id"])
        if k is None:
            continue
        if k in seen:
            keep[i] = False
            r0 = by_id[seen[k]]
            if (r0["sa"], r0["sb"]) == (r["sa"], r["sb"]):
                dup_same_score += 1
            else:
                dup_diff_score += 1
        else:
            seen[k] = r["id"]
    m = zone & keep
    r_, ip_, z_ = zstat(win[m], p_imp[m])
    out["dedup"] = {"n_dropped": int((~keep).sum()), "dup_same_score": dup_same_score,
                    "dup_diff_score": dup_diff_score,
                    "zone_n": int(m.sum()), "real": round(r_, 4), "imp": round(ip_, 4),
                    "z": round(z_, 2), "roi_yes_pct": round(float(ret_y[m].mean()) * 100, 2)}
    print("(a) dedup: drop=%d (same_score=%d diff=%d) -> zone n=%d real=%.4f imp=%.4f z=%+.2f ROIyes=%+.2f%%"
          % ((~keep).sum(), dup_same_score, dup_diff_score, m.sum(), r_, ip_, z_,
             float(ret_y[m].mean()) * 100))

    # (b) OOS re-pull
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
    seen_fix = dict(seen)  # fixtures deja vues dans l'echantillon original
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
        k = meta.get(eid)
        if k is not None and k in seen_fix:
            continue  # dup d'une fixture deja comptee
        if k is not None:
            seen_fix[k] = eid
        new_rows.append({"id": eid, "comp": cp, "oh": oh, "od": od, "oa": oa,
                         "o_yes": float(gng["Oui"]), "o_no": float(gng["Non"]),
                         "sa": int(a), "sb": int(b)})
    print("(b) OOS re-pull: n_total=%d (dedup inclus)" % len(new_rows))
    noh = np.array([r["oh"] for r in new_rows], float)
    nod = np.array([r["od"] for r in new_rows], float)
    noa = np.array([r["oa"] for r in new_rows], float)
    ns = 1 / noh + 1 / nod + 1 / noa
    nlh, nla, res_h, res_d = invert_lambdas((1 / noh) / ns, (1 / nod) / ns)
    print("    residu inversion max: %.2e / %.2e" % (res_h, res_d))
    ngap = np.abs(nlh - nla)
    nzone = (ngap >= 0.6) & (ngap < 1.3)
    noy = np.array([r["o_yes"] for r in new_rows], float)
    non_ = np.array([r["o_no"] for r in new_rows], float)
    npimp = (1 / noy) / (1 / noy + 1 / non_)
    nwin = (np.array([r["sa"] for r in new_rows]) > 0) & (np.array([r["sb"] for r in new_rows]) > 0)
    ncomp = np.array([r["comp"] for r in new_rows])
    mz = nzone
    r_, ip_, z_ = zstat(nwin[mz], npimp[mz])
    nroi = float((noy[mz] * nwin[mz] - 1).mean()) * 100
    out["oos"] = {"n_total": len(new_rows), "zone_n": int(mz.sum()), "real": round(r_, 4),
                  "imp": round(ip_, 4), "z": round(z_, 2), "roi_yes_pct": round(nroi, 2),
                  "max_resid": max(res_h, res_d)}
    print("    OOS zone: n=%d real=%.4f imp=%.4f z=%+.2f ROIyes=%+.2f%%" % (
        mz.sum(), r_, ip_, z_, nroi))
    print("    breakdown par ligue (OOS zone):")
    out["oos_by_league"] = {}
    for lg in LEAGUES:
        m2 = mz & (ncomp == lg)
        if m2.sum() < 20:
            continue
        r2, ip2, z2 = zstat(nwin[m2], npimp[m2])
        out["oos_by_league"][lg] = {"n": int(m2.sum()), "real": round(r2, 4),
                                    "imp": round(ip2, 4), "z": round(z2, 2)}
        print("      %-22s n=%4d real=%.4f imp=%.4f z=%+.2f" % (lg, m2.sum(), r2, ip2, z2))

    # estimation combinee (orig dedup + OOS dedup)
    all_win = np.concatenate([win[zone & keep], nwin[mz]])
    all_imp = np.concatenate([p_imp[zone & keep], npimp[mz]])
    all_ret = np.concatenate([ret_y[zone & keep], (noy * nwin - 1)[mz]])
    r_, ip_, z_ = zstat(all_win, all_imp)
    out["combined"] = {"n": len(all_win), "real": round(r_, 4), "imp": round(ip_, 4),
                       "z": round(z_, 2), "dev_pts": round((r_ - ip_) * 100, 2),
                       "roi_yes_pct": round(float(all_ret.mean()) * 100, 2)}
    print("    COMBINE (orig dedup + OOS): n=%d real=%.4f imp=%.4f z=%+.2f ROIyes=%+.2f%%" % (
        len(all_win), r_, ip_, z_, float(all_ret.mean()) * 100))

    with open("exports/wf4_btts_advrefute2.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
