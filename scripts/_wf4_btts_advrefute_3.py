# WF4 - BTTS - ADVERSARIAL REFUTE etape 3: l'OOS negatif est-il specifique a la zone
# ou un effet global de la fenetre (~1h)?  dev OOS in-zone vs out-zone + zone-excess OOS.
# Sortie: stdout uniquement (complement de advrefute2). LECTURE SEULE.
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
    return lh, la


def main():
    with open("exports/wf4_btts_data.json", encoding="utf-8") as f:
        rows = json.load(f)["rows"]
    ids_orig = set(r["id"] for r in rows)
    corrupted = set()
    with open("exports/corrupted_events.json", encoding="utf-8") as f:
        corrupted = set(int(k) for k in json.load(f)["events"].keys())

    from scraper.config import load_settings
    from sqlalchemy import create_engine, text
    e = create_engine(load_settings().db_url)
    in_list = ",".join("'%s'" % l for l in LEAGUES)
    with e.connect() as c:
        meta = {r[0]: (r[1], r[2], r[3], str(r[4])) for r in c.execute(text(
            "SELECT id, competition, team_a, team_b, expected_start FROM events "
            "WHERE competition IN (%s)" % in_list)).fetchall()}
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
    seen_fix = {}
    for r in rows:
        k = meta.get(r["id"])
        if k is not None and k not in seen_fix:
            seen_fix[k] = r["id"]
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
        k = meta.get(eid)
        if k is not None and k in seen_fix:
            continue
        if k is not None:
            seen_fix[k] = eid
        new_rows.append({"comp": cp, "oh": oh, "od": od, "oa": oa, "start": str(est),
                         "o_yes": float(gng["Oui"]), "o_no": float(gng["Non"]),
                         "sa": int(a), "sb": int(b)})
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
    starts = sorted(r["start"] for r in new_rows)
    print("OOS n=%d ; start range: %s -> %s" % (len(new_rows), starts[0], starts[-1]))
    for nm, m in (("in-zone ", nzone), ("out-zone", ~nzone)):
        r = float(nwin[m].mean()); ip = float(npimp[m].mean())
        se = math.sqrt(max(r * (1 - r), 1e-9) / m.sum())
        print("  %s n=%4d real=%.4f imp=%.4f dev=%+.4f z=%+.2f" % (
            nm, m.sum(), r, ip, r - ip, (r - ip) / se))
    di = float(nwin[nzone].mean() - npimp[nzone].mean())
    do = float(nwin[~nzone].mean() - npimp[~nzone].mean())
    sei = math.sqrt(nwin[nzone].mean() * (1 - nwin[nzone].mean()) / nzone.sum())
    seo = math.sqrt(nwin[~nzone].mean() * (1 - nwin[~nzone].mean()) / (~nzone).sum())
    print("  zone-excess OOS = %+.4f (z=%+.2f)" % (di - do, (di - do) / math.sqrt(sei**2 + seo**2)))


if __name__ == "__main__":
    main()
