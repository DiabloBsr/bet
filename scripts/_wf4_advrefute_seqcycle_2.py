# -*- coding: utf-8 -*-
"""
WF4 ADVERSARIAL (2) - finding "cycle saisonnier du favori": suite des contre-tests.
 1. distribution (captured_at - expected_start) du snapshot d'ouverture par ligue (quantiles)
    -> verifier si "post-kickoff" est reel ou un offset de fuseau.
 2. newchamps: deficit late selon capture pre/post kickoff de l'ouverture.
 3. newchamps: residu late PAR instance de saison (la 1ere saison partielle = warm-up scraper ?).
 4. 8035: contraste early-late par QUART calendaire (periode chanceuse ?).
 5. meta par clusters separee 8035 vs newchamps.
 6. ou va le deficit du favori en late (taux draw / dog) pooled.
Sortie: exports/wf4_advrefute_seqcycle2.json. LECTURE SEULE.
"""
import sys, json, math, importlib.util, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
from datetime import datetime
import numpy as np
from scipy import stats
from sqlalchemy import create_engine, text
from scraper.config import load_settings

spec = importlib.util.spec_from_file_location("wf4seq1", "scripts/_wf4_seq_1.py")
m1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m1)

MAXR_CLAIM = {"InstantLeague-8035": 38, "InstantLeague-8036": 38, "InstantLeague-8037": 38,
              "InstantLeague-8042": 34, "InstantLeague-8043": 34, "InstantLeague-8044": 34}
RNG = np.random.default_rng(321)

def main():
    out = {}
    eng = create_engine(load_settings().db_url)
    with eng.connect() as c:
        snap = c.execute(text("""
            SELECT e.id, e.competition, e.expected_start, o.captured_at
            FROM events e
            JOIN odds_snapshots o ON o.id = (SELECT MIN(o2.id) FROM odds_snapshots o2 WHERE o2.event_id = e.id)
            JOIN results r ON r.event_id = e.id""")).fetchall()
    delta_by_lg = {}
    delta_by_id = {}
    for eid, comp, st, cap in snap:
        if comp not in MAXR_CLAIM:
            continue
        try:
            d = (datetime.fromisoformat(str(cap)) - datetime.fromisoformat(str(st))).total_seconds() / 60.0
        except Exception:
            continue
        delta_by_lg.setdefault(comp, []).append(d)
        delta_by_id[eid] = d
    out["open_capture_minus_start_minutes"] = {
        lg: dict(n=len(v), q05=float(np.quantile(v, .05)), q25=float(np.quantile(v, .25)),
                 q50=float(np.quantile(v, .50)), q75=float(np.quantile(v, .75)),
                 q95=float(np.quantile(v, .95)), pct_pos=float(100 * np.mean(np.array(v) > 0)))
        for lg, v in delta_by_lg.items()}

    rows = m1.load_data()
    oh = np.array([r["oh"] for r in rows]); od = np.array([r["od"] for r in rows]); oa = np.array([r["oa"] for r in rows])
    inv = 1 / oh + 1 / od + 1 / oa
    ph, pd_, pa = (1 / oh) / inv, (1 / od) / inv, (1 / oa) / inv
    for i, r in enumerate(rows):
        r.update(ph=float(ph[i]), pd=float(pd_[i]), pa=float(pa[i]))
    champs = [r for r in rows if r["comp"] in MAXR_CLAIM and 1 <= r["rnd"] <= MAXR_CLAIM[r["comp"]]]
    for r in champs:
        r["pfav"] = max(r["ph"], r["pa"])
        r["fav_home"] = r["ph"] >= r["pa"]
        r["favwin"] = 1.0 if ((r["sa"] > r["sb"]) if r["fav_home"] else (r["sb"] > r["sa"])) else 0.0
        r["draw"] = 1.0 if r["sa"] == r["sb"] else 0.0
        r["ofav"] = r["oh"] if r["fav_home"] else r["oa"]
        r["res"] = r["favwin"] - r["pfav"]
        r["early"] = r["rnd"] <= 5
        r["late"] = r["rnd"] >= MAXR_CLAIM[r["comp"]] - 4
        r["dt_open"] = delta_by_id.get(r["id"])

    new = [r for r in champs if r["comp"] != "InstantLeague-8035"]
    e35 = [r for r in champs if r["comp"] == "InstantLeague-8035"]

    # ---- 2. newchamps late: pre vs post kickoff opening capture ----
    def seg_stats(seg):
        if len(seg) < 20:
            return dict(n=len(seg))
        res = np.array([r["res"] for r in seg])
        pnl = np.array([(r["ofav"] - 1) if r["favwin"] else -1.0 for r in seg])
        return dict(n=len(seg), res_fav=float(res.mean()),
                    se=float(res.std(ddof=1) / math.sqrt(len(seg))),
                    roi_pct=float(100 * pnl.mean()), pfav=float(np.mean([r["pfav"] for r in seg])))
    out["newchamps_late_by_capture"] = {
        "late_pre_kick": seg_stats([r for r in new if r["late"] and r["dt_open"] is not None and r["dt_open"] <= 0]),
        "late_post_kick": seg_stats([r for r in new if r["late"] and r["dt_open"] is not None and r["dt_open"] > 0]),
        "mid_pre_kick": seg_stats([r for r in new if not r["late"] and not r["early"] and r["dt_open"] is not None and r["dt_open"] <= 0]),
        "mid_post_kick": seg_stats([r for r in new if not r["late"] and not r["early"] and r["dt_open"] is not None and r["dt_open"] > 0]),
    }
    out["e35_late_by_capture"] = {
        "late_pre_kick": seg_stats([r for r in e35 if r["late"] and r["dt_open"] is not None and r["dt_open"] <= 0]),
        "late_post_kick": seg_stats([r for r in e35 if r["late"] and r["dt_open"] is not None and r["dt_open"] > 0]),
        "mid_pre_kick": seg_stats([r for r in e35 if not r["late"] and not r["early"] and r["dt_open"] is not None and r["dt_open"] <= 0]),
        "mid_post_kick": seg_stats([r for r in e35 if not r["late"] and not r["early"] and r["dt_open"] is not None and r["dt_open"] > 0]),
    }

    # ---- 3. saisons (scan sequentiel seq_4) ----
    def tag_seasons(sub):
        sub = sorted(sub, key=lambda r: (r["start"], r["id"]))
        season, prev = 0, None
        for r in sub:
            if prev is not None and r["rnd"] < prev - 10:
                season += 1
            r["season"] = season
            prev = max(prev, r["rnd"]) if prev is not None and r["rnd"] >= prev - 10 else r["rnd"]
        return sub

    out["newchamps_late_by_season"] = {}
    for lg in sorted(set(r["comp"] for r in new)):
        sub = tag_seasons([r for r in new if r["comp"] == lg])
        per = {}
        for sid in sorted(set(r["season"] for r in sub)):
            seg = [r for r in sub if r["season"] == sid]
            late = [r for r in seg if r["late"]]
            early = [r for r in seg if r["early"]]
            per[f"s{sid}"] = dict(n=len(seg), rounds=f"{min(r['rnd'] for r in seg)}-{max(r['rnd'] for r in seg)}",
                                  late=seg_stats(late), early=seg_stats(early))
        out["newchamps_late_by_season"][lg] = per

    # ---- 4. 8035 par quart calendaire ----
    e35s = sorted(e35, key=lambda r: (r["start"], r["id"]))
    qs = {}
    k = len(e35s) // 4
    for qi in range(4):
        seg = e35s[qi * k: (qi + 1) * k if qi < 3 else len(e35s)]
        e = np.array([r["res"] for r in seg if r["early"]])
        l = np.array([r["res"] for r in seg if r["late"]])
        if len(e) > 20 and len(l) > 20:
            t, p = stats.ttest_ind(e, l, equal_var=False)
            qs[f"Q{qi+1}"] = dict(n_e=len(e), n_l=len(l), res_e=float(e.mean()), res_l=float(l.mean()),
                                  contrast=float(e.mean() - l.mean()), p=float(p),
                                  span=f"{seg[0]['start'][:16]} -> {seg[-1]['start'][:16]}")
    out["e35_calendar_quarters"] = qs

    # ---- 5. cluster meta separee ----
    def cluster_meta(pop):
        clusters = {}
        for lg in sorted(set(r["comp"] for r in pop)):
            sub = tag_seasons([r for r in pop if r["comp"] == lg])
            for r in sub:
                clusters.setdefault((lg, r["season"]), []).append(r)
        contrasts, weights = [], []
        for lst in clusters.values():
            e = [r["res"] for r in lst if r["early"]]
            l = [r["res"] for r in lst if r["late"]]
            if len(e) >= 5 and len(l) >= 5:
                contrasts.append(np.mean(e) - np.mean(l))
                weights.append(1.0 / (1.0 / len(e) + 1.0 / len(l)))
        cv, w = np.array(contrasts), np.array(weights)
        if len(cv) < 4:
            return dict(n_instances=len(cv))
        bi = RNG.integers(0, len(cv), size=(8000, len(cv)))
        bm = (w[bi] * cv[bi]).sum(axis=1) / w[bi].sum(axis=1)
        return dict(n_instances=len(cv), weighted_mean=float((w * cv).sum() / w.sum()),
                    p_boot_le_0=float((bm <= 0).mean()), n_pos=int((cv > 0).sum()),
                    p_sign=float(stats.binomtest(int((cv > 0).sum()), len(cv), 0.5).pvalue))
    out["cluster_meta_8035"] = cluster_meta(e35)
    out["cluster_meta_newchamps"] = cluster_meta(new)

    # ---- 6. decomposition du deficit late (pooled): favwin/draw/dogwin early-mid-late ----
    dec = {}
    for tag, flt in (("early", lambda r: r["early"]),
                     ("mid", lambda r: not r["early"] and not r["late"]),
                     ("late", lambda r: r["late"])):
        seg = [r for r in champs if flt(r)]
        dec[tag] = dict(n=len(seg),
                        favwin=float(np.mean([r["favwin"] for r in seg])),
                        draw=float(np.mean([r["draw"] for r in seg])),
                        dogwin=float(np.mean([1.0 - r["favwin"] - r["draw"] for r in seg])),
                        pfav=float(np.mean([r["pfav"] for r in seg])),
                        pd=float(np.mean([r["pd"] for r in seg])))
    out["outcome_decomposition_pooled"] = dec

    with open("exports/wf4_advrefute_seqcycle2.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
