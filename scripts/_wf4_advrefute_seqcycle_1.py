# -*- coding: utf-8 -*-
"""
WF4 ADVERSARIAL - refutation du finding "cycle saisonnier du favori" (_wf4_seq_2/3/4).
Angles d'attaque:
 A. Structure reelle des rounds: completion resultats par round/ligue (censure au bord de saison).
    Constat SQL prealable: round 38 (resp. 34) n'existe pas avec resultats; rounds 36-37 a ~60%.
 B. res_fav + ROI backfav PAR ROUND individuel autour des bords (1-5, 30-37 sur 8035).
 C. Fenetre "late" corrigee: (i) late_true = 5 derniers rounds OBSERVES (33-37 / 29-33),
    (ii) late_complete = 5 derniers rounds a couverture complete (31-35 / 27-31 etc, mesure par
    completion >= 85%). Si l'effet disparait hors de la zone censuree -> artefact de bord.
 D. Test pertinent pour un FILTRE: late vs RESTE (pas late vs early) - t-test + bootstrap du delta ROI.
 E. Scrambling test: pente logistique favwin ~ logit(pfav) par bucket (early/mid/late) -
    une desattribution de resultats aplatit la pente; un vrai effet moteur deplace l'intercept.
 F. Cluster bootstrap par instance de saison (detection seq_4) du contraste early-late (pooled).
 G. Split temporel calendrier (2 moities par expected_start) du contraste sur 8035.
 H. Sanity look-ahead: part des snapshots d'ouverture posterieurs au coup d'envoi, par bucket.
Sortie: exports/wf4_advrefute_seqcycle.json. LECTURE SEULE sur la DB.
"""
import sys, json, math, importlib.util, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import numpy as np
from scipy import stats
from sqlalchemy import create_engine, text
from scraper.config import load_settings

spec = importlib.util.spec_from_file_location("wf4seq1", "scripts/_wf4_seq_1.py")
m1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m1)

MAXR_CLAIM = {"InstantLeague-8035": 38, "InstantLeague-8036": 38, "InstantLeague-8037": 38,
              "InstantLeague-8042": 34, "InstantLeague-8043": 34, "InstantLeague-8044": 34}
RNG = np.random.default_rng(123)

def roi_stats(bets):
    if not bets:
        return dict(n=0)
    pnl = np.array([(o - 1) if w else -1.0 for o, w in bets])
    n = len(pnl)
    boot = RNG.choice(pnl, size=(6000, n), replace=True).mean(axis=1)
    return dict(n=n, roi_pct=float(100 * pnl.mean()),
                se_roi_pct=float(100 * pnl.std(ddof=1) / math.sqrt(n)),
                p_boot_le_0=float((boot <= 0).mean()),
                wr=float(np.mean([w for _, w in bets])),
                avg_odds=float(np.mean([o for o, _ in bets])))

def diff_roi_boot(betsA, betsB, nb=6000):
    """bootstrap du delta ROI(A)-ROI(B); p = P(delta >= 0) si on attend A<B"""
    pa = np.array([(o - 1) if w else -1.0 for o, w in betsA])
    pb = np.array([(o - 1) if w else -1.0 for o, w in betsB])
    da = RNG.choice(pa, size=(nb, len(pa)), replace=True).mean(axis=1)
    db = RNG.choice(pb, size=(nb, len(pb)), replace=True).mean(axis=1)
    d = da - db
    return dict(delta_roi_pct=float(100 * (pa.mean() - pb.mean())),
                p_boot_ge_0=float((d >= 0).mean()))

def slope_fit(sub):
    """logistic favwin ~ logit(pfav): retourne (intercept, slope)"""
    from sklearn.linear_model import LogisticRegression
    X = np.array([[math.log(r["pfav"] / (1 - r["pfav"]))] for r in sub])
    y = np.array([r["favwin"] for r in sub])
    if y.std() == 0:
        return None
    m = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000).fit(X, y)
    return dict(n=len(sub), intercept=float(m.intercept_[0]), slope=float(m.coef_[0][0]))

def main():
    out = {}

    # ---------- A. completion par round (tous events vs finis) ----------
    eng = create_engine(load_settings().db_url)
    comp_table = {}
    with eng.connect() as c:
        allev = c.execute(text("""
            SELECT e.competition, CAST(e.round_info AS INTEGER) rnd, COUNT(*),
                   SUM(CASE WHEN r.event_id IS NOT NULL THEN 1 ELSE 0 END)
            FROM events e LEFT JOIN results r ON r.event_id = e.id
            GROUP BY e.competition, rnd""")).fetchall()
        # H. snapshots d'ouverture post-kickoff
        snap = c.execute(text("""
            SELECT e.competition, CAST(e.round_info AS INTEGER) rnd,
                   SUM(CASE WHEN o.captured_at > e.expected_start THEN 1 ELSE 0 END), COUNT(*)
            FROM events e
            JOIN odds_snapshots o ON o.id = (SELECT MIN(o2.id) FROM odds_snapshots o2 WHERE o2.event_id = e.id)
            JOIN results r ON r.event_id = e.id
            GROUP BY e.competition, rnd""")).fetchall()
    for comp, rnd, n_all, n_fin in allev:
        if comp in MAXR_CLAIM:
            comp_table.setdefault(comp, {})[rnd] = dict(events=n_all, finished=n_fin,
                                                        pct=round(100 * n_fin / max(n_all, 1), 1))
    out["completion_by_round"] = comp_table
    late_open = {}
    for comp, rnd, n_post, n_tot in snap:
        if comp in MAXR_CLAIM and rnd is not None and rnd >= 1:
            key = "late_claimwin" if rnd >= MAXR_CLAIM[comp] - 4 else "rest"
            d = late_open.setdefault(comp, {}).setdefault(key, [0, 0])
            d[0] += n_post; d[1] += n_tot
    out["opening_snapshot_post_kickoff"] = {
        c: {k: dict(post=v[0], tot=v[1], pct=round(100 * v[0] / max(v[1], 1), 2)) for k, v in d.items()}
        for c, d in late_open.items()}

    # ---------- charge le meme dataset que l'auteur ----------
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
        r["ofav"] = r["oh"] if r["fav_home"] else r["oa"]
        r["res"] = r["favwin"] - r["pfav"]

    # bornes empiriques (max round OBSERVE avec resultats dans ce dataset)
    obs_max = {}
    for r in champs:
        obs_max[r["comp"]] = max(obs_max.get(r["comp"], 0), r["rnd"])
    out["observed_max_round"] = obs_max

    # ---------- B. res_fav / ROI par round individuel ----------
    per_round = {}
    for scope, flt in (("8035", lambda r: r["comp"] == "InstantLeague-8035"),
                       ("newchamps", lambda r: r["comp"] != "InstantLeague-8035")):
        sub = [r for r in champs if flt(r)]
        tbl = {}
        # position relative depuis la fin OBSERVEE: 0 = dernier round observe
        for r in sub:
            r["from_end"] = obs_max[r["comp"]] - r["rnd"]
        for fe in range(0, 8):
            seg = [r for r in sub if r["from_end"] == fe]
            if len(seg) < 30:
                continue
            res = np.array([r["res"] for r in seg])
            tbl[f"end-{fe}"] = dict(n=len(seg), res_fav=float(res.mean()),
                                    se=float(res.std(ddof=1) / math.sqrt(len(seg))),
                                    roi=roi_stats([(r["ofav"], r["favwin"] == 1.0) for r in seg])["roi_pct"])
        for rd in range(1, 6):
            seg = [r for r in sub if r["rnd"] == rd]
            if len(seg) < 30:
                continue
            res = np.array([r["res"] for r in seg])
            tbl[f"J{rd}"] = dict(n=len(seg), res_fav=float(res.mean()),
                                 se=float(res.std(ddof=1) / math.sqrt(len(seg))),
                                 roi=roi_stats([(r["ofav"], r["favwin"] == 1.0) for r in seg])["roi_pct"])
        per_round[scope] = tbl
    out["per_round_edges"] = per_round

    # ---------- C/D. fenetres late alternatives + late vs RESTE ----------
    def completion_of(comp, rnd):
        d = comp_table.get(comp, {}).get(rnd)
        return d["pct"] if d else 0.0

    windows = {}
    for r in champs:
        mx_claim = MAXR_CLAIM[r["comp"]]
        mx_obs = obs_max[r["comp"]]
        r["late_claim"] = r["rnd"] >= mx_claim - 4            # fenetre de l'auteur (34-38 / 30-34)
        r["late_true5"] = r["rnd"] >= mx_obs - 4              # 5 derniers rounds OBSERVES
        # rounds "complets" (>=85% de resultats): late_complete = 5 derniers complets
        complete_rounds = [k for k in range(1, mx_obs + 1) if completion_of(r["comp"], k) >= 85.0]
        last5_complete = set(complete_rounds[-5:]) if len(complete_rounds) >= 5 else set()
        r["late_complete"] = r["rnd"] in last5_complete
        r["censored_zone"] = r["rnd"] <= mx_obs and completion_of(r["comp"], r["rnd"]) < 85.0
        r["early"] = r["rnd"] <= 5

    for scope, flt in (("pooled-6champs", lambda r: True),
                       ("8035", lambda r: r["comp"] == "InstantLeague-8035"),
                       ("newchamps", lambda r: r["comp"] != "InstantLeague-8035")):
        sub = [r for r in champs if flt(r)]
        sc = {}
        for wname in ("late_claim", "late_true5", "late_complete", "censored_zone"):
            late = [r for r in sub if r[wname]]
            rest = [r for r in sub if not r[wname] and not r["early"]]
            if len(late) < 50:
                sc[wname] = dict(n=len(late))
                continue
            res_l = np.array([r["res"] for r in late])
            res_r = np.array([r["res"] for r in rest])
            t, p = stats.ttest_ind(res_l, res_r, equal_var=False)
            bets_l = [(r["ofav"], r["favwin"] == 1.0) for r in late]
            bets_r = [(r["ofav"], r["favwin"] == 1.0) for r in rest]
            sc[wname] = dict(
                roi_late=roi_stats(bets_l), roi_rest_mid=roi_stats(bets_r),
                res_fav_late=float(res_l.mean()), res_fav_restmid=float(res_r.mean()),
                t_late_vs_restmid=float(t), p_late_vs_restmid=float(p),
                boot_delta=diff_roi_boot(bets_l, bets_r))
        windows[scope] = sc
    out["late_windows"] = windows

    # ---------- E. scrambling test: pente logistique par bucket ----------
    slopes = {}
    for scope, flt in (("8035", lambda r: r["comp"] == "InstantLeague-8035"),
                       ("newchamps", lambda r: r["comp"] != "InstantLeague-8035")):
        sub = [r for r in champs if flt(r)]
        slopes[scope] = {
            "early_J1-5": slope_fit([r for r in sub if r["early"]]),
            "mid": slope_fit([r for r in sub if not r["early"] and not r["late_claim"]]),
            "late_claim": slope_fit([r for r in sub if r["late_claim"]]),
            "late_censored_only": slope_fit([r for r in sub if r["censored_zone"]]),
            "late_complete": slope_fit([r for r in sub if r["late_complete"]]),
        }
    out["calibration_slopes"] = slopes

    # ---------- F. cluster bootstrap par instance de saison (contraste early-late_claim) ----------
    # detection sequentielle comme seq_4
    clusters = {}
    for lg in MAXR_CLAIM:
        sub = sorted([r for r in champs if r["comp"] == lg], key=lambda r: (r["start"], r["id"]))
        season, prev = 0, None
        for r in sub:
            if prev is not None and r["rnd"] < prev - 10:
                season += 1
            r["season"] = season
            prev = max(prev, r["rnd"]) if prev is not None and r["rnd"] >= prev - 10 else r["rnd"]
        for r in sub:
            clusters.setdefault((lg, r["season"]), []).append(r)
    cl_list = list(clusters.values())
    contrasts = []
    for lst in cl_list:
        e = [r["res"] for r in lst if r["early"]]
        l = [r["res"] for r in lst if r["late_claim"]]
        if len(e) >= 5 and len(l) >= 5:
            contrasts.append((np.mean(e) - np.mean(l), len(e), len(l)))
    cvals = np.array([c[0] for c in contrasts])
    w = np.array([1.0 / (1.0 / c[1] + 1.0 / c[2]) for c in contrasts])  # poids ~ n harmonique
    wmean = float(np.sum(w * cvals) / np.sum(w))
    boot_idx = RNG.integers(0, len(cvals), size=(8000, len(cvals)))
    bm = (w[boot_idx] * cvals[boot_idx]).sum(axis=1) / w[boot_idx].sum(axis=1)
    out["cluster_meta"] = dict(
        n_instances=len(cvals), weighted_mean_contrast=wmean,
        p_clusterboot_le_0=float((bm <= 0).mean()),
        n_positive=int((cvals > 0).sum()),
        p_sign=float(stats.binomtest(int((cvals > 0).sum()), len(cvals), 0.5).pvalue))

    # ---------- G. split calendrier en 2 moities (8035) ----------
    sub8 = sorted([r for r in champs if r["comp"] == "InstantLeague-8035"], key=lambda r: (r["start"], r["id"]))
    halves = {}
    half = len(sub8) // 2
    for tag, seg in (("H1", sub8[:half]), ("H2", sub8[half:])):
        e = np.array([r["res"] for r in seg if r["early"]])
        l = np.array([r["res"] for r in seg if r["late_claim"]])
        t, p = stats.ttest_ind(e, l, equal_var=False)
        halves[tag] = dict(n_e=len(e), n_l=len(l), res_e=float(e.mean()), res_l=float(l.mean()),
                           contrast=float(e.mean() - l.mean()), p=float(p),
                           roi_late=roi_stats([(r["ofav"], r["favwin"] == 1.0) for r in seg if r["late_claim"]]))
    out["calendar_halves_8035"] = halves

    # ---------- replication directe des chiffres du claim ----------
    sub = champs
    bets_late = [(r["ofav"], r["favwin"] == 1.0) for r in sub if r["late_claim"]]
    bets_early = [(r["ofav"], r["favwin"] == 1.0) for r in sub if r["early"]]
    e = np.array([r["res"] for r in sub if r["early"]]); l = np.array([r["res"] for r in sub if r["late_claim"]])
    t, p = stats.ttest_ind(e, l, equal_var=False)
    out["claim_replication"] = dict(
        backfav_last5=roi_stats(bets_late), backfav_J1_5=roi_stats(bets_early),
        res_fav_early=float(e.mean()), res_fav_late=float(l.mean()), p_ttest=float(p))

    with open("exports/wf4_advrefute_seqcycle.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    # console
    print("\n=== observed max rounds ===", obs_max)
    print("\n=== claim replication (live DB) ===")
    print(json.dumps(out["claim_replication"], indent=1))
    print("\n=== per-round edges ===")
    print(json.dumps(per_round, indent=1))
    print("\n=== late windows (late vs rest-mid) ===")
    for sc, d in windows.items():
        for w_, v in d.items():
            if "roi_late" in v:
                print(f"{sc:<16} {w_:<16} n={v['roi_late']['n']:>4} roi_late={v['roi_late']['roi_pct']:+6.2f} "
                      f"roi_mid={v['roi_rest_mid']['roi_pct']:+6.2f} res_late={v['res_fav_late']:+.4f} "
                      f"res_mid={v['res_fav_restmid']:+.4f} p={v['p_late_vs_restmid']:.4g} "
                      f"p_boot_delta={v['boot_delta']['p_boot_ge_0']:.4g}")
    print("\n=== calibration slopes ===")
    print(json.dumps(slopes, indent=1))
    print("\n=== cluster meta ===", json.dumps(out["cluster_meta"]))
    print("\n=== calendar halves 8035 ===", json.dumps(halves, default=str))
    print("\n=== opening snapshot post kickoff ===")
    print(json.dumps(out["opening_snapshot_post_kickoff"], indent=1))

if __name__ == "__main__":
    main()
