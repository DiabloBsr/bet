"""Audit quantitatif V5 — walk-forward 70/30 temporel, metriques OOS only.

Quantifie les gains attendus des fixes proposes dans l'audit de predictor_v5.py :
  A. 1X2 : baseline (blend V3 0.4 Poisson + 0.6 multi-market) vs variantes
     - poids multi-market w_mm in {0.6, 0.7, 0.8, 0.9, 1.0} (recupere algebriquement)
     - blend avec probas 1X2 devig (jamais utilisees dans le pick aujourd'hui)
  B. Hautes cotes : ROI OOS des picks cote >= 2.0 par variante + strategie edge
  C. Over 2.5 : Poisson pur (prod) vs grille blend vs IPF market
  D. HT 1X2 : Poisson independant (prod) vs Dixon-Coles tau applique aux lambdas HT
  E. Score exact top1/top3 : blend V5 vs market-only vs poisson-only
  F. Stats buckets empirical_score_dist (min n=5 -> bruit)
NE MODIFIE AUCUN fichier de scraper/.
"""
from __future__ import annotations
import sys, json
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sqlalchemy import create_engine

from scraper.config import load_settings
from scraper.predictor_v5 import fit_model_v5, predict_match_v5, _cote_bucket
from scraper.predictor_v2 import devig, poisson_score_grid, grid_to_1x2, _dc_tau
from scraper.predictor_v3 import multi_market_score_grid

EPS = 1e-9


def outcome_of(sa, sb):
    return "1" if sa > sb else ("X" if sa == sb else "2")


def roi_line(name, rows):
    """rows: list of (won_bool, cote). Print n, wr, avg_cote, roi."""
    if not rows:
        print(f"  {name:<46} n=0")
        return
    n = len(rows)
    wr = np.mean([w for w, _ in rows])
    avg_c = np.mean([c for _, c in rows])
    roi = np.mean([w * (c - 1) - (1 - w) for w, c in rows])
    flag = "" if n >= 30 else "  [INSTABLE n<30]"
    print(f"  {name:<46} n={n:<5} wr={wr*100:5.1f}%  cote_moy={avg_c:.2f}  ROI={roi*100:+6.1f}%{flag}")


def main():
    settings = load_settings()
    engine = create_engine(settings.db_url)
    df = pd.read_sql("""
        SELECT e.id ev_id, e.team_a, e.team_b, e.expected_start,
               o.odds_home, o.odds_draw, o.odds_away, o.extra_markets,
               r.score_a, r.score_b, r.ht_score_a, r.ht_score_b
        FROM events e
        JOIN odds_snapshots o ON o.id = (SELECT MIN(id) FROM odds_snapshots WHERE event_id = e.id)
        JOIN results r ON r.event_id = e.id
        WHERE r.score_a IS NOT NULL AND o.odds_home IS NOT NULL
              AND o.odds_draw IS NOT NULL AND o.odds_away IS NOT NULL
        ORDER BY e.expected_start
    """, engine)
    df = df.reset_index(drop=True)
    n = len(df)
    cut = int(n * 0.70)
    train, oos = df.iloc[:cut].copy(), df.iloc[cut:].copy()
    print(f"total={n}  train={len(train)}  oos={len(oos)}")

    ht_train = train[train.ht_score_a.notna()].copy()
    model = fit_model_v5(train, ht_history=ht_train, form_alpha=0.0)
    print(f"rho={model.rho:.4f}  mu_h={model.mu_h:.3f}  mu_a={model.mu_a:.3f}  ht_ratio={model.ht_lambda_ratio:.3f}")
    print(f"cal_h={model.cal_h:+.4f} cal_d={model.cal_d:+.4f} cal_a={model.cal_a:+.4f}")

    # F. bucket stats
    dist = model.empirical_score_dist or {}
    # recount totals by refitting counts quickly
    from collections import defaultdict
    counts = defaultdict(int)
    for r in train.itertuples():
        counts[(_cote_bucket(r.odds_home), _cote_bucket(r.odds_away))] += 1
    kept = {k: v for k, v in counts.items() if v >= 5}
    small = {k: v for k, v in kept.items() if v < 30}
    print(f"empirical_score_dist: {len(kept)} buckets gardes (min 5), dont {len(small)} avec n<30 "
          f"({sum(small.values())} matchs concernes)")

    # ---- OOS loop ----
    recs = []
    for r in oos.itertuples():
        pred = predict_match_v5(model, r.team_a, r.team_b,
                                 float(r.odds_home), float(r.odds_draw), float(r.odds_away),
                                 extra_markets=r.extra_markets)
        if pred.get("p_h_blend") is None:
            continue
        out = outcome_of(r.score_a, r.score_b)
        p_pois = np.array([pred["p_h_pois"], pred["p_d_pois"], pred["p_a_pois"]])
        p_blend = np.array([pred["p_h_blend"], pred["p_d_blend"], pred["p_a_blend"]])
        p_cote_cal = np.array([pred["p_h_cote"], pred["p_d_cote"], pred["p_a_cote"]])
        p_devig = np.array(devig(r.odds_home, r.odds_draw, r.odds_away))
        # recover multi-market marginals: blend = 0.4 pois + 0.6 mm
        if np.abs(p_blend - p_pois).max() > 1e-9:
            p_mm = (p_blend - 0.4 * p_pois) / 0.6
            p_mm = np.clip(p_mm, 0, None)
            s = p_mm.sum()
            p_mm = p_mm / s if s > 0 else p_pois
            has_mm = True
        else:
            p_mm, has_mm = p_pois.copy(), False

        # extra markets parse for O/U
        em = r.extra_markets
        if isinstance(em, str):
            try: em = json.loads(em)
            except Exception: em = None
        em = em if isinstance(em, dict) else {}

        # C. over 2.5 truth + probas
        tot_goals = int(r.score_a) + int(r.score_b)
        lam_h, lam_a = pred["lam_h"], pred["lam_a"]
        p_o25_pois = 1 - poisson.cdf(2, lam_h + lam_a)
        gp = poisson_score_grid(lam_h, lam_a, model.rho)
        mm_grid = multi_market_score_grid(em)
        if mm_grid is not None:
            gb = 0.4 * gp + 0.6 * mm_grid
            gb = np.clip(gb, 0, None); gb /= gb.sum()
        else:
            gb = gp
        idx = np.add.outer(np.arange(8), np.arange(8))
        p_o25_grid = float(gb[idx >= 3].sum())
        p_o25_mkt = float(mm_grid[idx >= 3].sum()) if mm_grid is not None else None

        # D. HT with vs without DC
        ht_ok = pd.notna(r.ht_score_a)
        ht_rec = None
        if ht_ok and pred.get("lam_h_ht"):
            lh, la = pred["lam_h_ht"], pred["lam_a_ht"]
            p_ind = np.zeros(3); p_dc = np.zeros(3)
            for h in range(6):
                for a in range(6):
                    p = poisson.pmf(h, lh) * poisson.pmf(a, la)
                    k = 0 if h > a else (1 if h == a else 2)
                    p_ind[k] += p
                    p_dc[k] += p * _dc_tau(h, a, lh, la, model.rho)
            p_ind /= p_ind.sum(); p_dc = np.clip(p_dc, 0, None); p_dc /= p_dc.sum()
            ht_out = outcome_of(r.ht_score_a, r.ht_score_b)
            ht_rec = (["1","X","2"][int(p_ind.argmax())], ["1","X","2"][int(p_dc.argmax())], ht_out)

        # E. scores
        top5 = pred.get("top5_scores_enriched") or []
        true_score = f"{int(r.score_a)}-{int(r.score_b)}"
        # market-only / poisson-only top
        sc_mkt = em.get("Score exact")
        top_mkt = None
        if isinstance(sc_mkt, dict) and sc_mkt:
            best, bestp = None, -1
            for k, c in sc_mkt.items():
                try:
                    p = 1.0 / float(c)
                    if p > bestp: bestp, best = p, str(k).strip()
                except Exception: pass
            top_mkt = best.replace(" ", "") if best else None
        top_pois = f"{int(np.unravel_index(gp.argmax(), gp.shape)[0])}-{int(np.unravel_index(gp.argmax(), gp.shape)[1])}"

        recs.append(dict(out=out, oh=r.odds_home, od=r.odds_draw, oa=r.odds_away,
                          p_pois=p_pois, p_blend=p_blend, p_mm=p_mm, has_mm=has_mm,
                          p_cote_cal=p_cote_cal, p_devig=p_devig,
                          o25=tot_goals >= 3, p_o25_pois=p_o25_pois, p_o25_grid=p_o25_grid,
                          p_o25_mkt=p_o25_mkt, ht=ht_rec,
                          top5=[s for s, _ in top5], top_mkt=top_mkt, top_pois=top_pois,
                          true_score=true_score))

    print(f"\nOOS evaluables : {len(recs)} (skip equipes inconnues : {len(oos) - len(recs)})")
    O = ["1", "X", "2"]

    def eval_1x2(name, get_p, conf_bins=(0.5, 0.6, 0.7)):
        hits, rows_all, rows_hi = [], [], []
        conf = {t: [] for t in conf_bins}
        for rec in recs:
            p = get_p(rec)
            k = int(np.argmax(p))
            pick = O[k]
            cote = [rec["oh"], rec["od"], rec["oa"]][k]
            won = pick == rec["out"]
            hits.append(won)
            rows_all.append((won, cote))
            if cote >= 2.0:
                rows_hi.append((won, cote))
            for t in conf_bins:
                if p[k] >= t:
                    conf[t].append((won, cote))
        acc = np.mean(hits) * 100
        print(f"\n[{name}] acc_oos={acc:.2f}%  (n={len(hits)})")
        roi_line("  tous picks", rows_all)
        roi_line("  picks cote>=2.0", rows_hi)
        for t in conf_bins:
            roi_line(f"  picks p>={t:.0%}", conf[t])
        return acc

    print("\n" + "=" * 90)
    print("A/B. 1X2 — variantes de blending (OOS)")
    print("=" * 90)
    eval_1x2("BASELINE prod: 0.4 pois + 0.6 mm", lambda r: r["p_blend"])
    eval_1x2("poisson pur", lambda r: r["p_pois"])
    eval_1x2("multi-market pur (w_mm=1.0)", lambda r: r["p_mm"])
    for w in (0.7, 0.8, 0.9):
        eval_1x2(f"w_mm={w}", lambda r, w=w: (1 - w) * r["p_pois"] + w * r["p_mm"])
    eval_1x2("devig 1X2 pur (favori)", lambda r: r["p_devig"])
    eval_1x2("cote calibree (cal_h/d/a)", lambda r: r["p_cote_cal"])
    for wc in (0.3, 0.5, 0.7):
        eval_1x2(f"{wc} devig + {round(1-wc,1)} blend_prod",
                 lambda r, wc=wc: wc * r["p_devig"] + (1 - wc) * r["p_blend"])

    # B2. strategie hautes cotes par edge (p modele vs cote)
    print("\n" + "=" * 90)
    print("B2. Hautes cotes — bets edge-positif (p*cote-1 >= seuil) ET cote >= 2.0 (OOS)")
    print("=" * 90)
    for pname, get_p in [("blend_prod", lambda r: r["p_blend"]),
                          ("0.5 devig+0.5 blend", lambda r: 0.5 * r["p_devig"] + 0.5 * r["p_blend"]),
                          ("w_mm=0.9", lambda r: 0.1 * r["p_pois"] + 0.9 * r["p_mm"])]:
        for edge_min in (0.05, 0.15):
            rows = []
            for rec in recs:
                p = get_p(rec)
                cotes = [rec["oh"], rec["od"], rec["oa"]]
                edges = [p[i] * cotes[i] - 1 for i in range(3)]
                k = int(np.argmax(edges))
                if edges[k] >= edge_min and cotes[k] >= 2.0:
                    rows.append((O[k] == rec["out"], cotes[k]))
            roi_line(f"{pname} edge>={edge_min}", rows)

    # C. Over 2.5
    print("\n" + "=" * 90)
    print("C. Over 2.5 (OOS) — accuracy binaire (p>0.5) + Brier")
    print("=" * 90)
    y = np.array([rec["o25"] for rec in recs], dtype=float)
    for nm, key in [("Poisson pur (prod _predict_one_round)", "p_o25_pois"),
                     ("grille blend 0.4/0.6", "p_o25_grid")]:
        p = np.array([rec[key] for rec in recs], dtype=float)
        print(f"  {nm:<42} acc={(np.round(p) == y).mean()*100:5.1f}%  brier={np.mean((p-y)**2):.4f}  n={len(y)}")
    sub = [(rec["p_o25_mkt"], rec["o25"]) for rec in recs if rec["p_o25_mkt"] is not None]
    if sub:
        pm = np.array([a for a, _ in sub]); ym = np.array([float(b) for _, b in sub])
        print(f"  {'IPF market seul':<42} acc={(np.round(pm) == ym).mean()*100:5.1f}%  brier={np.mean((pm-ym)**2):.4f}  n={len(ym)}")

    # D. HT
    print("\n" + "=" * 90)
    print("D. HT 1X2 (OOS)")
    print("=" * 90)
    hts = [rec["ht"] for rec in recs if rec["ht"]]
    if hts:
        acc_ind = np.mean([a == o for a, _, o in hts]) * 100
        acc_dc = np.mean([b == o for _, b, o in hts]) * 100
        print(f"  Poisson independant (prod) : {acc_ind:.2f}%  (n={len(hts)})")
        print(f"  avec Dixon-Coles tau       : {acc_dc:.2f}%")
        dist_ht = pd.Series([o for _, _, o in hts]).value_counts(normalize=True)
        print(f"  distribution reelle HT : {dict((k, round(v*100,1)) for k, v in dist_ht.items())}")
        # toujours X a HT (base rate)
        accX = np.mean([o == 'X' for _, _, o in hts]) * 100
        print(f"  baseline 'toujours X'      : {accX:.2f}%")

    # E. score exact
    print("\n" + "=" * 90)
    print("E. Score exact (OOS)")
    print("=" * 90)
    top1 = np.mean([rec["top5"][0] == rec["true_score"] if rec["top5"] else False for rec in recs]) * 100
    top3 = np.mean([rec["true_score"] in rec["top5"][:3] if rec["top5"] else False for rec in recs]) * 100
    print(f"  V5 blend enrichi : top1={top1:.1f}%  top3={top3:.1f}%  (n={len(recs)})")
    smkt = [rec for rec in recs if rec["top_mkt"]]
    if smkt:
        t1m = np.mean([rec["top_mkt"] == rec["true_score"] for rec in smkt]) * 100
        print(f"  market-only modal : top1={t1m:.1f}%  (n={len(smkt)})")
    t1p = np.mean([rec["top_pois"] == rec["true_score"] for rec in recs]) * 100
    print(f"  poisson-only modal: top1={t1p:.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
