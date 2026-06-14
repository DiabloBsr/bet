"""Backtest tier-based Poisson classifier (now with Dixon-Coles).

Methode :
  1. Split chronologique 70/30
  2. Fit model sur train (forces equipes + rho DC + calibration cotes)
  3. Predire test, comparer 3 modeles : Poisson+DC, Cotes calibrees, Blend
  4. Reporter accuracy, ROI, Brier, par tier mismatch et par seuil de confiance
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from scraper.config import load_settings
from scraper.predictor import fit_model, predict_match


def _outcome(sa, sb):
    return "1" if sa > sb else ("X" if sa == sb else "2")


def main() -> int:
    settings = load_settings()
    engine = create_engine(settings.db_url)

    df = pd.read_sql(
        """
        SELECT e.id, e.team_a, e.team_b, e.round_info,
               o.odds_home, o.odds_draw, o.odds_away,
               r.score_a, r.score_b, r.scrape_run_id
        FROM events e
        JOIN odds_snapshots o ON o.event_id = e.id
        JOIN results r ON r.event_id = e.id
        ORDER BY r.scrape_run_id, r.id
        """,
        engine,
    )
    if df.empty:
        print("aucun match joint")
        return 1
    df["outcome"] = df.apply(lambda r: _outcome(r.score_a, r.score_b), axis=1)

    split = int(len(df) * 0.7)
    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy().reset_index(drop=True)
    print(f"== Split chronologique : train={len(train)} | test={len(test)} ==")

    model = fit_model(train)
    print(f"\n== Modele fit (train) ==")
    print(f"  mu_h={model.mu_h:.3f}  mu_a={model.mu_a:.3f}  rho_DC={model.rho:+.4f}")
    print(f"  cal_cotes : shift_h={model.cal_h:+.3f}  shift_d={model.cal_d:+.3f}  shift_a={model.cal_a:+.3f}")

    rows = []
    for _, r in test.iterrows():
        pred = predict_match(model, r.team_a, r.team_b,
                              r.odds_home, r.odds_draw, r.odds_away)
        if pred["p_h_pois"] is None:
            continue
        # blend
        p_h_bl = (pred["p_h_pois"] + pred["p_h_cote"]) / 2
        p_d_bl = (pred["p_d_pois"] + pred["p_d_cote"]) / 2
        p_a_bl = (pred["p_a_pois"] + pred["p_a_cote"]) / 2

        rows.append({
            "team_a": r.team_a, "team_b": r.team_b,
            "outcome": r.outcome,
            "score_a": r.score_a, "score_b": r.score_b,
            "score_pred_pois": pred["score_pois"],
            "odds_home": r.odds_home, "odds_draw": r.odds_draw, "odds_away": r.odds_away,
            "p_h_pois": pred["p_h_pois"], "p_d_pois": pred["p_d_pois"], "p_a_pois": pred["p_a_pois"],
            "p_h_cote": pred["p_h_cote"], "p_d_cote": pred["p_d_cote"], "p_a_cote": pred["p_a_cote"],
            "p_h_bl": p_h_bl, "p_d_bl": p_d_bl, "p_a_bl": p_a_bl,
            "attack_diff": pred["attack_diff"],
        })

    te = pd.DataFrame(rows)
    if te.empty:
        print("aucun match testable")
        return 1
    print(f"\n== Test set : {len(te)} matchs testables ==")

    def evaluate(prefix, label):
        cols = [f"p_h_{prefix}", f"p_d_{prefix}", f"p_a_{prefix}"]
        idx = te[cols].idxmax(axis=1)
        pick = idx.map({f"p_h_{prefix}": "1", f"p_d_{prefix}": "X", f"p_a_{prefix}": "2"})

        def picked_odds(r):
            if pick[r.name] == "1": return r.odds_home
            if pick[r.name] == "X": return r.odds_draw
            return r.odds_away
        odds = te.apply(picked_odds, axis=1)
        payoff = np.where(pick == te["outcome"], odds - 1, -1)
        roi = payoff.mean() * 100
        acc = (pick == te["outcome"]).mean() * 100

        p_mat = te[cols].values
        y = pd.get_dummies(te["outcome"]).reindex(columns=["1", "X", "2"], fill_value=0).values
        brier = ((p_mat - y) ** 2).sum(axis=1).mean()

        cm = pd.crosstab(pick.rename("pred"), te["outcome"].rename("actual"))
        for c in ("1", "X", "2"):
            if c not in cm.columns: cm[c] = 0
            if c not in cm.index: cm.loc[c] = 0
        cm = cm.reindex(["1", "X", "2"], axis=0).reindex(["1", "X", "2"], axis=1).fillna(0).astype(int)

        print(f"\n--- {label} ---")
        print(f"  accuracy : {acc:.1f}%   ROI : {roi:+.1f}%   Brier : {brier:.4f}")
        print(f"  Matrice (pred lignes / actual colonnes) :")
        print("    " + cm.to_string().replace("\n", "\n    "))
        return acc, roi, brier, pick, payoff

    acc_p, roi_p, b_p, pick_p, pay_p = evaluate("pois", "(A) Poisson+DC")
    acc_c, roi_c, b_c, pick_c, pay_c = evaluate("cote", "(B) Cotes + calibration")
    acc_b, roi_b, b_b, pick_b, pay_b = evaluate("bl",   "(C) Blend / 2")

    print(f"\n== Resume ==")
    print(f"  {'Modele':<35} {'Accuracy':>10} {'ROI':>9} {'Brier':>8}")
    for label, a, r_, b in [("(A) Poisson + Dixon-Coles", acc_p, roi_p, b_p),
                              ("(B) Cotes + calibration", acc_c, roi_c, b_c),
                              ("(C) Blend", acc_b, roi_b, b_b)]:
        print(f"  {label:<35} {a:>9.1f}% {r_:>+8.1f}% {b:>8.4f}")

    # Mismatch
    te["mismatch"] = pd.cut(te["attack_diff"], bins=[-99, -0.5, -0.2, 0.2, 0.5, 99],
                              labels=["away++", "away+", "even", "home+", "home++"])
    te["hit_p"] = (pick_p == te["outcome"]).astype(int)
    te["payoff_p"] = pay_p

    print(f"\n== Accuracy + ROI par ecart de tier (Poisson) ==")
    by_mis = te.groupby("mismatch", observed=True).agg(
        n=("hit_p", "size"),
        accuracy=("hit_p", lambda s: s.mean() * 100),
        roi=("payoff_p", lambda s: s.mean() * 100),
    ).round(1)
    print(by_mis.to_string())

    # Confidence thresholds
    te["max_p_pois"] = te[["p_h_pois", "p_d_pois", "p_a_pois"]].max(axis=1)
    print(f"\n== Filtre par seuil de confiance Poisson ==")
    for th in (0.50, 0.55, 0.60, 0.70, 0.75):
        sub = te[te["max_p_pois"] >= th]
        if len(sub) == 0:
            continue
        sub_pick = pick_p.loc[sub.index]
        sub_pay = pay_p[sub.index]
        acc = (sub_pick == sub["outcome"]).mean() * 100
        roi = sub_pay.mean() * 100
        print(f"  p_pois >= {th:.0%}  n={len(sub):>3}  accuracy={acc:.1f}%  ROI={roi:+.1f}%")

    # Accord entre Poisson et Cote
    print(f"\n== Filtre accord Poisson + Cote ==")
    agree = pick_p == pick_c
    sub_agree = te[agree]
    sub_disagree = te[~agree]
    print(f"  Accord (mm pick)    : n={len(sub_agree):>3}  "
          f"accuracy={(pick_p.loc[sub_agree.index] == sub_agree['outcome']).mean()*100:.1f}%  "
          f"ROI={pay_p[sub_agree.index].mean()*100:+.1f}%")
    if len(sub_disagree) > 0:
        print(f"  Desaccord           : n={len(sub_disagree):>3}  "
              f"accuracy_pois={(pick_p.loc[sub_disagree.index] == sub_disagree['outcome']).mean()*100:.1f}%  "
              f"accuracy_cote={(pick_c.loc[sub_disagree.index] == sub_disagree['outcome']).mean()*100:.1f}%")

    # Combined filter
    print(f"\n== FILTRE PREMIUM (accord ET p_pois>=70%) ==")
    premium = agree & (te["max_p_pois"] >= 0.70)
    sub = te[premium]
    if len(sub) > 0:
        acc = (pick_p.loc[sub.index] == sub["outcome"]).mean() * 100
        roi = pay_p[sub.index].mean() * 100
        print(f"  n={len(sub)}  accuracy={acc:.1f}%  ROI={roi:+.1f}%")
    else:
        print(f"  aucun match ne passe ce double filtre")

    # Score exact
    te["actual_score"] = te["score_a"].astype(str) + "-" + te["score_b"].astype(str)
    score_hit = (te["score_pred_pois"] == te["actual_score"]).mean() * 100
    print(f"\n  Score exact Poisson-DC : hit={score_hit:.1f}%   (baseline 1/49 = 2.0%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
