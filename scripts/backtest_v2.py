"""A/B backtest V1 vs V2 sur le meme split chronologique.

V1 = predictor (Poisson global + DC + calibration cote)
V2 = predictor_v2 (home/away split + market score blend + time decay optionnel)

Reporting :
  - Accuracy 1X2
  - ROI 1X2 (mise 1 sur argmax)
  - Brier (calibration)
  - Score exact accuracy
  - Filtre haute confiance (>=55% et >=70%)
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
from scraper.predictor_v2 import fit_model_v2, predict_match_v2


def _outcome(sa, sb):
    return "1" if sa > sb else ("X" if sa == sb else "2")


def _summary(name, picks, te, odds_h, odds_d, odds_a):
    """Compute accuracy + ROI + Brier from a Series of picks."""
    odds_map = {"1": odds_h, "X": odds_d, "2": odds_a}
    picked_odds = pd.Series([odds_map[p].iloc[i] for i, p in enumerate(picks)])
    actual = te["outcome"].reset_index(drop=True)
    hit = (picks.reset_index(drop=True) == actual)
    payoff = np.where(hit, picked_odds - 1, -1)
    acc = hit.mean() * 100
    roi = payoff.mean() * 100
    return acc, roi


def main() -> int:
    settings = load_settings()
    engine = create_engine(settings.db_url)

    df = pd.read_sql(
        """
        SELECT e.id, e.team_a, e.team_b, e.round_info,
               o.odds_home, o.odds_draw, o.odds_away, o.extra_markets,
               r.score_a, r.score_b, r.scrape_run_id
        FROM events e
        JOIN odds_snapshots o ON o.event_id = e.id
        JOIN results r ON r.event_id = e.id
        ORDER BY r.scrape_run_id, r.id
        """,
        engine,
    )
    df["outcome"] = df.apply(lambda r: _outcome(r.score_a, r.score_b), axis=1)
    print(f"== Sample : {len(df)} matchs joints ==")

    split = int(len(df) * 0.7)
    train = df.iloc[:split].reset_index(drop=True)
    test = df.iloc[split:].reset_index(drop=True)
    print(f"   train = {len(train)}  | test = {len(test)}")

    # =======================================================
    # V1
    # =======================================================
    print("\n=== Fit V1 ===")
    m1 = fit_model(train)
    print(f"   rho_DC = {m1.rho:+.4f}")
    rows_v1 = []
    for _, r in test.iterrows():
        p = predict_match(m1, r.team_a, r.team_b, r.odds_home, r.odds_draw, r.odds_away)
        rows_v1.append(p)
    pv1 = pd.DataFrame(rows_v1)

    # =======================================================
    # V2 (multiple configs to compare)
    # =======================================================
    print("\n=== Fit V2 ===")

    configs = [
        ("V2 no-decay, market=0.5",    None, 0.5),
        ("V2 no-decay, market=0.3",    None, 0.3),
        ("V2 no-decay, market=0.7",    None, 0.7),
        ("V2 decay=500, market=0.5",   500.0, 0.5),
        ("V2 decay=300, market=0.5",   300.0, 0.5),
    ]

    score_market_col = test["extra_markets"]

    def evaluate_v2(half_life, market_weight):
        m2 = fit_model_v2(train, half_life=half_life, score_market_weight=market_weight)
        rows = []
        for i, r in test.iterrows():
            p = predict_match_v2(
                m2, r.team_a, r.team_b,
                r.odds_home, r.odds_draw, r.odds_away,
                score_exact_market=score_market_col.iloc[i],
            )
            # extract score exact market for storage
            try:
                em = score_market_col.iloc[i]
                if isinstance(em, str):
                    em = pd.io.json.loads(em) if em else {}
                sc_market = (em or {}).get("Score exact")
            except Exception:
                sc_market = None
            p["has_market"] = sc_market is not None
            rows.append(p)
        return m2, pd.DataFrame(rows)

    # =======================================================
    # Helper to print metrics
    # =======================================================
    def metrics(prefix, label, pv):
        cols = [f"p_h_{prefix}", f"p_d_{prefix}", f"p_a_{prefix}"]
        # rows with missing prediction
        valid = pv[cols].notna().all(axis=1)
        sub = pv[valid].reset_index(drop=True)
        actual_sub = test["outcome"].reset_index(drop=True)[valid.values].reset_index(drop=True)
        odds_h_sub = test["odds_home"].reset_index(drop=True)[valid.values].reset_index(drop=True)
        odds_d_sub = test["odds_draw"].reset_index(drop=True)[valid.values].reset_index(drop=True)
        odds_a_sub = test["odds_away"].reset_index(drop=True)[valid.values].reset_index(drop=True)

        idx = sub[cols].idxmax(axis=1)
        pick = idx.map({cols[0]: "1", cols[1]: "X", cols[2]: "2"})

        def pi_odds(i):
            return odds_h_sub.iloc[i] if pick.iloc[i] == "1" else (
                odds_d_sub.iloc[i] if pick.iloc[i] == "X" else odds_a_sub.iloc[i]
            )
        odds_pick = pd.Series([pi_odds(i) for i in range(len(sub))])
        hit = (pick == actual_sub)
        payoff = np.where(hit, odds_pick - 1, -1)
        acc = hit.mean() * 100
        roi = payoff.mean() * 100

        p_mat = sub[cols].values
        y = pd.get_dummies(actual_sub).reindex(columns=["1", "X", "2"], fill_value=0).values
        brier = ((p_mat - y) ** 2).sum(axis=1).mean()

        # high-conf
        sub["max_p"] = sub[cols].max(axis=1)
        for th in (0.55, 0.70):
            hc = sub[sub["max_p"] >= th]
            if len(hc) > 0:
                hc_pick = idx.loc[hc.index].map({cols[0]: "1", cols[1]: "X", cols[2]: "2"})
                hc_actual = actual_sub.loc[hc.index]
                hc_odds = odds_pick.loc[hc.index]
                hc_hit = (hc_pick == hc_actual)
                hc_pay = np.where(hc_hit, hc_odds - 1, -1)
                hc_acc = hc_hit.mean() * 100
                hc_roi = hc_pay.mean() * 100
                yield f"{label:<35} n={len(sub):>3}  acc={acc:5.1f}%  ROI={roi:+6.1f}%  Brier={brier:.4f}  [p>={th:.0%}: n={len(hc):>3} acc={hc_acc:.1f}% ROI={hc_roi:+.1f}%]"
                if th == 0.55: continue
                else: return

    # =======================================================
    # 1X2 V1 vs V2 variants
    # =======================================================
    print("\n=== 1X2 ACCURACY / ROI ===")
    # V1 cote / V1 pois
    for line in metrics("cote", "V1 Cote+calib (baseline)", pv1):
        print(line)
    for line in metrics("pois", "V1 Poisson tier", pv1):
        print(line)

    best_v2 = None
    best_v2_score = -np.inf
    for label, hl, mw in configs:
        m2, pv2 = evaluate_v2(hl, mw)

        # V2 cote (same as V1 essentially)
        # V2 poisson seul (sans market)
        for line in metrics("pois", f"{label} - Poisson seul", pv2):
            print(line)
        # V2 blend (Poisson + market score)
        for line in metrics("blend", f"{label} - Blend Pois+Market", pv2):
            print(line)
        # V2 market seul
        if pv2["p_h_market"].notna().any():
            for line in metrics("market", f"{label} - Market scores seul", pv2):
                print(line)

    # =======================================================
    # SCORE EXACT
    # =======================================================
    print("\n=== SCORE EXACT ACCURACY ===")
    test_actual_score = test.apply(lambda r: f"{int(r.score_a)}-{int(r.score_b)}", axis=1).reset_index(drop=True)
    # V1
    v1_score = pv1["score_pois"].reset_index(drop=True)
    v1_acc = (v1_score == test_actual_score).mean() * 100
    print(f"  V1 Poisson seul                            : {v1_acc:.1f}%  (baseline 1/49 = 2.0%)")

    for label, hl, mw in configs:
        m2, pv2 = evaluate_v2(hl, mw)
        s_p = pv2["score_pois"].reset_index(drop=True)
        s_m = pv2["score_market"].reset_index(drop=True)
        s_b = pv2["score_blend"].reset_index(drop=True)
        acc_p = (s_p == test_actual_score).mean() * 100
        acc_m = (s_m.dropna() == test_actual_score[s_m.notna()].reset_index(drop=True)).mean() * 100 if s_m.notna().any() else float("nan")
        acc_b = (s_b == test_actual_score).mean() * 100
        n_mkt = s_m.notna().sum()
        print(f"  {label:<35} : Pois={acc_p:5.1f}%  Market={acc_m:5.1f}%(n={n_mkt})  Blend={acc_b:5.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
