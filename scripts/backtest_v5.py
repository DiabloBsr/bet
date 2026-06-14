"""Backtest V5 : trouve les marches HT/HT-FT/FTTS rentables."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from scraper.config import load_settings
from scraper.predictor_v5 import fit_model_v5, predict_match_v5


def _out(sa, sb):
    return "1" if sa > sb else ("X" if sa == sb else "2")


def main():
    settings = load_settings()
    engine = create_engine(settings.db_url)
    df = pd.read_sql("""
        SELECT e.team_a, e.team_b,
               o.odds_home, o.odds_draw, o.odds_away, o.extra_markets,
               r.score_a, r.score_b, r.ht_score_a, r.ht_score_b,
               r.scrape_run_id
        FROM events e
        JOIN odds_snapshots o ON o.event_id = e.id
        JOIN results r ON r.event_id = e.id
        ORDER BY r.scrape_run_id, r.id
    """, engine)
    df["outcome"] = df.apply(lambda r: _out(r.score_a, r.score_b), axis=1)
    df["ht_outcome"] = df.apply(
        lambda r: _out(r.ht_score_a, r.ht_score_b) if pd.notna(r.ht_score_a) else None,
        axis=1,
    )
    df["htft"] = df.apply(lambda r: f"{r.ht_outcome}/{r.outcome}" if r.ht_outcome else None, axis=1)

    print(f"== Sample n={len(df)} | avec HT score n={df['ht_outcome'].notna().sum()} ==\n")

    # Split using only matches with HT data for fairness
    ht_df = df[df["ht_outcome"].notna()].reset_index(drop=True)
    split = int(len(ht_df) * 0.7)
    train_ht = ht_df.iloc[:split].copy()
    test_ht = ht_df.iloc[split:].reset_index(drop=True)

    train_all = df.iloc[:int(len(df)*0.7)].reset_index(drop=True)

    print(f"   train_all={len(train_all)} | train_ht={len(train_ht)} | test_ht={len(test_ht)}\n")

    # Fit V5 (need HT data for HT/FT matrix)
    model = fit_model_v5(train_all, ht_history=train_ht, engine=engine, form_alpha=0.0)
    print(f"V5 fit : rho={model.rho:+.4f}  ht_lambda_ratio={model.ht_lambda_ratio:.3f}")
    print(f"HT/FT matrix from training : {len(model.ht_ft_matrix or {})} entries\n")
    if model.ht_ft_matrix:
        for ht, fts in sorted(model.ht_ft_matrix.items()):
            print(f"   HT={ht} -> FT: " + " ".join(f"{ft}={p*100:.1f}%" for ft, p in fts.items()))

    rows = []
    for _, r in test_ht.iterrows():
        pred = predict_match_v5(model, r.team_a, r.team_b,
                                  r.odds_home, r.odds_draw, r.odds_away,
                                  extra_markets=r.extra_markets)
        em = pred.get("ht_ft_probs") or {}
        rows.append({
            "outcome": r.outcome,
            "ht_outcome": r.ht_outcome,
            "htft_actual": r.htft,
            "ht_pick": pred.get("ht_pick"),
            "ht_p": pred.get("ht_p") or 0,
            "htft_pick": pred.get("htft_pick"),
            "htft_p": pred.get("htft_p") or 0,
            "extra_markets": r.extra_markets,
        })
    te = pd.DataFrame(rows)

    # ============ HT 1X2 ============
    print(f"\n=== HT 1X2 (Mi-tps 1X2) ===")
    ht_hit = (te["ht_pick"] == te["ht_outcome"]).mean() * 100
    print(f"  Accuracy globale : {ht_hit:.1f}%")

    # Distribution V5 picks vs actual
    print(f"  Picks V5 : 1={(te['ht_pick']=='1').mean()*100:.1f}%  X={(te['ht_pick']=='X').mean()*100:.1f}%  2={(te['ht_pick']=='2').mean()*100:.1f}%")
    print(f"  Actual   : 1={(te['ht_outcome']=='1').mean()*100:.1f}%  X={(te['ht_outcome']=='X').mean()*100:.1f}%  2={(te['ht_outcome']=='2').mean()*100:.1f}%")

    # Backtest pari HT X (pari le nul HT)
    print(f"\n  === BACKTEST HT X (parier le nul HT systematiquement) ===")
    em_data = te["extra_markets"].apply(
        lambda x: x if isinstance(x, dict) else (json.loads(x) if x else {}))
    ht_x_cotes = em_data.apply(
        lambda em: em.get("Mi-tps 1X2", {}).get("X") if isinstance(em.get("Mi-tps 1X2"), dict) else None)
    if ht_x_cotes.notna().any():
        valid = ht_x_cotes.notna()
        actual_x = te.loc[valid, "ht_outcome"] == "X"
        cotes = ht_x_cotes[valid]
        payoff = np.where(actual_x, cotes - 1, -1)
        print(f"  Global :")
        print(f"    n={valid.sum()}  hit={actual_x.mean()*100:.1f}%  cote_moy={cotes.mean():.2f}  ROI={payoff.mean()*100:+.2f}%")

        # Filtre par confiance V5
        print(f"\n  Filtre par confiance V5 p_d_ht :")
        for th in (0.40, 0.45, 0.50, 0.55, 0.60):
            mask_th = valid & (te["ht_p"] >= th) & (te["ht_pick"] == "X")
            if mask_th.sum() < 5: continue
            sub_actual = te.loc[mask_th, "ht_outcome"] == "X"
            sub_cotes = ht_x_cotes[mask_th]
            sub_payoff = np.where(sub_actual, sub_cotes - 1, -1)
            print(f"    p>={th:.0%}  n={mask_th.sum():>3}  hit={sub_actual.mean()*100:5.1f}%  cote={sub_cotes.mean():.2f}  ROI={sub_payoff.mean()*100:+6.2f}%")

        # Filter par cote (parier X seulement quand cote dans certain range)
        print(f"\n  Filtre par cote HT-X :")
        for cote_range in [(1.5, 2.0), (2.0, 2.3), (2.3, 2.6), (2.6, 3.0), (3.0, 5.0)]:
            mask_c = valid & (ht_x_cotes >= cote_range[0]) & (ht_x_cotes < cote_range[1])
            if mask_c.sum() < 10: continue
            sub_actual = te.loc[mask_c, "ht_outcome"] == "X"
            sub_cotes = ht_x_cotes[mask_c]
            sub_payoff = np.where(sub_actual, sub_cotes - 1, -1)
            print(f"    cote [{cote_range[0]:.1f}-{cote_range[1]:.1f}[  n={mask_c.sum():>3}  hit={sub_actual.mean()*100:5.1f}%  ROI={sub_payoff.mean()*100:+6.2f}%")

        # Combine: match equilibre + cote X >= 2.0 + p_d_ht >= 0.45
        # = exclure les matchs avec gros favori (qui pousse a HT victoire)
        mask_combo = valid & (te["ht_p"] >= 0.45) & (te["ht_pick"] == "X") & (ht_x_cotes >= 2.0)
        if mask_combo.sum() > 10:
            sub_actual = te.loc[mask_combo, "ht_outcome"] == "X"
            sub_cotes = ht_x_cotes[mask_combo]
            sub_payoff = np.where(sub_actual, sub_cotes - 1, -1)
            print(f"\n  COMBO p>=45% + pick=X + cote>=2.0 :")
            print(f"    n={mask_combo.sum()}  hit={sub_actual.mean()*100:5.1f}%  ROI={sub_payoff.mean()*100:+6.2f}%")

    # ============ HT/FT ============
    print(f"\n=== HT/FT ===")
    if te["htft_pick"].notna().any():
        htft_hit = (te["htft_pick"] == te["htft_actual"]).mean() * 100
        print(f"  Accuracy globale (modal) : {htft_hit:.1f}%")
        # Distribution top picks
        print(f"  Top 5 picks V5 : ")
        for pick, n in te["htft_pick"].value_counts().head(5).items():
            actual_when_picked = (te[te["htft_pick"] == pick]["htft_actual"] == pick).mean() * 100
            print(f"    {pick} : pique {n}x  realisation {actual_when_picked:.1f}%")

        # Most-frequent ACTUAL htft
        print(f"\n  Top 5 HT/FT REELS du test :")
        for pick, n in te["htft_actual"].value_counts().head(5).items():
            print(f"    {pick} : {n}x ({n/len(te)*100:.1f}%)")

    # Backtest specifically X/1 (signal from analysis)
    print(f"\n  === BACKTEST 'X/1' (pari : nul HT puis home gagne FT) ===")
    em_data2 = te["extra_markets"].apply(
        lambda x: x if isinstance(x, dict) else (json.loads(x) if x else {}))
    htft_cotes = em_data2.apply(
        lambda em: em.get("HT/FT", {}).get("X/1") if isinstance(em.get("HT/FT"), dict) else None)
    if htft_cotes.notna().any():
        valid = htft_cotes.notna()
        actual_x1 = te.loc[valid, "htft_actual"] == "X/1"
        cotes = htft_cotes[valid]
        payoff = np.where(actual_x1, cotes - 1, -1)
        print(f"    n={valid.sum()}  hit={actual_x1.mean()*100:.1f}%  cote_moy={cotes.mean():.2f}  ROI={payoff.mean()*100:+.2f}%")

    # Backtest 1/1
    print(f"\n  === BACKTEST '1/1' (home HT puis home FT) ===")
    htft_cotes_11 = em_data2.apply(
        lambda em: em.get("HT/FT", {}).get("1/1") if isinstance(em.get("HT/FT"), dict) else None)
    if htft_cotes_11.notna().any():
        valid = htft_cotes_11.notna()
        actual_11 = te.loc[valid, "htft_actual"] == "1/1"
        cotes = htft_cotes_11[valid]
        payoff = np.where(actual_11, cotes - 1, -1)
        print(f"    n={valid.sum()}  hit={actual_11.mean()*100:.1f}%  cote_moy={cotes.mean():.2f}  ROI={payoff.mean()*100:+.2f}%")

    # Tous les HT/FT
    print(f"\n  === BACKTEST tous les HT/FT (ROI par cellule) ===")
    htft_cells = ["1/1", "1/X", "1/2", "X/1", "X/X", "X/2", "2/1", "2/X", "2/2"]
    for cell in htft_cells:
        cotes_cell = em_data2.apply(
            lambda em, c=cell: em.get("HT/FT", {}).get(c) if isinstance(em.get("HT/FT"), dict) else None)
        if not cotes_cell.notna().any(): continue
        valid = cotes_cell.notna()
        actual = te.loc[valid, "htft_actual"] == cell
        cotes = cotes_cell[valid]
        payoff = np.where(actual, cotes - 1, -1)
        roi = payoff.mean() * 100
        flag = " <-- ROI POSITIF" if roi > 0 else ""
        print(f"    {cell} : n={valid.sum():>3}  hit={actual.mean()*100:5.1f}%  cote_moy={cotes.mean():5.2f}  ROI={roi:+6.2f}%{flag}")

    # ============ Mi-tps DC (Double chance HT) ============
    print(f"\n=== Mi-tps Double Chance ===")
    for cell in ("1X", "12", "X2"):
        cotes = em_data2.apply(
            lambda em, c=cell: em.get("Mi-tps DC", {}).get(c) if isinstance(em.get("Mi-tps DC"), dict) else None)
        if not cotes.notna().any(): continue
        valid = cotes.notna()
        # 1X hit = ht_outcome in {1,X}
        if cell == "1X":
            hits = te.loc[valid, "ht_outcome"].isin(["1", "X"])
        elif cell == "12":
            hits = te.loc[valid, "ht_outcome"].isin(["1", "2"])
        elif cell == "X2":
            hits = te.loc[valid, "ht_outcome"].isin(["X", "2"])
        else:
            continue
        payoff = np.where(hits, cotes[valid] - 1, -1)
        roi = payoff.mean() * 100
        flag = " <-- ROI POSITIF" if roi > 0 else ""
        print(f"    {cell:<3} : n={valid.sum():>3}  hit={hits.mean()*100:5.1f}%  cote_moy={cotes[valid].mean():5.2f}  ROI={roi:+6.2f}%{flag}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
