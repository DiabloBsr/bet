"""Predictor V3 — toutes ameliorations cumulees vs V2.

Changements vs V2 :
  1. Multi-market IPF (Score exact + Total de buts + G/NG -> grille score plus precise)
  2. X-aware pick rule (bascule X quand p_X assez haute et match equilibre)
  3. Form integration (ajuste forces via history des 5 derniers matchs - rankings_snapshots)

Toutes les ameliorations sont parametrables, defauts conservateurs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson

from scraper.predictor_v2 import (
    FittedModelV2,
    _dc_tau,
    compute_home_away_strengths,
    cote_probs,
    devig,
    estimate_calibration,
    estimate_rho_v2,
    grid_modal_score,
    grid_to_1x2,
    grid_top_k_scores,
    market_score_grid,
    poisson_score_grid,
)


@dataclass
class FittedModelV3(FittedModelV2):
    form_data: dict[str, dict] = field(default_factory=dict)
    form_alpha: float = 0.10  # impact du form sur attack/defense


# ---------------------------------------------------------------------------
# Form integration (last 5 results -> form score [0, 1])
# ---------------------------------------------------------------------------

def compute_form_from_history(history_list: list[str]) -> float:
    """Convert ranking.history (e.g. ['Won','Lost','Won','Draw','Won']) to [0, 1]."""
    if not history_list: return 0.5
    score = 0
    n = 0
    for o in history_list:
        if not isinstance(o, str): continue
        n += 1
        s = o.strip().lower()
        if s.startswith("won") or s == "w": score += 3
        elif s.startswith("draw") or s == "d": score += 1
        # lost = 0
    return score / (3 * n) if n > 0 else 0.5


def load_form_from_rankings(engine, competition: str | None = None) -> dict[str, dict]:
    """Read latest rankings_snapshots per team -> form score."""
    query = "SELECT team_name, competition, history, MAX(captured_at) FROM rankings_snapshots"
    if competition:
        query += f" WHERE competition = '{competition}'"
    query += " GROUP BY team_name"
    df = pd.read_sql(query, engine)
    out = {}
    for _, r in df.iterrows():
        if r.history:
            try:
                hist = r.history if isinstance(r.history, list) else json.loads(r.history)
                form = compute_form_from_history(hist)
                out[r.team_name] = {"form": form, "history": hist}
            except Exception:
                continue
    return out


# ---------------------------------------------------------------------------
# Multi-market IPF
# ---------------------------------------------------------------------------

def multi_market_score_grid(em: dict, max_iter: int = 30, max_goals: int = 8) -> np.ndarray | None:
    """IPF: Score exact + Total de buts + G/NG constraints sur la grille."""
    sc_exact = em.get("Score exact") if isinstance(em, dict) else None
    base = market_score_grid(sc_exact, max_goals)
    if base is None:
        return None
    grid = base.copy()

    # Total de buts marginal (sum h+a)
    tot_target = None
    tot_market = em.get("Total de buts") if isinstance(em, dict) else None
    if isinstance(tot_market, dict):
        tot_target = {}
        for k, cote in tot_market.items():
            try:
                n = int(str(k).rstrip("+").strip())
                tot_target[n] = 1.0 / float(cote)
            except (ValueError, TypeError):
                continue
        s = sum(tot_target.values())
        if s > 0:
            tot_target = {k: v / s for k, v in tot_target.items()}
        else:
            tot_target = None

    # G/NG marginal (P(BTTS))
    btts_target = None
    gng = em.get("G/NG") if isinstance(em, dict) else None
    if isinstance(gng, dict) and "Oui" in gng and "Non" in gng:
        try:
            p_oui = 1.0 / float(gng["Oui"])
            p_non = 1.0 / float(gng["Non"])
            btts_target = p_oui / (p_oui + p_non)
        except (ValueError, TypeError, ZeroDivisionError):
            btts_target = None

    if tot_target is None and btts_target is None:
        return grid  # rien a contraindre, retour grille market

    for _ in range(max_iter):
        old = grid.copy()

        if tot_target:
            for n_goals, target in tot_target.items():
                cells = [(h, a) for h in range(max_goals) for a in range(max_goals) if h + a == n_goals]
                current = sum(grid[h, a] for h, a in cells)
                if current > 0 and target > 0:
                    factor = target / current
                    for h, a in cells:
                        grid[h, a] *= factor

        if btts_target is not None:
            btts_cells = [(h, a) for h in range(1, max_goals) for a in range(1, max_goals)]
            no_btts_cells = [(h, a) for h in range(max_goals) for a in range(max_goals) if h == 0 or a == 0]
            current_btts = sum(grid[h, a] for h, a in btts_cells)
            current_no = sum(grid[h, a] for h, a in no_btts_cells)
            if current_btts > 0:
                factor = btts_target / current_btts
                for h, a in btts_cells:
                    grid[h, a] *= factor
            if current_no > 0:
                factor = (1 - btts_target) / current_no
                for h, a in no_btts_cells:
                    grid[h, a] *= factor

        grid = np.clip(grid, 0, None)
        s = grid.sum()
        if s > 0:
            grid /= s

        if np.abs(grid - old).max() < 1e-4:
            break

    return grid


# ---------------------------------------------------------------------------
# Fit + predict
# ---------------------------------------------------------------------------

def fit_model_v3(
    history: pd.DataFrame,
    engine=None,
    smoothing: int = 5,
    half_life: float | None = None,
    score_market_weight: float = 0.5,
    form_alpha: float = 0.10,
    competition: str | None = "InstantLeague-8035",
) -> FittedModelV3:
    """Fit V2 puis ajuste forces avec data form."""
    home_s, away_s, mu_h, mu_a = compute_home_away_strengths(history, smoothing, half_life)
    rho = estimate_rho_v2(history, home_s, away_s, mu_h, mu_a)
    cal_h, cal_d, cal_a = estimate_calibration(history)

    # Load form from rankings if engine provided
    form_data = {}
    if engine is not None:
        form_data = load_form_from_rankings(engine, competition)
        # Apply form adjustment : team chaude (form>0.5) -> attack boost
        for team, info in form_data.items():
            form_score = info["form"]
            boost = (form_score - 0.5) * form_alpha * 2  # range [-form_alpha, +form_alpha]
            for strengths in (home_s, away_s):
                if team in strengths:
                    strengths[team]["attack"] *= (1 + boost)
                    strengths[team]["defense"] *= max(0.5, (1 - boost))

    return FittedModelV3(
        home_strengths=home_s, away_strengths=away_s,
        mu_h=mu_h, mu_a=mu_a, rho=rho,
        cal_h=cal_h, cal_d=cal_d, cal_a=cal_a,
        n_train=len(history),
        half_life=half_life,
        score_market_weight=score_market_weight,
        form_data=form_data,
        form_alpha=form_alpha,
        config={"smoothing": smoothing},
    )


def predict_match_v3(
    model: FittedModelV3, team_a: str, team_b: str,
    odds_home: float, odds_draw: float, odds_away: float,
    extra_markets: dict | str | None = None,
    x_threshold: float = 0.27,
    x_balance: float = 0.10,
) -> dict[str, Any]:
    """V3 prediction avec multi-market IPF + X-aware."""
    # Parse extra_markets
    if isinstance(extra_markets, str):
        try: extra_markets = json.loads(extra_markets)
        except Exception: extra_markets = None
    em = extra_markets if isinstance(extra_markets, dict) else {}

    out: dict[str, Any] = {
        "team_a": team_a, "team_b": team_b,
        "odds_home": odds_home, "odds_draw": odds_draw, "odds_away": odds_away,
    }

    sh = model.home_strengths.get(team_a)
    sa = model.away_strengths.get(team_b)

    if sh and sa:
        lam_h = sh["attack"] * sa["defense"] * model.mu_h
        lam_a = sa["attack"] * sh["defense"] * model.mu_a

        # Poisson grid
        grid_p = poisson_score_grid(lam_h, lam_a, model.rho)
        p_h_p, p_d_p, p_a_p = grid_to_1x2(grid_p)

        # Multi-market grid (Score exact + Total + G/NG)
        mm_grid = multi_market_score_grid(em)
        if mm_grid is not None:
            # Blend Poisson + multi-market grid
            grid_blend = 0.4 * grid_p + 0.6 * mm_grid
            grid_blend = np.clip(grid_blend, 0, None)
            s = grid_blend.sum()
            if s > 0: grid_blend /= s
        else:
            grid_blend = grid_p

        p_h_bl, p_d_bl, p_a_bl = grid_to_1x2(grid_blend)
        score_pois = grid_modal_score(grid_p)
        score_blend = grid_modal_score(grid_blend)
        score_mm = grid_modal_score(mm_grid) if mm_grid is not None else None
        top3_blend = grid_top_k_scores(grid_blend, 3)
        top3_mm = grid_top_k_scores(mm_grid, 3) if mm_grid is not None else None

        # X-aware pick rule
        if p_h_bl >= p_d_bl and p_h_bl >= p_a_bl:
            pick_xaware = "1"
        elif p_a_bl >= p_d_bl:
            pick_xaware = "2"
        else:
            pick_xaware = "X"
        # Override: bascule X si p_X assez haute et match equilibre
        if p_d_bl >= x_threshold and abs(p_h_bl - p_a_bl) < x_balance:
            pick_xaware = "X"

        # Form info
        form_h = model.form_data.get(team_a, {}).get("form")
        form_a = model.form_data.get(team_b, {}).get("form")

        # O/U + BTTS marginales depuis la grille BLEND (IPF market-contrainte) :
        # +1.8pp acc Over 2.5 OOS vs le recalcul Poisson indépendant
        n_goals = grid_blend.shape[0]
        totals = np.add.outer(np.arange(n_goals), np.arange(n_goals))
        p_over_15_bl = float(grid_blend[totals > 1.5].sum())
        p_over_25_bl = float(grid_blend[totals > 2.5].sum())
        p_over_35_bl = float(grid_blend[totals > 3.5].sum())
        p_btts_bl = float(grid_blend[1:, 1:].sum())

        out.update({
            "lam_h": lam_h, "lam_a": lam_a,
            "p_h_pois": p_h_p, "p_d_pois": p_d_p, "p_a_pois": p_a_p,
            "p_h_blend": p_h_bl, "p_d_blend": p_d_bl, "p_a_blend": p_a_bl,
            "p_over_15_blend": p_over_15_bl, "p_over_25_blend": p_over_25_bl,
            "p_over_35_blend": p_over_35_bl, "p_btts_blend": p_btts_bl,
            "score_pois": score_pois,
            "score_blend": score_blend,
            "score_market": score_mm,
            "top3_blend": top3_blend,
            "top3_market": top3_mm,
            "pick_xaware": pick_xaware,
            "attack_diff": sh["attack"] - sa["attack"],
            "form_home": form_h, "form_away": form_a,
            "team_a_n": sh["n"], "team_b_n": sa["n"],
        })
    else:
        out.update({k: None for k in [
            "lam_h", "lam_a", "p_h_pois", "p_d_pois", "p_a_pois",
            "p_h_blend", "p_d_blend", "p_a_blend",
            "p_over_15_blend", "p_over_25_blend", "p_over_35_blend", "p_btts_blend",
            "score_pois", "score_blend", "score_market",
            "top3_blend", "top3_market",
            "pick_xaware", "attack_diff", "form_home", "form_away",
        ]})
        out.update({"team_a_n": 0, "team_b_n": 0})

    # Cotes calibrees (toujours)
    p_h_c, p_d_c, p_a_c = cote_probs(
        odds_home, odds_draw, odds_away,
        model.cal_h, model.cal_d, model.cal_a,
    )
    out.update({"p_h_cote": p_h_c, "p_d_cote": p_d_c, "p_a_cote": p_a_c})
    return out
