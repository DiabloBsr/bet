"""Shared prediction logic.

Combine 3 strategies :
  - Team strengths (Poisson) avec Dixon-Coles regularisation
  - Vig-free + calibration cotes
  - Convenience helpers pour cote -> probas

Dixon-Coles tau corrige les scores faibles (0-0, 0-1, 1-0, 1-1) que le Poisson
sous-estime en realite. rho est estime par maximum de vraisemblance sur le train.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson


@dataclass
class FittedModel:
    strengths: dict[str, dict[str, float]]
    mu_h: float
    mu_a: float
    rho: float       # Dixon-Coles correlation
    cal_h: float     # cote calibration residuals
    cal_d: float
    cal_a: float
    n_train: int


# ---------------------------------------------------------------------------
# Team strengths
# ---------------------------------------------------------------------------

def compute_strengths(history: pd.DataFrame, smoothing: int = 5) -> tuple[dict, float, float]:
    """Returns (strengths, mu_h, mu_a) where strengths[team] = {attack, defense, n}."""
    teams = defaultdict(lambda: {"gf": 0.0, "ga": 0.0, "n": 0})
    for r in history.itertuples():
        sa, sb = float(r.score_a), float(r.score_b)
        teams[r.team_a]["gf"] += sa
        teams[r.team_a]["ga"] += sb
        teams[r.team_a]["n"] += 1
        teams[r.team_b]["gf"] += sb
        teams[r.team_b]["ga"] += sa
        teams[r.team_b]["n"] += 1

    mu_h = float(history["score_a"].mean())
    mu_a = float(history["score_b"].mean())
    avg_total = mu_h + mu_a

    strengths: dict[str, dict[str, float]] = {}
    for name, s in teams.items():
        if s["n"] == 0:
            continue
        attack = ((s["gf"] + smoothing * avg_total / 2) / (s["n"] + smoothing)) / (avg_total / 2)
        defense = ((s["ga"] + smoothing * avg_total / 2) / (s["n"] + smoothing)) / (avg_total / 2)
        strengths[name] = {"attack": attack, "defense": defense, "n": s["n"]}
    return strengths, mu_h, mu_a


# ---------------------------------------------------------------------------
# Dixon-Coles
# ---------------------------------------------------------------------------

def _dc_tau(h: int, a: int, lam_h: float, lam_a: float, rho: float) -> float:
    """Dixon-Coles low-score correction. rho<0 boosts {0-0,1-1}, rho>0 boosts {0-1,1-0}."""
    if h == 0 and a == 0:
        return 1.0 - lam_h * lam_a * rho
    if h == 0 and a == 1:
        return 1.0 + lam_h * rho
    if h == 1 and a == 0:
        return 1.0 + lam_a * rho
    if h == 1 and a == 1:
        return 1.0 - rho
    return 1.0


def estimate_rho(history: pd.DataFrame, strengths: dict, mu_h: float, mu_a: float) -> float:
    """Maximum likelihood estimate of rho on training data."""
    rows = []
    for r in history.itertuples():
        sh = strengths.get(r.team_a)
        sa = strengths.get(r.team_b)
        if sh is None or sa is None:
            continue
        lam_h = sh["attack"] * sa["defense"] * mu_h
        lam_a = sa["attack"] * sh["defense"] * mu_a
        rows.append((int(r.score_a), int(r.score_b), lam_h, lam_a))

    def neg_log_lik(rho: float) -> float:
        ll = 0.0
        for h, a, lh, la in rows:
            p_p = poisson.pmf(h, lh) * poisson.pmf(a, la)
            tau = _dc_tau(h, a, lh, la, rho)
            if p_p > 0 and tau > 0:
                ll += np.log(p_p * tau)
        return -ll

    result = minimize_scalar(neg_log_lik, bounds=(-0.3, 0.3), method="bounded")
    return float(result.x)


def poisson_probs(
    lam_h: float, lam_a: float, rho: float = 0.0, max_goals: int = 8
) -> tuple[float, float, float, str, np.ndarray]:
    """Returns (P(1), P(X), P(2), most_likely_score, full_grid)."""
    p_grid = np.zeros((max_goals, max_goals))
    for h in range(max_goals):
        for a in range(max_goals):
            p_grid[h, a] = (
                poisson.pmf(h, lam_h)
                * poisson.pmf(a, lam_a)
                * _dc_tau(h, a, lam_h, lam_a, rho)
            )
    p_grid = np.clip(p_grid, 0, None)
    total = p_grid.sum()
    if total > 0:
        p_grid /= total

    p_h = float(sum(p_grid[h, a] for h in range(max_goals) for a in range(max_goals) if h > a))
    p_d = float(sum(p_grid[h, a] for h in range(max_goals) for a in range(max_goals) if h == a))
    p_a = float(sum(p_grid[h, a] for h in range(max_goals) for a in range(max_goals) if h < a))

    idx = np.unravel_index(p_grid.argmax(), p_grid.shape)
    return p_h, p_d, p_a, f"{idx[0]}-{idx[1]}", p_grid


# ---------------------------------------------------------------------------
# Cotes -> probas vig-free + calibration
# ---------------------------------------------------------------------------

def devig(oh: float, od: float, oa: float) -> tuple[float, float, float]:
    ih, id_, ia = 1 / oh, 1 / od, 1 / oa
    s = ih + id_ + ia
    return ih / s, id_ / s, ia / s


def estimate_calibration(history: pd.DataFrame) -> tuple[float, float, float]:
    train = history.copy()
    probs = train.apply(
        lambda r: pd.Series(devig(r.odds_home, r.odds_draw, r.odds_away),
                              index=["p_h", "p_d", "p_a"]),
        axis=1,
    )
    train = pd.concat([train, probs], axis=1)
    train["outcome"] = train.apply(
        lambda r: "1" if r.score_a > r.score_b
        else ("X" if r.score_a == r.score_b else "2"), axis=1,
    )
    cal_h = (train["outcome"] == "1").mean() - train["p_h"].mean()
    cal_d = (train["outcome"] == "X").mean() - train["p_d"].mean()
    cal_a = (train["outcome"] == "2").mean() - train["p_a"].mean()
    return float(cal_h), float(cal_d), float(cal_a)


def cote_probs(
    oh: float, od: float, oa: float,
    cal_h: float = 0.0, cal_d: float = 0.0, cal_a: float = 0.0,
) -> tuple[float, float, float]:
    p_h, p_d, p_a = devig(oh, od, oa)
    p_h = max(0.01, p_h + cal_h)
    p_d = max(0.01, p_d + cal_d)
    p_a = max(0.01, p_a + cal_a)
    s = p_h + p_d + p_a
    return p_h / s, p_d / s, p_a / s


# ---------------------------------------------------------------------------
# Fit + predict (high-level)
# ---------------------------------------------------------------------------

def fit_model(history: pd.DataFrame) -> FittedModel:
    """Fit complete model: team strengths + DC rho + cote calibration."""
    strengths, mu_h, mu_a = compute_strengths(history)
    rho = estimate_rho(history, strengths, mu_h, mu_a)
    cal_h, cal_d, cal_a = estimate_calibration(history)
    return FittedModel(
        strengths=strengths, mu_h=mu_h, mu_a=mu_a, rho=rho,
        cal_h=cal_h, cal_d=cal_d, cal_a=cal_a, n_train=len(history),
    )


def predict_match(
    model: FittedModel, team_a: str, team_b: str,
    odds_home: float, odds_draw: float, odds_away: float,
) -> dict[str, Any]:
    """Return a dict with both Poisson and Cote predictions."""
    out: dict[str, Any] = {
        "team_a": team_a, "team_b": team_b,
        "odds_home": odds_home, "odds_draw": odds_draw, "odds_away": odds_away,
    }

    sh = model.strengths.get(team_a)
    sa = model.strengths.get(team_b)
    if sh and sa:
        lam_h = sh["attack"] * sa["defense"] * model.mu_h
        lam_a = sa["attack"] * sh["defense"] * model.mu_a
        p_h_p, p_d_p, p_a_p, score_pred, _ = poisson_probs(lam_h, lam_a, model.rho)
        out.update({
            "lam_h": lam_h, "lam_a": lam_a,
            "p_h_pois": p_h_p, "p_d_pois": p_d_p, "p_a_pois": p_a_p,
            "score_pois": score_pred,
            "attack_diff": sh["attack"] - sa["attack"],
            "team_a_n": sh["n"], "team_b_n": sa["n"],
        })
    else:
        out.update({
            "lam_h": None, "lam_a": None,
            "p_h_pois": None, "p_d_pois": None, "p_a_pois": None,
            "score_pois": None, "attack_diff": None,
            "team_a_n": 0, "team_b_n": 0,
        })

    p_h_c, p_d_c, p_a_c = cote_probs(
        odds_home, odds_draw, odds_away,
        model.cal_h, model.cal_d, model.cal_a,
    )
    out.update({"p_h_cote": p_h_c, "p_d_cote": p_d_c, "p_a_cote": p_a_c})
    return out
