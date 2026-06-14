"""Predictor V2 — ameliorations sans overfit.

Differences vs V1 :
  1. Forces SEPAREES home/away par equipe.
     attack_home / defense_home / attack_away / defense_away.
     Le moteur peut traiter une equipe forte a domicile mais faible exterieur.

  2. Time-decay optionnel (parametre half_life en nombre de matchs).
     Sans decay (defaut), tous les matchs pesent pareil.
     Avec decay, les matchs anciens decroissent exponentiellement.

  3. Score exact via le marche `Score exact` du bookmaker.
     Le bookmaker propose ~28 cotes par match pour chaque (h-a).
     On le devig pour avoir la distribution implicite, puis on blend
     avec la grille Poisson+DC. Pondere par `score_market_weight`.

  4. 1X2 derive de la grille de scores blendee (au lieu de cotes 1X2 separees).
     Plus precis car on agrege l'info de tous les marches.

Aucun hyperparametre n'est tune sur le test : valeurs par defaut conservatrices.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson


@dataclass
class FittedModelV2:
    home_strengths: dict[str, dict[str, float]]
    away_strengths: dict[str, dict[str, float]]
    mu_h: float
    mu_a: float
    rho: float
    cal_h: float
    cal_d: float
    cal_a: float
    n_train: int
    half_life: float | None = None
    score_market_weight: float = 0.5
    # Mode lookup tier-bucket -> liste de (score, frequency) pour fallback no-market
    tier_bucket_scores: dict[str, list[tuple[str, float]]] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Home/Away separated strengths
# ---------------------------------------------------------------------------

def compute_home_away_strengths(
    history: pd.DataFrame, smoothing: int = 5, half_life: float | None = None,
) -> tuple[dict, dict, float, float]:
    """Return (home_strengths, away_strengths, mu_h, mu_a).

    home_strengths[team]    = {attack, defense, n} pour les matchs ou team joue a domicile
    away_strengths[team]    = {attack, defense, n} pour les matchs ou team joue a l'exterieur

    Si half_life fourni, les matchs sont ponderes par exp(-rank/half_life).
    rank = 0 pour le match le plus recent, croissant pour les plus anciens.
    """
    # Assume history is ordered chronologically (oldest first)
    n = len(history)
    if half_life is not None and half_life > 0:
        ranks = np.arange(n - 1, -1, -1)  # most recent = rank 0
        weights = np.exp(-ranks / half_life)
    else:
        weights = np.ones(n)

    home_t = defaultdict(lambda: {"gf": 0.0, "ga": 0.0, "n": 0.0})
    away_t = defaultdict(lambda: {"gf": 0.0, "ga": 0.0, "n": 0.0})

    weighted_home_goals = 0.0
    weighted_away_goals = 0.0
    sum_w = 0.0

    for w, r in zip(weights, history.itertuples()):
        sa, sb = float(r.score_a), float(r.score_b)
        home_t[r.team_a]["gf"] += sa * w
        home_t[r.team_a]["ga"] += sb * w
        home_t[r.team_a]["n"] += w
        away_t[r.team_b]["gf"] += sb * w
        away_t[r.team_b]["ga"] += sa * w
        away_t[r.team_b]["n"] += w
        weighted_home_goals += sa * w
        weighted_away_goals += sb * w
        sum_w += w

    mu_h = weighted_home_goals / sum_w
    mu_a = weighted_away_goals / sum_w

    home_strengths = {}
    for name, s in home_t.items():
        if s["n"] < 1:
            continue
        # Bayesian shrinkage : on prior = league average (1.0)
        attack = ((s["gf"] + smoothing * mu_h) / (s["n"] + smoothing)) / mu_h
        defense = ((s["ga"] + smoothing * mu_a) / (s["n"] + smoothing)) / mu_a
        home_strengths[name] = {"attack": attack, "defense": defense, "n": s["n"]}

    away_strengths = {}
    for name, s in away_t.items():
        if s["n"] < 1:
            continue
        attack = ((s["gf"] + smoothing * mu_a) / (s["n"] + smoothing)) / mu_a
        defense = ((s["ga"] + smoothing * mu_h) / (s["n"] + smoothing)) / mu_h
        away_strengths[name] = {"attack": attack, "defense": defense, "n": s["n"]}

    return home_strengths, away_strengths, mu_h, mu_a


# ---------------------------------------------------------------------------
# Dixon-Coles
# ---------------------------------------------------------------------------

def _dc_tau(h: int, a: int, lam_h: float, lam_a: float, rho: float) -> float:
    if h == 0 and a == 0:
        return 1.0 - lam_h * lam_a * rho
    if h == 0 and a == 1:
        return 1.0 + lam_h * rho
    if h == 1 and a == 0:
        return 1.0 + lam_a * rho
    if h == 1 and a == 1:
        return 1.0 - rho
    return 1.0


def estimate_rho_v2(
    history: pd.DataFrame, home_s: dict, away_s: dict, mu_h: float, mu_a: float,
) -> float:
    """ML rho with home/away separated strengths."""
    rows = []
    for r in history.itertuples():
        sh = home_s.get(r.team_a); sa = away_s.get(r.team_b)
        if sh is None or sa is None:
            continue
        lam_h = sh["attack"] * sa["defense"] * mu_h
        lam_a = sa["attack"] * sh["defense"] * mu_a
        rows.append((int(r.score_a), int(r.score_b), lam_h, lam_a))

    def neg_log_lik(rho: float) -> float:
        ll = 0.0
        for h, a, lh, la in rows:
            p = poisson.pmf(h, lh) * poisson.pmf(a, la)
            tau = _dc_tau(h, a, lh, la, rho)
            if p > 0 and tau > 0:
                ll += np.log(p * tau)
        return -ll

    return float(minimize_scalar(neg_log_lik, bounds=(-0.3, 0.3), method="bounded").x)


# ---------------------------------------------------------------------------
# Score grid (Poisson + DC + market blend)
# ---------------------------------------------------------------------------

def poisson_score_grid(lam_h: float, lam_a: float, rho: float, max_goals: int = 8) -> np.ndarray:
    grid = np.zeros((max_goals, max_goals))
    for h in range(max_goals):
        for a in range(max_goals):
            grid[h, a] = poisson.pmf(h, lam_h) * poisson.pmf(a, lam_a) * _dc_tau(h, a, lam_h, lam_a, rho)
    grid = np.clip(grid, 0, None)
    total = grid.sum()
    if total > 0:
        grid /= total
    return grid


def market_score_grid(score_exact_market: dict | None, max_goals: int = 8) -> np.ndarray | None:
    """Convert the bookmaker's Score exact market into a vig-free probability grid.

    Le marche envoie : {'0-0': cote, '0-1': cote, ...}.
    On extrait p_imp = 1/cote pour chaque score connu, on normalise.
    Si un score n'est pas liste, on lui donne 0.
    """
    if not isinstance(score_exact_market, dict) or not score_exact_market:
        return None
    grid = np.zeros((max_goals, max_goals))
    for k, cote in score_exact_market.items():
        try:
            h_str, a_str = str(k).split("-")
            h, a = int(h_str.strip()), int(a_str.strip())
            if 0 <= h < max_goals and 0 <= a < max_goals and cote > 0:
                grid[h, a] = 1.0 / float(cote)
        except (ValueError, AttributeError):
            continue
    total = grid.sum()
    if total <= 0:
        return None
    grid /= total
    return grid


def blended_score_grid(
    lam_h: float, lam_a: float, rho: float,
    score_market: dict | None,
    market_weight: float = 0.5,
    max_goals: int = 8,
) -> np.ndarray:
    """Combine Poisson grid + market grid. Si market absent, Poisson seul."""
    p_grid = poisson_score_grid(lam_h, lam_a, rho, max_goals)
    m_grid = market_score_grid(score_market, max_goals)
    if m_grid is None:
        return p_grid
    blended = market_weight * m_grid + (1 - market_weight) * p_grid
    blended /= blended.sum()
    return blended


# ---------------------------------------------------------------------------
# 1X2 from grid + cote calibration
# ---------------------------------------------------------------------------

def grid_to_1x2(grid: np.ndarray) -> tuple[float, float, float]:
    n = grid.shape[0]
    p_h = float(sum(grid[h, a] for h in range(n) for a in range(n) if h > a))
    p_d = float(sum(grid[h, a] for h in range(n) for a in range(n) if h == a))
    p_a = float(sum(grid[h, a] for h in range(n) for a in range(n) if h < a))
    return p_h, p_d, p_a


def grid_modal_score(grid: np.ndarray) -> str:
    idx = np.unravel_index(grid.argmax(), grid.shape)
    return f"{idx[0]}-{idx[1]}"


def grid_top_k_scores(grid: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
    """Return top-k most likely scores with their probabilities."""
    n = grid.shape[0]
    flat = grid.flatten()
    top_idx = np.argsort(flat)[-k:][::-1]
    return [(f"{i // n}-{i % n}", float(flat[i])) for i in top_idx]


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
# Fit + predict
# ---------------------------------------------------------------------------

def _tier_bucket(d: float) -> str:
    if d < -0.4: return "away++"
    if d < -0.15: return "away+"
    if d < 0.15: return "even"
    if d < 0.4: return "home+"
    return "home++"


def _build_tier_bucket_scores(
    history: pd.DataFrame, home_s: dict, away_s: dict,
) -> dict[str, list[tuple[str, float]]]:
    """Pour chaque bucket de tier mismatch, retourne le top-3 scores empiriques."""
    from collections import Counter, defaultdict
    buckets = defaultdict(list)
    for r in history.itertuples():
        sh = home_s.get(r.team_a); sa = away_s.get(r.team_b)
        if sh is None or sa is None:
            continue
        d = sh["attack"] - sa["attack"]
        buckets[_tier_bucket(d)].append(f"{int(r.score_a)}-{int(r.score_b)}")
    out = {}
    for k, scores in buckets.items():
        n = len(scores)
        c = Counter(scores)
        top = [(s, c[s] / n) for s, _ in c.most_common(3)]
        out[k] = top
    return out


def fit_model_v2(
    history: pd.DataFrame,
    smoothing: int = 5,
    half_life: float | None = None,
    score_market_weight: float = 0.5,
) -> FittedModelV2:
    home_s, away_s, mu_h, mu_a = compute_home_away_strengths(history, smoothing, half_life)
    rho = estimate_rho_v2(history, home_s, away_s, mu_h, mu_a)
    cal_h, cal_d, cal_a = estimate_calibration(history)
    tier_scores = _build_tier_bucket_scores(history, home_s, away_s)
    return FittedModelV2(
        home_strengths=home_s, away_strengths=away_s,
        mu_h=mu_h, mu_a=mu_a, rho=rho,
        cal_h=cal_h, cal_d=cal_d, cal_a=cal_a,
        n_train=len(history),
        half_life=half_life,
        score_market_weight=score_market_weight,
        tier_bucket_scores=tier_scores,
        config={"smoothing": smoothing},
    )


def predict_match_v2(
    model: FittedModelV2, team_a: str, team_b: str,
    odds_home: float, odds_draw: float, odds_away: float,
    score_exact_market: dict | str | None = None,
) -> dict[str, Any]:
    """Predict with Poisson+DC, market score blend, and cote calibration.

    score_exact_market : le dict du marche `Score exact` (ou JSON-string), ou None.
    """
    if isinstance(score_exact_market, str):
        try:
            score_exact_market = json.loads(score_exact_market)
        except (ValueError, json.JSONDecodeError):
            score_exact_market = None
    # Si on passe l'extra_markets complet (dict avec plusieurs marches),
    # on extrait le sous-marche "Score exact".
    if isinstance(score_exact_market, dict) and "Score exact" in score_exact_market:
        score_exact_market = score_exact_market.get("Score exact")

    # Helpers internes pour le pick edge-max + edge par outcome
    def _picks_with_edges(p_h, p_d, p_a, oh, od, oa):
        edges = [("1", p_h, oh, p_h * oh - 1),
                 ("X", p_d, od, p_d * od - 1),
                 ("2", p_a, oa, p_a * oa - 1)]
        argmax_pick = max(edges, key=lambda x: x[1])  # by probability
        edgemax_pick = max(edges, key=lambda x: x[3])  # by expected value
        return argmax_pick, edgemax_pick, edges

    out: dict[str, Any] = {
        "team_a": team_a, "team_b": team_b,
        "odds_home": odds_home, "odds_draw": odds_draw, "odds_away": odds_away,
    }

    sh = model.home_strengths.get(team_a)
    sa = model.away_strengths.get(team_b)

    if sh and sa:
        lam_h = sh["attack"] * sa["defense"] * model.mu_h
        lam_a = sa["attack"] * sh["defense"] * model.mu_a

        # Poisson seul
        grid_pois = poisson_score_grid(lam_h, lam_a, model.rho)
        p_h_pois, p_d_pois, p_a_pois = grid_to_1x2(grid_pois)
        score_pois = grid_modal_score(grid_pois)

        # Marche seul
        grid_market = market_score_grid(score_exact_market)
        if grid_market is not None:
            p_h_market, p_d_market, p_a_market = grid_to_1x2(grid_market)
            score_market_pick = grid_modal_score(grid_market)
        else:
            p_h_market = p_d_market = p_a_market = None
            score_market_pick = None

        # Blended (Poisson + Market)
        grid_blend = blended_score_grid(
            lam_h, lam_a, model.rho, score_exact_market, model.score_market_weight,
        )
        p_h_bl, p_d_bl, p_a_bl = grid_to_1x2(grid_blend)
        score_blend = grid_modal_score(grid_blend)
        top3_blend = grid_top_k_scores(grid_blend, 3)
        top3_pois = grid_top_k_scores(grid_pois, 3)
        top3_market = grid_top_k_scores(grid_market, 3) if grid_market is not None else None

        # Edge-max pick (Option A : maximise p*cote - 1 au lieu de p seul)
        _, edgemax_blend, _ = _picks_with_edges(p_h_bl, p_d_bl, p_a_bl,
                                                  odds_home, odds_draw, odds_away)
        out.update({
            "lam_h": lam_h, "lam_a": lam_a,
            "p_h_pois": p_h_pois, "p_d_pois": p_d_pois, "p_a_pois": p_a_pois,
            "p_h_market": p_h_market, "p_d_market": p_d_market, "p_a_market": p_a_market,
            "p_h_blend": p_h_bl, "p_d_blend": p_d_bl, "p_a_blend": p_a_bl,
            "score_pois": score_pois,
            "score_market": score_market_pick,
            "score_blend": score_blend,
            "top3_blend": top3_blend,
            "top3_pois": top3_pois,
            "top3_market": top3_market,
            "pick_edgemax": edgemax_blend[0],
            "p_edgemax": edgemax_blend[1],
            "edge_edgemax": edgemax_blend[3],
            "attack_diff": sh["attack"] - sa["attack"],
            "team_a_n": sh["n"], "team_b_n": sa["n"],
        })
    else:
        out.update({k: None for k in [
            "lam_h", "lam_a", "p_h_pois", "p_d_pois", "p_a_pois",
            "p_h_market", "p_d_market", "p_a_market",
            "p_h_blend", "p_d_blend", "p_a_blend",
            "score_pois", "score_market", "score_blend",
            "top3_blend", "top3_pois", "top3_market",
            "pick_edgemax", "p_edgemax", "edge_edgemax",
            "attack_diff",
        ]})
        out.update({"team_a_n": 0, "team_b_n": 0})

    p_h_c, p_d_c, p_a_c = cote_probs(
        odds_home, odds_draw, odds_away, model.cal_h, model.cal_d, model.cal_a,
    )
    out.update({"p_h_cote": p_h_c, "p_d_cote": p_d_c, "p_a_cote": p_a_c})
    return out
