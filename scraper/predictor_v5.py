"""Predictor V5 — etend V4 avec predictions HT, HT/FT, FTTS.

Nouvelles dimensions :
  1. HT 1X2 (Mi-tps 1X2) - 45% des matchs sont nuls a HT
  2. HT/FT (combo 9 cellules) - "X/1" arrive 19.6% du temps
  3. FTTS (premier marqueur) - home marque 1er -> 70.9% wins
  4. Premier but minute - distribution non-uniforme

Le modele combine :
  - V4 (Poisson FT + Multi-market + H2H)
  - HT Poisson (lambda HT ≈ 0.43 × lambda FT, calibre sur train)
  - HT/FT matrix empirique + Bayesian smoothing
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sqlalchemy.engine import Engine

from scraper.predictor_v4 import (
    FittedModelV4,
    fit_model_v4,
    predict_match_v4,
)
from scraper.predictor_v2 import _dc_tau  # Dixon-Coles helper


@dataclass
class FittedModelV5(FittedModelV4):
    ht_lambda_ratio: float = 0.43      # ratio buts HT / FT (empirique)
    ht_ft_matrix: dict | None = None    # matrice empirique HT->FT
    first_goal_home_rate: float = 0.61  # 912/(912+590) approx
    first_goal_when_home_wins: float = 0.709
    first_goal_when_away_wins: float = 0.539
    # Score distribution conditionnelle: { (cote_h_bucket, cote_a_bucket): {(h, a): freq} }
    empirical_score_dist: dict | None = None


def _cote_bucket(c):
    if c < 1.3: return "<1.3"
    if c < 1.5: return "<1.5"
    if c < 1.8: return "<1.8"
    if c < 2.1: return "<2.1"
    if c < 2.5: return "<2.5"
    if c < 3.0: return "<3.0"
    if c < 4.0: return "<4.0"
    if c < 6.0: return "<6.0"
    return "6+"


def compute_empirical_score_dist(history: "pd.DataFrame") -> dict:
    """Distribution empirique des scores conditionnellement au profil de cotes."""
    from collections import defaultdict
    buckets = defaultdict(lambda: defaultdict(int))
    for r in history.itertuples():
        oh, oa = r.odds_home, r.odds_away
        if oh is None or oa is None:
            continue
        key = (_cote_bucket(oh), _cote_bucket(oa))
        score_key = (int(r.score_a), int(r.score_b))
        buckets[key][score_key] += 1
    # normalize per bucket
    out = {}
    for key, sc_counter in buckets.items():
        total = sum(sc_counter.values())
        if total >= 5:  # min sample size
            out[key] = {s: c / total for s, c in sc_counter.items()}
    return out


def get_top_scores_blended(model: FittedModelV5, lam_h: float, lam_a: float,
                             odds_home: float, odds_away: float,
                             score_market: dict | None = None,
                             k: int = 5, max_goals: int = 8) -> list[tuple[str, float]]:
    """Top-K scores using Poisson + empirical + market blend.

    Returns list of (score_str, probability).
    """
    import numpy as np
    from scipy.stats import poisson as _pois

    # 1. Poisson grid with Dixon-Coles
    grid_p = np.zeros((max_goals, max_goals))
    for h in range(max_goals):
        for a in range(max_goals):
            grid_p[h, a] = _pois.pmf(h, lam_h) * _pois.pmf(a, lam_a) * _dc_tau(h, a, lam_h, lam_a, model.rho)
    grid_p = np.clip(grid_p, 0, None)
    grid_p /= grid_p.sum() if grid_p.sum() > 0 else 1

    # 2. Empirical grid (conditional on cote profile)
    grid_e = np.zeros((max_goals, max_goals))
    if model.empirical_score_dist:
        key = (_cote_bucket(odds_home), _cote_bucket(odds_away))
        emp = model.empirical_score_dist.get(key)
        if emp:
            for (h, a), p in emp.items():
                if 0 <= h < max_goals and 0 <= a < max_goals:
                    grid_e[h, a] = p
    if grid_e.sum() == 0:
        # fallback: global score distribution from training (using buckets close)
        for key, dist in (model.empirical_score_dist or {}).items():
            for (h, a), p in dist.items():
                if 0 <= h < max_goals and 0 <= a < max_goals:
                    grid_e[h, a] += p
        if grid_e.sum() > 0:
            grid_e /= grid_e.sum()

    # 3. Market grid
    grid_m = None
    if score_market:
        grid_m = np.zeros((max_goals, max_goals))
        for k_score, cote in (score_market or {}).items():
            try:
                h_str, a_str = str(k_score).split("-")
                h, a = int(h_str.strip()), int(a_str.strip())
                if 0 <= h < max_goals and 0 <= a < max_goals and cote > 0:
                    grid_m[h, a] = 1.0 / float(cote)
            except (ValueError, AttributeError):
                continue
        if grid_m.sum() > 0:
            grid_m /= grid_m.sum()
        else:
            grid_m = None

    # 4. Blend: 0.3 Poisson + 0.4 Empirical + 0.3 Market (or 0.5/0.5 if no market)
    if grid_m is not None:
        blend = 0.3 * grid_p + 0.4 * grid_e + 0.3 * grid_m
    elif grid_e.sum() > 0:
        blend = 0.4 * grid_p + 0.6 * grid_e
    else:
        blend = grid_p
    if blend.sum() > 0:
        blend /= blend.sum()

    # Top-K
    flat = [(h * max_goals + a, blend[h, a]) for h in range(max_goals) for a in range(max_goals)]
    flat.sort(key=lambda x: -x[1])
    top = []
    for idx, p in flat[:k]:
        h = idx // max_goals
        a = idx % max_goals
        top.append((f"{h}-{a}", float(p)))
    return top


def fit_model_v5(history: pd.DataFrame, ht_history: pd.DataFrame | None = None, **kwargs) -> FittedModelV5:
    """Fit V4 puis ajoute calibrations HT."""
    v4 = fit_model_v4(history, **kwargs)

    ht_lambda_ratio = 0.43
    ht_ft_matrix = None
    first_goal_home_rate = 0.61

    if ht_history is not None and len(ht_history) > 100:
        # Calibrate HT lambda ratio
        ht_total = ht_history["ht_score_a"] + ht_history["ht_score_b"]
        ft_total = ht_history["score_a"] + ht_history["score_b"]
        if ft_total.sum() > 0:
            ht_lambda_ratio = float(ht_total.sum() / ft_total.sum())

        # HT/FT empirical matrix
        ht_history = ht_history.copy()
        ht_history["ht_outcome"] = ht_history.apply(
            lambda r: "1" if r.ht_score_a > r.ht_score_b
            else ("X" if r.ht_score_a == r.ht_score_b else "2"), axis=1)
        ht_history["ft_outcome"] = ht_history.apply(
            lambda r: "1" if r.score_a > r.score_b
            else ("X" if r.score_a == r.score_b else "2"), axis=1)
        ct = pd.crosstab(ht_history["ht_outcome"], ht_history["ft_outcome"], normalize="index")
        ht_ft_matrix = {ht: {ft: float(ct.loc[ht, ft]) if ft in ct.columns else 0
                              for ft in ("1", "X", "2")} for ht in ct.index}

    # Empirical score distribution conditionnelle (cote_h, cote_a) -> top scores
    empirical_score_dist = compute_empirical_score_dist(history)

    return FittedModelV5(
        **{k: v for k, v in vars(v4).items() if k != "config"},
        config=v4.config,
        ht_lambda_ratio=ht_lambda_ratio,
        ht_ft_matrix=ht_ft_matrix,
        first_goal_home_rate=first_goal_home_rate,
        empirical_score_dist=empirical_score_dist,
    )


def predict_match_v5(model: FittedModelV5, team_a: str, team_b: str,
                      odds_home: float, odds_draw: float, odds_away: float,
                      extra_markets: dict | str | None = None) -> dict[str, Any]:
    """V5 prediction : ajoute HT et HT/FT."""
    base = predict_match_v4(model, team_a, team_b, odds_home, odds_draw, odds_away,
                              extra_markets=extra_markets)

    if isinstance(extra_markets, str):
        try: extra_markets = json.loads(extra_markets)
        except Exception: extra_markets = None
    em = extra_markets if isinstance(extra_markets, dict) else {}

    sh = model.home_strengths.get(team_a)
    sa = model.away_strengths.get(team_b)

    if sh and sa:
        lam_h = sh["attack"] * sa["defense"] * model.mu_h
        lam_a = sa["attack"] * sh["defense"] * model.mu_a

        # HT Poisson : lambdas HT = ratio × lambdas FT
        lam_h_ht = lam_h * model.ht_lambda_ratio
        lam_a_ht = lam_a * model.ht_lambda_ratio

        # HT 1X2 probabilities (Poisson + Dixon-Coles : +1.7pp acc OOS vs indépendant,
        # car les lambdas HT ~0.77/0.57 concentrent la masse sur 0-0/1-0/0-1/1-1)
        p_h_ht = p_d_ht = p_a_ht = 0.0
        for h in range(6):
            for a in range(6):
                p = (poisson.pmf(h, lam_h_ht) * poisson.pmf(a, lam_a_ht)
                     * _dc_tau(h, a, lam_h_ht, lam_a_ht, model.rho))
                if h > a: p_h_ht += p
                elif h == a: p_d_ht += p
                else: p_a_ht += p
        total = p_h_ht + p_d_ht + p_a_ht
        if total > 0:
            p_h_ht, p_d_ht, p_a_ht = p_h_ht/total, p_d_ht/total, p_a_ht/total

        # HT/FT : chain rule avec matrice empirique
        ht_ft_probs = {}
        if model.ht_ft_matrix:
            for ht_o, ht_p in [("1", p_h_ht), ("X", p_d_ht), ("2", p_a_ht)]:
                for ft_o, cond_p in model.ht_ft_matrix.get(ht_o, {}).items():
                    key = f"{ht_o}/{ft_o}"
                    ht_ft_probs[key] = ht_p * cond_p

        # HT pick = argmax
        ht_picks = [("1", p_h_ht), ("X", p_d_ht), ("2", p_a_ht)]
        ht_pick = max(ht_picks, key=lambda x: x[1])

        # HT/FT modal pick
        if ht_ft_probs:
            htft_modal = max(ht_ft_probs.items(), key=lambda kv: kv[1])
            htft_pick = htft_modal[0]
            htft_p = htft_modal[1]
        else:
            htft_pick = None
            htft_p = 0

        # FTTS estimation : qui marque le premier ?
        # P(home marque premier) ≈ lam_h / (lam_h + lam_a)
        if (lam_h + lam_a) > 0:
            p_ftts_home = lam_h / (lam_h + lam_a)
            p_ftts_away = lam_a / (lam_h + lam_a)
        else:
            p_ftts_home = 0.5
            p_ftts_away = 0.5
        # Plus P(no goal at all)
        p_no_goal = poisson.pmf(0, lam_h) * poisson.pmf(0, lam_a)
        p_ftts_home *= (1 - p_no_goal)
        p_ftts_away *= (1 - p_no_goal)

        # Enriched score predictions (top-5) using empirical + Poisson + market blend
        score_market = em.get("Score exact") if isinstance(em, dict) else None
        top5_scores = get_top_scores_blended(
            model, lam_h, lam_a,
            odds_home=odds_home, odds_away=odds_away,
            score_market=score_market, k=5,
        )

        base.update({
            "lam_h_ht": lam_h_ht, "lam_a_ht": lam_a_ht,
            "p_h_ht": p_h_ht, "p_d_ht": p_d_ht, "p_a_ht": p_a_ht,
            "ht_pick": ht_pick[0], "ht_p": ht_pick[1],
            "ht_ft_probs": ht_ft_probs,
            "htft_pick": htft_pick, "htft_p": htft_p,
            "p_ftts_home": p_ftts_home, "p_ftts_away": p_ftts_away,
            "p_no_goal": p_no_goal,
            "top5_scores_enriched": top5_scores,
            "score_enriched_modal": top5_scores[0][0] if top5_scores else None,
        })
    else:
        base.update({k: None for k in [
            "lam_h_ht", "lam_a_ht", "p_h_ht", "p_d_ht", "p_a_ht",
            "ht_pick", "ht_p", "ht_ft_probs", "htft_pick", "htft_p",
            "p_ftts_home", "p_ftts_away", "p_no_goal",
        ]})

    return base
