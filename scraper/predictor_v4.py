"""Predictor V4 — V3 + signal H2H pour stratégie de double pari sur les X probables.

Strategy :
  - Primary pick = V3 argmax (87.5% acc sur p>=70%, robuste)
  - Secondary X pick recommended si :
      (a) Paire a >= 5 historiques ET >= 30% de nuls passes
      OU
      (b) V3 X-aware se declenche (p_X >= 27% ET match equilibre)
  - Output : primary_pick + x_recommended (booleen)

Le user place :
  - 1 unite sur le primary pick (toujours)
  - 0.5 ou 1 unite supplementaire sur X si x_recommended (couverture)

Backtest a montre :
  - V3 pur : ROI -8.3%
  - H2H+Poisson : ROI -1.5% (mais accuracy lower)
  - V4 hybride attendu : ROI > 0 grace aux X picks ciblesreuf.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sqlalchemy.engine import Engine

from scraper.predictor_v3 import FittedModelV3, fit_model_v3, predict_match_v3


@dataclass
class FittedModelV4(FittedModelV3):
    h2h_stats: dict[tuple[str, str], dict] = field(default_factory=dict)
    h2h_min_n: int = 5
    h2h_x_threshold: float = 0.30


def compute_h2h_from_history(history: pd.DataFrame) -> dict:
    """Pre-compute outcomes per fixture pair."""
    pairs = defaultdict(lambda: {"n": 0, "1": 0, "X": 0, "2": 0})
    for r in history.itertuples():
        key = (r.team_a, r.team_b)
        pairs[key]["n"] += 1
        outcome = "1" if r.score_a > r.score_b else ("X" if r.score_a == r.score_b else "2")
        pairs[key][outcome] += 1
    return dict(pairs)


def fit_model_v4(
    history: pd.DataFrame,
    engine: Engine | None = None,
    h2h_min_n: int = 5,
    h2h_x_threshold: float = 0.30,
    **kwargs,
) -> FittedModelV4:
    """Fit V3 + add H2H stats lookup."""
    v3 = fit_model_v3(history, engine=engine, **kwargs)
    h2h = compute_h2h_from_history(history)
    return FittedModelV4(
        **{k: v for k, v in vars(v3).items() if k != "config"},
        config=v3.config,
        h2h_stats=h2h,
        h2h_min_n=h2h_min_n,
        h2h_x_threshold=h2h_x_threshold,
    )


def predict_match_v4(
    model: FittedModelV4, team_a: str, team_b: str,
    odds_home: float, odds_draw: float, odds_away: float,
    extra_markets: dict | str | None = None,
) -> dict[str, Any]:
    """V4 prediction : V3 + signal X-recommendation."""
    base = predict_match_v3(
        model, team_a, team_b,
        odds_home, odds_draw, odds_away,
        extra_markets=extra_markets,
    )

    # Primary pick from V3 argmax
    if base.get("p_h_blend") is not None:
        probs = [("1", base["p_h_blend"]), ("X", base["p_d_blend"]), ("2", base["p_a_blend"])]
        primary_pick = max(probs, key=lambda x: x[1])[0]
        primary_p = max(probs, key=lambda x: x[1])[1]
    else:
        primary_pick = "1"; primary_p = 0.0

    # Check H2H signal
    h2h_x_rate = None
    h2h_n = 0
    pair_stats = model.h2h_stats.get((team_a, team_b))
    h2h_recommends_x = False
    if pair_stats and pair_stats["n"] >= model.h2h_min_n:
        h2h_n = pair_stats["n"]
        h2h_x_rate = pair_stats["X"] / pair_stats["n"]
        if h2h_x_rate >= model.h2h_x_threshold:
            h2h_recommends_x = True

    # Check X-aware rule
    x_aware = base.get("pick_xaware") == "X"

    # Combined X recommendation
    x_recommended = h2h_recommends_x or x_aware
    x_reason = []
    if h2h_recommends_x: x_reason.append(f"H2H n={h2h_n} X={h2h_x_rate:.0%}")
    if x_aware: x_reason.append("X-aware balanced match")

    base.update({
        "primary_pick": primary_pick,
        "primary_p": primary_p,
        "x_recommended": x_recommended,
        "x_reason": "; ".join(x_reason) if x_reason else None,
        "h2h_n": h2h_n,
        "h2h_x_rate": h2h_x_rate,
    })
    return base
