"""Predictor V8 — synthèse finale avec cote movement + portfolio reasoning.

Signaux validés (héritage V7) :
  1X2 main : HOME 1.7-1.8 (+6%), HOME 4-5 (+6%), AWAY 2.5-2.9 (+3%)
  Exotiques : HT/FT 1/2 (+104%), 2/1 (+28%), Score 1-0 upset (+42%)

NOUVEAU V8 :
  Cote movement : Δ_away baisse (smart money), Δ_X monte → boosts EV
"""
from __future__ import annotations
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class FittedModelV8:
    p_h_calibration: dict
    p_a_calibration: dict
    p_x_calibration: dict
    p_12_global: float; p_21_global: float
    p_12_by_pair_bucket: dict; p_21_by_pair_bucket: dict
    p_12_by_home_bucket: dict; p_21_by_away_bucket: dict
    p_1_0_when_away_fav: float
    p_1_0_by_away_bucket: dict
    # NEW : cote movement signals (boost in pp)
    movement_boosts: dict = field(default_factory=dict)
    n_train: int = 0


def _bucket_p(p):
    edges = [0, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
             0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 1.01]
    for i in range(len(edges) - 1):
        if edges[i] <= p < edges[i+1]:
            return f"[{edges[i]:.2f};{edges[i+1]:.2f}]"
    return "[0.95;1.00]"


def _bucket_ft(c):
    if c < 1.3: return "<1.3"
    if c < 1.5: return "<1.5"
    if c < 1.8: return "<1.8"
    if c < 2.1: return "<2.1"
    if c < 2.5: return "<2.5"
    if c < 3.0: return "<3.0"
    if c < 4.0: return "<4.0"
    if c < 6.0: return "<6.0"
    return "6+"


def fit_model_v8(history: pd.DataFrame, movement_history: pd.DataFrame | None = None) -> FittedModelV8:
    h = history.copy()
    h["ft_o"] = np.where(h.score_a > h.score_b, "1",
                  np.where(h.score_a == h.score_b, "X", "2"))
    h["ht_o"] = np.where(h.ht_score_a > h.ht_score_b, "1",
                  np.where(h.ht_score_a == h.ht_score_b, "X", "2"))
    inv_sum = 1/h.odds_home + 1/h.odds_draw + 1/h.odds_away
    h["p_1"] = (1/h.odds_home) / inv_sum
    h["p_x"] = (1/h.odds_draw) / inv_sum
    h["p_2"] = (1/h.odds_away) / inv_sum

    # Calibration 1X2
    p_h_cal, p_a_cal, p_x_cal = {}, {}, {}
    edges = [0, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
             0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 1.01]
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i+1]
        key = f"[{lo:.2f};{hi:.2f}]"
        sub_h = h[(h.p_1 >= lo) & (h.p_1 < hi)]
        if len(sub_h) >= 80: p_h_cal[key] = float((sub_h.ft_o == "1").mean())
        sub_a = h[(h.p_2 >= lo) & (h.p_2 < hi)]
        if len(sub_a) >= 80: p_a_cal[key] = float((sub_a.ft_o == "2").mean())
        sub_x = h[(h.p_x >= lo) & (h.p_x < hi)]
        if len(sub_x) >= 80: p_x_cal[key] = float((sub_x.ft_o == "X").mean())

    # V6 signals
    won_12 = (h.ht_o == "1") & (h.ft_o == "2")
    won_21 = (h.ht_o == "2") & (h.ft_o == "1")
    p_12_global = float(won_12.mean()); p_21_global = float(won_21.mean())
    p_12_by_home, p_21_by_away = {}, {}
    for bk in ["<1.3", "<1.5", "<1.8", "<2.1", "<2.5", "<3.0", "<4.0", "<6.0", "6+"]:
        m1 = h.odds_home.apply(_bucket_ft) == bk
        p_12_by_home[bk] = float(won_12[m1].mean()) if m1.sum() >= 100 else p_12_global
        m2 = h.odds_away.apply(_bucket_ft) == bk
        p_21_by_away[bk] = float(won_21[m2].mean()) if m2.sum() >= 100 else p_21_global

    p_12_pair, p_21_pair = {}, {}
    pc = defaultdict(lambda: {"n": 0, "won_12": 0, "won_21": 0})
    for r in h.itertuples():
        k = (_bucket_ft(r.odds_home), _bucket_ft(r.odds_away))
        pc[k]["n"] += 1
        if r.ht_o == "1" and r.ft_o == "2": pc[k]["won_12"] += 1
        if r.ht_o == "2" and r.ft_o == "1": pc[k]["won_21"] += 1
    for k, s in pc.items():
        if s["n"] >= 30:
            p_12_pair[k] = s["won_12"] / s["n"]
            p_21_pair[k] = s["won_21"] / s["n"]

    fav_away = h[h.odds_away < 1.7]
    p_1_0_when_away_fav = float(((fav_away.score_a == 1) & (fav_away.score_b == 0)).mean()) if len(fav_away) > 0 else 0.0
    p_1_0_by_away = {}
    for bk in ["<1.3", "<1.5", "<1.8"]:
        m = h.odds_away.apply(_bucket_ft) == bk
        if m.sum() >= 50:
            p_1_0_by_away[bk] = float(((h.score_a == 1) & (h.score_b == 0))[m].mean())

    # Cote movement boosts (empirique)
    # boost = ROI mesuré sur l'historique, à utiliser comme +pp à ajouter à p_model
    movement_boosts = {
        "away_drop": 0.10,   # +10pp boost si Δ_away ∈ [-0.20;-0.05]
        "x_rise": 0.08,      # +8pp boost si Δ_X ∈ [+0.05;+0.20]
        "home_rise": 0.02,   # +2pp si Δ_h ∈ [+0.05;+0.20] (marginal)
    }

    return FittedModelV8(
        p_h_calibration=p_h_cal, p_a_calibration=p_a_cal, p_x_calibration=p_x_cal,
        p_12_global=p_12_global, p_21_global=p_21_global,
        p_12_by_pair_bucket=p_12_pair, p_21_by_pair_bucket=p_21_pair,
        p_12_by_home_bucket=p_12_by_home, p_21_by_away_bucket=p_21_by_away,
        p_1_0_when_away_fav=p_1_0_when_away_fav,
        p_1_0_by_away_bucket=p_1_0_by_away,
        movement_boosts=movement_boosts,
        n_train=len(h),
    )


def _get(em, market, label):
    if isinstance(em, str):
        try: em = json.loads(em)
        except Exception: return None
    if not isinstance(em, dict): return None
    md = em.get(market)
    if not isinstance(md, dict): return None
    v = md.get(label)
    try: return float(v) if v is not None else None
    except Exception: return None


def predict_v8(model: FittedModelV8,
                odds_home: float, odds_draw: float, odds_away: float,
                extra_markets=None,
                first_odds_home: float | None = None,
                first_odds_draw: float | None = None,
                first_odds_away: float | None = None) -> dict:
    """Prédit avec calibration + cote movement boost si disponible."""
    if isinstance(extra_markets, str):
        try: extra_markets = json.loads(extra_markets)
        except Exception: extra_markets = None
    em = extra_markets if isinstance(extra_markets, dict) else {}

    inv_sum = 1/odds_home + 1/odds_draw + 1/odds_away
    p_market_h = (1/odds_home) / inv_sum
    p_market_x = (1/odds_draw) / inv_sum
    p_market_a = (1/odds_away) / inv_sum

    p_raw_h = model.p_h_calibration.get(_bucket_p(p_market_h), p_market_h)
    p_raw_x = model.p_x_calibration.get(_bucket_p(p_market_x), p_market_x)
    p_raw_a = model.p_a_calibration.get(_bucket_p(p_market_a), p_market_a)

    # Cote movement boost
    movement_info = None
    if first_odds_home is not None and first_odds_away is not None:
        delta_h = odds_home - first_odds_home
        delta_a = odds_away - first_odds_away
        delta_x = (odds_draw or 0) - (first_odds_draw or 0)
        movement_info = {"delta_h": delta_h, "delta_a": delta_a, "delta_x": delta_x,
                          "first": (first_odds_home, first_odds_draw, first_odds_away),
                          "last":  (odds_home, odds_draw, odds_away)}
        # Apply boosts
        if -0.20 <= delta_a < -0.05:
            p_raw_a = min(1.0, p_raw_a + model.movement_boosts["away_drop"])
            movement_info["boost"] = ("away_drop", model.movement_boosts["away_drop"])
        if 0.05 < delta_x <= 0.20:
            p_raw_x = min(1.0, p_raw_x + model.movement_boosts["x_rise"])
            movement_info["boost"] = ("x_rise", model.movement_boosts["x_rise"])
        if 0.05 < delta_h <= 0.20:
            p_raw_h = min(1.0, p_raw_h + model.movement_boosts["home_rise"])
            movement_info["boost"] = ("home_rise", model.movement_boosts["home_rise"])

    ev_1 = p_raw_h * odds_home - 1
    ev_x = p_raw_x * odds_draw - 1
    ev_2 = p_raw_a * odds_away - 1

    # Affichage normalisé
    total = p_raw_h + p_raw_x + p_raw_a
    p_model_h = p_raw_h / total if total > 0 else p_raw_h
    p_model_x = p_raw_x / total if total > 0 else p_raw_x
    p_model_a = p_raw_a / total if total > 0 else p_raw_a

    # Exotiques V6
    bh, ba = _bucket_ft(odds_home), _bucket_ft(odds_away)
    cote_12 = _get(em, "HT/FT", "1/2")
    p_12 = (model.p_12_by_pair_bucket.get((bh, ba))
            or model.p_12_by_home_bucket.get(bh) or model.p_12_global)
    ev_12 = (p_12 * cote_12 - 1) if cote_12 else None

    cote_21 = _get(em, "HT/FT", "2/1")
    p_21 = (model.p_21_by_pair_bucket.get((bh, ba))
            or model.p_21_by_away_bucket.get(ba) or model.p_21_global)
    ev_21 = (p_21 * cote_21 - 1) if cote_21 else None

    cote_1_0 = _get(em, "Score exact", "1-0")
    ev_1_0 = None; p_1_0 = None
    if cote_1_0 and odds_away < 1.7:
        p_1_0 = model.p_1_0_by_away_bucket.get(ba) or model.p_1_0_when_away_fav
        ev_1_0 = p_1_0 * cote_1_0 - 1

    return {
        "p_market": {"1": p_market_h, "X": p_market_x, "2": p_market_a},
        "p_model":  {"1": p_model_h,  "X": p_model_x,  "2": p_model_a},
        "ev_1x2":   {"1": ev_1, "X": ev_x, "2": ev_2},
        "cotes":    {"1": odds_home, "X": odds_draw, "2": odds_away},
        "movement": movement_info,
        "exotics": {
            "HT/FT 1/2": {"cote": cote_12, "p_emp": p_12, "ev": ev_12},
            "HT/FT 2/1": {"cote": cote_21, "p_emp": p_21, "ev": ev_21},
            "Score 1-0 (upset)": {"cote": cote_1_0, "p_emp": p_1_0, "ev": ev_1_0},
        },
    }
