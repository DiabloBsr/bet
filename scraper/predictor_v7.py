"""Predictor V7 — synthèse finale : value-betting omni-directionnel.

V7 ne suit PAS le favori par défaut et ne le combat PAS non plus.
Il calcule l'EV de CHAQUE issue possible et recommande UNIQUEMENT celles avec value.

Signaux validés 5-fold out-of-sample :

  1X2 main (corrections calibration bookmaker)
  ──────────────────────────────────────────
    SIGNAL A : HOME cote 1.7-1.8 (p_market 0.55-0.60)
      → Bookmaker dit 57.1%, réel 64.3%.  Parier 1.  ROI +6.08%  (n=378)
    SIGNAL B : HOME cote 4.0-5.0 (outsider home modéré)
      → Bookmaker dit 22.4%, réel 24.4%.  Parier 1.  ROI +6.45%  (n=141)
    SIGNAL C : AWAY cote 2.5-2.9 (semi-favori away)
      → Bookmaker dit 37.2%, réel 40.9%.  Parier 2.  ROI +3.33%  (n=257)

  Marchés exotiques (V6)
  ─────────────────────
    SIGNAL D : HT/FT 1/2 (away revient au score)  ROI +104%
    SIGNAL E : HT/FT 2/1 (home revient)            ROI  +28%
    SIGNAL F : Score 1-0 quand away favori         ROI  +42%

Philosophie : 'tout est possible' — V7 n'écarte aucune issue. Il identifie
juste celle(s) qui ont positive EV après comparaison avec la BDD réelle.
"""
from __future__ import annotations
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class FittedModelV7:
    # 1X2 calibration corrections : p_market_bucket -> p_actual
    p_h_calibration: dict   # p_market_h bucket → p_actual home win
    p_a_calibration: dict   # p_market_a bucket → p_actual away win
    p_x_calibration: dict
    # V6 signals (héritage)
    p_12_global: float
    p_21_global: float
    p_12_by_pair_bucket: dict
    p_21_by_pair_bucket: dict
    p_12_by_home_bucket: dict
    p_21_by_away_bucket: dict
    p_1_0_when_away_fav: float
    p_1_0_by_away_bucket: dict
    n_train: int


def _bucket_p(p, edges=None):
    """Bucket p_market 1X2 par tranches de 5%."""
    if edges is None:
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


def fit_model_v7(history: pd.DataFrame) -> FittedModelV7:
    """history : odds_home, odds_draw, odds_away, score_a, score_b, ht_score_a, ht_score_b."""
    h = history.copy()
    h["ft_o"] = np.where(h.score_a > h.score_b, "1",
                  np.where(h.score_a == h.score_b, "X", "2"))
    h["ht_o"] = np.where(h.ht_score_a > h.ht_score_b, "1",
                  np.where(h.ht_score_a == h.ht_score_b, "X", "2"))
    inv_sum = 1/h.odds_home + 1/h.odds_draw + 1/h.odds_away
    h["p_1"] = (1/h.odds_home) / inv_sum
    h["p_x"] = (1/h.odds_draw) / inv_sum
    h["p_2"] = (1/h.odds_away) / inv_sum

    # 1X2 calibration : pour chaque bucket de p_market, calculer p_actual
    p_h_cal, p_a_cal, p_x_cal = {}, {}, {}
    edges = [0, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
             0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 1.01]
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i+1]
        key = f"[{lo:.2f};{hi:.2f}]"
        sub_h = h[(h.p_1 >= lo) & (h.p_1 < hi)]
        if len(sub_h) >= 80:
            p_h_cal[key] = float((sub_h.ft_o == "1").mean())
        sub_a = h[(h.p_2 >= lo) & (h.p_2 < hi)]
        if len(sub_a) >= 80:
            p_a_cal[key] = float((sub_a.ft_o == "2").mean())
        sub_x = h[(h.p_x >= lo) & (h.p_x < hi)]
        if len(sub_x) >= 80:
            p_x_cal[key] = float((sub_x.ft_o == "X").mean())

    # V6 signals
    won_12 = (h.ht_o == "1") & (h.ft_o == "2")
    won_21 = (h.ht_o == "2") & (h.ft_o == "1")
    p_12_global = float(won_12.mean())
    p_21_global = float(won_21.mean())

    p_12_by_home_bucket, p_21_by_away_bucket = {}, {}
    for bk in ["<1.3", "<1.5", "<1.8", "<2.1", "<2.5", "<3.0", "<4.0", "<6.0", "6+"]:
        m1 = h.odds_home.apply(_bucket_ft) == bk
        if m1.sum() >= 100:
            p_12_by_home_bucket[bk] = float(won_12[m1].mean())
        else:
            p_12_by_home_bucket[bk] = p_12_global
        m2 = h.odds_away.apply(_bucket_ft) == bk
        if m2.sum() >= 100:
            p_21_by_away_bucket[bk] = float(won_21[m2].mean())
        else:
            p_21_by_away_bucket[bk] = p_21_global

    p_12_by_pair_bucket, p_21_by_pair_bucket = {}, {}
    pc = defaultdict(lambda: {"n": 0, "won_12": 0, "won_21": 0})
    for r in h.itertuples():
        k = (_bucket_ft(r.odds_home), _bucket_ft(r.odds_away))
        pc[k]["n"] += 1
        if r.ht_o == "1" and r.ft_o == "2": pc[k]["won_12"] += 1
        if r.ht_o == "2" and r.ft_o == "1": pc[k]["won_21"] += 1
    for k, s in pc.items():
        if s["n"] >= 30:
            p_12_by_pair_bucket[k] = s["won_12"] / s["n"]
            p_21_by_pair_bucket[k] = s["won_21"] / s["n"]

    fav_away = h[h.odds_away < 1.7]
    p_1_0_when_away_fav = float(((fav_away.score_a == 1) & (fav_away.score_b == 0)).mean()) if len(fav_away) > 0 else 0.0
    p_1_0_by_away_bucket = {}
    for bk in ["<1.3", "<1.5", "<1.8"]:
        m = h.odds_away.apply(_bucket_ft) == bk
        if m.sum() >= 50:
            p_1_0_by_away_bucket[bk] = float(((h.score_a == 1) & (h.score_b == 0))[m].mean())

    return FittedModelV7(
        p_h_calibration=p_h_cal,
        p_a_calibration=p_a_cal,
        p_x_calibration=p_x_cal,
        p_12_global=p_12_global,
        p_21_global=p_21_global,
        p_12_by_pair_bucket=p_12_by_pair_bucket,
        p_21_by_pair_bucket=p_21_by_pair_bucket,
        p_12_by_home_bucket=p_12_by_home_bucket,
        p_21_by_away_bucket=p_21_by_away_bucket,
        p_1_0_when_away_fav=p_1_0_when_away_fav,
        p_1_0_by_away_bucket=p_1_0_by_away_bucket,
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


def predict_v7(model: FittedModelV7, odds_home: float, odds_draw: float, odds_away: float,
                extra_markets: dict | str | None = None) -> dict[str, Any]:
    """Prédit p_modèle et EV pour TOUTES les issues."""
    if isinstance(extra_markets, str):
        try: extra_markets = json.loads(extra_markets)
        except Exception: extra_markets = None
    em = extra_markets if isinstance(extra_markets, dict) else {}

    inv_sum = 1/odds_home + 1/odds_draw + 1/odds_away
    p_market_h = (1/odds_home) / inv_sum
    p_market_x = (1/odds_draw) / inv_sum
    p_market_a = (1/odds_away) / inv_sum

    # Calibration corrigée (raw, indépendante par issue)
    p_raw_h = model.p_h_calibration.get(_bucket_p(p_market_h), p_market_h)
    p_raw_x = model.p_x_calibration.get(_bucket_p(p_market_x), p_market_x)
    p_raw_a = model.p_a_calibration.get(_bucket_p(p_market_a), p_market_a)
    # EV : utiliser les probas raw (indépendantes par issue, vrais signaux)
    ev_1 = p_raw_h * odds_home - 1
    ev_x = p_raw_x * odds_draw - 1
    ev_2 = p_raw_a * odds_away - 1
    # Probas pour affichage : normalisées pour cohérence somme=1
    total = p_raw_h + p_raw_x + p_raw_a
    p_model_h = p_raw_h / total if total > 0 else p_raw_h
    p_model_x = p_raw_x / total if total > 0 else p_raw_x
    p_model_a = p_raw_a / total if total > 0 else p_raw_a

    # V6 signaux exotiques
    bh, ba = _bucket_ft(odds_home), _bucket_ft(odds_away)
    cote_12 = _get(em, "HT/FT", "1/2")
    p_12 = (model.p_12_by_pair_bucket.get((bh, ba))
            or model.p_12_by_home_bucket.get(bh)
            or model.p_12_global)
    ev_12 = (p_12 * cote_12 - 1) if cote_12 else None

    cote_21 = _get(em, "HT/FT", "2/1")
    p_21 = (model.p_21_by_pair_bucket.get((bh, ba))
            or model.p_21_by_away_bucket.get(ba)
            or model.p_21_global)
    ev_21 = (p_21 * cote_21 - 1) if cote_21 else None

    cote_1_0 = _get(em, "Score exact", "1-0")
    ev_1_0 = None; p_1_0 = None
    if cote_1_0 and odds_away < 1.7:
        p_1_0 = model.p_1_0_by_away_bucket.get(ba) or model.p_1_0_when_away_fav
        ev_1_0 = p_1_0 * cote_1_0 - 1

    return {
        # 1X2 main
        "p_market": {"1": p_market_h, "X": p_market_x, "2": p_market_a},
        "p_model":  {"1": p_model_h,  "X": p_model_x,  "2": p_model_a},
        "ev_1x2":   {"1": ev_1, "X": ev_x, "2": ev_2},
        "cotes":    {"1": odds_home, "X": odds_draw, "2": odds_away},
        # Marchés exotiques
        "exotics": {
            "HT/FT 1/2": {"cote": cote_12, "p_emp": p_12, "ev": ev_12},
            "HT/FT 2/1": {"cote": cote_21, "p_emp": p_21, "ev": ev_21},
            "Score 1-0 (upset)": {"cote": cote_1_0, "p_emp": p_1_0, "ev": ev_1_0},
        },
    }
