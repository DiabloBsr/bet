"""Predictor V6 — exploite les inefficiences du bookmaker (signaux indépendants de la cote 1X2).

Inefficiences validées 5-fold out-of-sample (n=2720) :

  SIGNAL 1 — HT/FT 1/2 (away revient au score)
    Bookmaker pense 1.8%, réel 3.4%. Cote méd 57. ROI +102.8%.

  SIGNAL 2 — HT/FT 2/1 (home revient au score)
    Bookmaker pense 2.9%, réel 3.9%. Cote méd 35. ROI +28.5%.

  SIGNAL 3 — UPSET 1-0 quand away est favori
    Bookmaker sous-estime home outsider gagne 1-0. ROI +22.1% (n=361, cote méd 22).

V6 NE SUIT PAS LE FAVORI : il parie systématiquement la cote la PLUS HAUTE
quand l'inefficience le justifie.
"""
from __future__ import annotations
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class FittedModelV6:
    p_12_by_home_bucket: dict
    p_21_by_away_bucket: dict
    p_12_by_pair_bucket: dict
    p_21_by_pair_bucket: dict
    p_12_global: float
    p_21_global: float
    # Signal upset 1-0 quand away favori
    p_1_0_when_away_fav: float       # probabilité empirique
    p_1_0_by_away_bucket: dict        # par bucket cote_away < 1.7
    # Signal upset HT-1 outsider home
    p_ht1_by_home_bucket: dict
    n_train: int


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


def fit_model_v6(history: pd.DataFrame) -> FittedModelV6:
    """history : odds_home, odds_away, ht_score_a, ht_score_b, score_a, score_b."""
    h = history.copy()
    h["ht_outcome"] = np.where(h.ht_score_a > h.ht_score_b, "1",
                       np.where(h.ht_score_a == h.ht_score_b, "X", "2"))
    h["ft_outcome"] = np.where(h.score_a > h.score_b, "1",
                       np.where(h.score_a == h.score_b, "X", "2"))

    won_12 = (h.ht_outcome == "1") & (h.ft_outcome == "2")
    won_21 = (h.ht_outcome == "2") & (h.ft_outcome == "1")
    won_1_0 = (h.score_a == 1) & (h.score_b == 0)
    won_ht1 = h.ht_outcome == "1"

    p_12_global = float(won_12.mean())
    p_21_global = float(won_21.mean())

    # Buckets cote home pour HT/FT 1/2
    p_12_by_home_bucket = {}
    for bk in ["<1.3", "<1.5", "<1.8", "<2.1", "<2.5", "<3.0", "<4.0", "<6.0", "6+"]:
        m = h.odds_home.apply(_bucket_ft) == bk
        if m.sum() >= 100:
            p_12_by_home_bucket[bk] = float(won_12[m].mean())
        else:
            p_12_by_home_bucket[bk] = p_12_global

    # Buckets cote away pour HT/FT 2/1
    p_21_by_away_bucket = {}
    for bk in ["<1.3", "<1.5", "<1.8", "<2.1", "<2.5", "<3.0", "<4.0", "<6.0", "6+"]:
        m = h.odds_away.apply(_bucket_ft) == bk
        if m.sum() >= 100:
            p_21_by_away_bucket[bk] = float(won_21[m].mean())
        else:
            p_21_by_away_bucket[bk] = p_21_global

    # Pair bucket (plus précis)
    p_12_by_pair_bucket = {}
    p_21_by_pair_bucket = {}
    pc = defaultdict(lambda: {"n": 0, "won_12": 0, "won_21": 0})
    for r in h.itertuples():
        k = (_bucket_ft(r.odds_home), _bucket_ft(r.odds_away))
        pc[k]["n"] += 1
        if r.ht_outcome == "1" and r.ft_outcome == "2": pc[k]["won_12"] += 1
        if r.ht_outcome == "2" and r.ft_outcome == "1": pc[k]["won_21"] += 1
    for k, s in pc.items():
        if s["n"] >= 30:
            p_12_by_pair_bucket[k] = s["won_12"] / s["n"]
            p_21_by_pair_bucket[k] = s["won_21"] / s["n"]

    # SIGNAL UPSET 1-0 quand away favori (cote_away < 1.7)
    fav_away = h[h.odds_away < 1.7]
    p_1_0_when_away_fav = float(((fav_away.score_a == 1) & (fav_away.score_b == 0)).mean()) if len(fav_away) > 0 else 0.0
    p_1_0_by_away_bucket = {}
    for bk in ["<1.3", "<1.5", "<1.8"]:
        m = h.odds_away.apply(_bucket_ft) == bk
        if m.sum() >= 50:
            p_1_0_by_away_bucket[bk] = float(((h.score_a == 1) & (h.score_b == 0))[m].mean())

    # SIGNAL HT-1 quand cote_home ∈ [3;5]
    p_ht1_by_home_bucket = {}
    for bk in ["<3.0", "<4.0", "<6.0", "6+"]:
        m = h.odds_home.apply(_bucket_ft) == bk
        if m.sum() >= 100:
            p_ht1_by_home_bucket[bk] = float(won_ht1[m].mean())

    return FittedModelV6(
        p_12_by_home_bucket=p_12_by_home_bucket,
        p_21_by_away_bucket=p_21_by_away_bucket,
        p_12_by_pair_bucket=p_12_by_pair_bucket,
        p_21_by_pair_bucket=p_21_by_pair_bucket,
        p_12_global=p_12_global,
        p_21_global=p_21_global,
        p_1_0_when_away_fav=p_1_0_when_away_fav,
        p_1_0_by_away_bucket=p_1_0_by_away_bucket,
        p_ht1_by_home_bucket=p_ht1_by_home_bucket,
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


def predict_edges_v6(model: FittedModelV6, odds_home: float, odds_away: float,
                      extra_markets: dict | str | None) -> dict[str, Any]:
    """Calcule EV pour TOUS les signaux V6 (HT/FT comebacks + upset 1-0 + HT-1 outsider)."""
    em = extra_markets
    if isinstance(em, str):
        try: em = json.loads(em)
        except Exception: em = None
    if not isinstance(em, dict): em = {}

    bh, ba = _bucket_ft(odds_home), _bucket_ft(odds_away)

    # SIGNAL 1 — HT/FT 1/2
    cote_12 = _get(em, "HT/FT", "1/2")
    p_12 = (model.p_12_by_pair_bucket.get((bh, ba))
            or model.p_12_by_home_bucket.get(bh)
            or model.p_12_global)
    ev_12 = (p_12 * cote_12 - 1) if cote_12 else None

    # SIGNAL 2 — HT/FT 2/1
    cote_21 = _get(em, "HT/FT", "2/1")
    p_21 = (model.p_21_by_pair_bucket.get((bh, ba))
            or model.p_21_by_away_bucket.get(ba)
            or model.p_21_global)
    ev_21 = (p_21 * cote_21 - 1) if cote_21 else None

    # SIGNAL 3 — UPSET 1-0 quand away favori
    cote_1_0 = _get(em, "Score exact", "1-0")
    ev_1_0 = None
    p_1_0 = None
    if cote_1_0 and odds_away < 1.7:
        p_1_0 = model.p_1_0_by_away_bucket.get(ba) or model.p_1_0_when_away_fav
        ev_1_0 = p_1_0 * cote_1_0 - 1

    # SIGNAL 4 — HT-1 quand outsider home
    cote_ht1 = _get(em, "Mi-tps 1X2", "1")
    ev_ht1 = None
    p_ht1 = None
    if cote_ht1 and odds_home >= 3.0:
        p_ht1 = model.p_ht1_by_home_bucket.get(bh)
        if p_ht1: ev_ht1 = p_ht1 * cote_ht1 - 1

    return {
        "ht_ft_1_2": {"cote": cote_12, "p_emp": p_12, "ev": ev_12, "label": "HT/FT 1/2"},
        "ht_ft_2_1": {"cote": cote_21, "p_emp": p_21, "ev": ev_21, "label": "HT/FT 2/1"},
        "upset_1_0": {"cote": cote_1_0, "p_emp": p_1_0, "ev": ev_1_0, "label": "Score 1-0 (upset)"},
        "outsider_ht1": {"cote": cote_ht1, "p_emp": p_ht1, "ev": ev_ht1, "label": "HT-1 outsider"},
        "bucket_home": bh, "bucket_away": ba,
    }
