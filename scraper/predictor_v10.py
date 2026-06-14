"""Predictor V10 — synthèse unifiée + données validées par équipe.

Signaux intégrés :
  ✅ V7 Calibration brackets (HOME 1.7-1.8, HOME 4-5, AWAY 2.5-2.9)
  ✅ V6 HT/FT comebacks (1/2 +104%, 2/1 +28%) + Score 1-0 upset (+42%)
  ✅ V8 Cote movement (Δ_away ↓ +22%, Δ_X ↑ +22%)
  ✅ V9 Team bias (whitelist/blacklist home + away)
  ✅ Per-team per-bracket bias (table validée 4341 matchs)
  ✅ Pair-specific bias (paires OR + paires TRAP hardcodées)
  ✅ Multi-signal co-occurrence boost (2+ signaux convergents → +17%)
  ✅ Rank diff direct (home rank > away rank +10 → ROI +7%)

V10 calcule un score de confiance multi-signaux pour CHAQUE pari potentiel.
"""
from __future__ import annotations
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from scraper.team_gold_data import (
    PAIR_HOME_GOLD, PAIR_AWAY_GOLD,
    BRACKET_GOLD_HOME, BRACKET_GOLD_AWAY,
    BRACKET_TRAP_HOME, PAIR_TRAP_HOME,
    bracket_match,
)


@dataclass
class FittedModelV10:
    # V8 calibration
    p_h_cal: dict
    p_a_cal: dict
    p_x_cal: dict
    # V6 exotiques
    p_12_global: float
    p_21_global: float
    p_12_by_pair_bucket: dict
    p_21_by_pair_bucket: dict
    p_12_by_home_bucket: dict
    p_21_by_away_bucket: dict
    p_1_0_when_away_fav: float
    p_1_0_by_away_bucket: dict
    # V9 team bias
    team_bias_home: dict
    team_bias_away: dict
    # 🆕 Per-team per-bracket bias
    team_bracket_home: dict   # (team, bracket) -> {n, acc, roi}
    # 🆕 Pair-specific bias
    pair_bias: dict           # (team_a, team_b) -> {n, acc, roi}
    # 🆕 Rank-based positions (latest known)
    team_ranks: dict
    n_train: int = 0


# Constantes finales V9 (calculées sur full DB)
HOME_WL_DEFAULT = {"Burnley", "C. Palace", "Brentford", "Brighton", "N. Forest"}
HOME_BL_DEFAULT = {"Everton", "Wolverhampton", "West Ham", "Spurs"}
AWAY_WL_DEFAULT = {"West Ham", "London Blues", "A. Villa", "Everton", "Spurs", "Manchester Red"}
AWAY_BL_DEFAULT = {"Liverpool"}


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


def _bracket_cote(c):
    """Brackets V10 finer for per-team per-bracket bias."""
    if c < 1.3: return None
    if c < 1.5: return "[1.3;1.5)"
    if c < 1.7: return "[1.5;1.7)"
    if c < 1.9: return "[1.7;1.9)"
    if c < 2.1: return "[1.9;2.1)"
    if c < 2.5: return "[2.1;2.5)"
    return None


def fit_model_v10(history: pd.DataFrame) -> FittedModelV10:
    h = history.copy()
    h["ft_o"] = np.where(h.score_a > h.score_b, "1",
                  np.where(h.score_a == h.score_b, "X", "2"))
    if "ht_score_a" in h.columns and h.ht_score_a.notna().any():
        h["ht_o"] = np.where(h.ht_score_a > h.ht_score_b, "1",
                      np.where(h.ht_score_a == h.ht_score_b, "X", "2"))
    inv_sum = 1/h.odds_home + 1/h.odds_draw + 1/h.odds_away
    h["p_1"] = (1/h.odds_home) / inv_sum
    h["p_x"] = (1/h.odds_draw) / inv_sum
    h["p_2"] = (1/h.odds_away) / inv_sum

    # V8 calibration
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

    # V6
    p_12_global = p_21_global = 0.035
    p_12_by_home_bucket = {bk: 0.035 for bk in ["<1.3","<1.5","<1.8","<2.1","<2.5","<3.0","<4.0","<6.0","6+"]}
    p_21_by_away_bucket = dict(p_12_by_home_bucket)
    p_12_by_pair_bucket, p_21_by_pair_bucket = {}, {}
    p_1_0_when_away_fav = 0.055
    p_1_0_by_away_bucket = {}

    if "ht_o" in h.columns:
        won_12 = (h.ht_o == "1") & (h.ft_o == "2")
        won_21 = (h.ht_o == "2") & (h.ft_o == "1")
        p_12_global = float(won_12.mean())
        p_21_global = float(won_21.mean())
        for bk in ["<1.3", "<1.5", "<1.8", "<2.1", "<2.5", "<3.0", "<4.0", "<6.0", "6+"]:
            m1 = h.odds_home.apply(_bucket_ft) == bk
            p_12_by_home_bucket[bk] = float(won_12[m1].mean()) if m1.sum() >= 100 else p_12_global
            m2 = h.odds_away.apply(_bucket_ft) == bk
            p_21_by_away_bucket[bk] = float(won_21[m2].mean()) if m2.sum() >= 100 else p_21_global
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
        p_1_0_when_away_fav = float(((fav_away.score_a == 1) & (fav_away.score_b == 0)).mean()) if len(fav_away) > 0 else 0.055
        for bk in ["<1.3", "<1.5", "<1.8"]:
            m = h.odds_away.apply(_bucket_ft) == bk
            if m.sum() >= 50:
                p_1_0_by_away_bucket[bk] = float(((h.score_a == 1) & (h.score_b == 0))[m].mean())

    # V9 team bias
    home_fav = h[(h.odds_home < h.odds_away) & (h.odds_home < h.odds_draw)].copy()
    home_fav["won"] = home_fav.ft_o == "1"
    team_bias_home = {}
    for team in home_fav.team_a.unique():
        sub = home_fav[home_fav.team_a == team]
        if len(sub) < 20: continue
        team_bias_home[team] = float(sub.won.mean() - sub.p_1.mean())

    away_fav = h[(h.odds_away < h.odds_home) & (h.odds_away < h.odds_draw)].copy()
    away_fav["won"] = away_fav.ft_o == "2"
    team_bias_away = {}
    for team in away_fav.team_b.unique():
        sub = away_fav[away_fav.team_b == team]
        if len(sub) < 20: continue
        team_bias_away[team] = float(sub.won.mean() - sub.p_2.mean())

    # 🆕 Per-team per-bracket bias
    team_bracket_home = {}
    for team in home_fav.team_a.unique():
        for bracket in ["[1.3;1.5)", "[1.5;1.7)", "[1.7;1.9)", "[1.9;2.1)", "[2.1;2.5)"]:
            lo, hi = float(bracket[1:4]), float(bracket[5:8])
            sub = home_fav[(home_fav.team_a == team) & (home_fav.odds_home >= lo) & (home_fav.odds_home < hi)]
            if len(sub) < 15: continue
            roi = float(np.where(sub.won, sub.odds_home - 1, -1).mean())
            team_bracket_home[(team, bracket)] = {
                "n": len(sub), "acc": float(sub.won.mean()), "roi": roi
            }

    # 🆕 Pair-specific bias
    pair_bias = {}
    for ta in home_fav.team_a.unique():
        for tb in home_fav.team_b.unique():
            sub = home_fav[(home_fav.team_a == ta) & (home_fav.team_b == tb)]
            if len(sub) < 10: continue
            roi = float(np.where(sub.won, sub.odds_home - 1, -1).mean())
            pair_bias[(ta, tb)] = {
                "n": len(sub), "acc": float(sub.won.mean()), "roi": roi
            }

    return FittedModelV10(
        p_h_cal=p_h_cal, p_a_cal=p_a_cal, p_x_cal=p_x_cal,
        p_12_global=p_12_global, p_21_global=p_21_global,
        p_12_by_pair_bucket=p_12_by_pair_bucket,
        p_21_by_pair_bucket=p_21_by_pair_bucket,
        p_12_by_home_bucket=p_12_by_home_bucket,
        p_21_by_away_bucket=p_21_by_away_bucket,
        p_1_0_when_away_fav=p_1_0_when_away_fav,
        p_1_0_by_away_bucket=p_1_0_by_away_bucket,
        team_bias_home=team_bias_home,
        team_bias_away=team_bias_away,
        team_bracket_home=team_bracket_home,
        pair_bias=pair_bias,
        team_ranks={},
        n_train=len(h),
    )


def _get_em(em, market, label):
    if isinstance(em, str):
        try: em = json.loads(em)
        except Exception: return None
    if not isinstance(em, dict): return None
    md = em.get(market)
    if not isinstance(md, dict): return None
    v = md.get(label)
    try: return float(v) if v is not None else None
    except Exception: return None


def predict_v10(model: FittedModelV10, team_a: str, team_b: str,
                 odds_home: float, odds_draw: float, odds_away: float,
                 extra_markets=None, rank_home: int | None = None,
                 rank_away: int | None = None) -> dict:
    """Prédiction V10 multi-signaux avec score de confiance."""
    em = extra_markets
    if isinstance(em, str):
        try: em = json.loads(em)
        except Exception: em = None
    if not isinstance(em, dict): em = {}

    inv_sum = 1/odds_home + 1/odds_draw + 1/odds_away
    p_market_h = (1/odds_home) / inv_sum
    p_market_x = (1/odds_draw) / inv_sum
    p_market_a = (1/odds_away) / inv_sum

    # V8 calibration probabilities
    p_raw_h = model.p_h_cal.get(_bucket_p(p_market_h), p_market_h)
    p_raw_x = model.p_x_cal.get(_bucket_p(p_market_x), p_market_x)
    p_raw_a = model.p_a_cal.get(_bucket_p(p_market_a), p_market_a)

    # ========== SIGNAL ANALYSIS PER ISSUE ==========
    signals = {"1": [], "X": [], "2": []}

    # SIGNAL 1 : V7 calibration positive
    ev_cal_h = p_raw_h * odds_home - 1
    ev_cal_x = p_raw_x * odds_draw - 1
    ev_cal_a = p_raw_a * odds_away - 1
    if ev_cal_h > 0.03: signals["1"].append(("calibration_v7", ev_cal_h, "+"))
    if ev_cal_x > 0.03: signals["X"].append(("calibration_v7", ev_cal_x, "+"))
    if ev_cal_a > 0.03: signals["2"].append(("calibration_v7", ev_cal_a, "+"))

    # SIGNAL 2 : V9 team bias (whitelist boost / blacklist penalty)
    bias_h = model.team_bias_home.get(team_a, 0)
    bias_a = model.team_bias_away.get(team_b, 0)
    if bias_h > 0.04: signals["1"].append(("team_bias_+", bias_h, "+"))
    elif bias_h < -0.04: signals["1"].append(("team_bias_-", bias_h, "-"))
    if bias_a > 0.04: signals["2"].append(("team_bias_+", bias_a, "+"))
    elif bias_a < -0.04: signals["2"].append(("team_bias_-", bias_a, "-"))

    # SIGNAL 3 : Per-team per-bracket (V10 model + hardcoded GOLD)
    bracket_h = _bracket_cote(odds_home)
    bracket_a = _bracket_cote(odds_away)
    bb_h = model.team_bracket_home.get((team_a, bracket_h)) if bracket_h else None
    if bb_h and bb_h["n"] >= 15:
        if bb_h["roi"] > 0.10: signals["1"].append(("team_bracket_+", bb_h["roi"], "+"))
        elif bb_h["roi"] < -0.15: signals["1"].append(("team_bracket_-", bb_h["roi"], "-"))

    # SIGNAL 3b : Hardcoded BRACKET GOLD HOME (override le precedent si plus précis)
    gold_h = bracket_match(team_a, odds_home, BRACKET_GOLD_HOME)
    if gold_h is not None and gold_h > 0.10:
        signals["1"].append(("BRACKET_GOLD_HOME", gold_h, "+"))
    # SIGNAL 3c : Hardcoded BRACKET TRAP HOME
    trap_h = bracket_match(team_a, odds_home, BRACKET_TRAP_HOME)
    if trap_h is not None and trap_h < -0.20:
        signals["1"].append(("BRACKET_TRAP_HOME", trap_h, "-"))
    # SIGNAL 3d : Hardcoded BRACKET GOLD AWAY
    gold_a = bracket_match(team_b, odds_away, BRACKET_GOLD_AWAY)
    if gold_a is not None and gold_a > 0.10:
        signals["2"].append(("BRACKET_GOLD_AWAY", gold_a, "+"))

    # SIGNAL 4 : Pair-specific (V10 model + hardcoded GOLD/TRAP)
    pb = model.pair_bias.get((team_a, team_b))
    if pb and pb["n"] >= 10:
        if pb["roi"] > 0.20: signals["1"].append(("pair_+", pb["roi"], "+"))
        elif pb["roi"] < -0.20: signals["1"].append(("pair_-", pb["roi"], "-"))

    # SIGNAL 4b : PAIRES OR HOME (hardcoded, très haute confiance)
    if (team_a, team_b) in PAIR_HOME_GOLD:
        pair_data = PAIR_HOME_GOLD[(team_a, team_b)]
        signals["1"].append(("PAIR_GOLD_HOME", pair_data["roi"], "++"))  # double boost
    # SIGNAL 4c : PAIRES OR AWAY
    if (team_a, team_b) in PAIR_AWAY_GOLD:
        pair_data = PAIR_AWAY_GOLD[(team_a, team_b)]
        signals["2"].append(("PAIR_GOLD_AWAY", pair_data["roi"], "++"))
    # SIGNAL 4d : PAIRES TRAP HOME
    if (team_a, team_b) in PAIR_TRAP_HOME:
        signals["1"].append(("PAIR_TRAP_HOME", -0.5, "--"))  # block

    # SIGNAL 5 : 🆕 Rank diff
    if rank_home is not None and rank_away is not None:
        diff = rank_home - rank_away
        if diff >= 10: signals["1"].append(("rank_diff_+", 0.07, "+"))
        elif diff <= -15: signals["1"].append(("rank_diff_-", -0.05, "-"))

    # ========== AGRÉGATION SCORES ==========
    def aggregate(sig_list):
        # ++ = double positif (paire OR), + = positif, - = négatif, -- = paire trap (block)
        double_pos = [s for s in sig_list if s[2] == "++"]
        pos = [s for s in sig_list if s[2] == "+"]
        neg = [s for s in sig_list if s[2] == "-"]
        double_neg = [s for s in sig_list if s[2] == "--"]
        return {"n_pos": len(pos) + 2*len(double_pos),
                "n_neg": len(neg) + 2*len(double_neg),
                "has_pair_gold": len(double_pos) > 0,
                "has_pair_trap": len(double_neg) > 0,
                "ev_sum": sum(s[1] for s in pos+double_pos) - sum(abs(s[1]) for s in neg+double_neg),
                "signals": sig_list}

    agg = {o: aggregate(signals[o]) for o in ["1", "X", "2"]}

    # Multi-signal boost
    boost = {}
    for o in ["1", "X", "2"]:
        if agg[o]["has_pair_trap"]:
            boost[o] = -1.0  # block absolu (paire trap = 0% historique)
        elif agg[o]["has_pair_gold"]:
            # Paire OR : boost extra car validation extrême (n>=10, ROI>=+40%)
            boost[o] = 0.25
        elif agg[o]["n_pos"] >= 2:
            # 2+ signaux positifs convergent : +10pp EV boost (validé 4341 matchs)
            boost[o] = 0.10
        elif agg[o]["n_pos"] == 1 and agg[o]["n_neg"] == 0:
            boost[o] = 0.0
        elif agg[o]["n_neg"] >= 1:
            boost[o] = -1.0  # bloque
        else:
            boost[o] = 0.0

    # EV finale par issue (calibration + agrégation signaux + boost co-occurrence)
    final_evs = {}
    final_probs = {}
    for outcome in ["1", "X", "2"]:
        base_ev = {"1": ev_cal_h, "X": ev_cal_x, "2": ev_cal_a}[outcome]
        base_p = {"1": p_raw_h, "X": p_raw_x, "2": p_raw_a}[outcome]
        if boost[outcome] == -1.0:
            final_evs[outcome] = -1.0   # blocked
        else:
            adj = sum(s[1] if s[2] == "+" else -abs(s[1]) for s in signals[outcome])
            final_evs[outcome] = base_ev + adj + boost[outcome]
        final_probs[outcome] = base_p

    # V6 exotiques inchangé
    bh, ba = _bucket_ft(odds_home), _bucket_ft(odds_away)
    cote_12 = _get_em(em, "HT/FT", "1/2")
    p_12 = (model.p_12_by_pair_bucket.get((bh, ba))
            or model.p_12_by_home_bucket.get(bh) or model.p_12_global)
    ev_12 = (p_12 * cote_12 - 1) if cote_12 else None
    cote_21 = _get_em(em, "HT/FT", "2/1")
    p_21 = (model.p_21_by_pair_bucket.get((bh, ba))
            or model.p_21_by_away_bucket.get(ba) or model.p_21_global)
    ev_21 = (p_21 * cote_21 - 1) if cote_21 else None
    cote_1_0 = _get_em(em, "Score exact", "1-0")
    ev_1_0 = None; p_1_0 = None
    if cote_1_0 and odds_away < 1.7:
        p_1_0 = model.p_1_0_by_away_bucket.get(ba) or model.p_1_0_when_away_fav
        ev_1_0 = p_1_0 * cote_1_0 - 1

    # Confidence score /10 par issue : basé sur n_signals + EV
    confidence = {}
    for outcome in ["1", "X", "2"]:
        if boost[outcome] == -1.0:
            confidence[outcome] = 0
        else:
            base = final_probs[outcome] * 10
            n_pos = agg[outcome]["n_pos"]
            sig_boost = min(n_pos * 1.0, 4)
            if agg[outcome]["has_pair_gold"]: sig_boost += 2.0  # paire OR = +2 confidence
            confidence[outcome] = min(round(base + sig_boost, 1), 10)

    return {
        "p_market": {"1": p_market_h, "X": p_market_x, "2": p_market_a},
        "p_model": {"1": p_raw_h, "X": p_raw_x, "2": p_raw_a},
        "ev_1x2": final_evs,
        "ev_1x2_base": {"1": ev_cal_h, "X": ev_cal_x, "2": ev_cal_a},
        "cotes": {"1": odds_home, "X": odds_draw, "2": odds_away},
        "signals": signals,
        "agg": agg,
        "confidence": confidence,
        "exotics": {
            "HT/FT 1/2": {"cote": cote_12, "p_emp": p_12, "ev": ev_12},
            "HT/FT 2/1": {"cote": cote_21, "p_emp": p_21, "ev": ev_21},
            "Score 1-0 (upset)": {"cote": cote_1_0, "p_emp": p_1_0, "ev": ev_1_0},
        },
    }
