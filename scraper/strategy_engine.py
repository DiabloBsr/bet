"""STRATEGY ENGINE — Système de prédiction segmenté par phase de saison.

Conçu à partir d'une analyse profonde de 4 543 matchs (J1-J37) avec :
- 5 segments saison : DS / MS_early / MS_mid / MS_late / FS
- Patterns par équipe × segment
- Buckets de cote × segment (favoris + non-favoris)
- Paires GOLD + Score COMBO + Paires TRAP par segment
- HT/FT transitions

Usage:
    from scraper.strategy_engine import StrategyEngine, label_segment
    engine = StrategyEngine()
    signals = engine.evaluate(team_a, team_b, journee, odds_h, odds_d, odds_a)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# ============================================================================
# 🎯 SEGMENT DEFINITIONS
# ============================================================================
SEGMENTS = [
    ("DS",       1,  3),
    ("MS_early", 4,  12),
    ("MS_mid",   13, 25),
    ("MS_late",  26, 33),
    ("FS",       34, 38),
]

def label_segment(journee: Optional[int]) -> Optional[str]:
    if journee is None: return None
    for name, lo, hi in SEGMENTS:
        if lo <= journee <= hi:
            return name
    return None


# ============================================================================
# 🌟 TEAM × SEGMENT PROFILES (calibrés sur 4 543 matchs)
# ============================================================================
# WR_HOME_DELTA[team][segment] = delta vs WR_home global
TEAM_SEGMENT_DELTA = {
    # 🔥 DS PEAKERS (J1-J3, +pp vs global)
    "West Ham":     {"DS": +0.20, "MS_early": -0.14, "MS_mid":  0.00, "MS_late": +0.06, "FS": -0.09},
    "London Blues": {"DS": +0.16, "MS_early": -0.03, "MS_mid": -0.05, "MS_late": +0.02, "FS": +0.13},
    "A. Villa":     {"DS": +0.09, "MS_early": +0.01, "MS_mid": +0.02, "MS_late": -0.06, "FS": -0.11},

    # ❄️ DS DROPPERS (catastrophiques en J1-J3)
    "Leeds":        {"DS": -0.15, "MS_early": +0.04, "MS_mid": -0.01, "MS_late": +0.03, "FS": +0.01},
    "Fulham":       {"DS": -0.14, "MS_early": +0.04, "MS_mid":  0.00, "MS_late": +0.02, "FS": -0.03},
    "Brentford":    {"DS": -0.13, "MS_early": -0.04, "MS_mid": -0.01, "MS_late": +0.08, "FS": +0.06},

    # 🔥 FS PEAKERS (J34-J38 — bonus monstrueux en fin)
    "Spurs":        {"DS": -0.08, "MS_early": -0.06, "MS_mid": +0.05, "MS_late": -0.07, "FS": +0.23},
    "C. Palace":    {"DS":  0.00, "MS_early": -0.04, "MS_mid": -0.01, "MS_late": -0.01, "FS": +0.21},
    "Sunderland":   {"DS": -0.02, "MS_early": -0.01, "MS_mid": -0.02, "MS_late": +0.01, "FS": +0.11},
    "Brighton":     {"DS": +0.05, "MS_early": -0.05, "MS_mid": -0.01, "MS_late": +0.01, "FS": +0.08},

    # ❄️ FS DROPPERS (à éviter en fin de saison)
    "Everton":      {"DS": +0.03, "MS_early": +0.02, "MS_mid":  0.00, "MS_late": +0.03, "FS": -0.17},
    "Manchester Red": {"DS": +0.03, "MS_early": -0.01, "MS_mid": -0.02, "MS_late": +0.06, "FS": -0.10},
    "Manchester Blue": {"DS": +0.05, "MS_early": -0.02, "MS_mid": -0.01, "MS_late": +0.05, "FS": -0.08},

    # Stables (variations < 8pp)
    "Liverpool":      {"DS": -0.08, "MS_early": +0.05, "MS_mid": +0.03, "MS_late": -0.03, "FS": -0.08},
    "Newcastle":      {"DS": +0.05, "MS_early": +0.03, "MS_mid": -0.02, "MS_late":  0.00, "FS": -0.03},
    "Bournemouth":    {"DS": -0.05, "MS_early": +0.10, "MS_mid": -0.01, "MS_late": -0.04, "FS":  0.00},
    "Burnley":        {"DS": -0.02, "MS_early": +0.03, "MS_mid": +0.03, "MS_late": -0.07, "FS": -0.06},
    "Wolverhampton":  {"DS": +0.01, "MS_early": +0.02, "MS_mid": -0.03, "MS_late": +0.01, "FS":  0.00},
    "N. Forest":      {"DS": -0.05, "MS_early": +0.03, "MS_mid":  0.00, "MS_late": -0.03, "FS": +0.04},
    "London Reds":    {"DS": +0.01, "MS_early": +0.06, "MS_mid": -0.01, "MS_late": -0.09, "FS": +0.07},
}


# ============================================================================
# 💰 EDGES PAR BUCKET DE COTE × SEGMENT (calibrés)
# (cote bucket, edge en pp, ROI)
# ============================================================================
COTE_EDGES = {
    "DS": {
        # 🔥 EDGES POSITIFS
        "home_equilibre_2.2_2.7":   {"edge": +0.082, "roi": +0.198, "min": 2.20, "max": 2.70, "side": "home"},
        "away_favori_modere_1.5_1.8": {"edge": +0.144, "roi": +0.235, "min": 1.50, "max": 1.80, "side": "away"},
        "away_underdog_3.5_5":      {"edge": +0.059, "roi": +0.245, "min": 3.50, "max": 5.00, "side": "away"},
        "away_favori_solide_1.3_1.5": {"edge": +0.065, "roi": +0.090, "min": 1.30, "max": 1.50, "side": "away"},
        # ❄️ TRAPS
        "home_leger_favori_1.8_2.2_TRAP": {"edge": -0.127, "roi": -0.250, "min": 1.80, "max": 2.20, "side": "home"},
        "home_underdog_3.5_5_TRAP":  {"edge": -0.155, "roi": -0.637, "min": 3.50, "max": 5.00, "side": "home"},
    },
    "MS_early": {
        "home_long_shot_5plus":     {"edge": +0.056, "roi": +0.439, "min": 5.00, "max": 50.0, "side": "home"},
        "home_non_favori_2.7_3.5":  {"edge": +0.060, "roi": +0.181, "min": 2.70, "max": 3.50, "side": "home"},
        "away_leger_favori_1.8_2.2_TRAP": {"edge": -0.169, "roi": -0.331, "min": 1.80, "max": 2.20, "side": "away"},
        "home_equilibre_2.2_2.7_TRAP":  {"edge": -0.079, "roi": -0.191, "min": 2.20, "max": 2.70, "side": "home"},
    },
    "MS_mid": {
        "home_long_shot_5plus":     {"edge": +0.032, "roi": +0.259, "min": 5.00, "max": 50.0, "side": "home"},
        "away_long_shot_5plus":     {"edge": +0.024, "roi": +0.196, "min": 5.00, "max": 50.0, "side": "away"},
        "home_favori_extreme_TRAP": {"edge": -0.101, "roi": -0.123, "min": 1.00, "max": 1.30, "side": "home"},
        "home_underdog_3.5_5_TRAP": {"edge": -0.051, "roi": -0.213, "min": 3.50, "max": 5.00, "side": "home"},
    },
    "MS_late": {
        "home_long_shot_5plus":     {"edge": +0.036, "roi": +0.284, "min": 5.00, "max": 50.0, "side": "home"},
        "away_favori_solide_1.3_1.5": {"edge": +0.062, "roi": +0.087, "min": 1.30, "max": 1.50, "side": "away"},
        "home_favori_solide_1.3_1.5_TRAP": {"edge": -0.082, "roi": -0.114, "min": 1.30, "max": 1.50, "side": "home"},
        "away_leger_favori_1.8_2.2_TRAP": {"edge": -0.055, "roi": -0.107, "min": 1.80, "max": 2.20, "side": "away"},
    },
    "FS": {
        "home_long_shot_5plus":     {"edge": +0.068, "roi": +0.554, "min": 5.00, "max": 50.0, "side": "home"},
        "home_underdog_3.5_5":      {"edge": +0.055, "roi": +0.226, "min": 3.50, "max": 5.00, "side": "home"},
        "away_favori_solide_1.3_1.5": {"edge": +0.066, "roi": +0.092, "min": 1.30, "max": 1.50, "side": "away"},
        "away_favori_modere_1.5_1.8_TRAP": {"edge": -0.265, "roi": -0.436, "min": 1.50, "max": 1.80, "side": "away"},
        "away_leger_favori_1.8_2.2_TRAP": {"edge": -0.185, "roi": -0.367, "min": 1.80, "max": 2.20, "side": "away"},
        "home_favori_extreme_TRAP": {"edge": -0.093, "roi": -0.113, "min": 1.00, "max": 1.30, "side": "home"},
        "home_equilibre_2.2_2.7_TRAP": {"edge": -0.127, "roi": -0.308, "min": 2.20, "max": 2.70, "side": "home"},
    },
}


# ============================================================================
# 📊 BASE RATES PAR SEGMENT
# ============================================================================
SEGMENT_BASE_RATES = {
    "DS":       {"rate_1": 0.480, "rate_X": 0.197, "rate_2": 0.323, "avg_goals": 3.00, "btts": 0.558, "ht_X_rate": 0.413},
    "MS_early": {"rate_1": 0.481, "rate_X": 0.227, "rate_2": 0.292, "avg_goals": 2.99, "btts": 0.592, "ht_X_rate": 0.423},
    "MS_mid":   {"rate_1": 0.477, "rate_X": 0.218, "rate_2": 0.305, "avg_goals": 2.94, "btts": 0.581, "ht_X_rate": 0.423},
    "MS_late":  {"rate_1": 0.485, "rate_X": 0.227, "rate_2": 0.288, "avg_goals": 2.94, "btts": 0.585, "ht_X_rate": 0.409},
    "FS":       {"rate_1": 0.491, "rate_X": 0.251, "rate_2": 0.258, "avg_goals": 3.01, "btts": 0.565, "ht_X_rate": 0.458},
}


# ============================================================================
# 🔄 HT→FT TRANSITIONS PAR SEGMENT
# ============================================================================
HT_FT_TRANSITIONS = {
    "DS":       {"HT_1": {"1": 0.86, "X": 0.11, "2": 0.03}, "HT_X": {"1": 0.38, "X": 0.30, "2": 0.32}, "HT_2": {"1": 0.09, "X": 0.13, "2": 0.78}},
    "MS_early": {"HT_1": {"1": 0.84, "X": 0.11, "2": 0.05}, "HT_X": {"1": 0.37, "X": 0.35, "2": 0.28}, "HT_2": {"1": 0.13, "X": 0.18, "2": 0.69}},
    "MS_mid":   {"HT_1": {"1": 0.81, "X": 0.11, "2": 0.07}, "HT_X": {"1": 0.37, "X": 0.33, "2": 0.30}, "HT_2": {"1": 0.16, "X": 0.17, "2": 0.67}},
    "MS_late":  {"HT_1": {"1": 0.85, "X": 0.11, "2": 0.04}, "HT_X": {"1": 0.35, "X": 0.35, "2": 0.29}, "HT_2": {"1": 0.14, "X": 0.18, "2": 0.68}},
    "FS":       {"HT_1": {"1": 0.77, "X": 0.14, "2": 0.09}, "HT_X": {"1": 0.44, "X": 0.34, "2": 0.22}, "HT_2": {"1": 0.13, "X": 0.23, "2": 0.64}},
}


# ============================================================================
# ⏰ MINUTE PREMIER BUT PAR SEGMENT
# ============================================================================
FIRST_GOAL_DIST = {
    # Toutes saisons confondues : ~3% pas de but, médiane 27', 23% avant 15'
    "DS":       {"pct_no_goal": 0.030, "median": 27, "mean": 31.5, "pct_15": 0.236, "pct_30": 0.598, "pct_45": 0.781},
    "MS_early": {"pct_no_goal": 0.030, "median": 27, "mean": 31.5, "pct_15": 0.236, "pct_30": 0.598, "pct_45": 0.781},
    "MS_mid":   {"pct_no_goal": 0.030, "median": 27, "mean": 31.9, "pct_15": 0.236, "pct_30": 0.573, "pct_45": 0.768},
    "MS_late":  {"pct_no_goal": 0.021, "median": 27, "mean": 31.9, "pct_15": 0.227, "pct_30": 0.580, "pct_45": 0.762},
    "FS":       {"pct_no_goal": 0.021, "median": 28, "mean": 32.8, "pct_15": 0.234, "pct_30": 0.550, "pct_45": 0.750},
}


# ============================================================================
# 🎯 SCORES + OVER/UNDER + BTTS PAR PROFIL × SEGMENT
# (calibrés sur 4 543 matchs avec profils home_crush / home_strong / ...)
# ============================================================================
PROFILE_SCORES = {
    "DS": {
        "home_crush":  {"n": 41,  "over_25": 0.63, "btts": 0.44, "top3": {"2-0": 0.22, "3-0": 0.15, "3-1": 0.12}},
        "home_strong": {"n": 133, "over_25": 0.66, "btts": 0.47, "top3": {"2-0": 0.13, "2-1": 0.12, "3-0": 0.11}},
        "home_slight": {"n": 143, "over_25": 0.59, "btts": 0.64, "top3": {"1-2": 0.13, "1-1": 0.12, "2-0": 0.09}},
        "away_slight": {"n": 51,  "over_25": 0.51, "btts": 0.59, "top3": {"1-1": 0.14, "0-1": 0.14, "0-2": 0.12}},
        "away_strong": {"n": 35,  "over_25": 0.63, "btts": 0.31, "top3": {"0-3": 0.20, "0-2": 0.17, "0-4": 0.11}},
    },
    "MS_early": {
        "home_crush":  {"n": 102, "over_25": 0.76, "btts": 0.46, "top3": {"3-0": 0.15, "2-1": 0.12, "4-0": 0.12}},
        "home_strong": {"n": 317, "over_25": 0.73, "btts": 0.58, "top3": {"2-1": 0.13, "3-0": 0.10, "3-1": 0.08}},
        "home_slight": {"n": 352, "over_25": 0.55, "btts": 0.57, "top3": {"1-1": 0.13, "2-1": 0.11, "1-0": 0.11}},
        "away_slight": {"n": 146, "over_25": 0.64, "btts": 0.62, "top3": {"2-2": 0.11, "1-1": 0.10, "1-2": 0.10}},
        "away_strong": {"n": 88,  "over_25": 0.69, "btts": 0.62, "top3": {"1-2": 0.16, "2-3": 0.11, "0-3": 0.10}},
        "away_crush":  {"n": 21,  "over_25": 0.95, "btts": 0.62, "top3": {"0-3": 0.29, "2-1": 0.14, "1-3": 0.10}},
    },
    "MS_mid": {
        "home_crush":  {"n": 143, "over_25": 0.62, "btts": 0.46, "top3": {"2-0": 0.13, "2-1": 0.09, "3-1": 0.09}},
        "home_strong": {"n": 486, "over_25": 0.63, "btts": 0.52, "top3": {"2-1": 0.10, "1-1": 0.10, "2-0": 0.10}},
        "home_slight": {"n": 493, "over_25": 0.63, "btts": 0.58, "top3": {"2-1": 0.13, "3-1": 0.08, "1-1": 0.08}},
        "away_slight": {"n": 221, "over_25": 0.65, "btts": 0.65, "top3": {"1-2": 0.15, "1-1": 0.12, "1-3": 0.09}},
        "away_strong": {"n": 118, "over_25": 0.71, "btts": 0.58, "top3": {"1-2": 0.19, "0-2": 0.10, "0-3": 0.10}},
        "away_crush":  {"n": 26,  "over_25": 0.77, "btts": 0.54, "top3": {"1-2": 0.31, "0-1": 0.15, "0-4": 0.12}},
    },
    "MS_late": {
        "home_crush":  {"n": 94,  "over_25": 0.76, "btts": 0.54, "top3": {"3-1": 0.16, "2-1": 0.11, "1-0": 0.10}},
        "home_strong": {"n": 305, "over_25": 0.67, "btts": 0.57, "top3": {"3-1": 0.12, "2-1": 0.12, "1-0": 0.09}},
        "home_slight": {"n": 314, "over_25": 0.64, "btts": 0.63, "top3": {"2-1": 0.13, "1-1": 0.12, "2-0": 0.08}},
        "away_slight": {"n": 123, "over_25": 0.59, "btts": 0.59, "top3": {"0-1": 0.13, "1-2": 0.12, "1-1": 0.09}},
        "away_strong": {"n": 76,  "over_25": 0.58, "btts": 0.45, "top3": {"0-1": 0.14, "0-2": 0.13, "0-3": 0.13}},
        "away_crush":  {"n": 19,  "over_25": 0.63, "btts": 0.26, "top3": {"0-3": 0.32, "0-1": 0.16, "1-3": 0.11}},
    },
    "FS": {
        "home_crush":  {"n": 40,  "over_25": 0.68, "btts": 0.40, "top3": {"4-0": 0.12, "5-0": 0.10, "1-0": 0.10}},
        "home_strong": {"n": 130, "over_25": 0.72, "btts": 0.52, "top3": {"2-1": 0.11, "3-0": 0.10, "4-0": 0.09}},
        "home_slight": {"n": 123, "over_25": 0.62, "btts": 0.58, "top3": {"2-1": 0.14, "1-0": 0.12, "2-0": 0.09}},
        "away_slight": {"n": 47,  "over_25": 0.60, "btts": 0.66, "top3": {"1-1": 0.19, "2-2": 0.13, "0-2": 0.11}},
        "away_strong": {"n": 34,  "over_25": 0.59, "btts": 0.41, "top3": {"0-3": 0.12, "0-2": 0.12, "1-0": 0.12}},
    },
}


def classify_profile(odds_h: float, odds_a: float) -> str:
    """Classifie un match selon le profil cote."""
    if odds_h < 1.3 and odds_a > 7:  return "home_crush"
    if odds_h < 1.6 and odds_a > 4:  return "home_strong"
    if 1.6 <= odds_h < 2.2 and odds_a >= 2.5: return "home_slight"
    if 1.9 <= odds_h < 2.5 and 1.9 <= odds_a < 2.5: return "balanced"
    if 1.6 <= odds_a < 2.2 and odds_h >= 2.5: return "away_slight"
    if odds_a < 1.6 and odds_h > 4:  return "away_strong"
    if odds_a < 1.3 and odds_h > 7:  return "away_crush"
    return "other"


# ============================================================================
# 🎯 STRATEGY ENGINE
# ============================================================================
@dataclass
class StrategySignal:
    """Un signal stratégique avec pick + confiance + raison."""
    category: str       # "TEAM_SEGMENT" | "COTE_BUCKET" | "TRAP" | "HT_FT" | ...
    pick: str           # "1" | "X" | "2" | "Over 2.5" | "BTTS NON" | ...
    strength: float     # 0.0 à 1.0
    edge: float         # estimated edge vs market (pp)
    reason: str         # human-readable
    bonus_action: Optional[str] = None  # e.g. "increase_stake" | "skip" | "double_chance"


@dataclass
class StrategyEvaluation:
    """Résultat de l'evaluation d'un match."""
    segment: str
    team_a: str
    team_b: str
    profile: str = "other"
    base_signals: list = field(default_factory=list)    # Signaux qui SUPPORTENT un pari
    traps: list = field(default_factory=list)           # TRAPS à éviter
    score_signals: list = field(default_factory=list)   # Scores recommandés par profil
    expected_total: float = 3.0                          # buts attendus moyens segment
    recommended_picks: list = field(default_factory=list)  # Picks finaux consolidés
    notes: list = field(default_factory=list)


class StrategyEngine:
    """Moteur stratégique segmenté par phase de saison."""

    def evaluate(self, team_a: str, team_b: str, journee: int,
                  odds_h: float, odds_d: float, odds_a: float) -> StrategyEvaluation:
        segment = label_segment(journee) or "MS_mid"
        ev = StrategyEvaluation(segment=segment, team_a=team_a, team_b=team_b)

        base = SEGMENT_BASE_RATES.get(segment, SEGMENT_BASE_RATES["MS_mid"])
        ev.expected_total = base["avg_goals"]

        # ------------------------ 1) TEAM × SEGMENT signals ------------------------
        d_ha = TEAM_SEGMENT_DELTA.get(team_a, {}).get(segment, 0.0)
        d_ab = TEAM_SEGMENT_DELTA.get(team_b, {}).get(segment, 0.0)

        if d_ha >= 0.10:
            ev.base_signals.append(StrategySignal(
                category="TEAM_PEAK_HOME", pick="1", strength=min(1.0, d_ha * 5),
                edge=d_ha, reason=f"🔥 {team_a} PEAK en {segment} (+{d_ha*100:.0f}pp vs global)",
                bonus_action="increase_stake",
            ))
        elif d_ha <= -0.10:
            ev.traps.append(StrategySignal(
                category="TEAM_DROP_HOME", pick="1", strength=min(1.0, -d_ha * 5),
                edge=d_ha, reason=f"❄️ {team_a} CHUTE en {segment} ({d_ha*100:.0f}pp vs global) — TRAP sur '1'",
                bonus_action="skip_home_pick",
            ))
        if d_ab >= 0.10:
            ev.base_signals.append(StrategySignal(
                category="TEAM_PEAK_AWAY", pick="2", strength=min(1.0, d_ab * 5),
                edge=d_ab, reason=f"🔥 {team_b} PEAK en {segment} (+{d_ab*100:.0f}pp vs global)",
                bonus_action="increase_stake",
            ))
        elif d_ab <= -0.10:
            ev.traps.append(StrategySignal(
                category="TEAM_DROP_AWAY", pick="2", strength=min(1.0, -d_ab * 5),
                edge=d_ab, reason=f"❄️ {team_b} CHUTE en {segment} ({d_ab*100:.0f}pp vs global) — TRAP sur '2'",
                bonus_action="skip_away_pick",
            ))

        # ------------------------ 2) COTE BUCKET × SEGMENT signals ------------------
        edges_seg = COTE_EDGES.get(segment, {})
        for name, e in edges_seg.items():
            side = e["side"]
            cote = odds_h if side == "home" else odds_a
            if e["min"] <= cote < e["max"]:
                pick = "1" if side == "home" else "2"
                if e["edge"] >= 0:
                    ev.base_signals.append(StrategySignal(
                        category=f"COTE_BUCKET_{name}",
                        pick=pick, strength=min(1.0, e["edge"] * 8),
                        edge=e["edge"],
                        reason=f"💰 Bucket cote @{cote:.2f} en {segment} — EDGE {e['edge']*100:+.1f}pp, ROI {e['roi']*100:+.1f}%",
                    ))
                else:
                    ev.traps.append(StrategySignal(
                        category=f"COTE_BUCKET_TRAP_{name}",
                        pick=pick, strength=min(1.0, -e["edge"] * 4),
                        edge=e["edge"],
                        reason=f"⚠️ TRAP : Bucket cote @{cote:.2f} en {segment} — EDGE {e['edge']*100:.1f}pp historique, ROI {e['roi']*100:.1f}%",
                        bonus_action="skip" if "TRAP" in name else None,
                    ))

        # ------------------------ 3) Consolidation : recommended picks --------------
        # Priorité : Team PEAK > COTE bucket positif > base rates
        pick_scores = {"1": 0.0, "X": 0.0, "2": 0.0}
        for sig in ev.base_signals:
            if sig.pick in pick_scores:
                pick_scores[sig.pick] += sig.strength
        # Pénalité traps
        for trap in ev.traps:
            if trap.pick in pick_scores:
                pick_scores[trap.pick] -= trap.strength * 0.5

        # Pick recommandé : meilleur score si > 0.3
        best_pick = max(pick_scores, key=pick_scores.get)
        best_score = pick_scores[best_pick]
        if best_score >= 0.30:
            cote = {"1": odds_h, "X": odds_d, "2": odds_a}[best_pick]
            ev.recommended_picks.append({
                "pick": best_pick, "cote": cote,
                "strength": best_score,
                "n_supporting": sum(1 for s in ev.base_signals if s.pick == best_pick),
            })

        # ------------------------ 3.5) PROFILE-BASED SCORES + O/U + BTTS -----------
        profile = classify_profile(odds_h, odds_a)
        ev.profile = profile
        prof_data = PROFILE_SCORES.get(segment, {}).get(profile)
        if prof_data:
            # Top scores recommandés
            for score, rate in prof_data["top3"].items():
                if rate >= 0.12:
                    ev.score_signals.append({
                        "score": score, "rate": rate,
                        "reason": f"📊 {profile} en {segment} → {score} à {rate*100:.0f}% (n={prof_data['n']})",
                    })
            # OVER/UNDER signal
            if prof_data["over_25"] >= 0.70:
                ev.base_signals.append(StrategySignal(
                    category="OVER_PROFILE", pick="Over 2.5", strength=prof_data["over_25"],
                    edge=prof_data["over_25"] - 0.58, reason=f"📈 {profile} en {segment} → O2.5 {prof_data['over_25']*100:.0f}%",
                ))
            elif prof_data["over_25"] <= 0.55:
                ev.base_signals.append(StrategySignal(
                    category="UNDER_PROFILE", pick="Under 2.5", strength=1-prof_data["over_25"],
                    edge=0.58 - prof_data["over_25"], reason=f"📉 {profile} en {segment} → U2.5 {(1-prof_data['over_25'])*100:.0f}%",
                ))
            # BTTS signal
            if prof_data["btts"] >= 0.65:
                ev.base_signals.append(StrategySignal(
                    category="BTTS_OUI_PROFILE", pick="BTTS OUI", strength=prof_data["btts"],
                    edge=prof_data["btts"] - 0.58, reason=f"🎯 {profile} en {segment} → BTTS OUI {prof_data['btts']*100:.0f}%",
                ))
            elif prof_data["btts"] <= 0.45:
                ev.base_signals.append(StrategySignal(
                    category="BTTS_NON_PROFILE", pick="BTTS NON", strength=1-prof_data["btts"],
                    edge=0.58 - prof_data["btts"], reason=f"🎯 {profile} en {segment} → BTTS NON {(1-prof_data['btts'])*100:.0f}%",
                ))

        # ------------------------ 4) Notes contextuelles ---------------------------
        ev.notes.append(f"Segment : {segment} (base : 1={base['rate_1']*100:.0f}% / X={base['rate_X']*100:.0f}% / 2={base['rate_2']*100:.0f}%)")
        ev.notes.append(f"Buts attendus segment : {base['avg_goals']:.2f}  BTTS : {base['btts']*100:.0f}%")
        ev.notes.append(f"Profile match : {profile}")
        # Highlight FS HT_X
        if segment == "FS":
            ev.notes.append("⭐ FS : HT_X → FT_1 à 44%! (forte chance d'home win après nul HT)")

        return ev


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================
def print_evaluation(ev: StrategyEvaluation):
    """Affichage lisible d'une évaluation."""
    print(f"\n{'═'*100}")
    print(f"  📊 {ev.team_a} vs {ev.team_b}  ({ev.segment})")
    print(f"{'═'*100}")
    for note in ev.notes:
        print(f"  ℹ️  {note}")
    print(f"\n  ✅ SIGNAUX POSITIFS ({len(ev.base_signals)}) :")
    for sig in ev.base_signals:
        print(f"     [{sig.pick}] {sig.reason}  (strength {sig.strength:.2f})")
    print(f"\n  ⚠️  TRAPS ({len(ev.traps)}) :")
    for trap in ev.traps:
        print(f"     [{trap.pick}] {trap.reason}  (strength {trap.strength:.2f})")
    print(f"\n  🎯 RECOMMENDED PICKS :")
    if not ev.recommended_picks:
        print(f"     (aucun pick conviction suffisante — skip ou attendre)")
    for p in ev.recommended_picks:
        print(f"     • {p['pick']} @{p['cote']:.2f}  (strength {p['strength']:.2f}, {p['n_supporting']} signaux)")
    if ev.score_signals:
        print(f"\n  📊 SCORES PROBABLES (profile {ev.profile}) :")
        for s in ev.score_signals:
            print(f"     • {s['score']:<6} : {s['rate']*100:.0f}%  — {s['reason']}")
