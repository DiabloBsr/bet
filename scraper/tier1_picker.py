"""TIER 1 PICKER — sélectionne uniquement les picks haute confiance (WR ≥ 72%).

Filtres backtest validés sur 1 621 matchs out-of-sample :
- TIER 1 ULTRA (~5% picks, 82% WR) : FT V5 ≥ 75%
- TIER 1 STRICT (~11% picks, 78% WR) : FT V5 ≥ 70%
- TIER 1 STANDARD (~24% picks, 72% WR) : cote ≤ 1.7 + V5 ≥ 60% + 0 trap
- TIER 2 CONFIRM (V5 ≥ 60% + signal GOLD/PAIRE OR) : ~75% WR

Usage:
    from scraper.tier1_picker import classify_pick
    tier, reason = classify_pick(ft_pick, ft_p, cote_ft, n_traps, n_gold_signals)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


TIER_DEFINITIONS = {
    "TIER_1_ULTRA":    {"min_v5": 0.75, "max_cote": 1.30, "label": "🟢🟢🟢 ULTRA — 82% WR attendu"},
    "TIER_1_STRICT":   {"min_v5": 0.70, "max_cote": 1.50, "label": "🟢🟢 STRICT — 78% WR attendu"},
    "TIER_1_STANDARD": {"min_v5": 0.60, "max_cote": 1.70, "label": "🟢 STANDARD — 72% WR attendu"},
    "TIER_2_MODERATE": {"min_v5": 0.55, "max_cote": 2.00, "label": "🟡 MODERATE — 67% WR attendu"},
}


@dataclass
class Tier1Pick:
    pick: str           # "1" | "X" | "2"
    cote: float
    v5_conf: float
    tier: str
    reason: str
    expected_wr: float


def classify_pick(ft_pick: str, ft_p: float, cote_ft: Optional[float],
                   n_traps: int = 0, n_gold_signals: int = 0,
                   team_form_drop: bool = False) -> Optional[Tier1Pick]:
    """Classe un pick FT V5 dans son TIER de confiance.

    Args:
        ft_pick: "1" / "X" / "2"
        ft_p: probabilité V5 (0-1)
        cote_ft: cote du pick (1X2)
        n_traps: nb de TRAP signaux qui ciblent ce pick
        n_gold_signals: nb de signaux GOLD (PAIRE OR, MULTI, BRACKET GOLD)
        team_form_drop: l'équipe est en perte de forme saison

    Returns:
        Tier1Pick si éligible, None sinon (skip)
    """
    if cote_ft is None or ft_pick not in ("1", "X", "2"):
        return None

    # Filtres bloquants
    if n_traps >= 1 and ft_p < 0.65:
        return None  # TRAP signal trop fort sans confirmation
    if team_form_drop and ft_p < 0.65:
        return None  # équipe en perte

    # Promote si signaux GOLD multiples
    boosted_p = ft_p + min(0.10, n_gold_signals * 0.05)

    # Classement par TIER
    if boosted_p >= 0.75 and cote_ft <= 1.30 and n_traps == 0:
        return Tier1Pick(ft_pick, cote_ft, ft_p, "TIER_1_ULTRA",
                          f"V5 {ft_p*100:.0f}% (boost +{n_gold_signals*5}%) ≥ 75%, cote ≤ 1.30",
                          expected_wr=0.82)
    if boosted_p >= 0.70 and cote_ft <= 1.50 and n_traps == 0:
        return Tier1Pick(ft_pick, cote_ft, ft_p, "TIER_1_STRICT",
                          f"V5 {ft_p*100:.0f}% ≥ 70%, cote ≤ 1.50",
                          expected_wr=0.78)
    if boosted_p >= 0.60 and cote_ft <= 1.70 and n_traps == 0:
        return Tier1Pick(ft_pick, cote_ft, ft_p, "TIER_1_STANDARD",
                          f"V5 {ft_p*100:.0f}% ≥ 60%, cote ≤ 1.70, 0 trap",
                          expected_wr=0.72)
    if boosted_p >= 0.55 and cote_ft <= 2.00 and n_traps == 0 and n_gold_signals >= 1:
        return Tier1Pick(ft_pick, cote_ft, ft_p, "TIER_2_MODERATE",
                          f"V5 {ft_p*100:.0f}% ≥ 55% + {n_gold_signals} GOLD",
                          expected_wr=0.67)
    return None


def expected_outcome(picks: list[Tier1Pick]) -> dict:
    """Calcule l'outcome attendu d'une session de picks."""
    if not picks:
        return {"n": 0, "expected_wins": 0, "accuracy": 0}
    total_wr = sum(p.expected_wr for p in picks)
    expected_wins = total_wr
    accuracy = total_wr / len(picks)
    expected_pnl = sum(p.expected_wr * (p.cote - 1) - (1 - p.expected_wr) for p in picks)
    return {
        "n": len(picks),
        "expected_wins": round(expected_wins, 1),
        "accuracy": round(accuracy * 100, 1),
        "expected_pnl": round(expected_pnl, 2),
        "tiers": {tier: sum(1 for p in picks if p.tier == tier) for tier in TIER_DEFINITIONS},
    }
