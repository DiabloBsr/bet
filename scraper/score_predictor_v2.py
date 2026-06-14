"""SCORE PREDICTOR V2 — Ensemble multi-source pour score exact.

Combine 4 sources :
1. V5 Poisson + Dixon-Coles (déjà existant — base théorique)
2. PAIR-SPECIFIC empirical (P(score | team_a, team_b, segment) live depuis BDD)
3. PROFILE_SCORES par segment×profile (heuristique global)
4. Market cotes "Score exact" (extra_markets)

Output : top-N scores avec proba ensemble (réduit la variance, capture les patterns pair-spécifiques).
"""
from __future__ import annotations
from collections import Counter
import math
from typing import Optional
import pandas as pd
from sqlalchemy.engine import Engine
from scraper.strategy_engine import label_segment, PROFILE_SCORES, classify_profile


# ============================================================================
# 🎯 1. PAIR-SPECIFIC EMPIRICAL — dynamique depuis BDD
# ============================================================================

class PairScoreCache:
    """Cache des distributions de scores par paire (team_a, team_b, segment)."""

    def __init__(self, engine: Engine, min_n: int = 5):
        self.engine = engine
        self.min_n = min_n
        self._cache: dict[tuple[str, str, str], dict[str, float]] = {}
        self._loaded = False

    def _load(self):
        """Charge l'historique complet en mémoire (une seule fois)."""
        df = pd.read_sql("""
            SELECT e.round_info, e.team_a, e.team_b, r.score_a, r.score_b
            FROM events e
            JOIN results r ON r.event_id = e.id
            WHERE r.score_a IS NOT NULL AND e.round_info IS NOT NULL AND e.round_info != '0' AND e.competition = 'InstantLeague-8035'
        """, self.engine)
        df["journee"] = pd.to_numeric(df.round_info, errors="coerce")
        df["segment"] = df.journee.apply(label_segment)
        df = df[df.segment.notna()].copy()
        df["score"] = df.apply(lambda r: f"{int(r.score_a)}-{int(r.score_b)}", axis=1)

        # Group by (team_a, team_b, segment)
        for (ta, tb, seg), grp in df.groupby(["team_a", "team_b", "segment"]):
            if len(grp) < self.min_n: continue
            counts = grp.score.value_counts().to_dict()
            total = sum(counts.values())
            self._cache[(ta, tb, seg)] = {s: c/total for s, c in counts.items()}

        # Cache aussi par paire toutes saisons confondues
        for (ta, tb), grp in df.groupby(["team_a", "team_b"]):
            if len(grp) < self.min_n: continue
            counts = grp.score.value_counts().to_dict()
            total = sum(counts.values())
            self._cache[(ta, tb, "ALL")] = {s: c/total for s, c in counts.items()}
        self._loaded = True

    def get(self, team_a: str, team_b: str, segment: str) -> tuple[dict[str, float], int]:
        """Retourne (distribution_scores, n_samples) — segment-spécifique si possible."""
        if not self._loaded:
            self._load()
        # Priorité 1 : segment-spécifique
        d = self._cache.get((team_a, team_b, segment))
        if d:
            n = sum(1 for _ in d)  # approximation
            return d, n
        # Priorité 2 : toutes saisons
        d = self._cache.get((team_a, team_b, "ALL"))
        if d:
            return d, sum(1 for _ in d)
        return {}, 0


# ============================================================================
# 🎯 2. FORM-ADJUSTED Poisson lambdas
# ============================================================================

def adjust_lambdas_for_form(lam_h: float, lam_a: float,
                              wr_h_season_delta: float = 0.0,
                              wr_a_season_delta: float = 0.0) -> tuple[float, float]:
    """Ajuste les lambdas Poisson en fonction de l'écart de forme saison.

    Si home en grosse forme (Δ +15pp), on boost lam_h (+15%).
    Si home en perte (Δ -10pp), on réduit lam_h (-10%).
    """
    boost_h = 1.0 + max(-0.20, min(0.20, wr_h_season_delta * 1.5))
    boost_a = 1.0 + max(-0.20, min(0.20, wr_a_season_delta * 1.5))
    return lam_h * boost_h, lam_a * boost_a


# ============================================================================
# 🎯 3. MARKET-INFERRED probabilities depuis cotes Score exact
# ============================================================================

def market_score_distribution(extra_markets) -> dict[str, float]:
    """Extrait P(score) à partir des cotes opérateur (avec dévigorisation)."""
    if not extra_markets: return {}
    # Si stocké en string JSON
    if isinstance(extra_markets, str):
        try:
            import json
            extra_markets = json.loads(extra_markets)
        except Exception:
            return {}
    if not isinstance(extra_markets, dict):
        return {}
    score_market = None
    for key in ("Score exact", "score_exact", "exactScore"):
        if key in extra_markets:
            score_market = extra_markets[key]
            break
    if not score_market or not isinstance(score_market, dict):
        return {}
    # Inverser cotes → proba implicite
    raw = {s: 1/c for s, c in score_market.items() if isinstance(c, (int, float)) and c > 1}
    if not raw: return {}
    # Normaliser (retirer la marge bookmaker)
    total = sum(raw.values())
    if total <= 0: return {}
    return {s: p/total for s, p in raw.items()}


# ============================================================================
# 🎯 4. ENSEMBLE PREDICTOR
# ============================================================================

class ScorePredictorV2:
    """Score predictor ensemble : V5 + pair + profile + market."""

    def __init__(self, engine: Engine,
                  weights: Optional[dict] = None):
        self.engine = engine
        self.pair_cache = PairScoreCache(engine)
        # Poids par défaut (à tuner via backtest)
        self.weights = weights or {
            "v5": 0.40,
            "pair": 0.30,
            "profile": 0.15,
            "market": 0.15,
        }

    def predict(self, team_a: str, team_b: str, journee: int,
                  v5_score_grid: dict[str, float],
                  extra_markets: Optional[dict] = None,
                  odds_h: Optional[float] = None,
                  odds_a: Optional[float] = None,
                  wr_h_season_delta: float = 0.0,
                  wr_a_season_delta: float = 0.0,
                  top_n: int = 5) -> list[tuple[str, float, dict]]:
        """Prédit top-N scores. Retourne [(score, proba, sources_breakdown), ...].

        Args:
            v5_score_grid: dict {score_str: proba_v5} déjà calculé par V5 (form-adjusted)
            extra_markets: cotes opérateur (incl Score exact)
            odds_h / odds_a: cotes 1X2 pour classify_profile
            wr_h_season_delta, wr_a_season_delta: déjà appliqués au V5 grid (optionnel)
        """
        segment = label_segment(journee) or "MS_mid"

        # 1. V5 grid (déjà form-adjusted upstream)
        v5_dist = v5_score_grid.copy() if v5_score_grid else {}

        # 2. Pair empirical
        pair_dist, pair_n = self.pair_cache.get(team_a, team_b, segment)

        # 3. Profile scores (top 3 only — heuristique)
        profile_dist = {}
        if odds_h and odds_a:
            profile = classify_profile(odds_h, odds_a)
            pdata = PROFILE_SCORES.get(segment, {}).get(profile)
            if pdata:
                profile_dist = pdata["top3"].copy()  # déjà normalisé

        # 4. Market
        market_dist = market_score_distribution(extra_markets) if extra_markets else {}

        # Ensemble : merge all distributions
        all_scores = set(v5_dist) | set(pair_dist) | set(profile_dist) | set(market_dist)
        ensemble = {}
        breakdown = {}
        for s in all_scores:
            v_score = v5_dist.get(s, 0)
            p_score = pair_dist.get(s, 0)
            pr_score = profile_dist.get(s, 0)
            m_score = market_dist.get(s, 0)
            # Si pair_dist disponible, réduire le poids du profile
            if pair_dist:
                w = self.weights
            else:
                # Pas de pair → redistribuer
                w = {"v5": 0.55, "pair": 0.0, "profile": 0.25, "market": 0.20}
            score = (w["v5"] * v_score + w["pair"] * p_score
                     + w["profile"] * pr_score + w["market"] * m_score)
            ensemble[s] = score
            breakdown[s] = {"v5": v_score, "pair": p_score, "profile": pr_score, "market": m_score}

        # Normaliser
        total = sum(ensemble.values())
        if total > 0:
            ensemble = {s: p/total for s, p in ensemble.items()}

        # Top N
        sorted_scores = sorted(ensemble.items(), key=lambda x: -x[1])[:top_n]
        return [(s, p, breakdown.get(s, {})) for s, p in sorted_scores]
