"""Détection saison courante + blending stats globales × stats saison.

Une "saison" Bet261 virtual = 38 journées, ~30 min temps réel.
Les forces d'équipes oscillent de ±20pp WR home entre saisons.

Logique :
- Détecter dernière saison (depuis dernier reset J38 → J1 dans la BDD)
- Pour chaque équipe, calculer perf saison courante (typ. 1-5 matchs disponibles)
- Blending bayésien : effective = (n_season * season + prior_weight * global) / (n_season + prior_weight)
- prior_weight = 8 (équivalent à 8 matchs de prior global)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


PRIOR_WEIGHT = 8  # Force du prior global vs saison


@dataclass
class TeamSeasonStats:
    team: str
    n_home_season: int
    wr_home_season: float
    avg_goals_for_season: float
    avg_goals_against_season: float
    wr_home_global: float
    # Stats blendées (utilisées par le predictor)
    wr_home_effective: float
    goals_for_effective: float
    goals_against_effective: float
    # Score de confiance (basé sur n_season)
    season_confidence: float


def _bayesian_blend(season_val: float, n_season: int,
                     global_val: float, prior_weight: int = PRIOR_WEIGHT) -> float:
    """Blending bayésien : pondère saison vs global selon échantillon saison."""
    if n_season == 0:
        return global_val
    return (n_season * season_val + prior_weight * global_val) / (n_season + prior_weight)


def detect_current_season(engine: Engine, lookback_seasons: int = 5) -> tuple[pd.Timestamp, int]:
    """Trouve le début de la fenêtre "saison récente" (par défaut 5 dernières saisons).

    Une saison virtuelle Bet261 = ~30 min, ~38 journées. Une seule saison donne trop
    peu de matchs (souvent 0 fini). On combine donc les N dernières saisons pour
    capturer la "forme du moment" tout en gardant la réactivité.

    Returns: (start_timestamp_of_lookback_window, last_season_id_diagnostic)
    """
    df = pd.read_sql("""
        SELECT expected_start, round_info
        FROM events
        WHERE competition = 'InstantLeague-8035' AND round_info IS NOT NULL
        ORDER BY expected_start
    """, engine)
    df["expected_start"] = pd.to_datetime(df.expected_start)
    df["journee"] = pd.to_numeric(df.round_info, errors="coerce")
    df = df.dropna(subset=["journee"]).sort_values("expected_start").reset_index(drop=True)
    df["prev_j"] = df.journee.shift(1)
    df["jump_back"] = (df.prev_j > df.journee + 5)
    df["season_id"] = df.jump_back.cumsum().astype(int)

    last_season_id = int(df.season_id.max())
    # Démarre la fenêtre N saisons en arrière
    window_start_id = max(0, last_season_id - lookback_seasons + 1)
    window = df[df.season_id >= window_start_id]
    return window.expected_start.min(), last_season_id


def compute_season_stats(engine: Engine,
                          since_ts: Optional[pd.Timestamp] = None
                          ) -> Dict[str, TeamSeasonStats]:
    """Calcule les stats blendées (saison × global) pour chaque équipe.

    Si since_ts est None, détecte automatiquement la dernière saison.
    Retourne un dict {team_name: TeamSeasonStats}.
    """
    if since_ts is None:
        since_ts, _ = detect_current_season(engine)

    # ========= Stats GLOBALES (tout l'historique fini) =========
    global_df = pd.read_sql("""
        SELECT e.team_a, e.team_b, r.score_a, r.score_b
        FROM events e
        JOIN results r ON r.event_id = e.id
        WHERE r.score_a IS NOT NULL AND e.competition = 'InstantLeague-8035'
    """, engine)
    global_df["won_home"] = (global_df.score_a > global_df.score_b).astype(int)

    global_home = global_df.groupby("team_a").agg(
        n=("won_home", "count"),
        wr=("won_home", "mean"),
        gf=("score_a", "mean"),
        ga=("score_b", "mean"),
    ).rename(columns={"team_a": "team"})

    global_away = global_df.groupby("team_b").agg(
        gf_away=("score_b", "mean"),
        ga_away=("score_a", "mean"),
    )

    # ========= Stats SAISON COURANTE (matchs depuis since_ts) =========
    since_str = since_ts.strftime("%Y-%m-%d %H:%M:%S")  # format SQLite (espace, pas T)
    season_df = pd.read_sql(f"""
        SELECT e.team_a, e.team_b, r.score_a, r.score_b
        FROM events e
        JOIN results r ON r.event_id = e.id
        WHERE r.score_a IS NOT NULL AND e.competition = 'InstantLeague-8035'
          AND e.expected_start >= '{since_str}'
    """, engine)

    if len(season_df) > 0:
        season_df["won_home"] = (season_df.score_a > season_df.score_b).astype(int)
        season_home = season_df.groupby("team_a").agg(
            n_s=("won_home", "count"),
            wr_s=("won_home", "mean"),
            gf_s=("score_a", "mean"),
            ga_s=("score_b", "mean"),
        )
    else:
        season_home = pd.DataFrame(columns=["n_s", "wr_s", "gf_s", "ga_s"])

    # ========= BLENDING =========
    out: Dict[str, TeamSeasonStats] = {}
    all_teams = set(global_home.index) | set(season_home.index)
    for team in all_teams:
        g = global_home.loc[team] if team in global_home.index else None
        s = season_home.loc[team] if team in season_home.index else None

        wr_g = float(g["wr"]) if g is not None else 0.48
        gf_g = float(g["gf"]) if g is not None else 1.3
        ga_g = float(g["ga"]) if g is not None else 1.2

        n_s = int(s["n_s"]) if s is not None else 0
        wr_s = float(s["wr_s"]) if s is not None and n_s > 0 else wr_g
        gf_s = float(s["gf_s"]) if s is not None and n_s > 0 else gf_g
        ga_s = float(s["ga_s"]) if s is not None and n_s > 0 else ga_g

        wr_eff = _bayesian_blend(wr_s, n_s, wr_g)
        gf_eff = _bayesian_blend(gf_s, n_s, gf_g)
        ga_eff = _bayesian_blend(ga_s, n_s, ga_g)
        confidence = min(1.0, n_s / 5.0)  # 5 matchs = full confidence

        out[team] = TeamSeasonStats(
            team=team,
            n_home_season=n_s,
            wr_home_season=wr_s,
            avg_goals_for_season=gf_s,
            avg_goals_against_season=ga_s,
            wr_home_global=wr_g,
            wr_home_effective=wr_eff,
            goals_for_effective=gf_eff,
            goals_against_effective=ga_eff,
            season_confidence=confidence,
        )
    return out


def season_adjustment_factor(team_home: str, team_away: str,
                              stats: Dict[str, TeamSeasonStats]) -> dict:
    """Calcule un facteur multiplicatif à appliquer aux probas 1X2.

    Si l'équipe home performe MIEUX cette saison qu'en global → boost 1.
    Si elle performe MOINS BIEN → boost 2.
    """
    sh = stats.get(team_home)
    sa = stats.get(team_away)
    if sh is None or sa is None:
        return {"home_factor": 1.0, "away_factor": 1.0, "note": "no_data"}

    # Différence saison vs global
    home_delta = sh.wr_home_effective - sh.wr_home_global  # positif = forme
    home_strength = home_delta * sh.season_confidence
    away_strength = -home_delta * sh.season_confidence  # inverse

    # Facteurs (max ±15%)
    home_factor = 1.0 + max(-0.15, min(0.15, home_strength * 0.5))
    away_factor = 1.0 + max(-0.15, min(0.15, away_strength * 0.5))

    note = ""
    if sh.season_confidence >= 0.6:
        if home_delta > 0.10:
            note = f"🔥 {team_home} EN FORME saison ({sh.wr_home_season*100:.0f}% vs {sh.wr_home_global*100:.0f}% global, n={sh.n_home_season})"
        elif home_delta < -0.10:
            note = f"❄️  {team_home} EN PERTE saison ({sh.wr_home_season*100:.0f}% vs {sh.wr_home_global*100:.0f}% global, n={sh.n_home_season})"

    return {
        "home_factor": home_factor,
        "away_factor": away_factor,
        "home_delta": home_delta,
        "season_confidence": sh.season_confidence,
        "n_home_season": sh.n_home_season,
        "note": note,
    }
