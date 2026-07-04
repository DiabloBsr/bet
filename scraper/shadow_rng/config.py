"""Configuration centrale du Shadow-RNG Ensemble System.

Tout est paramétrable ici : aucune valeur magique en dur dans le code.
Surcharge possible à l'instanciation : DistributionProfiler(config={...}).
Les surcharges sont fusionnées (deep merge) sur DEFAULT_CONFIG.
"""
from __future__ import annotations

import copy

DEFAULT_CONFIG: dict = {
    # ---- espace des scores ----
    "max_goals": 7,          # grille 0..6 sur chaque axe ; >=7 buts -> clampé à 6
    # ---- fenêtres glissantes (en nb de matchs, du plus récent) ----
    "windows": [50, 200, 500, 1000],
    "default_window": 200,   # fenêtre par défaut des snapshots / get_bias
    "min_window_matches": 30,  # en-dessous : pas de stat fiable -> low confidence
    # ---- statistiques ----
    "alpha": 0.05,           # seuil de significativité (tests bilatéraux)
    "fdr_q": 0.10,           # Benjamini-Hochberg pour le snapshot multi-scores
    # ---- détection de régime (entropie de la distribution réalisée) ----
    "regime_window": 50,
    "regime_z_high": 1.5,    # z(entropie) > +seuil -> haute entropie (dispersé)
    "regime_z_low": -1.5,    # z(entropie) < -seuil -> basse entropie (concentré)
    # ---- tests de mémoire ----
    "memory_lags": [1, 2],
    # ---- mapping des colonnes du DataFrame fourni à fit() ----
    "columns": {
        "odds_home": "oh",
        "odds_draw": "od",
        "odds_away": "oa",
        "score_home": "sa",
        "score_away": "sb",
        "timestamp": "es",   # optionnel ; si absent, on suppose df déjà trié
    },
    # ---- perf ----
    "round_odds_cache": 2,   # arrondi des cotes pour le cache d'inversion
    "log_level": "INFO",
    # ---- BRIQUE B : simulateurs ----
    "simulators": {
        "n_iterations": 10000,       # tirages Monte-Carlo par simulateur (1000 pour tests)
        "random_seed": None,         # int -> reproductible ; None -> non
        "kl_epsilon": 1e-4,          # KL(cible||baseline) < eps -> simulateur "= baseline"
        # TREND
        "trend_window": 50,          # fenêtre de biais utilisée par TREND
        "trend_ratio_clip": [0.3, 3.0],  # bornes du facteur réel/théo (anti-explosion)
        # MEMORY
        "memory_strength": 1.0,      # exposant du lift de transition (0 = ignore)
        "memory_min_count": 30,      # occurrences min d'un score précédent pour une transition fiable
        "memory_smoothing": 0.5,     # lissage de Laplace de la matrice de transition
        "memory_lift_clip": [0.25, 4.0],  # bornes du lift cond/marg (anti-explosion scores rares)
        # REGIME
        "regime_strength": 0.15,     # mélange vers l'uniforme si haute entropie
        "regime_sharpen": 0.15,      # exposant de concentration si basse entropie
        # poids du vote (consommés par la Brique C)
        "weights": {"BASELINE": 0.15, "TREND": 0.40, "MEMORY": 0.25, "REGIME": 0.20},
    },
    # ---- BRIQUE C : vote ensemble + divergence ----
    "ensemble": {
        "weights": {"BASELINE": 0.15, "TREND": 0.40, "MEMORY": 0.25, "REGIME": 0.20},
        "baseline_floor": 0.15,      # poids plancher garanti à BASELINE
        "main_window": 500,          # fenêtre principale : 500 pour la PUISSANCE FDR
                                     #   (à 200, le FDR rate même un biais fort — cf. validation)
        "confirm_window": 200,       # 2e fenêtre pour la confirmation multi-window (sens du biais)
        "consensus_top_n": 3,
        # condition (a) de divergence : ampleur GLOBALE du biais FDR (pas la KL per-match)
        "divergence_min_scores": 2,  # N : nb min de scores FDR-significatifs
        "divergence_amplitude": 0.02,  # ampleur moyenne min des biais FDR (fraction, 0.02 = 2pp)
        "kl_threshold_info": 0.05,   # INFORMATIF seulement (loggué, ne conditionne plus l'alerte)
    },
    # ---- slot cross-marché (réservé, non implémenté) ----
    "cross_market_slot": {
        "available": False,
        "reason": "correct_score_odds_not_available",
        "data_required": ["correct_score_odds", "btts_odds", "ou25_odds"],
    },
    # ---- ÉTAPE 7 : runtime du script d'intégration (production) ----
    "runtime": {
        "league": "InstantLeague-8035",  # ligue surveillée
        "history_size": 2000,            # nb de matchs récents pour le fit initial
        "refit_every": 50,               # re-fit du profiler toutes les M itérations
        "refit_window": 2000,            # taille de la fenêtre glissante au re-fit
        "sleep_seconds": 30,             # pause entre 2 itérations de la boucle
        "max_iterations": None,          # None = boucle infinie ; int = arrêt (tests)
        "predictions_table": "shadow_rng_predictions",
    },
}


def merge_config(overrides: dict | None) -> dict:
    """Deep-merge des surcharges sur DEFAULT_CONFIG (copie ; ne mute pas le défaut)."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if not overrides:
        return cfg
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k] = {**cfg[k], **v}
        else:
            cfg[k] = v
    return cfg
