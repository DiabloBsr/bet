"""VFoot-ML — PHASE 2 : Feature Engineering avancé.

Construit, à partir du dataset nettoyé (Phase 1), une matrice de features
**strictement causale** (aucune fuite : chaque variable n'utilise que le passé).

Groupes de features :
  A) séquences      — séries, écarts, streaks Over/Under
  B) cotes          — probas implicites, ratios, spread, volatilité
  C) temporelles    — heure, jour, position dans la journée, délai
  D) par équipe     — winrate/buts/forme historiques (expanding/rolling, shift(1))
  E) détection RNG  — entropie de Shannon, runs test, répétition de score

RÈGLE ABSOLUE : toute feature au temps t n'utilise QUE les matchs < t (shift(1)).
Les premières lignes (warmup des fenêtres) sont retirées.

Usage :
    from phase2_features import FeatureBuilder
    fb = FeatureBuilder("data/vfoot_ml/matches.csv")
    X = fb.build_all()                      # DataFrame features + cibles
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase1_eda import VirtualFootballEDA   # réutilise le nettoyage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
logger = logging.getLogger("vfoot.features")


class FeatureBuilder:
    """Génère la matrice de features causale pour la modélisation."""

    def __init__(self, source, warmup: int = 20):
        self.source = source
        self.warmup = warmup          # nb de lignes initiales à retirer (fenêtres incomplètes)
        self.df: pd.DataFrame | None = None

    # ------------------------------------------------------------------ #
    def _load(self) -> pd.DataFrame:
        """Charge + nettoie via le pipeline Phase 1, ajoute les cibles."""
        eda = VirtualFootballEDA(self.source)
        df = eda.load_and_clean().copy()
        df["exact_score"] = df.score_home.astype(str) + "-" + df.score_away.astype(str)
        # cibles
        df["y_1x2"] = df.result                     # classif principale
        df["y_home_goals"] = df.score_home          # Poisson
        df["y_away_goals"] = df.score_away
        df["y_over25"] = (df.total_goals > 2.5).astype(int)
        df["y_btts"] = df.btts
        self.df = df.reset_index(drop=True)
        return self.df

    # ------------------------------------------------------------------ #
    @staticmethod
    def _streak_before(cond: pd.Series) -> pd.Series:
        """Longueur de la série de True se terminant JUSTE AVANT chaque ligne (causal)."""
        s = cond.astype(int)
        grp = (s == 0).cumsum()                      # nouvel id à chaque False
        run = s.groupby(grp).cumsum()                # longueur de run incluant la ligne
        return run.shift(1).fillna(0)                # run se terminant avant la ligne

    # ------------------------------------------------------------------ #
    def build_sequence_features(self, df):
        """A) Séquences globales (causales)."""
        h = (df.result == "1")
        df["seq_home_wins_5"] = h.shift(1).rolling(5, min_periods=1).sum()
        df["seq_home_wins_10"] = h.shift(1).rolling(10, min_periods=1).sum()
        df["seq_no_draw_streak"] = self._streak_before(df.result != "X")
        df["seq_over25_streak"] = self._streak_before(df.total_goals > 2.5)
        df["seq_gap_since_draw"] = self._streak_before(df.result != "X")   # = matchs depuis dernier nul
        return df

    def build_odds_features(self, df):
        """B) Cotes (le signal le plus fort attendu)."""
        inv = 1 / df.odd_1 + 1 / df.odd_x + 1 / df.odd_2
        df["imp_1"] = (1 / df.odd_1) / inv
        df["imp_x"] = (1 / df.odd_x) / inv
        df["imp_2"] = (1 / df.odd_2) / inv
        df["odds_ratio_1_2"] = df.odd_1 / df.odd_2
        df["odds_spread"] = df[["odd_1", "odd_x", "odd_2"]].max(axis=1) - df[["odd_1", "odd_x", "odd_2"]].min(axis=1)
        df["fav_strength"] = df[["imp_1", "imp_2"]].max(axis=1)
        df["lambda_tot_impl"] = -np.log(df.imp_x.clip(1e-6)) * 1.0   # proxy d'intensité (info implicite)
        return df

    def build_temporal_features(self, df):
        """C) Temporelles."""
        df["hour"] = df.datetime.dt.hour
        df["dayofweek"] = df.datetime.dt.dayofweek
        df["date"] = df.datetime.dt.date
        df["match_pos_in_day"] = df.groupby("date").cumcount() + 1
        df["minutes_since_prev"] = df.datetime.diff().dt.total_seconds().div(60).fillna(0).clip(0, 600)
        return df

    def build_team_features(self, df):
        """D) Par équipe — historiques CAUSAUX (expanding/rolling + shift(1))."""
        df = df.reset_index(drop=True).copy()
        df["mid"] = df.index
        recs = []
        for r in df.itertuples():
            recs.append({"mid": r.mid, "role": "home", "team": r.home_team,
                         "gf": r.score_home, "ga": r.score_away, "is_home": 1,
                         "win": int(r.score_home > r.score_away),
                         "pts": 3 if r.score_home > r.score_away else (1 if r.score_home == r.score_away else 0)})
            recs.append({"mid": r.mid, "role": "away", "team": r.away_team,
                         "gf": r.score_away, "ga": r.score_home, "is_home": 0,
                         "win": int(r.score_away > r.score_home),
                         "pts": 3 if r.score_away > r.score_home else (1 if r.score_home == r.score_away else 0)})
        L = pd.DataFrame(recs).sort_values(["team", "mid"]).reset_index(drop=True)
        g = L.groupby("team")
        L["winrate_hist"] = g["win"].transform(lambda s: s.shift(1).expanding().mean())
        L["gf_avg"] = g["gf"].transform(lambda s: s.shift(1).expanding().mean())
        L["ga_avg"] = g["ga"].transform(lambda s: s.shift(1).expanding().mean())
        L["form5"] = g["pts"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        L["venue_winrate"] = L.groupby(["team", "is_home"])["win"].transform(
            lambda s: s.shift(1).expanding().mean())
        feats = ["winrate_hist", "gf_avg", "ga_avg", "form5", "venue_winrate"]
        home = L[L.role == "home"].set_index("mid")[feats].add_prefix("home_")
        away = L[L.role == "away"].set_index("mid")[feats].add_prefix("away_")
        df = df.set_index("mid").join(home).join(away).reset_index(drop=True)
        df["winrate_diff"] = df.home_winrate_hist - df.away_winrate_hist
        df["form_diff"] = df.home_form5 - df.away_form5
        return df

    @staticmethod
    def _shannon(window_vals):
        v, c = np.unique(window_vals, return_counts=True)
        p = c / c.sum()
        return float(-(p * np.log2(p)).sum())

    @staticmethod
    def _runs_z(binary_window):
        b = np.asarray(binary_window); n1, n0 = b.sum(), len(b) - b.sum()
        if n1 == 0 or n0 == 0:
            return 0.0
        runs = 1 + (b[1:] != b[:-1]).sum()
        mu = 1 + 2 * n1 * n0 / (n1 + n0)
        var = (2 * n1 * n0 * (2 * n1 * n0 - n1 - n0)) / ((n1 + n0) ** 2 * (n1 + n0 - 1))
        return float((runs - mu) / np.sqrt(var)) if var > 0 else 0.0

    def build_rng_features(self, df):
        """E) Détection RNG (rolling, causal). Benford = non applicable (scores 0-6,
        un seul chiffre) -> on retient entropie + runs + répétition de score."""
        code = df.result.map({"1": 0, "X": 1, "2": 2})
        df["rng_entropy_10"] = code.shift(1).rolling(10).apply(self._shannon, raw=True)
        hb = (df.result == "1").astype(int)
        df["rng_runs_z_20"] = hb.shift(1).rolling(20).apply(self._runs_z, raw=True)
        prev = df.exact_score.shift(1)
        df["rng_score_repeat"] = (prev == df.exact_score.shift(2)).astype(int)
        return df

    # ------------------------------------------------------------------ #
    def build_all(self, save: str = "data/vfoot_ml/features.parquet") -> pd.DataFrame:
        """Assemble toutes les features, retire le warmup, sauvegarde."""
        df = self._load()
        df = self.build_sequence_features(df)
        df = self.build_odds_features(df)
        df = self.build_temporal_features(df)
        df = self.build_team_features(df)
        df = self.build_rng_features(df)

        df = df.iloc[self.warmup:].reset_index(drop=True)   # retire les fenêtres incomplètes
        # impute le résiduel (1ères apparitions d'équipe) par des valeurs neutres
        feat_cols = [c for c in df.columns if c.startswith(
            ("seq_", "imp_", "odds_", "fav_", "lambda_", "hour", "day", "match_pos",
             "minutes_", "home_", "away_", "winrate_", "form_", "rng_"))]
        df[feat_cols] = df[feat_cols].fillna(df[feat_cols].median(numeric_only=True))
        try:
            df.to_parquet(save)
        except Exception:
            save = save.replace(".parquet", ".csv"); df.to_csv(save, index=False)
        logger.info("features: %d lignes x %d features -> %s", len(df), len(feat_cols), save)
        self.feature_cols = feat_cols
        return df


if __name__ == "__main__":
    fb = FeatureBuilder("data/vfoot_ml/matches.csv")
    X = fb.build_all()
    print("\n" + "=" * 70)
    print(f"  PHASE 2 — {len(X)} lignes x {len(fb.feature_cols)} features")
    print("=" * 70)
    print("Features:", fb.feature_cols)
    # aperçu corrélation (point-biserial) avec victoire domicile -- HONNÊTE
    hw = (X.y_1x2 == "1").astype(int)
    cor = {c: round(float(X[c].corr(hw)), 3) for c in fb.feature_cols
           if pd.api.types.is_numeric_dtype(X[c])}
    cor = dict(sorted(cor.items(), key=lambda kv: -abs(kv[1])))
    print("\nTop 12 |corr| avec victoire domicile :")
    for k, v in list(cor.items())[:12]:
        print(f"   {k:<24} {v:+.3f}")
    print("\nFeatures de séquence/RNG (corr attendue ~0) :")
    for k in ["seq_home_wins_10", "seq_no_draw_streak", "seq_over25_streak",
              "rng_entropy_10", "rng_runs_z_20", "rng_score_repeat", "minutes_since_prev"]:
        if k in cor:
            print(f"   {k:<24} {cor[k]:+.3f}")
