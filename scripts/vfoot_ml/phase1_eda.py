"""VFoot-ML — PHASE 1 : Analyse Exploratoire Complète (EDA).

Module modulaire et réutilisable pour l'analyse exploratoire d'un dataset de
football virtuel (Bet261 / RNG). Couvre : nettoyage, statistiques descriptives,
analyse des cotes (value + marge), analyse de séquences/cycles, visualisations.

Conçu pour tourner sur un PC modeste (8 Go RAM). Pandas/NumPy/SciPy/Matplotlib.

Usage :
    from phase1_eda import VirtualFootballEDA
    eda = VirtualFootballEDA("data/vfoot_ml/matches.csv")
    eda.run_all()                       # tout + plots + résumé
    print(eda.summary)                  # dict récapitulatif

Ou en ligne de commande :
    python scripts/vfoot_ml/phase1_eda.py
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")               # backend headless (pas de display requis)
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
logger = logging.getLogger("vfoot.eda")


class VirtualFootballEDA:
    """Pipeline d'analyse exploratoire pour un dataset de foot virtuel.

    Le dataset attendu contient au minimum :
        datetime, home_team, away_team, odd_1, odd_x, odd_2, score_home, score_away

    Tous les résultats numériques sont accumulés dans `self.summary` (dict
    JSON-sérialisable) et les figures sauvegardées dans `plots_dir`.
    """

    REQUIRED = ["datetime", "home_team", "away_team",
                "odd_1", "odd_x", "odd_2", "score_home", "score_away"]

    def __init__(self, source, plots_dir: str = "data/vfoot_ml/plots"):
        """source : chemin CSV (str/Path) OU DataFrame déjà chargé."""
        self.source = source
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.df: Optional[pd.DataFrame] = None
        self.summary: dict = {}

    # ------------------------------------------------------------------ #
    # 1) CHARGEMENT + NETTOYAGE
    # ------------------------------------------------------------------ #
    def load_and_clean(self) -> pd.DataFrame:
        """Charge, valide le schéma, convertit les types, traite manquants,
        doublons et anomalies. Retourne le DataFrame nettoyé."""
        try:
            df = self.source.copy() if isinstance(self.source, pd.DataFrame) \
                else pd.read_csv(self.source)
        except Exception as exc:
            logger.error("Échec du chargement : %s", exc)
            raise

        missing_cols = [c for c in self.REQUIRED if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Colonnes manquantes : {missing_cols}")

        n0 = len(df)
        # types
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        for c in ["odd_1", "odd_x", "odd_2", "score_home", "score_away"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # valeurs manquantes : on supprime les lignes inexploitables
        before = len(df)
        df = df.dropna(subset=self.REQUIRED)
        n_missing = before - len(df)

        # anomalies : cotes <= 1 (impossibles), scores négatifs
        anom = (df.odd_1 <= 1) | (df.odd_x <= 1) | (df.odd_2 <= 1) | \
               (df.score_home < 0) | (df.score_away < 0)
        n_anom = int(anom.sum())
        df = df[~anom]

        # doublons : même match (équipes + datetime) — JAMAIS sur le score
        before = len(df)
        df = df.drop_duplicates(subset=["datetime", "home_team", "away_team"])
        n_dup = before - len(df)

        df = df.sort_values("datetime").reset_index(drop=True)

        # colonnes dérivées de base (réutilisées partout)
        df["total_goals"] = df.score_home + df.score_away
        df["result"] = np.where(df.score_home > df.score_away, "1",
                        np.where(df.score_home == df.score_away, "X", "2"))
        df["btts"] = ((df.score_home > 0) & (df.score_away > 0)).astype(int)
        df["hour"] = df.datetime.dt.hour

        self.df = df
        self.summary["cleaning"] = {
            "n_lignes_brutes": n0, "n_manquantes_supprimees": int(n_missing),
            "n_anomalies_supprimees": n_anom, "n_doublons_supprimes": int(n_dup),
            "n_final": len(df),
            "periode": [str(df.datetime.min()), str(df.datetime.max())],
            "n_equipes": int(pd.concat([df.home_team, df.away_team]).nunique()),
        }
        logger.info("Nettoyage : %d -> %d lignes (%d manquantes, %d anomalies, %d doublons)",
                    n0, len(df), n_missing, n_anom, n_dup)
        return df

    # ------------------------------------------------------------------ #
    # 2) STATISTIQUES DESCRIPTIVES
    # ------------------------------------------------------------------ #
    def descriptive_stats(self) -> dict:
        """Distribution 1/X/2, scores exacts fréquents, buts moyens, Over/Under,
        BTTS, victoires domicile vs extérieur."""
        d = self.df
        res_dist = (d.result.value_counts(normalize=True) * 100).round(2).to_dict()
        d["exact_score"] = d.score_home.astype(str) + "-" + d.score_away.astype(str)
        top_scores = (d.exact_score.value_counts(normalize=True).head(10) * 100).round(2).to_dict()
        ou = {f"Over_{ln}": round(100 * (d.total_goals > ln).mean(), 2)
              for ln in (1.5, 2.5, 3.5)}
        ou.update({f"Under_{ln}": round(100 * (d.total_goals < ln).mean(), 2)
                   for ln in (1.5, 2.5, 3.5)})
        out = {
            "distribution_1X2_pct": res_dist,
            "top10_scores_exacts_pct": top_scores,
            "buts_moyens": {"domicile": round(d.score_home.mean(), 3),
                            "exterieur": round(d.score_away.mean(), 3),
                            "total": round(d.total_goals.mean(), 3)},
            "over_under_pct": ou,
            "btts_oui_pct": round(100 * d.btts.mean(), 2),
            "victoire_domicile_pct": round(100 * (d.result == "1").mean(), 2),
            "victoire_exterieur_pct": round(100 * (d.result == "2").mean(), 2),
            "nul_pct": round(100 * (d.result == "X").mean(), 2),
        }
        self.summary["descriptif"] = out
        return out

    # ------------------------------------------------------------------ #
    # 3) ANALYSE DES COTES (calibration, value, marge)
    # ------------------------------------------------------------------ #
    def odds_analysis(self, n_bins: int = 10) -> dict:
        """Probabilités implicites vs réalisé (calibration), marge bookmaker,
        détection de tranches de cotes sous/sur-évaluées (value bets)."""
        d = self.df
        inv = 1 / d.odd_1 + 1 / d.odd_x + 1 / d.odd_2
        d["margin"] = inv - 1.0                       # overround
        # proba implicite dé-marginée de l'issue domicile
        d["p_home_impl"] = (1 / d.odd_1) / inv
        d["home_win"] = (d.result == "1").astype(int)

        # calibration par déciles de proba implicite domicile
        d["p_bin"] = pd.qcut(d.p_home_impl, n_bins, labels=False, duplicates="drop")
        calib = d.groupby("p_bin").apply(
            lambda g: pd.Series({"n": len(g),
                                 "implied": round(100 * g.p_home_impl.mean(), 2),
                                 "realized": round(100 * g.home_win.mean(), 2)}))
        calib["gap_pp"] = (calib.realized - calib.implied).round(2)

        # value : ROI flat en pariant la victoire domicile par tranche de cote
        d["odd_band"] = pd.cut(d.odd_1, [1, 1.5, 2, 2.5, 3, 4, 6, 100])
        value = d.groupby("odd_band", observed=True).apply(
            lambda g: pd.Series({"n": len(g),
                                 "roi_home_pct": round(100 * (g.home_win * g.odd_1 - 1).mean(), 2)}))

        out = {
            "marge_bookmaker_pct": {"mediane": round(100 * d.margin.median(), 3),
                                    "min": round(100 * d.margin.min(), 3),
                                    "max": round(100 * d.margin.max(), 3)},
            "calibration_par_decile": calib.reset_index().to_dict("records"),
            "value_par_tranche_cote_domicile": value.reset_index().astype(str).to_dict("records"),
            "ecart_calibration_moyen_abs_pp": round(float(calib.gap_pp.abs().mean()), 3),
        }
        self.summary["cotes"] = out
        return out

    # ------------------------------------------------------------------ #
    # 4) SÉQUENCES & CYCLES (test de hasard du RNG)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _max_streak(mask: np.ndarray) -> int:
        """Plus longue série consécutive de True."""
        best = cur = 0
        for v in mask:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        return best

    @staticmethod
    def _runs_test(binary: np.ndarray) -> tuple[float, float]:
        """Test des runs de Wald-Wolfowitz : z et p-value (H0 = séquence aléatoire)."""
        b = np.asarray(binary)
        n1, n0 = int(b.sum()), int((1 - b).sum())
        if n1 == 0 or n0 == 0:
            return 0.0, 1.0
        runs = 1 + int((b[1:] != b[:-1]).sum())
        mu = 1 + 2 * n1 * n0 / (n1 + n0)
        var = (2 * n1 * n0 * (2 * n1 * n0 - n1 - n0)) / ((n1 + n0) ** 2 * (n1 + n0 - 1))
        z = (runs - mu) / np.sqrt(var) if var > 0 else 0.0
        return float(z), float(2 * (1 - stats.norm.cdf(abs(z))))

    def sequence_cycles(self) -> dict:
        """Séries, écarts entre occurrences, autocorrélation N->N+1, runs test,
        heatmap horaire (le cœur de la détection de pattern RNG)."""
        d = self.df
        code = d.result.map({"1": 1, "X": 0, "2": -1}).to_numpy()

        # 1) autocorrélation lag-1 du résultat encodé + indépendance N->N+1 (chi2)
        ac = float(pd.Series(code).autocorr(lag=1))
        ct = pd.crosstab(d.result.iloc[:-1].to_numpy(), d.result.iloc[1:].to_numpy())
        chi2, p_chi2, _, _ = stats.chi2_contingency(ct)

        # 2) runs test sur la binaire "victoire domicile vs pas"
        z_runs, p_runs = self._runs_test((d.result == "1").to_numpy().astype(int))

        # 3) séries max
        streaks = {r: self._max_streak((d.result == r).to_numpy()) for r in ["1", "X", "2"]}
        streaks["over25"] = self._max_streak((d.total_goals > 2.5).to_numpy())

        # 4) écarts moyens entre deux occurrences du même résultat
        gaps = {}
        for r in ["1", "X", "2"]:
            idx = np.where(d.result.to_numpy() == r)[0]
            gaps[r] = round(float(np.diff(idx).mean()), 2) if len(idx) > 1 else None

        out = {
            "autocorr_lag1_resultat": round(ac, 4),
            "independance_N_N+1_chi2_p": round(float(p_chi2), 4),
            "runs_test_victoire_dom": {"z": round(z_runs, 3), "p": round(p_runs, 4)},
            "series_max_consecutives": streaks,
            "ecart_moyen_entre_memes_resultats": gaps,
            "interpretation": "autocorr~0, chi2 p>0.05, runs p>0.05 => séquence indistinguable "
                              "du hasard (RNG sans mémoire). Un signal apparaîtrait ici.",
        }
        self.summary["sequences"] = out
        return out

    # ------------------------------------------------------------------ #
    # 5) VISUALISATIONS
    # ------------------------------------------------------------------ #
    def visualizations(self) -> list[str]:
        """Génère et sauvegarde les figures. Retourne la liste des chemins."""
        d = self.df
        saved = []

        def _save(fig, name):
            p = self.plots_dir / name
            fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig); saved.append(str(p))

        try:
            # distribution 1X2
            fig, ax = plt.subplots(figsize=(6, 4))
            (d.result.value_counts(normalize=True).reindex(["1", "X", "2"]) * 100).plot.bar(
                ax=ax, color=["#2a9d8f", "#e9c46a", "#e76f51"])
            ax.set_title("Distribution 1 / X / 2 (%)"); ax.set_ylabel("%")
            _save(fig, "01_distribution_1x2.png")

            # top scores exacts
            fig, ax = plt.subplots(figsize=(7, 4))
            (d.exact_score.value_counts().head(12)).plot.bar(ax=ax, color="#264653")
            ax.set_title("Top 12 scores exacts"); ax.set_ylabel("occurrences")
            _save(fig, "02_top_scores.png")

            # distribution du total de buts
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(d.total_goals, bins=range(0, 11), ax=ax, color="#457b9d")
            ax.set_title("Distribution du total de buts"); ax.set_xlabel("buts")
            _save(fig, "03_total_buts.png")

            # calibration : implicite vs réalisé
            if "p_home_impl" in d:
                cal = d.groupby(pd.qcut(d.p_home_impl, 10, duplicates="drop")).apply(
                    lambda g: pd.Series({"impl": g.p_home_impl.mean(), "real": (g.result == "1").mean()}))
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.plot([0, 1], [0, 1], "k--", alpha=.5, label="calibration parfaite")
                ax.scatter(cal.impl, cal.real, color="#e76f51", s=60)
                ax.set_xlabel("proba implicite (cote)"); ax.set_ylabel("fréquence réalisée")
                ax.set_title("Calibration domicile"); ax.legend()
                _save(fig, "04_calibration.png")

            # heatmap horaire : résultat par heure
            piv = pd.crosstab(d.hour, d.result, normalize="index") * 100
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.heatmap(piv[["1", "X", "2"]], annot=True, fmt=".1f", cmap="viridis", ax=ax)
            ax.set_title("% de 1/X/2 par heure (heatmap)")
            _save(fig, "05_heatmap_horaire.png")
        except Exception as exc:
            logger.warning("Visualisation partielle (%s)", exc)

        self.summary["plots"] = saved
        logger.info("%d figures sauvegardées dans %s", len(saved), self.plots_dir)
        return saved

    # ------------------------------------------------------------------ #
    # ORCHESTRATION
    # ------------------------------------------------------------------ #
    def run_all(self) -> dict:
        """Exécute toute la Phase 1 et retourne le résumé complet."""
        self.load_and_clean()
        self.descriptive_stats()
        self.odds_analysis()
        self.sequence_cycles()
        self.visualizations()
        logger.info("PHASE 1 terminée.")
        return self.summary


if __name__ == "__main__":
    import json
    eda = VirtualFootballEDA("data/vfoot_ml/matches.csv")
    s = eda.run_all()
    print("\n" + "=" * 70)
    print("  RÉSUMÉ PHASE 1 — EDA")
    print("=" * 70)
    print(json.dumps(s, ensure_ascii=False, indent=1, default=str)[:2600])
    Path("data/vfoot_ml/phase1_summary.json").write_text(
        json.dumps(s, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
