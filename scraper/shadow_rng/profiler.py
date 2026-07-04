"""BRIQUE A — DistributionProfiler (Signature Extractor).

Extrait en continu la "signature comportementale" du RNG adverse :
  - biais par score = fréquence RÉELLE observée vs fréquence THÉORIQUE des cotes
    (théorique = 1X2 -> devig -> lambda -> grille Poisson pure), sur fenêtres
    glissantes, avec significativité statistique (binomial + BH-FDR) ;
  - régime = haute / basse entropie de la distribution réalisée récente ;
  - mémoire = le match N influence-t-il N+1 ? (autocorrélation, runs test, chi2).

PHILOSOPHIE : la théorique (BASELINE) est la référence calibrée. Tout "biais"
mesuré ici n'est exploitable QUE s'il est significatif APRÈS correction ET
stable dans le temps. Sur ce dataset, l'attendu honnête est "aucun biais
significatif" — mais le profiler est prêt à en détecter un si le RNG dérive.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2_contingency

from ..market_inversion import devig, exact_invert_1x2, _fast_grid, apply_sim_deviations
from .config import merge_config

logger = logging.getLogger("shadow_rng.profiler")


def _shannon_entropy(counts: np.ndarray) -> float:
    """Entropie de Shannon (base e) d'un vecteur de comptages. 0 si vide."""
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-np.sum(p * np.log(p)))


def _runs_test_z(binary: np.ndarray) -> float:
    """Z du test des runs de Wald-Wolfowitz sur une séquence binaire.
    z>0 = trop alterné (anti-persistance) ; z<0 = trop de séries (persistance).
    Renvoie 0.0 si la séquence est dégénérée."""
    x = np.asarray(binary).astype(int)
    n1 = int(x.sum())
    n0 = len(x) - n1
    if n1 < 10 or n0 < 10:
        return 0.0
    runs = 1 + int(np.sum(x[1:] != x[:-1]))
    n = len(x)
    mu = 2 * n1 * n0 / n + 1
    var = 2 * n1 * n0 * (2 * n1 * n0 - n) / (n ** 2 * (n - 1))
    return float((runs - mu) / np.sqrt(var)) if var > 0 else 0.0


def _autocorr(series: np.ndarray, lag: int) -> tuple[float, float]:
    """Autocorrélation (Pearson) à `lag` + z approx (corr*sqrt(n)).
    Renvoie (corr, z). (0,0) si trop court."""
    s = np.asarray(series, dtype=float)
    s = s - np.nanmean(s)
    if len(s) <= lag + 30:
        return 0.0, 0.0
    a, b = s[:-lag], s[lag:]
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if denom <= 0:
        return 0.0, 0.0
    corr = float(np.sum(a * b) / denom)
    return corr, float(corr * np.sqrt(len(a)))


def _bh_fdr(pvals: dict[str, float], q: float) -> set[str]:
    """Benjamini-Hochberg : renvoie l'ensemble des clés significatives à FDR q."""
    items = [(k, p) for k, p in pvals.items() if p is not None and not np.isnan(p)]
    if not items:
        return set()
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    sig: set[str] = set()
    for rank, (k, p) in enumerate(items, start=1):
        if p <= rank / m * q:
            sig = {kk for kk, _ in items[:rank]}
    return sig


class DistributionProfiler:
    """Extracteur de signature du RNG (Brique A).

    Usage :
        prof = DistributionProfiler(config={...})
        prof.fit(df_historique)                 # df trié chronologiquement
        prof.get_bias("0-0", window=200)         # biais d'un score
        prof.detect_regime(window=50)            # régime entropique courant
        prof.test_memory(lag=1)                  # mémoire N -> N+1
        snap = prof.get_full_snapshot()          # dict complet (consommé par Brique B/C)

    Toutes les sorties sont JSON-sérialisables (floats Python, pas numpy).
    """

    def __init__(self, config: Optional[dict] = None,
                 theoretical_fn: Optional[Callable[[float, float, float], np.ndarray]] = None):
        """
        Args:
            config: surcharges de DEFAULT_CONFIG (deep-merge).
            theoretical_fn: fonction (oh,od,oa) -> vecteur de probas théoriques
                sur l'espace des scores (longueur max_goals**2, ordre row-major h*MAXG+a).
                Par défaut : 1X2 -> devig -> lambda -> grille Poisson pure.
        """
        self.cfg = merge_config(config)
        logging.getLogger("shadow_rng").setLevel(self.cfg["log_level"])
        self.MAXG: int = int(self.cfg["max_goals"])
        self.scores: list[str] = [f"{h}-{a}" for h in range(self.MAXG) for a in range(self.MAXG)]
        self._score_idx = {s: i for i, s in enumerate(self.scores)}
        self._theo_fn = theoretical_fn or self._default_theoretical
        self._cache: dict[tuple, np.ndarray] = {}
        # état après fit
        self._fitted = False
        self._theo: Optional[np.ndarray] = None      # (n, n_scores) probas théoriques
        self._real_idx: Optional[np.ndarray] = None   # (n,) index du score réalisé
        self._sa: Optional[np.ndarray] = None
        self._sb: Optional[np.ndarray] = None
        self._regime_baseline: Optional[tuple[float, float]] = None  # (mean, std) entropie

    # ------------------------------------------------------------------ #
    # théorique
    # ------------------------------------------------------------------ #
    def _default_theoretical(self, oh: float, od: float, oa: float) -> np.ndarray:
        """RÉFÉRENCE = comportement NORMAL du RNG = 1X2 -> devig -> (lam_h,lam_a)
        -> apply_sim_deviations('cells') (Poisson + Dixon-Coles + boosts, modèle
        calibré sur tout l'historique, sigma=0.75pp). PAS Poisson pur : Poisson pur
        ignore les déviations DC connues et stables -> fausses anomalies permanentes.
        Le biais mesuré = réel - CE modèle = 0 quand tout va bien, != 0 si vraie dérive."""
        key = (round(oh, self.cfg["round_odds_cache"]),
               round(od, self.cfg["round_odds_cache"]),
               round(oa, self.cfg["round_odds_cache"]))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        lh, la = exact_invert_1x2(oh, od, oa)          # inversion exacte (le pricing 1X2 EST Poisson pur)
        g = apply_sim_deviations(lh, la, "cells")      # modèle calibré du RNG = BASELINE
        sub = g[: self.MAXG, : self.MAXG].astype(float)
        s = sub.sum()
        vec = (sub / s).ravel() if s > 0 else np.full(self.MAXG ** 2, 1.0 / self.MAXG ** 2)
        self._cache[key] = vec
        return vec

    def pure_poisson(self, oh: float, od: float, oa: float) -> np.ndarray:
        """Grille Poisson PURE (rho=0) — INFORMATIVE seulement (reference_pure_poisson),
        n'entre PAS dans le calcul du biais ni des alertes."""
        lh, la = exact_invert_1x2(oh, od, oa)
        g = _fast_grid(lh, la, 0.0)[: self.MAXG, : self.MAXG].astype(float)
        s = g.sum()
        return (g / s).ravel() if s > 0 else np.full(self.MAXG ** 2, 1.0 / self.MAXG ** 2)

    # ------------------------------------------------------------------ #
    # fit
    # ------------------------------------------------------------------ #
    def fit(self, df: pd.DataFrame) -> "DistributionProfiler":
        """Calcule, pour chaque match, la distribution théorique des scores et
        l'index du score réalisé. Construit aussi la baseline d'entropie (régime).

        Args:
            df: DataFrame trié CHRONOLOGIQUEMENT (du plus ancien au plus récent).
                Colonnes requises (noms via cfg['columns']) :
                odds_home/draw/away + score_home/away.
        Returns: self.
        Raises: ValueError si colonnes manquantes ou df vide.
        """
        col = self.cfg["columns"]
        need = [col["odds_home"], col["odds_draw"], col["odds_away"],
                col["score_home"], col["score_away"]]
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise ValueError(f"colonnes manquantes pour fit(): {missing}")
        d = df.copy()
        ts = col.get("timestamp")
        if ts and ts in d.columns:
            d = d.sort_values(ts)
        d = d[(d[col["odds_home"]] > 1) & (d[col["odds_draw"]] > 1) & (d[col["odds_away"]] > 1)]
        d = d.dropna(subset=need).reset_index(drop=True)
        if len(d) == 0:
            raise ValueError("aucune ligne valide après filtrage des cotes")

        n = len(d)
        theo = np.empty((n, self.MAXG ** 2), dtype=float)
        oh = d[col["odds_home"]].to_numpy(float)
        od = d[col["odds_draw"]].to_numpy(float)
        oa = d[col["odds_away"]].to_numpy(float)
        n_err = 0
        for i in range(n):
            try:
                theo[i] = self._theo_fn(oh[i], od[i], oa[i])
            except Exception as exc:  # robustesse : une cote pourrie ne casse pas le fit
                logger.warning("théorique échouée match %d (%s) -> uniforme", i, exc)
                theo[i] = 1.0 / self.MAXG ** 2
                n_err += 1
        sa = np.clip(d[col["score_home"]].to_numpy(int), 0, self.MAXG - 1)
        sb = np.clip(d[col["score_away"]].to_numpy(int), 0, self.MAXG - 1)
        self._sa, self._sb = sa, sb
        self._real_idx = sa * self.MAXG + sb
        self._theo = theo
        self._fitted = True
        self._regime_baseline = self._compute_regime_baseline()
        logger.info("fit OK : %d matchs (%d cotes invalides -> uniforme)", n, n_err)
        return self

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("appelle fit() avant d'interroger le profiler")

    # ------------------------------------------------------------------ #
    # fenêtre
    # ------------------------------------------------------------------ #
    def _window(self, window: Optional[int]) -> tuple[np.ndarray, np.ndarray, int]:
        """Renvoie (théorique moyenne par score, fréquence réelle par score, n)
        sur les `window` derniers matchs. window=None -> tout l'historique."""
        self._check_fitted()
        n_tot = len(self._real_idx)
        w = n_tot if window is None else min(int(window), n_tot)
        theo_w = self._theo[n_tot - w:]                          # (w, n_scores)
        real_w = self._real_idx[n_tot - w:]
        theo_mean = theo_w.mean(axis=0)                          # proba théorique moy / score
        real_freq = np.bincount(real_w, minlength=self.MAXG ** 2).astype(float) / w
        return theo_mean, real_freq, w

    # ------------------------------------------------------------------ #
    # API publique
    # ------------------------------------------------------------------ #
    def get_bias(self, score: str, window: Optional[int] = None) -> dict:
        """Biais d'un score sur une fenêtre : réel vs théorique + significativité.

        Returns dict: real_freq, theo_freq, bias_pp (réel-théo, en points de %),
            ratio (réel/théo), z, p_value, n, significant (p<alpha, NON corrigé
            multi-tests — pour ça voir get_full_snapshot).
        """
        self._check_fitted()
        if score not in self._score_idx:
            raise ValueError(f"score hors espace 0..{self.MAXG-1}: {score!r}")
        window = window or self.cfg["default_window"]
        theo_mean, real_freq, w = self._window(window)
        idx = self._score_idx[score]
        p0 = float(theo_mean[idx])
        obs = int(round(real_freq[idx] * w))
        rf = float(real_freq[idx])
        # test binomial exact (null = proba théorique moyenne de la fenêtre)
        if 0.0 < p0 < 1.0 and w >= 1:
            pval = float(binomtest(obs, w, p0, alternative="two-sided").pvalue)
            z = float((rf - p0) / np.sqrt(p0 * (1 - p0) / w)) if w > 0 else 0.0
        else:
            pval, z = (1.0, 0.0) if obs == 0 else (0.0, np.inf)
        return {
            "score": score, "window": w,
            "real_freq": round(rf, 4), "theo_freq": round(p0, 4),
            "bias_pp": round((rf - p0) * 100, 2),
            "ratio": round(rf / p0, 3) if p0 > 0 else None,
            "z": round(z, 2), "p_value": round(pval, 4),
            "n_obs": obs,
            "significant": bool(pval < self.cfg["alpha"] and w >= self.cfg["min_window_matches"]),
        }

    def get_all_biases(self, window: Optional[int] = None) -> dict:
        """Biais de TOUS les scores + flag de significativité corrigée BH-FDR.
        Returns: {"window": w, "biases": {score: get_bias(...)}, "anomalies": [scores FDR-sig]}.
        """
        window = window or self.cfg["default_window"]
        biases = {s: self.get_bias(s, window) for s in self.scores}
        # FDR seulement sur les scores avec assez de masse théorique (évite le bruit pur)
        pvals = {s: b["p_value"] for s, b in biases.items() if b["theo_freq"] >= 0.005}
        sig = _bh_fdr(pvals, self.cfg["fdr_q"])
        _, _, w = self._window(window)
        for s, b in biases.items():
            b["fdr_significant"] = s in sig
        anomalies = sorted(sig, key=lambda s: -abs(biases[s]["bias_pp"]))
        return {"window": w, "biases": biases, "anomalies": anomalies}

    def _compute_regime_baseline(self) -> tuple[float, float]:
        """Distribution historique de l'entropie fenêtrée (pour z-scorer le régime)."""
        rw = int(self.cfg["regime_window"])
        n = len(self._real_idx)
        if n < rw * 2:
            return (0.0, 0.0)
        stride = max(1, rw // 2)
        ents = []
        for i in range(rw, n + 1, stride):
            counts = np.bincount(self._real_idx[i - rw:i], minlength=self.MAXG ** 2)
            ents.append(_shannon_entropy(counts))
        ents = np.asarray(ents)
        return (float(ents.mean()), float(ents.std() or 1e-9))

    def detect_regime(self, window: Optional[int] = None) -> dict:
        """Régime entropique courant : la distribution réalisée récente est-elle
        plus dispersée (haute entropie) ou concentrée (basse entropie) que d'habitude ?

        Returns dict: regime ('haute_entropie'|'basse_entropie'|'normal'),
            entropy, baseline_mean, z, confidence (|z| ramené à [0,1]), window.
        """
        self._check_fitted()
        rw = int(window or self.cfg["regime_window"])
        n = len(self._real_idx)
        rw = min(rw, n)
        counts = np.bincount(self._real_idx[n - rw:], minlength=self.MAXG ** 2)
        ent = _shannon_entropy(counts)
        mu, sd = self._regime_baseline or (0.0, 0.0)
        z = (ent - mu) / sd if sd > 0 else 0.0
        if z >= self.cfg["regime_z_high"]:
            regime = "haute_entropie"
        elif z <= self.cfg["regime_z_low"]:
            regime = "basse_entropie"
        else:
            regime = "normal"
        conf = float(min(abs(z) / 3.0, 1.0))  # |z|>=3 -> confiance saturée à 1
        return {"regime": regime, "entropy": round(ent, 4),
                "baseline_mean": round(mu, 4), "z": round(float(z), 2),
                "confidence": round(conf, 2), "window": rw}

    def test_memory(self, lag: int = 1, window: Optional[int] = None) -> dict:
        """Le match N influence-t-il N+lag ? Bat-on l'hypothèse 'sans mémoire' ?

        Tests : autocorrélation du total de buts, autocorrélation de home_win,
        runs test sur home_win, chi2 d'indépendance des issues 1X2 consécutives.

        Returns dict: par test (corr/z/p), + memory_detected (True seulement si
        AU MOINS un test ressort significatif au seuil alpha).
        """
        self._check_fitted()
        n = len(self._real_idx)
        sl = slice(n - int(window), n) if window else slice(0, n)
        sa, sb = self._sa[sl], self._sb[sl]
        total = (sa + sb).astype(float)
        hw = (sa > sb).astype(int)
        ac_tot_c, ac_tot_z = _autocorr(total, lag)
        ac_hw_c, ac_hw_z = _autocorr(hw, lag)
        runs_z = _runs_test_z(hw)
        # chi2 issues consécutives (lag fixé à 1 pour la contingence)
        outc = np.where(sa > sb, 0, np.where(sa == sb, 1, 2))  # 0=H,1=D,2=A
        chi2_p = 1.0
        try:
            prev, cur = outc[:-1], outc[1:]
            table = np.zeros((3, 3), dtype=int)
            for pcur, ccur in zip(prev, cur):
                table[pcur, ccur] += 1
            mask = (table.sum(axis=1) > 0)
            if mask.sum() >= 2 and (table.sum(axis=0) > 0).sum() >= 2:
                chi2_p = float(chi2_contingency(table[mask][:, table.sum(axis=0) > 0])[1])
        except Exception as exc:
            logger.warning("chi2 mémoire échoué (%s)", exc)

        def p_from_z(z):
            from math import erf, sqrt
            return float(2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2)))))

        tests = {
            "autocorr_total": {"corr": round(ac_tot_c, 4), "z": round(ac_tot_z, 2), "p": round(p_from_z(ac_tot_z), 4)},
            "autocorr_homewin": {"corr": round(ac_hw_c, 4), "z": round(ac_hw_z, 2), "p": round(p_from_z(ac_hw_z), 4)},
            "runs_homewin": {"z": round(runs_z, 2), "p": round(p_from_z(runs_z), 4)},
            "chi2_1x2_consecutif": {"p": round(chi2_p, 4)},
        }
        a = self.cfg["alpha"]
        detected = any(t.get("p", 1.0) < a for t in tests.values())
        return {"lag": lag, "n": int(len(total)), "tests": tests, "memory_detected": bool(detected)}

    def get_full_snapshot(self, window: Optional[int] = None) -> dict:
        """Snapshot complet de la signature courante (consommé par Brique B/C).

        Returns dict JSON-sérialisable : active_window, n_total, regime, memory
        (sur tous les lags configurés), biais (tous scores), anomalies (FDR-sig),
        et biais_snapshot compact {score: bias_pp} pour injection rapide.
        """
        self._check_fitted()
        window = window or self.cfg["default_window"]
        allb = self.get_all_biases(window)
        memory = {f"lag{l}": self.test_memory(l) for l in self.cfg["memory_lags"]}
        biais_snapshot = {s: allb["biases"][s]["bias_pp"] for s in self.scores}
        return {
            "active_window": allb["window"],
            "n_total": int(len(self._real_idx)),
            "regime": self.detect_regime(),
            "memory": memory,
            "anomalies": allb["anomalies"],
            "biais_snapshot": biais_snapshot,
            "biases_detail": allb["biases"],
        }


# ====================================================================== #
# EXEMPLE D'UTILISATION — DONNÉES SYNTHÉTIQUES
# (valide UNIQUEMENT que le code tourne et que le détecteur réagit ;
#  ne valide AUCUNE performance prédictive — données fabriquées.)
# ====================================================================== #
if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(levelname)s %(name)s | %(message)s")
    rng = np.random.RandomState(7)

    def _make_synthetic(n, inject_bias=None, inject_memory=False):
        """Fabrique n matchs : lambdas aléatoires -> cotes (avec marge) + score
        échantillonné depuis la grille. inject_bias: dict {score: facteur} appliqué
        à la grille d'ÉCHANTILLONNAGE (pas aux cotes) -> crée un écart réel/théorique.
        inject_memory: total_{N+1} dépend de total_N (brise l'indépendance)."""
        rows = []
        prev_high = False
        for _ in range(n):
            lh = rng.uniform(0.6, 2.6)
            la = rng.uniform(0.5, 2.1)
            if inject_memory and prev_high:      # mémoire artificielle : après match "haut", on baisse
                lh *= 0.6; la *= 0.6
            # cotes depuis le 1X2 POISSON PUR (le pricing 1X2 est Poisson pur) + marge 6%
            gp = _fast_grid(lh, la, 0.0)[:7, :7]; gp = gp / gp.sum()
            p1 = np.tril(gp, -1).sum(); pX = np.trace(gp); p2 = np.triu(gp, 1).sum()
            margin = 1.06
            oh, od, oa = margin / max(p1, 1e-3), margin / max(pX, 1e-3), margin / max(p2, 1e-3)
            # RNG "honnête" = échantillonné depuis le BASELINE calibré (apply_sim_deviations)
            greal = apply_sim_deviations(lh, la, "cells")[:7, :7]; greal = greal / greal.sum()
            gs = greal.copy()
            if inject_bias:
                for sc, f in inject_bias.items():
                    h, a = map(int, sc.split("-")); gs[h, a] *= f
                gs = gs / gs.sum()
            flat = gs.ravel(); k = rng.choice(len(flat), p=flat)
            sa, sb = divmod(k, 7)
            rows.append((oh, od, oa, sa, sb))
            prev_high = (sa + sb) >= 4
        return pd.DataFrame(rows, columns=["oh", "od", "oa", "sa", "sb"])

    print("\n########## CAS 1 : RNG honnête (aucun biais injecté) ##########")
    df0 = _make_synthetic(4000)
    p0 = DistributionProfiler().fit(df0)
    snap0 = p0.get_full_snapshot(window=1000)
    print(f"  régime: {snap0['regime']['regime']} (z={snap0['regime']['z']})")
    print(f"  mémoire lag1 détectée: {snap0['memory']['lag1']['memory_detected']}")
    print(f"  anomalies FDR (attendu: aucune): {snap0['anomalies']}")
    print(f"  ex. biais 0-0: {p0.get_bias('0-0', 1000)}")

    print("\n########## CAS 2 : biais INJECTÉ sur 0-0 (×1.6) et 2-1 (×1.5) ##########")
    df1 = _make_synthetic(4000, inject_bias={"0-0": 1.6, "2-1": 1.5})
    p1 = DistributionProfiler().fit(df1)
    snap1 = p1.get_full_snapshot(window=1000)
    print(f"  anomalies FDR (attendu: 0-0 et/ou 2-1): {snap1['anomalies']}")
    for s in ["0-0", "2-1"]:
        b = p1.get_bias(s, 1000)
        print(f"    {s}: réel={b['real_freq']} théo={b['theo_freq']} biais={b['bias_pp']:+}pp "
              f"z={b['z']} fdr_sig={snap1['biases_detail'][s]['fdr_significant']}")

    print("\n########## CAS 3 : mémoire INJECTÉE (total N -> N+1) ##########")
    df2 = _make_synthetic(4000, inject_memory=True)
    p2 = DistributionProfiler().fit(df2)
    mem = p2.test_memory(lag=1)
    print(f"  mémoire détectée (attendu: True): {mem['memory_detected']}")
    print(f"    autocorr_total: corr={mem['tests']['autocorr_total']['corr']} z={mem['tests']['autocorr_total']['z']}")

    print("\n-> Ces 3 cas valident que le CODE détecte correctement "
          "(biais présent/absent, mémoire présente/absente). Aucune conclusion prédictive.")
