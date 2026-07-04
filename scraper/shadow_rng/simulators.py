"""BRIQUE B — Multi-Simulateur Monte-Carlo (Shadow-RNG).

4 simulateurs branchables, chacun testable indépendamment :
  - BaselineSimulator : cotes pures -> Poisson. LA RÉFÉRENCE. Ignore le snapshot.
  - TrendSimulator    : baseline ajustée par les biais FDR-significatifs (fenêtre 50).
  - MemorySimulator   : baseline pondérée par la matrice de transition score_N -> N+1.
  - RegimeSimulator   : baseline aplatie (haute entropie) ou concentrée (basse entropie).

Tout simulateur SANS signal replie sur BASELINE (active=False) — pas d'ajustement
fantôme. La divergence KL(cible || baseline) montre immédiatement si un simulateur
apporte quelque chose. Conteneur : ShadowRNGSimulator (enable/disable par nom).
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from ..market_inversion import exact_invert_1x2, _fast_grid, apply_sim_deviations
from .config import merge_config

logger = logging.getLogger("shadow_rng.simulators")


# ---------------------------------------------------------------------- #
# helpers purs (réutilisables / testables)
# ---------------------------------------------------------------------- #
def score_list(max_goals: int) -> list[str]:
    return [f"{h}-{a}" for h in range(max_goals) for a in range(max_goals)]


def baseline_distribution(oh: float, od: float, oa: float, max_goals: int) -> np.ndarray:
    """BASELINE = comportement NORMAL du RNG = 1X2 -> devig -> (lam_h,lam_a) ->
    apply_sim_deviations('cells') (modèle calibré : Poisson + Dixon-Coles + boosts).
    PAS Poisson pur (qui ignorerait les déviations DC connues et stables).
    Identique à la théorique de la Brique A. Lève ValueError si cotes invalides."""
    if not (oh > 1 and od > 1 and oa > 1):
        raise ValueError(f"cotes invalides: {oh},{od},{oa}")
    lh, la = exact_invert_1x2(oh, od, oa)
    g = apply_sim_deviations(lh, la, "cells")[:max_goals, :max_goals].astype(float)
    s = g.sum()
    return (g / s).ravel() if s > 0 else np.full(max_goals ** 2, 1.0 / max_goals ** 2)


def pure_poisson_distribution(oh: float, od: float, oa: float, max_goals: int) -> np.ndarray:
    """Grille Poisson PURE (rho=0) — INFORMATIVE seulement (reference_pure_poisson).
    N'entre PAS dans le BASELINE ni dans les conditions d'alerte."""
    if not (oh > 1 and od > 1 and oa > 1):
        return np.full(max_goals ** 2, 1.0 / max_goals ** 2)
    lh, la = exact_invert_1x2(oh, od, oa)
    g = _fast_grid(lh, la, 0.0)[:max_goals, :max_goals].astype(float)
    s = g.sum()
    return (g / s).ravel() if s > 0 else np.full(max_goals ** 2, 1.0 / max_goals ** 2)


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    """KL(P||Q) en nats. Lissé pour éviter log(0)."""
    p = np.asarray(p, float) + eps
    q = np.asarray(q, float) + eps
    p /= p.sum(); q /= q.sum()
    return float(max(0.0, np.sum(p * np.log(p / q))))   # clamp >=0 (bruit eps quand p~q)


def build_transition_matrix(real_idx: np.ndarray, n_scores: int, lag: int = 1,
                            smoothing: float = 0.5, min_count: int = 30):
    """Matrice de transition score_N -> score_{N+lag} (lissée Laplace) + marginale.

    Args:
        real_idx: séquence (chrono) des index de score réalisés.
        min_count: si un score précédent apparaît < min_count fois, sa ligne est
            remplacée par la marginale (transition non fiable -> pas de mémoire).
    Returns: (T, marginal) avec T[i] = distribution du score suivant sachant prev=i.
    """
    # marginale LISSÉE (Laplace) : jamais 0 -> pas d'explosion du lift cond/marg
    marg_counts = np.bincount(real_idx, minlength=n_scores).astype(float) + smoothing
    marginal = marg_counts / marg_counts.sum()
    counts = np.full((n_scores, n_scores), smoothing, dtype=float)
    prev_counts = np.zeros(n_scores, dtype=int)
    prev, cur = real_idx[:-lag], real_idx[lag:]
    for p, c in zip(prev, cur):
        counts[p, c] += 1.0
        prev_counts[p] += 1
    T = counts / counts.sum(axis=1, keepdims=True)
    for i in range(n_scores):                       # transitions non fiables -> marginale
        if prev_counts[i] < min_count:
            T[i] = marginal
    return T, marginal


# ---------------------------------------------------------------------- #
# classe abstraite
# ---------------------------------------------------------------------- #
class BaseSimulator(ABC):
    """Interface commune. Sous-classer et implémenter `_target_dist`."""

    name: str = "BASE"

    def __init__(self, config: Optional[dict] = None):
        self.cfg = merge_config(config)
        self.MAXG: int = int(self.cfg["max_goals"])
        self.scores = score_list(self.MAXG)
        self._idx = {s: i for i, s in enumerate(self.scores)}
        self.scfg = self.cfg["simulators"]
        self._snapshot: dict = {}
        # état du dernier simulate()
        self.active: bool = False
        self.last_kl: float = 0.0
        self.last_target: Optional[np.ndarray] = None
        self.last_output: Optional[dict] = None

    def set_profiler_snapshot(self, snapshot: dict) -> "BaseSimulator":
        """Injecte le snapshot complet de la Brique A (peut contenir 'transition')."""
        self._snapshot = snapshot or {}
        return self

    def _baseline(self, odds_1x2) -> np.ndarray:
        """Distribution baseline robuste : uniforme + warning si cotes pourries."""
        try:
            return baseline_distribution(odds_1x2[0], odds_1x2[1], odds_1x2[2], self.MAXG)
        except Exception as exc:
            logger.warning("[%s] baseline impossible (%s) -> uniforme", self.name, exc)
            return np.full(self.MAXG ** 2, 1.0 / self.MAXG ** 2)

    @abstractmethod
    def _target_dist(self, baseline: np.ndarray, odds_1x2, snapshot: dict,
                     last_score: Optional[str]) -> tuple[np.ndarray, bool]:
        """Renvoie (distribution cible normalisée, applied) où applied=True si un
        ajustement réel a été appliqué (sinon la cible == baseline)."""
        ...

    def simulate(self, n_iterations: Optional[int] = None, odds_1x2=None,
                 profiler_snapshot: Optional[dict] = None, last_score: Optional[str] = None) -> dict:
        """Calcule la cible, mesure KL vs baseline, échantillonne Monte-Carlo.

        Returns: dict {score: probabilité simulée (fréquence MC)}.
        Effets de bord : self.active, self.last_kl, self.last_target, self.last_output.
        """
        n = int(n_iterations or self.scfg["n_iterations"])
        snap = profiler_snapshot if profiler_snapshot is not None else self._snapshot
        baseline = self._baseline(odds_1x2)
        try:
            target, applied = self._target_dist(baseline, odds_1x2, snap, last_score)
        except Exception as exc:                     # un ajustement foireux ne crash pas
            logger.warning("[%s] ajustement échoué (%s) -> repli BASELINE", self.name, exc)
            target, applied = baseline.copy(), False
        target = np.asarray(target, float)
        target = target / target.sum() if target.sum() > 0 else baseline
        kl = kl_divergence(target, baseline)
        self.active = bool(applied and kl > self.scfg["kl_epsilon"])
        self.last_kl = kl
        self.last_target = target
        # Monte-Carlo
        seed = self.scfg.get("random_seed")
        rng = np.random.RandomState(seed) if seed is not None else np.random
        draws = rng.choice(len(target), size=n, p=target)
        freq = np.bincount(draws, minlength=len(target)).astype(float) / n
        out = {self.scores[i]: float(freq[i]) for i in range(len(freq))}
        self.last_output = out
        logger.info("[%s] active=%s KL=%.4f", self.name, self.active, kl)
        return out

    def top_k(self, k: int = 5) -> list[tuple[str, float]]:
        if not self.last_output:
            return []
        return sorted(self.last_output.items(), key=lambda kv: -kv[1])[:k]


# ---------------------------------------------------------------------- #
# 1 - BASELINE
# ---------------------------------------------------------------------- #
class BaselineSimulator(BaseSimulator):
    """Cotes pures -> Poisson. LA RÉFÉRENCE. N'utilise jamais le snapshot."""
    name = "BASELINE"

    def _target_dist(self, baseline, odds_1x2, snapshot, last_score):
        return baseline.copy(), False   # baseline n'active JAMAIS d'ajustement


# ---------------------------------------------------------------------- #
# 2 - TREND
# ---------------------------------------------------------------------- #
class TrendSimulator(BaseSimulator):
    """Baseline × facteur(réel/théo) sur les scores FDR-significatifs (fenêtre 50)."""
    name = "TREND"

    def _target_dist(self, baseline, odds_1x2, snapshot, last_score):
        detail = snapshot.get("biases_detail") or {}
        lo, hi = self.scfg["trend_ratio_clip"]
        target = baseline.copy()
        applied = False
        for s, info in detail.items():
            if info.get("fdr_significant") and info.get("ratio"):
                ratio = float(min(max(info["ratio"], lo), hi))
                target[self._idx[s]] = baseline[self._idx[s]] * ratio
                applied = True
        return target, applied


# ---------------------------------------------------------------------- #
# 3 - MEMORY
# ---------------------------------------------------------------------- #
class MemorySimulator(BaseSimulator):
    """Baseline pondérée par le lift de transition (score précédent -> suivant).
    Replie sur BASELINE si pas de mémoire détectée / last_score inconnu / pas de matrice."""
    name = "MEMORY"

    def _target_dist(self, baseline, odds_1x2, snapshot, last_score):
        mem = (snapshot.get("memory") or {}).get("lag1") or {}
        trans = snapshot.get("transition")
        if not mem.get("memory_detected") or last_score is None or trans is None:
            return baseline.copy(), False
        if last_score not in self._idx:
            logger.warning("[MEMORY] last_score inconnu %r -> repli BASELINE", last_score)
            return baseline.copy(), False
        T = np.asarray(trans["matrix"], float)
        marg = np.asarray(trans["marginal"], float)
        cond = T[self._idx[last_score]]                 # dist du suivant | last_score
        eps = 1e-9
        lift = (cond + eps) / (marg + eps)
        lift = lift ** float(self.scfg["memory_strength"])
        lo, hi = self.scfg["memory_lift_clip"]          # borne le lift (anti-explosion)
        lift = np.clip(lift, lo, hi)
        target = baseline * lift
        return target, True


# ---------------------------------------------------------------------- #
# 4 - REGIME
# ---------------------------------------------------------------------- #
class RegimeSimulator(BaseSimulator):
    """Haute entropie -> aplatit (mélange uniforme) ; basse entropie -> concentre
    (puissance > 1) ; normal -> BASELINE."""
    name = "REGIME"

    def _target_dist(self, baseline, odds_1x2, snapshot, last_score):
        regime = (snapshot.get("regime") or {}).get("regime", "normal")
        if regime == "haute_entropie":
            a = float(self.scfg["regime_strength"])
            unif = np.full_like(baseline, 1.0 / len(baseline))
            return (1 - a) * baseline + a * unif, True
        if regime == "basse_entropie":
            b = float(self.scfg["regime_sharpen"])
            return np.power(baseline, 1.0 + b), True
        return baseline.copy(), False


# ---------------------------------------------------------------------- #
# conteneur — branchable/débranchable
# ---------------------------------------------------------------------- #
_REGISTRY = {
    "BASELINE": BaselineSimulator, "TREND": TrendSimulator,
    "MEMORY": MemorySimulator, "REGIME": RegimeSimulator,
}


class ShadowRNGSimulator:
    """Orchestre les 4 simulateurs. enabled=liste de noms (défaut: tous).
    Un simulateur débranché ne casse rien (il n'est juste pas instancié)."""

    def __init__(self, config: Optional[dict] = None, enabled: Optional[list[str]] = None):
        self.cfg = merge_config(config)
        names = enabled or list(_REGISTRY.keys())
        if "BASELINE" not in names:
            names = ["BASELINE"] + names    # BASELINE obligatoire (référence KL)
        self.sims = {n: _REGISTRY[n](config) for n in names if n in _REGISTRY}
        logger.info("ShadowRNG simulateurs actifs : %s", list(self.sims))

    def set_profiler_snapshot(self, snapshot: dict):
        for s in self.sims.values():
            s.set_profiler_snapshot(snapshot)
        return self

    def simulate_all(self, odds_1x2, profiler_snapshot: Optional[dict] = None,
                     last_score: Optional[str] = None, n_iterations: Optional[int] = None) -> dict:
        """Lance tous les simulateurs. Renvoie {name: {output, target, active,
        kl_vs_baseline, top5}}. La KL est mesurée sur les distributions CIBLES."""
        snap = profiler_snapshot if profiler_snapshot is not None else None
        # BASELINE d'abord (référence)
        base = self.sims["BASELINE"]
        base.simulate(n_iterations, odds_1x2, snap, last_score)
        base_target = base.last_target
        results = {}
        for name, sim in self.sims.items():
            if name != "BASELINE":
                sim.simulate(n_iterations, odds_1x2, snap, last_score)
            kl = kl_divergence(sim.last_target, base_target)
            results[name] = {
                "output": sim.last_output,
                "target": sim.last_target,
                "active": sim.active if name != "BASELINE" else False,
                "kl_vs_baseline": round(kl, 5),
                "top5": [(s, round(p, 3)) for s, p in sim.top_k(5)],
            }
        return results


# ====================================================================== #
# EXEMPLE / VALIDATION CODE (données synthétiques — pas de perf prédictive)
# ====================================================================== #
if __name__ == "__main__":
    logging.basicConfig(level="WARNING", format="%(levelname)s %(name)s | %(message)s")
    import pandas as pd
    from .profiler import DistributionProfiler

    rng = np.random.RandomState(11)
    MAXG = 7
    SC = score_list(MAXG)
    IDX = {s: i for i, s in enumerate(SC)}

    def make(n, inject_bias=None, inject_memory=False):
        rows = []; prev_high = False
        for _ in range(n):
            lh = rng.uniform(0.6, 2.6); la = rng.uniform(0.5, 2.1)
            if inject_memory and prev_high:
                lh *= 0.6; la *= 0.6
            gp = _fast_grid(lh, la, 0.0)[:7, :7]; gp = gp / gp.sum()   # cotes (Poisson pur)
            p1 = np.tril(gp, -1).sum(); pX = np.trace(gp); p2 = np.triu(gp, 1).sum()
            oh, od, oa = 1.06 / max(p1, 1e-3), 1.06 / max(pX, 1e-3), 1.06 / max(p2, 1e-3)
            greal = apply_sim_deviations(lh, la, "cells")[:7, :7]; greal = greal / greal.sum()
            gs = greal.copy()
            if inject_bias:
                for sc, f in inject_bias.items():
                    h, a = map(int, sc.split("-")); gs[h, a] *= f
                gs = gs / gs.sum()
            k = rng.choice(gs.size, p=gs.ravel()); sa, sb = divmod(k, 7)
            rows.append((oh, od, oa, sa, sb)); prev_high = (sa + sb) >= 4
        return pd.DataFrame(rows, columns=["oh", "od", "oa", "sa", "sb"])

    CFG = {"simulators": {"n_iterations": 2000}}      # MC réduit pour le test

    def run_case(title, df, last_score=None, window=1000):
        print(f"\n########## {title} ##########")
        prof = DistributionProfiler().fit(df)
        snap = prof.get_full_snapshot(window=window)
        # injecter la matrice de transition (la Brique C fera ça en prod)
        T, marg = build_transition_matrix(prof._real_idx, MAXG ** 2,
                                          lag=1, smoothing=0.5, min_count=30)
        snap["transition"] = {"matrix": T, "marginal": marg}
        # un match "moyen" pour simuler : cotes 1.65 / 3.8 / 5.5
        odds = (1.65, 3.8, 5.5)
        eng = ShadowRNGSimulator(CFG).set_profiler_snapshot(snap)
        res = eng.simulate_all(odds, last_score=last_score)
        print(f"  {'sim':<9}{'active':>8}{'KL':>9}   top5")
        for name in ["BASELINE", "TREND", "MEMORY", "REGIME"]:
            r = res[name]
            t5 = " ".join(f"{s}({p*100:.0f}%)" for s, p in r["top5"][:5])
            print(f"  {name:<9}{str(r['active']):>8}{r['kl_vs_baseline']:>9.4f}   {t5}")
        return res

    run_case("CAS 1 : RNG honnête (attendu: tous active=False, KL≈0)", make(4000))
    run_case("CAS 2 : biais 0-0×1.6, 2-1×1.5 (attendu: TREND active, KL>0)",
             make(4000, inject_bias={"0-0": 1.6, "2-1": 1.5}))
    run_case("CAS 3 : mémoire injectée + last_score='0-0' (attendu: MEMORY active)",
             make(4000, inject_memory=True), last_score="0-0")

    print("\n-> Valide que chaque simulateur s'active SEULEMENT quand le signal existe, "
          "et reste = BASELINE sinon. Aucune conclusion prédictive.")
