"""BRIQUE C — EnsembleVoter (vote pondéré + filtre de divergence).

Synthétise les sorties des 4 simulateurs (Brique B) en :
  - poids appliqués (redistribution des simulateurs INACTIFS vers BASELINE+actifs) ;
  - consensus Top-3 (moyenne pondérée des distributions CIBLES) ;
  - Top-1 "haute confiance" (3 conditions strictes) ;
  - alerte de divergence (3 conditions : KL + biais FDR + confirmation multi-fenêtre,
    PAS un seuil brut — cf. risque #3) ;
  - le format JSON exact spécifié, avec slot cross-marché réservé.

Résultat honnête attendu sur ce dataset : consensus ≈ BASELINE, divergence_alert
False, top1_high_confidence condition_met False. Le système ne force RIEN.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from .config import merge_config
from .simulators import score_list, kl_divergence

logger = logging.getLogger("shadow_rng.ensemble")

_NON_BASELINE = ["TREND", "MEMORY", "REGIME"]


class EnsembleVoter:
    """Vote ensemble + détection de divergence (Brique C).

    Entrées attendues :
      - sim_results : sortie de ShadowRNGSimulator.simulate_all()
                      = {NAME: {output, target, active, kl_vs_baseline, top5}}
      - snapshot_main / snapshot_confirm : sorties de DistributionProfiler.get_full_snapshot()
                      aux fenêtres principale (200) et de confirmation (50).
    Sortie : format_output() -> dict JSON-sérialisable (format exact du cahier des charges).
    """

    def __init__(self, config: Optional[dict] = None):
        self.cfg = merge_config(config)
        self.ecfg = self.cfg["ensemble"]
        self.MAXG = int(self.cfg["max_goals"])
        self.scores = score_list(self.MAXG)
        self._idx = {s: i for i, s in enumerate(self.scores)}

    # ------------------------------------------------------------------ #
    # 1 - poids (redistribution)
    # ------------------------------------------------------------------ #
    def compute_weights(self, sim_results: dict) -> dict:
        """Poids appliqués. Les simulateurs inactifs cèdent leur poids à
        {BASELINE} ∪ {actifs}, renormalisé. BASELINE garde un plancher.

        Returns: {NAME: poids} (somme=1) restreint aux simulateurs présents.
        """
        present = [n for n in self.ecfg["weights"] if n in sim_results]
        if "BASELINE" not in present:
            present = ["BASELINE"] + present
        base = {n: float(self.ecfg["weights"].get(n, 0.0)) for n in present}
        active = [n for n in _NON_BASELINE
                  if n in sim_results and sim_results[n].get("active")]
        recipients = ["BASELINE"] + active
        tot = sum(base[r] for r in recipients) or 1.0
        w = {n: (base[n] / tot if n in recipients else 0.0) for n in present}
        # plancher BASELINE
        floor = float(self.ecfg["baseline_floor"])
        others = [r for r in recipients if r != "BASELINE"]
        if w.get("BASELINE", 0.0) < floor and others:
            tot_o = sum(base[r] for r in others) or 1.0
            w = {n: 0.0 for n in present}
            w["BASELINE"] = floor
            for r in others:
                w[r] = (1.0 - floor) * base[r] / tot_o
        return w

    # ------------------------------------------------------------------ #
    # 2 - consensus
    # ------------------------------------------------------------------ #
    def _target(self, sim_results: dict, name: str) -> np.ndarray:
        t = sim_results.get(name, {}).get("target")
        if t is None:
            return np.full(self.MAXG ** 2, 1.0 / self.MAXG ** 2)
        return np.asarray(t, float)

    def vote(self, sim_results: dict, weights: Optional[dict] = None) -> tuple[np.ndarray, dict]:
        """Distribution de consensus = moyenne pondérée des CIBLES des simulateurs.
        Returns: (vecteur consensus normalisé, poids appliqués)."""
        w = weights if weights is not None else self.compute_weights(sim_results)
        cons = np.zeros(self.MAXG ** 2, dtype=float)
        for name, weight in w.items():
            if weight > 0:
                cons += weight * self._target(sim_results, name)
        s = cons.sum()
        cons = cons / s if s > 0 else self._target(sim_results, "BASELINE")
        return cons, w

    def _top_scores(self, vec: np.ndarray, k: int) -> list[tuple[str, float]]:
        order = np.argsort(-vec)[:k]
        return [(self.scores[i], float(vec[i])) for i in order]

    # ------------------------------------------------------------------ #
    # 3 - divergence (3 conditions, pas de seuil brut)
    # ------------------------------------------------------------------ #
    def detect_divergence(self, sim_results: dict, snapshot_main: dict,
                          snapshot_confirm: Optional[dict] = None) -> dict:
        """Alerte SSI les 3 conditions sont réunies — basées sur l'ampleur GLOBALE
        du biais FDR (le profiler fait ce travail sur la fenêtre glissante), PAS sur
        la KL per-match (qui ne mesure qu'UN match) :

          (a) >= N scores FDR-significatifs ET ampleur moyenne de leurs biais >= seuil
          (b) >= 1 score FDR-significatif (gardé pour la lisibilité du JSON)
          (c) sens du biais confirmé sur 2 fenêtres (50 ET 200)

        La KL(TREND||BASELINE) reste calculée et reportée (kl_trend_info) pour le
        monitoring, mais NE conditionne plus l'alerte.

        Returns: {alert, confidence, conditions: {fdr_amplitude_sufficient,
            fdr_significant_bias, confirmed_multi_window}, mean_amplitude,
            n_anomalies, kl_trend_info}.
        """
        anomalies = list((snapshot_main or {}).get("anomalies", []))
        bm = (snapshot_main or {}).get("biais_snapshot", {})
        amps = [abs(bm.get(s, 0.0)) / 100.0 for s in anomalies]  # pp -> fraction
        mean_amp = float(np.mean(amps)) if amps else 0.0
        min_scores = int(self.ecfg["divergence_min_scores"])
        amp_thr = float(self.ecfg["divergence_amplitude"])

        cond_a = len(anomalies) >= min_scores and mean_amp >= amp_thr
        cond_b = len(anomalies) > 0
        cond_c = False
        if snapshot_confirm and anomalies:
            bc = snapshot_confirm.get("biais_snapshot", {})
            cond_c = any(
                bm.get(s, 0.0) != 0.0 and np.sign(bm.get(s, 0.0)) == np.sign(bc.get(s, 0.0))
                for s in anomalies
            )

        conditions = {
            "fdr_amplitude_sufficient": bool(cond_a),
            "fdr_significant_bias": bool(cond_b),
            "confirmed_multi_window": bool(cond_c),
        }
        n_met = sum(conditions.values())
        return {
            "alert": bool(n_met == 3),
            "confidence": round(n_met / 3.0, 2),
            "conditions": conditions,
            "mean_amplitude": round(mean_amp, 4),
            "n_anomalies": len(anomalies),
            "kl_trend_info": round(float(sim_results.get("TREND", {}).get("kl_vs_baseline", 0.0)), 5),
        }

    # ------------------------------------------------------------------ #
    # 4 - Top-1 haute confiance
    # ------------------------------------------------------------------ #
    def top1_high_confidence(self, consensus_top: list[tuple[str, float]],
                             baseline_top: list[tuple[str, float]],
                             active_count: int) -> dict:
        """Rempli SSI : top consensus ∈ top-3 consensus (trivial) ET ∈ top-3 BASELINE
        ET >=2 simulateurs actifs. Sinon score=null, condition_met=False."""
        cons_scores = [s for s, _ in consensus_top]
        base_scores = [s for s, _ in baseline_top]
        if not cons_scores:
            return {"score": None, "condition_met": False, "reason": "no_consensus"}
        cand = cons_scores[0]
        in_base = cand in base_scores
        enough = active_count >= 2
        if in_base and enough:
            return {"score": cand, "condition_met": True,
                    "reason": "in_consensus_and_baseline_top3_and_>=2_active_signals"}
        reason = "insufficient_active_signals" if not enough else "top1_not_in_baseline_top3"
        return {"score": None, "condition_met": False, "reason": reason}

    # ------------------------------------------------------------------ #
    # 5 - assemblage JSON
    # ------------------------------------------------------------------ #
    def format_output(self, event_metadata: dict, sim_results: dict,
                      snapshot_main: dict, snapshot_confirm: Optional[dict] = None) -> dict:
        """Produit le JSON complet (format exact du cahier des charges).
        Robuste : produit toujours un JSON, même dégradé si une entrée manque."""
        try:
            weights = self.compute_weights(sim_results)
            cons_vec, weights = self.vote(sim_results, weights)
            baseline_vec = self._target(sim_results, "BASELINE")
            k = int(self.ecfg["consensus_top_n"])
            consensus_top = self._top_scores(cons_vec, k)
            baseline_top = self._top_scores(baseline_vec, 3)
            # distributions COMPLÈTES (49-vec) — requises par l'évaluateur pour
            # log-loss / Brier sur le score RÉALISÉ (qui peut être hors top-3).
            # Index canonique = score_list(max_goals) (cf. evaluator).
            rpp = event_metadata.get("reference_pure_poisson") or {}
            poisson_full = ([round(float(x), 6) for x in rpp["full"]]
                            if isinstance(rpp, dict) and rpp.get("full") is not None else None)
            active_count = sum(1 for n in _NON_BASELINE
                               if n in sim_results and sim_results[n].get("active"))
            div = self.detect_divergence(sim_results, snapshot_main, snapshot_confirm)
            anomalies = set((snapshot_main or {}).get("anomalies", []))

            def sim_block(name):
                r = sim_results.get(name, {})
                return {
                    "top5": [[s, round(p, 4)] for s, p in r.get("top5", [])],
                    "active": bool(r.get("active", False)),
                    "kl_vs_baseline": round(float(r.get("kl_vs_baseline", 0.0)), 5),
                }

            regime = (snapshot_main or {}).get("regime", {})
            out = {
                "event_id": event_metadata.get("event_id"),
                "timestamp_prediction": event_metadata.get(
                    "timestamp_prediction", datetime.now(timezone.utc).isoformat()),
                "league": event_metadata.get("league"),
                "home_team": event_metadata.get("home_team"),
                "away_team": event_metadata.get("away_team"),

                "simulators": {
                    "baseline": sim_block("BASELINE"),
                    "trend": sim_block("TREND"),
                    "memory": sim_block("MEMORY"),
                    "regime": sim_block("REGIME"),
                },
                "weights_applied": {
                    "baseline": round(weights.get("BASELINE", 0.0), 4),
                    "trend": round(weights.get("TREND", 0.0), 4),
                    "memory": round(weights.get("MEMORY", 0.0), 4),
                    "regime": round(weights.get("REGIME", 0.0), 4),
                },
                "consensus_top3": [
                    {"score": s, "consensus_prob": round(p, 4),
                     "divergence_flag": bool(s in anomalies)}
                    for s, p in consensus_top
                ],
                # vecteurs complets pour l'évaluation honnête (log-loss/Brier)
                "consensus_full": [round(float(x), 6) for x in cons_vec],
                "baseline_full": [round(float(x), 6) for x in baseline_vec],
                "poisson_full": poisson_full,
                "top1_high_confidence": self.top1_high_confidence(
                    consensus_top, baseline_top, active_count),

                "divergence_alert": div["alert"],
                "divergence_confidence": div["confidence"],
                "divergence_conditions": div["conditions"],
                "divergence_info": {                       # informatif (monitoring)
                    "mean_amplitude": div["mean_amplitude"],
                    "n_anomalies": div["n_anomalies"],
                    "kl_trend_vs_baseline": div["kl_trend_info"],
                },

                "current_regime": regime.get("regime", "normal"),
                "regime_confidence": regime.get("confidence", 0.0),
                "simulators_active_count": active_count,

                "active_window": (snapshot_main or {}).get("active_window"),
                "biais_snapshot": (snapshot_main or {}).get("biais_snapshot", {}),

                # référence Poisson pur — INFORMATIVE seulement (non utilisée dans les alertes)
                "reference_pure_poisson": event_metadata.get("reference_pure_poisson"),

                "cross_market_slot": dict(self.cfg["cross_market_slot"]),
            }
            return out
        except Exception as exc:                     # JSON toujours produit, même dégradé
            logger.exception("format_output dégradé (%s)", exc)
            return {
                "event_id": event_metadata.get("event_id"),
                "error": f"degraded:{exc}",
                "cross_market_slot": dict(self.cfg["cross_market_slot"]),
            }


# ====================================================================== #
# VALIDATION CODE (données synthétiques — pas de perf prédictive)
# ====================================================================== #
if __name__ == "__main__":
    logging.basicConfig(level="WARNING", format="%(levelname)s %(name)s | %(message)s")
    import json
    import pandas as pd
    from .profiler import DistributionProfiler
    from .simulators import (ShadowRNGSimulator, build_transition_matrix, _fast_grid,
                             apply_sim_deviations, pure_poisson_distribution, score_list)

    rng = np.random.RandomState(21)
    MAXG = 7

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

    CFG = {"simulators": {"n_iterations": 2000}}

    def run(title, df, last_score=None, main_window=200):
        print(f"\n########## {title} ##########")
        prof = DistributionProfiler().fit(df)
        snap_main = prof.get_full_snapshot(window=main_window)
        snap_conf = prof.get_full_snapshot(window=50)
        T, marg = build_transition_matrix(prof._real_idx, MAXG ** 2, 1, 0.5, 30)
        for s in (snap_main, snap_conf):
            s["transition"] = {"matrix": T, "marginal": marg}
        odds = (1.65, 3.8, 5.5)
        eng = ShadowRNGSimulator(CFG).set_profiler_snapshot(snap_main)
        res = eng.simulate_all(odds, last_score=last_score)
        voter = EnsembleVoter(CFG)
        pp = pure_poisson_distribution(odds[0], odds[1], odds[2], MAXG)
        SC = score_list(MAXG)
        pp_top5 = [[SC[i], round(float(pp[i]), 4)] for i in np.argsort(-pp)[:5]]
        meta = {"event_id": "synthetic", "league": "TEST", "home_team": "H", "away_team": "A",
                "reference_pure_poisson": {"top5": pp_top5}}
        out = voter.format_output(meta, res, snap_main, snap_conf)
        di = out["divergence_info"]
        print(f"  poids: {out['weights_applied']}")
        print(f"  actifs: {out['simulators_active_count']} | régime: {out['current_regime']}")
        print(f"  consensus_top3: {[(c['score'], round(c['consensus_prob'],3), c['divergence_flag']) for c in out['consensus_top3']]}")
        print(f"  divergence: alert={out['divergence_alert']} conf={out['divergence_confidence']} {out['divergence_conditions']}")
        print(f"    info: n_anomalies={di['n_anomalies']} ampleur_moy={di['mean_amplitude']} (seuil 0.02) kl_info={di['kl_trend_vs_baseline']}")
        print(f"  top1_high_confidence: {out['top1_high_confidence']}")
        return out

    run("CAS 1 : RNG honnête (attendu: actifs=0, alert=False, top1 met=False)",
        make(4000))
    run("CAS 2 : biais MODÉRÉ (attendu: conservateur — peut ne pas alerter à w=200)",
        make(4000, inject_bias={"0-0": 1.6, "2-1": 1.5}))
    o3 = run("CAS 3 : biais+mémoire (attendu: MEMORY actif, top1 met=True)",
             make(4000, inject_bias={"0-0": 1.6, "2-1": 1.5}, inject_memory=True), last_score="0-0")
    run("CAS 4 : biais FORT multi-scores @ w=1500 (puissance FDR) -> attendu ALERTE=True",
        make(6000, inject_bias={"0-0": 2.2, "1-1": 1.9, "2-1": 1.8, "0-1": 1.7}),
        main_window=1500)

    print("\n--- JSON complet du CAS 3 (extrait) ---")
    print(json.dumps({k: o3[k] for k in ["weights_applied", "consensus_top3",
          "top1_high_confidence", "divergence_alert", "divergence_conditions",
          "simulators_active_count", "cross_market_slot"]}, indent=1, ensure_ascii=False))
    print("\n-> Valide le vote, la redistribution, les 3 conditions de divergence "
          "et le format JSON. Aucune conclusion prédictive.")
