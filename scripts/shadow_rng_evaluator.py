"""ÉTAPE 8 — Évaluateur du Shadow-RNG Ensemble System (lecture seule + scoring).

Compare en continu, sur une fenêtre glissante, la qualité prédictive du CONSENSUS
(système actif : 4 simulateurs + vote) contre celle du BASELINE (modèle réalisé
calibré `apply_sim_deviations`, passif). Référence informative supplémentaire :
Poisson pur (le pricing brut).

Mesure honnête attendue sur ce RNG calibré : consensus ≈ baseline (aucun edge,
aucune dégradation). Le framework est là pour DÉTECTER une future dérive du RNG
(le jour où le consensus battrait — ou casserait — durablement le baseline).

========================  RÈGLE ABSOLUE — SÉPARATION  ========================
Cet évaluateur NE FAIT QUE :
  * marquer les prédictions échues (scored=1 + actual_score) — bookkeeping ;
  * lire les prédictions et calculer des métriques ;
  * afficher un tableau de bord + lever une alarme.
Il NE FAIT JAMAIS :
  * ajuster un poids, un seuil, une seed ;
  * re-fitter le profiler ;
  * modifier prediction_json, le code de production ou shadow_rng_main.py ;
  * "corriger" automatiquement quoi que ce soit.
Toute correction est une DÉCISION HUMAINE, déclenchée par l'alarme, jamais par lui.
=============================================================================

Usage :
  ./.venv/Scripts/python.exe scripts/shadow_rng_evaluator.py             # 1 passe + tableau
  ./.venv/Scripts/python.exe scripts/shadow_rng_evaluator.py --window 500
  ./.venv/Scripts/python.exe scripts/shadow_rng_evaluator.py --watch 60  # boucle (refresh 60s)
  ./.venv/Scripts/python.exe scripts/shadow_rng_evaluator.py --no-settle # n'écrit RIEN en base
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from scraper.config import load_settings
from scraper.shadow_rng.config import merge_config
from scraper.shadow_rng.simulators import score_list

logger = logging.getLogger("shadow_rng.evaluator")

# ---- paramètres de l'évaluateur (séparés de la config de PRODUCTION) ----
DEFAULTS = {
    "window": 500,            # fenêtre glissante (nb de prédictions échues récentes)
    "alarm_logloss_delta": 0.01,  # log_loss_cons > log_loss_base + delta -> ALARME ROUGE
    "min_n_alarm": 100,       # en-dessous : alarme INDICATIVE seulement (anti-bruit petit échantillon)
    "logloss_eps": 1e-6,      # plancher de proba (évite -log(0))
}


# ---------------------------------------------------------------------- #
# SQL
# ---------------------------------------------------------------------- #
_SETTLE_SQL = """
SELECT p.id pid, p.event_id, r.score_a sa, r.score_b sb
FROM {table} p
JOIN events e  ON CAST(e.id AS TEXT) = p.event_id
JOIN results r ON r.event_id = e.id
WHERE p.scored = 0 AND r.score_a IS NOT NULL
"""

_LOAD_SCORED_SQL = """
SELECT id, event_id, prediction_json, actual_score, created_at
FROM {table}
WHERE scored = 1 AND actual_score IS NOT NULL
ORDER BY id DESC
LIMIT :window
"""

_COUNT_SQL = "SELECT COUNT(*) n, SUM(scored) s FROM {table}"


# ---------------------------------------------------------------------- #
# settlement (seule écriture autorisée : scored + actual_score)
# ---------------------------------------------------------------------- #
def settle(engine, table: str, max_goals: int) -> int:
    """Marque les prédictions dont le résultat est désormais connu.
    N'écrit QUE les colonnes scored et actual_score. Retourne le nb settlé."""
    df = pd.read_sql(text(_SETTLE_SQL.format(table=table)), engine)
    if df.empty:
        return 0
    cap = max_goals - 1
    updates = []
    for r in df.itertuples():
        sa = min(int(r.sa), cap)
        sb = min(int(r.sb), cap)
        updates.append({"pid": int(r.pid), "sc": f"{sa}-{sb}"})
    with engine.begin() as cx:
        for u in updates:
            cx.execute(text(
                f"UPDATE {table} SET scored=1, actual_score=:sc WHERE id=:pid"), u)
    return len(updates)


# ---------------------------------------------------------------------- #
# métriques d'UNE prédiction
# ---------------------------------------------------------------------- #
def _logloss(p_actual: float, eps: float) -> float:
    return -math.log(max(float(p_actual), eps))


def _brier(vec: np.ndarray, idx: int) -> float:
    """Brier multiclasse = somme des carrés - 2*p_actual + 1 (one-hot en idx)."""
    return float(np.dot(vec, vec) - 2.0 * vec[idx] + 1.0)


def outcome_masks(max_goals: int):
    """Masques (home, draw, away) sur l'index des scores, pour projeter une
    grille 49-vec sur le marché 1X2."""
    sc = score_list(max_goals)
    hd = np.array([int(s.split("-")[0]) for s in sc])
    ad = np.array([int(s.split("-")[1]) for s in sc])
    return ((hd > ad).astype(float), (hd == ad).astype(float), (hd < ad).astype(float))


def _x12_from_vec(vec: np.ndarray, masks) -> tuple[str, float, float]:
    """(pick 1/X/2, proba du pick, proba de chaque) à partir d'une grille."""
    ph = float(vec @ masks[0]); px = float(vec @ masks[1]); pa = float(vec @ masks[2])
    probs = (ph, px, pa)
    j = int(np.argmax(probs))
    return ("1", "X", "2")[j], probs[j], probs


def _actual_x12(score: str) -> str:
    h, a = map(int, score.split("-"))
    return "1" if h > a else ("X" if h == a else "2")


def score_one(pred: dict, actual: str, idx_map: dict, eps: float, masks=None):
    """Métriques d'une prédiction vs le score réalisé. None si JSON dégradé."""
    if actual not in idx_map:
        return None
    j = idx_map[actual]
    cons_full = pred.get("consensus_full")
    base_full = pred.get("baseline_full")
    if not cons_full or not base_full:
        return None
    cv = np.asarray(cons_full, float)
    bv = np.asarray(base_full, float)
    pois = pred.get("poisson_full")
    pv = np.asarray(pois, float) if pois else None

    # ---- marché 1X2 (projeté depuis les grilles complètes) ----
    act_x12 = _actual_x12(actual)
    cx_pick, cx_pp, _ = _x12_from_vec(cv, masks) if masks else (None, None, None)
    bx_pick, bx_pp, _ = _x12_from_vec(bv, masks) if masks else (None, None, None)

    cons_top3 = pred.get("consensus_top3", [])
    cons_scores = [c["score"] for c in cons_top3]
    base_top5 = (pred.get("simulators", {}).get("baseline", {}).get("top5", []))
    base_scores = [s for s, _ in base_top5][:3]
    sims = pred.get("simulators", {})

    return {
        "actual": actual,
        # consensus
        "ll_cons": _logloss(cv[j], eps),
        "br_cons": _brier(cv, j),
        "top1_cons": bool(cons_scores[:1] == [actual]),
        "top3_cons": bool(actual in cons_scores),
        # baseline
        "ll_base": _logloss(bv[j], eps),
        "br_base": _brier(bv, j),
        "top1_base": bool(base_scores[:1] == [actual]),
        "top3_base": bool(actual in base_scores),
        # poisson (informatif)
        "ll_pois": _logloss(pv[j], eps) if pv is not None else None,
        # activité simulateurs
        "act_trend": bool(sims.get("trend", {}).get("active")),
        "act_memory": bool(sims.get("memory", {}).get("active")),
        "act_regime": bool(sims.get("regime", {}).get("active")),
        "n_active": int(pred.get("simulators_active_count", 0)),
        # divergence
        "alert": bool(pred.get("divergence_alert", False)),
        # ---- marché 1X2 ----
        "x12_actual": act_x12,
        "x12_cons": cx_pick, "x12_cons_prob": cx_pp,
        "x12_base": bx_pick, "x12_base_prob": bx_pp,
        "hit_x12_cons": (cx_pick == act_x12) if cx_pick else None,
        "hit_x12_base": (bx_pick == act_x12) if bx_pick else None,
        # ---- champs additionnels pour --export (non utilisés par l'agrégation) ----
        "event_id": pred.get("event_id"),
        "timestamp": pred.get("timestamp_prediction"),
        "league": pred.get("league"),
        "consensus_top1": cons_scores[0] if cons_scores else None,
        "consensus_top1_prob": (cons_top3[0].get("consensus_prob") if cons_top3 else None),
        "baseline_top1": base_scores[0] if base_scores else None,
        "baseline_top1_prob": (base_top5[0][1] if base_top5 else None),
        "div_conf": float(pred.get("divergence_confidence", 0.0)),
    }


# ---------------------------------------------------------------------- #
# agrégation fenêtre
# ---------------------------------------------------------------------- #
def aggregate(rows: list[dict], alarm_delta: float, min_n_alarm: int) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    a = pd.DataFrame(rows)

    def m(col):
        v = a[col].dropna()
        return float(v.mean()) if len(v) else float("nan")

    ll_cons, ll_base = m("ll_cons"), m("ll_base")
    res = {
        "n": n,
        "top1_cons": m("top1_cons"), "top3_cons": m("top3_cons"),
        "top1_base": m("top1_base"), "top3_base": m("top3_base"),
        "ll_cons": ll_cons, "ll_base": ll_base, "ll_pois": m("ll_pois"),
        "br_cons": m("br_cons"), "br_base": m("br_base"),
        "lift_logloss": ll_base - ll_cons,            # >0 : consensus meilleur
        "lift_top3": m("top3_cons") - m("top3_base"),
        # marché 1X2
        "x12_cons": m("hit_x12_cons"), "x12_base": m("hit_x12_base"),
        "x12_cons_prob": m("x12_cons_prob"),          # proba moyenne du pick (calibration)
        "lift_x12": m("hit_x12_cons") - m("hit_x12_base"),
        # activité
        "pct_trend": m("act_trend"), "pct_memory": m("act_memory"),
        "pct_regime": m("act_regime"), "mean_active": m("n_active"),
        "n_alerts": int(a["alert"].sum()),
    }

    # lift conditionnel par simulateur (association, PAS isolation causale)
    for sim, col in [("TREND", "act_trend"), ("MEMORY", "act_memory"), ("REGIME", "act_regime")]:
        sub = a[a[col]]
        res[f"cond_lift_{sim}"] = (float((sub["ll_base"] - sub["ll_cons"]).mean())
                                   if len(sub) else float("nan"))
        res[f"cond_n_{sim}"] = int(len(sub))

    # valeur prédictive de l'alerte de divergence
    alert = a[a["alert"]]
    res["alert_top3_cons"] = float(alert["top3_cons"].mean()) if len(alert) else float("nan")
    res["alert_lift_logloss"] = (float((alert["ll_base"] - alert["ll_cons"]).mean())
                                 if len(alert) else float("nan"))
    res["alert_precision"] = (float((alert["ll_cons"] < alert["ll_base"]).mean())
                              if len(alert) else float("nan"))

    # ALARME : le système ACTIF dégrade-t-il vs le baseline PASSIF ?
    res["alarm"] = bool(ll_cons > ll_base + alarm_delta)
    res["alarm_delta"] = alarm_delta
    res["alarm_reliable"] = bool(n >= min_n_alarm)   # sinon : indicatif (échantillon court)
    res["min_n_alarm"] = int(min_n_alarm)
    return res


# ---------------------------------------------------------------------- #
# rendu console
# ---------------------------------------------------------------------- #
def _pct(x):
    return "  n/a " if x != x else f"{100*x:5.1f}%"


def _f(x, d=4):
    return " n/a " if x != x else f"{x:.{d}f}"


def render(M: dict) -> str:
    if M["n"] == 0:
        return ("\n  Aucune prédiction échue à évaluer pour l'instant.\n"
                "  (le scraper/predicteur doit tourner et les résultats arriver en base)\n")
    L = []
    bar = "=" * 70
    L.append("")
    L.append(bar)
    L.append(f"  SHADOW-RNG — TABLEAU DE BORD (fenêtre glissante : {M['n']} matchs échus)")
    L.append(bar)
    L.append("  Métrique             | CONSENSUS (actif) | BASELINE (passif) |  Poisson")
    L.append("  ---------------------+-------------------+-------------------+---------")
    L.append(f"  Top-1 (score exact)  |      {_pct(M['top1_cons'])}        |      {_pct(M['top1_base'])}        |    -")
    L.append(f"  Top-3                |      {_pct(M['top3_cons'])}        |      {_pct(M['top3_base'])}        |    -")
    L.append(f"  1X2 (issue du match) |      {_pct(M['x12_cons'])}        |      {_pct(M['x12_base'])}        |    -")
    L.append(f"  Log-loss (v bas)     |       {_f(M['ll_cons'])}      |       {_f(M['ll_base'])}      |  {_f(M['ll_pois'])}")
    L.append(f"  Brier   (v bas)      |       {_f(M['br_cons'])}      |       {_f(M['br_base'])}      |    -")
    L.append("  ---------------------+-------------------+-------------------+---------")
    L.append(f"  LIFT log-loss (base - cons, >0 = consensus meilleur) : {M['lift_logloss']:+.4f}")
    L.append(f"  LIFT Top-3   (cons - base, >0 = consensus meilleur)  : {M['lift_top3']:+.4f}")
    L.append(f"  1X2 : consensus {_pct(M['x12_cons'])} réalisé vs {_pct(M['x12_cons_prob'])} prédit "
             f"(pick moyen) — lift vs base {M['lift_x12']:+.4f}")
    L.append("")
    L.append("  Activité des simulateurs (sur la fenêtre) :")
    L.append(f"    TREND  actif {_pct(M['pct_trend'])}  | lift conditionnel {_f(M['cond_lift_TREND'])} (n={M['cond_n_TREND']})")
    L.append(f"    MEMORY actif {_pct(M['pct_memory'])} | lift conditionnel {_f(M['cond_lift_MEMORY'])} (n={M['cond_n_MEMORY']})")
    L.append(f"    REGIME actif {_pct(M['pct_regime'])} | lift conditionnel {_f(M['cond_lift_REGIME'])} (n={M['cond_n_REGIME']})")
    L.append(f"    nb moyen de simulateurs actifs / match : {_f(M['mean_active'], 2)}")
    L.append("")
    L.append(f"  Alertes de divergence : {M['n_alerts']}")
    if M["n_alerts"]:
        L.append(f"    Top-3 consensus sur matchs alertés : {_pct(M['alert_top3_cons'])}")
        L.append(f"    lift log-loss sur alertes          : {_f(M['alert_lift_logloss'])}")
        L.append(f"    precision (cons<base sur alertes)  : {_pct(M['alert_precision'])}")
    else:
        L.append("    (aucune alerte — comportement attendu sur un RNG calibré)")
    L.append("")
    L.append(bar)
    reliable = M.get("alarm_reliable", True)
    short = "" if reliable else f"  [echantillon court n<{M.get('min_n_alarm')} -> INDICATIF, non fiable]"
    if M["alarm"]:
        head = "[!!!] ALARME ROUGE" if reliable else "[ ?? ] alerte INDICATIVE"
        L.append(f"  {head} : log-loss CONSENSUS ({_f(M['ll_cons'])}) > "
                 f"BASELINE ({_f(M['ll_base'])}) + {M['alarm_delta']}{short}")
        L.append("        Le systeme ACTIF degrade la prediction vs le baseline passif.")
        if reliable:
            L.append("        --> INVESTIGATION HUMAINE requise. AUCUN auto-ajustement effectue.")
        else:
            L.append("        --> attendre l'accumulation de prédictions avant toute conclusion.")
    elif M["lift_logloss"] > M["alarm_delta"]:
        head = "[+++] SIGNAL POTENTIEL" if reliable else "[ ?? ] signal INDICATIF"
        L.append(f"  {head} : consensus meilleur que baseline de "
                 f"{M['lift_logloss']:+.4f} log-loss.{short}")
        L.append("        Possible debut de derive RNG exploitable. A CONFIRMER sur fenetre longue")
        L.append("        avant toute decision (ne rien changer automatiquement).")
    else:
        L.append("  [ OK ] Consensus ~ baseline (|lift| sous le seuil). Aucune degradation,")
        L.append("         aucun edge. Comportement honnete attendu. Framework pret a detecter")
        L.append("         une future derive du RNG.")
    L.append(bar)
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------- #
# export CSV (1 ligne par round scoré) — lecture seule, pour notebook
# ---------------------------------------------------------------------- #
_EXPORT_COLS = [
    "event_id", "timestamp", "league", "actual_score",
    "consensus_top1", "consensus_top1_prob",
    "baseline_top1", "baseline_top1_prob",
    "hit_top1_consensus", "hit_top1_baseline",
    "hit_top3_consensus", "hit_top3_baseline",
    "log_loss_consensus", "log_loss_baseline",
    "brier_consensus", "brier_baseline",
    "trend_active", "memory_active", "regime_active",
    "divergence_alert", "divergence_confidence",
    "simulators_active_count",
    # ---- marché 1X2 (ajout) ----
    "actual_1x2", "consensus_1x2", "consensus_1x2_prob",
    "baseline_1x2", "hit_1x2_consensus", "hit_1x2_baseline",
]


def _export_row(r: dict) -> dict:
    """Mappe une ligne de score_one() vers une ligne CSV (ordre _EXPORT_COLS)."""
    def b(x):  # bool -> 0/1
        return 1 if x else 0
    def num(x, d):
        return "" if x is None else round(float(x), d)
    return {
        "event_id": r.get("event_id"),
        "timestamp": r.get("timestamp"),
        "league": r.get("league"),
        "actual_score": r.get("actual"),
        "consensus_top1": r.get("consensus_top1"),
        "consensus_top1_prob": num(r.get("consensus_top1_prob"), 4),
        "baseline_top1": r.get("baseline_top1"),
        "baseline_top1_prob": num(r.get("baseline_top1_prob"), 4),
        "hit_top1_consensus": b(r.get("top1_cons")),
        "hit_top1_baseline": b(r.get("top1_base")),
        "hit_top3_consensus": b(r.get("top3_cons")),
        "hit_top3_baseline": b(r.get("top3_base")),
        "log_loss_consensus": num(r.get("ll_cons"), 6),
        "log_loss_baseline": num(r.get("ll_base"), 6),
        "brier_consensus": num(r.get("br_cons"), 6),
        "brier_baseline": num(r.get("br_base"), 6),
        "trend_active": b(r.get("act_trend")),
        "memory_active": b(r.get("act_memory")),
        "regime_active": b(r.get("act_regime")),
        "divergence_alert": b(r.get("alert")),
        "divergence_confidence": num(r.get("div_conf"), 2),
        "simulators_active_count": int(r.get("n_active", 0)),
        "actual_1x2": r.get("x12_actual"),
        "consensus_1x2": r.get("x12_cons"),
        "consensus_1x2_prob": num(r.get("x12_cons_prob"), 4),
        "baseline_1x2": r.get("x12_base"),
        "hit_1x2_consensus": b(r.get("hit_x12_cons")),
        "hit_1x2_baseline": b(r.get("hit_x12_base")),
    }


def write_export(rows: list[dict], path: str) -> int:
    """Écrit le CSV (1 ligne/round, ordre chronologique). Retourne le nb de lignes.
    N'écrit RIEN d'autre — pur artefact d'analyse, jamais relu par la production."""
    p = Path(path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    ordered = list(reversed(rows))   # rows = id DESC (récent->ancien) -> chrono ascendant
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_EXPORT_COLS)
        w.writeheader()
        for r in ordered:
            w.writerow(_export_row(r))
    return len(ordered)


# ---------------------------------------------------------------------- #
# pilotage
# ---------------------------------------------------------------------- #
def evaluate_once(engine, table: str, max_goals: int, window: int,
                  alarm_delta: float, eps: float, do_settle: bool) -> dict:
    if do_settle:
        n_settled = settle(engine, table, max_goals)
        if n_settled:
            logger.info("settlement : %d prédiction(s) nouvellement échue(s)", n_settled)

    idx_map = {s: i for i, s in enumerate(score_list(max_goals))}
    masks = outcome_masks(max_goals)
    df = pd.read_sql(text(_LOAD_SCORED_SQL.format(table=table)), engine,
                     params={"window": int(window)})
    rows = []
    for r in df.itertuples():
        try:
            pred = json.loads(r.prediction_json)
        except Exception:
            continue
        s = score_one(pred, r.actual_score, idx_map, eps, masks)
        if s is not None:
            rows.append(s)
    M = aggregate(rows, alarm_delta, int(DEFAULTS["min_n_alarm"]))

    # contexte global
    try:
        c = pd.read_sql(text(_COUNT_SQL.format(table=table)), engine)
        M["_total_rows"] = int(c.n.iloc[0] or 0)
        M["_total_scored"] = int(c.s.iloc[0] or 0)
    except Exception:
        pass
    return M, rows


def run(config_path=None, window=None, watch=0, alarm_delta=None,
        do_settle=True, max_goals_override=None, export_path=None):
    cfg = merge_config(json.loads(Path(config_path).read_text(encoding="utf-8"))
                       if config_path else None)
    rt = cfg["runtime"]
    table = rt["predictions_table"]
    max_goals = int(max_goals_override or cfg["max_goals"])
    window = int(window or DEFAULTS["window"])
    alarm_delta = float(alarm_delta if alarm_delta is not None
                        else DEFAULTS["alarm_logloss_delta"])
    eps = float(DEFAULTS["logloss_eps"])

    engine = create_engine(load_settings().db_url)

    def one():
        M, rows = evaluate_once(engine, table, max_goals, window, alarm_delta, eps, do_settle)
        print(render(M))
        if "_total_scored" in M:
            print(f"  (base : {M['_total_scored']}/{M['_total_rows']} prédictions échues au total)\n")
        if export_path:
            n = write_export(rows, export_path)
            logger.info("export CSV : %d ligne(s) -> %s", n, export_path)
        return M

    if watch and watch > 0:
        logger.info("MODE WATCH : refresh toutes les %ds (Ctrl-C pour stopper)", watch)
        while True:
            try:
                one()
            except Exception as exc:
                logger.warning("itération évaluateur échouée (%s)", exc)
            time.sleep(int(watch))
    else:
        return one()


def main():
    ap = argparse.ArgumentParser(description="Évaluateur Shadow-RNG (lecture seule + scoring)")
    ap.add_argument("--config", default=None, help="config de prod (pour league/table/max_goals)")
    ap.add_argument("--window", type=int, default=None, help="taille fenêtre glissante")
    ap.add_argument("--once", action="store_true",
                    help="une seule passe puis sort (comportement par défaut)")
    ap.add_argument("--watch", type=int, default=0, help="boucle : refresh toutes les N secondes")
    ap.add_argument("--alarm-threshold", type=float, default=None,
                    help="delta log-loss déclenchant l'alarme rouge (défaut 0.01)")
    ap.add_argument("--no-settle", action="store_true",
                    help="n'écrit RIEN en base (pas de marquage scored)")
    ap.add_argument("--export", default=None, metavar="FICHIER.csv",
                    help="exporte 1 ligne/round scoré (fenêtre courante) vers ce CSV")
    a = ap.parse_args()
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    watch = 0 if a.once else a.watch        # --once force une seule passe
    run(config_path=a.config, window=a.window, watch=watch,
        alarm_delta=a.alarm_threshold, do_settle=not a.no_settle, export_path=a.export)


if __name__ == "__main__":
    main()
