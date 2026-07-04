"""ÉTAPE 7 — Script d'intégration PRODUCTION du Shadow-RNG Ensemble System.

Boucle continue : fit du profiler -> pour chaque round à venir, lance les 4
simulateurs + le vote, écrit le JSON dans `shadow_rng_predictions`. Re-fit du
profiler sur fenêtre glissante toutes les M itérations.

NE FAIT PAS : scoring des prédictions, métriques, ajustement de poids/seuils.
Tout ça est dans l'évaluateur (Étape 8). Séparation stricte production / évaluation.

Usage :
  ./.venv/Scripts/python.exe scripts/shadow_rng_main.py            # boucle continue
  ./.venv/Scripts/python.exe scripts/shadow_rng_main.py --smoke 5  # test : prédit 5 matchs récents
  ./.venv/Scripts/python.exe scripts/shadow_rng_main.py --config cfg.json --max-iter 3
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from sqlalchemy import create_engine, text

from scraper.config import load_settings
from scraper.shadow_rng.config import merge_config
from scraper.shadow_rng.profiler import DistributionProfiler
from scraper.shadow_rng.simulators import (ShadowRNGSimulator, build_transition_matrix,
                                           pure_poisson_distribution, score_list)
from scraper.shadow_rng.ensemble import EnsembleVoter
import numpy as np

logger = logging.getLogger("shadow_rng.main")

DDL = """
CREATE TABLE IF NOT EXISTS {table} (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id TEXT UNIQUE,
  timestamp_prediction TEXT,
  league TEXT,
  home_team TEXT,
  away_team TEXT,
  prediction_json TEXT,
  scored INTEGER DEFAULT 0,
  actual_score TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
)
"""

_HIST_SQL = """
SELECT e.id event_id, e.expected_start ts, e.competition league, e.team_a home, e.team_b away,
       o.odds_home oh, o.odds_draw od, o.odds_away oa, r.score_a sa, r.score_b sb
FROM events e
JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
JOIN results r ON r.event_id=e.id
WHERE r.score_a IS NOT NULL AND e.competition=:league
ORDER BY e.expected_start DESC LIMIT :n
"""

_ROUNDS_SQL = """
SELECT e.id event_id, e.expected_start ts, e.competition league, e.team_a home, e.team_b away,
       o.odds_home oh, o.odds_draw od, o.odds_away oa
FROM events e
JOIN odds_snapshots o ON o.id=(SELECT MAX(id) FROM odds_snapshots WHERE event_id=e.id)
LEFT JOIN results r ON r.event_id=e.id
WHERE r.id IS NULL AND e.competition=:league AND e.expected_start IS NOT NULL
  AND CAST(e.id AS TEXT) NOT IN (SELECT event_id FROM {table})
ORDER BY e.expected_start LIMIT :lim
"""

_LASTSCORE_SQL = """
SELECT r.score_a sa, r.score_b sb FROM events e JOIN results r ON r.event_id=e.id
WHERE e.competition=:league AND r.score_a IS NOT NULL AND e.expected_start < :ts
ORDER BY e.expected_start DESC LIMIT 1
"""


# ---------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------- #
def load_config(path: str | None) -> dict:
    """CONFIG de base + surcharge optionnelle depuis un fichier JSON externe."""
    overrides = None
    if path:
        overrides = json.loads(Path(path).read_text(encoding="utf-8"))
        logger.info("config externe chargée : %s", path)
    return merge_config(overrides)


def ensure_table(engine, table: str):
    with engine.begin() as cx:
        cx.execute(text(DDL.format(table=table)))


def load_history(engine, league: str, n: int) -> pd.DataFrame:
    df = pd.read_sql(text(_HIST_SQL), engine, params={"league": league, "n": int(n)})
    df = df.iloc[::-1].reset_index(drop=True)   # chrono ascendant pour le profiler
    logger.info("historique chargé : %d matchs (ligue %s)", len(df), league)
    return df


def fit_profiler(df: pd.DataFrame, cfg: dict):
    """Fit + snapshots (main/confirm) avec matrice de transition injectée."""
    prof = DistributionProfiler(cfg).fit(df)
    ecfg = cfg["ensemble"]
    snap_main = prof.get_full_snapshot(window=ecfg["main_window"])
    snap_conf = prof.get_full_snapshot(window=ecfg["confirm_window"])
    maxg2 = int(cfg["max_goals"]) ** 2
    T, marg = build_transition_matrix(
        prof._real_idx, maxg2, lag=1,
        smoothing=cfg["simulators"]["memory_smoothing"],
        min_count=cfg["simulators"]["memory_min_count"])
    for s in (snap_main, snap_conf):
        s["transition"] = {"matrix": T, "marginal": marg}
    logger.info("profiler fitté | régime=%s | anomalies=%s",
                snap_main["regime"]["regime"], snap_main["anomalies"])
    return prof, snap_main, snap_conf


def get_last_score(engine, league: str, ts, max_goals: int) -> str | None:
    try:
        r = pd.read_sql(text(_LASTSCORE_SQL), engine, params={"league": league, "ts": ts})
        if len(r) == 0:
            return None
        sa = min(int(r.sa.iloc[0]), max_goals - 1)
        sb = min(int(r.sb.iloc[0]), max_goals - 1)
        return f"{sa}-{sb}"
    except Exception as exc:
        logger.warning("last_score indisponible (%s)", exc)
        return None


def write_prediction(engine, table: str, out: dict):
    with engine.begin() as cx:
        cx.execute(text(f"""
            INSERT OR IGNORE INTO {table}
              (event_id, timestamp_prediction, league, home_team, away_team, prediction_json)
            VALUES (:eid, :ts, :lg, :h, :a, :js)"""),
            {"eid": str(out.get("event_id")), "ts": out.get("timestamp_prediction"),
             "lg": out.get("league"), "h": out.get("home_team"), "a": out.get("away_team"),
             "js": json.dumps(out, ensure_ascii=False)})


def predict_match(row, engine, league, sim_engine, voter, snap_main, snap_conf, max_goals):
    """Pipeline complet d'UN match -> JSON Brique C. Ne lève pas (robuste)."""
    oh, od, oa = float(row.oh), float(row.od), float(row.oa)
    if not (oh > 1 and od > 1 and oa > 1):
        raise ValueError(f"cotes manquantes/corrompues: {oh},{od},{oa}")
    last_score = get_last_score(engine, league, row.ts, max_goals)
    res = sim_engine.simulate_all((oh, od, oa), profiler_snapshot=snap_main, last_score=last_score)
    # référence Poisson pur (informative)
    pp = pure_poisson_distribution(oh, od, oa, max_goals)
    sc = score_list(max_goals)
    pp_top5 = [[sc[i], round(float(pp[i]), 4)] for i in np.argsort(-pp)[:5]]
    meta = {"event_id": row.event_id, "league": row.league,
            "home_team": row.home, "away_team": row.away,
            "reference_pure_poisson": {
                "top5": pp_top5,
                "full": [round(float(x), 6) for x in pp]}}
    out = voter.format_output(meta, res, snap_main, snap_conf)
    active = out["simulators_active_count"]
    logger.debug("[%s] poids=%s actifs=%d KL_trend=%s",
                 row.event_id, out["weights_applied"], active,
                 out["divergence_info"]["kl_trend_vs_baseline"])
    return out, last_score


# ---------------------------------------------------------------------- #
# boucle principale
# ---------------------------------------------------------------------- #
def run(config_path=None, smoke=0, max_iter=None):
    cfg = load_config(config_path)
    logging.getLogger("shadow_rng").setLevel(cfg["log_level"])
    rt = cfg["runtime"]
    league = rt["league"]
    table = rt["predictions_table"]
    max_goals = int(cfg["max_goals"])

    engine = create_engine(load_settings().db_url)
    ensure_table(engine, table)

    hist = load_history(engine, league, rt["history_size"])
    if len(hist) < cfg["min_window_matches"]:
        logger.error("historique trop court (%d) — abandon", len(hist))
        return
    prof, snap_main, snap_conf = fit_profiler(hist, cfg)
    sim_engine = ShadowRNGSimulator(cfg).set_profiler_snapshot(snap_main)
    voter = EnsembleVoter(cfg)

    # ---- mode SMOKE : prédit les K matchs récents (déjà settled) pour valider le pipeline ----
    if smoke > 0:
        logger.info("MODE SMOKE : prédiction de %d matchs récents (validation pipeline)", smoke)
        recent = hist.tail(smoke).copy()
        recent["league"] = recent["league"]
        n_written = 0
        for row in recent.itertuples():
            try:
                out, ls = predict_match(row, engine, league, sim_engine, voter,
                                        snap_main, snap_conf, max_goals)
                write_prediction(engine, table, out)
                n_written += 1
                top3 = [c["score"] for c in out["consensus_top3"]]
                logger.info("écrit %s | %s vs %s | consensus=%s last=%s alert=%s",
                            row.event_id, row.home, row.away, top3, ls, out["divergence_alert"])
            except Exception as exc:
                logger.warning("match %s skipé (%s)", getattr(row, "event_id", "?"), exc)
        # relire un exemple
        ex = pd.read_sql(text(f"SELECT prediction_json FROM {table} LIMIT 1"), engine)
        if len(ex):
            print("\n--- exemple de JSON écrit en base ---")
            print(json.dumps(json.loads(ex.prediction_json.iloc[0]), indent=1, ensure_ascii=False)[:1400])
        logger.info("SMOKE terminé : %d prédictions écrites dans %s", n_written, table)
        return

    # ---- boucle continue PRODUCTION ----
    it = 0
    while max_iter is None or it < max_iter:
        it += 1
        try:
            rounds = pd.read_sql(text(_ROUNDS_SQL.format(table=table)), engine,
                                 params={"league": league, "lim": 200})
        except Exception as exc:
            logger.warning("lecture des rounds échouée (%s)", exc)
            rounds = pd.DataFrame()
        for row in rounds.itertuples():
            try:
                out, ls = predict_match(row, engine, league, sim_engine, voter,
                                        snap_main, snap_conf, max_goals)
                write_prediction(engine, table, out)
                top3 = [c["score"] for c in out["consensus_top3"]]
                logger.info("prédit %s | consensus=%s | actifs=%d", row.event_id, top3,
                            out["simulators_active_count"])
                if out["divergence_alert"]:
                    logger.info("⚠️ DIVERGENCE ALERT sur %s : %s", row.event_id,
                                out["divergence_conditions"])
            except Exception as exc:
                logger.warning("match %s skipé (%s)", getattr(row, "event_id", "?"), exc)
        # re-fit périodique sur fenêtre glissante
        if it % int(rt["refit_every"]) == 0:
            hist = load_history(engine, league, rt["refit_window"])
            prof, snap_main, snap_conf = fit_profiler(hist, cfg)
            sim_engine.set_profiler_snapshot(snap_main)
            logger.info("RE-FIT profiler (itération %d, fenêtre %d)", it, len(hist))
        time.sleep(int(rt["sleep_seconds"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="fichier JSON de surcharge config")
    ap.add_argument("--smoke", type=int, default=0, help="prédit N matchs récents puis sort (test)")
    ap.add_argument("--max-iter", type=int, default=None, help="nb max d'itérations (test)")
    a = ap.parse_args()
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    run(config_path=a.config, smoke=a.smoke, max_iter=a.max_iter)


if __name__ == "__main__":
    main()
