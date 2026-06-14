"""WF5 — Diagnostic provenance du claim +9pp/+15pp : effet d'un pair cache LEAKY.

Identique a _wf5_score_v2_audit.py MAIS le PairScoreCache est charge via _load()
par defaut = TOUTE la BDD, y compris les resultats des matchs de TEST eux-memes
(comme le ferait ScorePredictorV2 utilise naivement en backtest).

Si le Top1/Top3 V2 bondit vers ~21%/45% avec ce cache, le claim "+9pp/+15pp"
est un artefact de fuite de donnees, pas une capacite predictive.
Sortie : exports/wf5_score_v2_leak_check.json. LECTURE SEULE.
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from sqlalchemy import create_engine, text

from scraper.config import load_settings
from scraper.predictor_v5 import fit_model_v5, predict_match_v5
from scraper.score_predictor_v2 import ScorePredictorV2
from scraper.strategy_engine import label_segment

TRAIN_END = "2026-06-10"
TEST_END = "2026-06-13"
NEEDED_MARKETS = ("Score exact", "Total de buts", "G/NG")
COMMON_SCORES = ["0-0", "1-0", "0-1", "1-1", "2-0", "0-2", "2-1", "1-2", "2-2",
                 "3-0", "0-3", "3-1", "1-3", "3-2", "2-3", "4-0", "0-4"]


def main() -> int:
    settings = load_settings()
    engine = create_engine(settings.db_url)

    corrupted = set()
    cj = ROOT / "exports" / "corrupted_events.json"
    if cj.exists():
        d = json.loads(cj.read_text(encoding="utf-8"))
        corrupted = {int(k) for k in d.get("events", {})}

    base = pd.read_sql(text("""
        SELECT e.id AS ev_id, e.round_info, e.team_a, e.team_b,
               o.id AS snap_id, o.odds_home, o.odds_draw, o.odds_away,
               r.score_a, r.score_b, r.ht_score_a, r.ht_score_b, r.finished_at
        FROM events e
        JOIN odds_snapshots o ON o.id = (SELECT MIN(id) FROM odds_snapshots
                                          WHERE event_id = e.id)
        JOIN results r ON r.event_id = e.id
        WHERE r.score_a IS NOT NULL
          AND e.competition = 'InstantLeague-8035'
          AND r.finished_at < :test_end
        ORDER BY r.finished_at
    """), engine, params={"test_end": TEST_END})
    base = base[~base.ev_id.isin(corrupted)].copy()
    base = base.dropna(subset=["odds_home", "odds_draw", "odds_away"])
    base["journee"] = pd.to_numeric(base.round_info, errors="coerce")
    base["segment"] = base.journee.apply(lambda j: label_segment(int(j)) if pd.notna(j) else None)
    base["actual"] = base.apply(lambda r: f"{int(r.score_a)}-{int(r.score_b)}", axis=1)

    train = base[base.finished_at < TRAIN_END].reset_index(drop=True)
    test = base[(base.finished_at >= TRAIN_END) & base.segment.notna()].reset_index(drop=True)
    print(f"TRAIN {len(train)} | TEST {len(test)}")

    train_ht = train[train.ht_score_a.notna()].reset_index(drop=True)
    model_v5 = fit_model_v5(train, ht_history=train_ht, engine=None, form_alpha=0.0)

    # ⚠️ PAIR CACHE LEAKY : _load() par defaut = TOUTE la BDD (test inclus)
    v2 = ScorePredictorV2(engine)
    v2.pair_cache._load()
    print(f"Pair cache FULL-DB (leaky) : {len(v2.pair_cache._cache)} entrees")

    snap_ids = test.snap_id.astype(int).tolist()
    em_by_snap: dict[int, dict] = {}
    with engine.connect() as conn:
        for i in range(0, len(snap_ids), 400):
            chunk = snap_ids[i:i + 400]
            ph = ",".join(str(x) for x in chunk)
            rows = conn.execute(text(
                f"SELECT id, extra_markets FROM odds_snapshots WHERE id IN ({ph})"
            )).fetchall()
            for sid, em in rows:
                if not em:
                    continue
                try:
                    dd = json.loads(em) if isinstance(em, str) else em
                except Exception:
                    continue
                if isinstance(dd, dict):
                    em_by_snap[int(sid)] = {k: dd[k] for k in NEEDED_MARKETS if k in dd}

    n = 0
    hits = {("v5", 1): 0, ("v5", 3): 0, ("v2", 1): 0, ("v2", 3): 0}
    hi_ann = []  # (annonce, hit) pour annonces >= 0.30
    for i, m in enumerate(test.itertuples()):
        if i % 500 == 0:
            print(f"  ... {i}/{len(test)}")
        em = em_by_snap.get(int(m.snap_id), {})
        pred5 = predict_match_v5(model_v5, m.team_a, m.team_b,
                                 float(m.odds_home), float(m.odds_draw), float(m.odds_away),
                                 extra_markets=em or None)
        v5_top5 = pred5.get("top5_scores_enriched") or []
        if not v5_top5:
            continue
        v5_grid = {s: p for s, p in v5_top5}
        for s in COMMON_SCORES:
            if s not in v5_grid:
                v5_grid[s] = 0.01
        tot = sum(v5_grid.values())
        v5_grid = {s: p / tot for s, p in v5_grid.items()}
        v2_top5 = v2.predict(m.team_a, m.team_b, int(m.journee),
                             v5_score_grid=v5_grid, extra_markets=em or None,
                             odds_h=float(m.odds_home), odds_a=float(m.odds_away),
                             top_n=5)
        n += 1
        v5s = [s for s, _ in v5_top5]
        v2s = [s for s, _, _ in v2_top5]
        if m.actual in v5s[:1]: hits[("v5", 1)] += 1
        if m.actual in v5s[:3]: hits[("v5", 3)] += 1
        if m.actual in v2s[:1]: hits[("v2", 1)] += 1
        if m.actual in v2s[:3]: hits[("v2", 3)] += 1
        p1 = float(v2_top5[0][1])
        if p1 >= 0.30:
            hi_ann.append((p1, 1 if m.actual == v2s[0] else 0))

    out = {"n": n,
           "v5_top1": hits[("v5", 1)] / n, "v2_top1_leaky": hits[("v2", 1)] / n,
           "v5_top3": hits[("v5", 3)] / n, "v2_top3_leaky": hits[("v2", 3)] / n,
           "delta_top1_pp": (hits[("v2", 1)] - hits[("v5", 1)]) / n * 100,
           "delta_top3_pp": (hits[("v2", 3)] - hits[("v5", 3)]) / n * 100,
           "n_annonce_ge30": len(hi_ann),
           "hit_annonce_ge30": (sum(h for _, h in hi_ann) / len(hi_ann)) if hi_ann else None,
           "annonce_moy_ge30": (sum(p for p, _ in hi_ann) / len(hi_ann)) if hi_ann else None}
    print(json.dumps(out, indent=2))
    dest = ROOT / "exports" / "wf5_score_v2_leak_check.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"-> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
