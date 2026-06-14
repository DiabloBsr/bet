"""WF5 — AUDIT Score Predictor V2 vs V5 brut sur les 3 derniers jours (8035).

Methodologie (leak-free, fidele au pipeline live _predict_one_round.py) :
- TRAIN  : tous les matchs 8035 finis AVANT 2026-06-10 (cotes ouverture MIN(id)).
- TEST   : matchs 8035 finis entre 2026-06-10 et 2026-06-12 inclus, avec cotes ouverture.
- Exclusion des ids de exports/corrupted_events.json.
- fit_model_v5(train, ht_history=train, form_alpha=0.0)  [comme en prod]
- PairScoreCache injecte TRAIN-only (anti-leakage, comme _backtest_score_v2.py).
- V2 invoque EXACTEMENT comme en live : v5_grid = top5 V5 + 17 scores communs a 0.01.

Mesures :
1. Top1/Top3/Top5 accuracy V5 vs V2 (+ McNemar paired test) vs claim +9pp/+15pp.
2. Calibration de la proba annoncee du Top1 V2 (et V5) par tranche.
3. Surconcentration : hit rate des Top1 annonces >=30% / >=35% ; cas '2-1'.
4. Calibration des picks 1X2 V5 (primary_p, p_cote, gate=max) par tranche annoncee.

Sortie : exports/wf5_score_v2_audit.json + rapport stdout. LECTURE SEULE sur la BDD.
"""
from __future__ import annotations
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from sqlalchemy import create_engine, text

from scraper.config import load_settings
from scraper.predictor_v5 import fit_model_v5, predict_match_v5
from scraper.score_predictor_v2 import ScorePredictorV2
from scraper.strategy_engine import label_segment

TRAIN_END = "2026-06-10"          # test = finished_at >= TRAIN_END
TEST_END = "2026-06-13"           # exclusif
NEEDED_MARKETS = ("Score exact", "Total de buts", "G/NG")
COMMON_SCORES = ["0-0", "1-0", "0-1", "1-1", "2-0", "0-2", "2-1", "1-2", "2-2",
                 "3-0", "0-3", "3-1", "1-3", "3-2", "2-3", "4-0", "0-4"]  # = live


def wilson_se(p: float, n: int) -> float:
    if n <= 0:
        return 0.0
    return math.sqrt(p * (1 - p) / n)


def main() -> int:
    settings = load_settings()
    engine = create_engine(settings.db_url)

    corrupted = set()
    cj = ROOT / "exports" / "corrupted_events.json"
    if cj.exists():
        d = json.loads(cj.read_text(encoding="utf-8"))
        corrupted = {int(k) for k in d.get("events", {})}
    print(f"Corrupted ids exclus : {len(corrupted)}")

    # ── 1. Base data (SANS extra_markets — RAM) ──────────────────────────────
    base = pd.read_sql(text("""
        SELECT e.id AS ev_id, e.round_info, e.team_a, e.team_b, e.expected_start,
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
    n0 = len(base)
    base = base[~base.ev_id.isin(corrupted)].copy()
    base = base.dropna(subset=["odds_home", "odds_draw", "odds_away"])
    base["journee"] = pd.to_numeric(base.round_info, errors="coerce")
    base["segment"] = base.journee.apply(lambda j: label_segment(int(j)) if pd.notna(j) else None)
    base["actual"] = base.apply(lambda r: f"{int(r.score_a)}-{int(r.score_b)}", axis=1)
    print(f"Matchs 8035 finis + cotes ouverture : {n0} -> {len(base)} apres exclusions")

    train = base[base.finished_at < TRAIN_END].reset_index(drop=True)
    test = base[(base.finished_at >= TRAIN_END) & base.segment.notna()].reset_index(drop=True)
    print(f"TRAIN (< {TRAIN_END}) : {len(train)}  |  TEST (3 derniers jours) : {len(test)}")
    if len(test) < 100:
        print("Test set trop petit — abandon.")
        return 1

    # ── 2. Fit V5 sur TRAIN uniquement ───────────────────────────────────────
    train_ht = train[train.ht_score_a.notna()].reset_index(drop=True)
    print(f"Fit V5 (n={len(train)}, ht={len(train_ht)}, form_alpha=0.0 comme prod)...")
    model_v5 = fit_model_v5(train, ht_history=train_ht, engine=None, form_alpha=0.0)
    print(f"  rho={model_v5.rho:.4f}  ht_ratio={model_v5.ht_lambda_ratio:.3f}  "
          f"emp_buckets={len(model_v5.empirical_score_dist or {})}")

    # ── 3. Pair cache TRAIN-only (anti-leakage) ──────────────────────────────
    v2 = ScorePredictorV2(engine)
    cache: dict = {}
    tr = train[train.segment.notna()].copy()
    min_n = v2.pair_cache.min_n
    for (ta, tb, seg), grp in tr.groupby(["team_a", "team_b", "segment"]):
        if len(grp) < min_n:
            continue
        counts = grp.actual.value_counts().to_dict()
        total = sum(counts.values())
        cache[(ta, tb, seg)] = {s: c / total for s, c in counts.items()}
    for (ta, tb), grp in tr.groupby(["team_a", "team_b"]):
        if len(grp) < min_n:
            continue
        counts = grp.actual.value_counts().to_dict()
        total = sum(counts.values())
        cache[(ta, tb, "ALL")] = {s: c / total for s, c in counts.items()}
    v2.pair_cache._cache = cache
    v2.pair_cache._loaded = True
    print(f"Pair cache train-only : {len(cache)} entrees")

    # ── 4. extra_markets du snapshot d'ouverture, par chunks (RAM) ──────────
    snap_ids = test.snap_id.astype(int).tolist()
    em_by_snap: dict[int, dict] = {}
    CHUNK = 400
    with engine.connect() as conn:
        for i in range(0, len(snap_ids), CHUNK):
            chunk = snap_ids[i:i + CHUNK]
            ph = ",".join(str(x) for x in chunk)
            rows = conn.execute(text(
                f"SELECT id, extra_markets FROM odds_snapshots WHERE id IN ({ph})"
            )).fetchall()
            for sid, em in rows:
                if not em:
                    continue
                try:
                    d = json.loads(em) if isinstance(em, str) else em
                except Exception:
                    continue
                if isinstance(d, dict):
                    em_by_snap[int(sid)] = {k: d[k] for k in NEEDED_MARKETS if k in d}
    print(f"extra_markets ouverture charges : {len(em_by_snap)}/{len(test)}")

    # ── 5. Replay des predictions ────────────────────────────────────────────
    recs = []
    skipped_no_model = 0
    for i, m in enumerate(test.itertuples()):
        if i % 500 == 0:
            print(f"  ... {i}/{len(test)}")
        em = em_by_snap.get(int(m.snap_id), {})
        pred5 = predict_match_v5(model_v5, m.team_a, m.team_b,
                                 float(m.odds_home), float(m.odds_draw), float(m.odds_away),
                                 extra_markets=em or None)
        v5_top5 = pred5.get("top5_scores_enriched") or []
        if not v5_top5:
            skipped_no_model += 1
            continue

        # V2 — construction de grille IDENTIQUE au live (_predict_one_round.py)
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
        pair_dist, _ = v2.pair_cache.get(m.team_a, m.team_b, m.segment)

        outcome = "1" if m.score_a > m.score_b else ("X" if m.score_a == m.score_b else "2")
        pick = pred5.get("primary_pick")
        p_cote_pick = {"1": pred5.get("p_h_cote"), "X": pred5.get("p_d_cote"),
                       "2": pred5.get("p_a_cote")}.get(pick) or 0.0
        recs.append({
            "actual": m.actual,
            "v5_scores": [s for s, _ in v5_top5],
            "v5_p1": float(v5_top5[0][1]),
            "v2_scores": [s for s, _, _ in v2_top5],
            "v2_probs": [float(p) for _, p, _ in v2_top5],
            "pair": bool(pair_dist),
            "pick": pick,
            "pick_p": float(pred5.get("primary_p") or 0.0),
            "pick_p_cote": float(p_cote_pick),
            "outcome": outcome,
            "odds_home": float(m.odds_home),
        })
    n = len(recs)
    print(f"\nPredictions rejouees : {n}  (skip sans modele : {skipped_no_model})")

    out: dict = {"train_n": len(train), "test_n": n, "train_end": TRAIN_END,
                 "skipped_no_model": skipped_no_model,
                 "pair_coverage": sum(r["pair"] for r in recs) / n}

    # ── 6. Top-k accuracy V5 vs V2 ───────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"  A. SCORE EXACT — V5 vs V2 (n={n}, pair dispo {out['pair_coverage']*100:.1f}%)")
    print("=" * 78)
    acc = {}
    for k in (1, 3, 5):
        h5 = sum(r["actual"] in r["v5_scores"][:k] for r in recs)
        h2 = sum(r["actual"] in r["v2_scores"][:k] for r in recs)
        a5, a2 = h5 / n, h2 / n
        se = math.sqrt(wilson_se(a5, n) ** 2 + wilson_se(a2, n) ** 2)
        acc[f"top{k}"] = {"v5": a5, "v2": a2, "delta_pp": (a2 - a5) * 100,
                          "v5_hits": h5, "v2_hits": h2}
        print(f"  Top{k}: V5 {a5*100:5.2f}% ({h5})  |  V2 {a2*100:5.2f}% ({h2})  "
              f"|  delta {(a2-a5)*100:+.2f}pp (+/-{se*100*1.96:.2f}pp 95%)")
    # McNemar top1 et top3
    for k in (1, 3):
        b = sum((r["actual"] in r["v5_scores"][:k]) and (r["actual"] not in r["v2_scores"][:k]) for r in recs)
        c = sum((r["actual"] not in r["v5_scores"][:k]) and (r["actual"] in r["v2_scores"][:k]) for r in recs)
        z = (c - b) / math.sqrt(b + c) if (b + c) > 0 else 0.0
        acc[f"top{k}"]["mcnemar"] = {"v5_only": b, "v2_only": c, "z": z}
        print(f"  McNemar top{k}: V5-seul={b}  V2-seul={c}  z={z:+.2f}")
    out["accuracy"] = acc
    # split pair / no pair
    for label, sel in (("avec pair", [r for r in recs if r["pair"]]),
                       ("sans pair", [r for r in recs if not r["pair"]])):
        if not sel:
            continue
        m_ = len(sel)
        t1v5 = sum(r["actual"] in r["v5_scores"][:1] for r in sel) / m_
        t1v2 = sum(r["actual"] in r["v2_scores"][:1] for r in sel) / m_
        t3v5 = sum(r["actual"] in r["v5_scores"][:3] for r in sel) / m_
        t3v2 = sum(r["actual"] in r["v2_scores"][:3] for r in sel) / m_
        print(f"  [{label} n={m_}] top1 V5 {t1v5*100:.2f}% / V2 {t1v2*100:.2f}%  "
              f"|  top3 V5 {t3v5*100:.2f}% / V2 {t3v2*100:.2f}%")
        out[f"acc_{label.replace(' ', '_')}"] = {"n": m_, "top1_v5": t1v5, "top1_v2": t1v2,
                                                 "top3_v5": t3v5, "top3_v2": t3v2}

    # ── 7. Calibration proba annoncee Top1 V2 ────────────────────────────────
    print("\n" + "=" * 78)
    print("  B. CALIBRATION DU TOP1 V2 (proba annoncee vs reel)")
    print("=" * 78)
    buckets = [(0, .10), (.10, .15), (.15, .20), (.20, .25), (.25, .30), (.30, .35), (.35, 1.01)]
    cal_v2 = []
    for lo, hi in buckets:
        sel = [r for r in recs if lo <= r["v2_probs"][0] < hi]
        if not sel:
            continue
        hr = sum(r["actual"] == r["v2_scores"][0] for r in sel) / len(sel)
        ann = sum(r["v2_probs"][0] for r in sel) / len(sel)
        cal_v2.append({"bucket": f"{lo*100:.0f}-{hi*100:.0f}%", "n": len(sel),
                       "annonce": ann, "reel": hr, "gap_pp": (hr - ann) * 100})
        print(f"  annonce [{lo*100:3.0f}-{hi*100:3.0f}%) n={len(sel):5d}  "
              f"annonce moy {ann*100:5.1f}%  reel {hr*100:5.1f}%  gap {(hr-ann)*100:+.1f}pp")
    out["calib_v2_top1"] = cal_v2
    hi_sel = [r for r in recs if r["v2_probs"][0] >= 0.35]
    if hi_sel:
        hr = sum(r["actual"] == r["v2_scores"][0] for r in hi_sel) / len(hi_sel)
        out["v2_top1_ge35"] = {"n": len(hi_sel), "reel": hr,
                               "annonce_moy": sum(r["v2_probs"][0] for r in hi_sel) / len(hi_sel)}
    # cas '2-1' annonce en Top1
    sel21 = [r for r in recs if r["v2_scores"][0] == "2-1"]
    if sel21:
        hr21 = sum(r["actual"] == "2-1" for r in sel21) / len(sel21)
        ann21 = sum(r["v2_probs"][0] for r in sel21) / len(sel21)
        freq21 = sum(r["actual"] == "2-1" for r in recs) / n
        out["v2_pick_21"] = {"n": len(sel21), "annonce_moy": ann21, "reel": hr21,
                             "freq_globale_21": freq21}
        print(f"\n  Top1='2-1' : n={len(sel21)}  annonce moy {ann21*100:.1f}%  "
              f"reel {hr21*100:.1f}%  (freq globale 2-1 dans test : {freq21*100:.1f}%)")
    # distribution des top1 V2 vs frequences reelles
    top1_dist = Counter(r["v2_scores"][0] for r in recs)
    real_dist = Counter(r["actual"] for r in recs)
    print("\n  Top1 V2 les plus annonces vs frequence reelle :")
    rows_dist = []
    for s, c in top1_dist.most_common(8):
        hit = sum(r["actual"] == s for r in recs if r["v2_scores"][0] == s) / c
        ann = sum(r["v2_probs"][0] for r in recs if r["v2_scores"][0] == s) / c
        rows_dist.append({"score": s, "n_pick": c, "annonce_moy": ann,
                          "hit_quand_pick": hit, "freq_reelle": real_dist[s] / n})
        print(f"    {s:>4} : picke {c:5d}x ({c/n*100:4.1f}%)  annonce {ann*100:5.1f}%  "
              f"hit {hit*100:5.1f}%  freq reelle {real_dist[s]/n*100:5.1f}%")
    out["v2_top1_distribution"] = rows_dist
    out["real_score_top10"] = [{"score": s, "freq": c / n} for s, c in real_dist.most_common(10)]

    # calibration V5 top1 annonce pour reference
    cal_v5 = []
    for lo, hi in buckets:
        sel = [r for r in recs if lo <= r["v5_p1"] < hi]
        if not sel:
            continue
        hr = sum(r["actual"] == r["v5_scores"][0] for r in sel) / len(sel)
        ann = sum(r["v5_p1"] for r in sel) / len(sel)
        cal_v5.append({"bucket": f"{lo*100:.0f}-{hi*100:.0f}%", "n": len(sel),
                       "annonce": ann, "reel": hr})
    out["calib_v5_top1"] = cal_v5

    # ── 8. Calibration 1X2 V5 ────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  C. CALIBRATION PICKS 1X2 V5 (primary_p annoncee vs WR reel)")
    print("=" * 78)
    b1x2 = [(0, .50), (.50, .55), (.55, .60), (.60, .65), (.65, .70),
            (.70, .75), (.75, .80), (.80, .85), (.85, 1.01)]
    for name, key in (("primary_p (annonce V5)", "pick_p"), ("p_cote calibree", "pick_p_cote")):
        print(f"\n  -- {name} --")
        cal = []
        brier = sum((r[key] - (1.0 if r["pick"] == r["outcome"] else 0.0)) ** 2 for r in recs) / n
        for lo, hi in b1x2:
            sel = [r for r in recs if lo <= r[key] < hi]
            if len(sel) < 10:
                continue
            wr = sum(r["pick"] == r["outcome"] for r in sel) / len(sel)
            ann = sum(r[key] for r in sel) / len(sel)
            cal.append({"bucket": f"{lo*100:.0f}-{hi*100:.0f}%", "n": len(sel),
                        "annonce": ann, "wr_reel": wr, "gap_pp": (wr - ann) * 100})
            print(f"  [{lo*100:3.0f}-{hi*100:3.0f}%) n={len(sel):5d}  annonce {ann*100:5.1f}%  "
                  f"WR reel {wr*100:5.1f}%  gap {(wr-ann)*100:+5.1f}pp  "
                  f"+/-{wilson_se(wr, len(sel))*196:.1f}pp")
        out[f"calib_1x2_{key}"] = {"buckets": cal, "brier": brier}
        print(f"  Brier : {brier:.4f}")
    # gate (max des deux) — ce que TIER1 utilise
    print("\n  -- gate = max(primary_p, p_cote) [utilise par tier1_picker] --")
    cal = []
    for lo, hi in b1x2:
        sel = [r for r in recs if lo <= max(r["pick_p"], r["pick_p_cote"]) < hi]
        if len(sel) < 10:
            continue
        wr = sum(r["pick"] == r["outcome"] for r in sel) / len(sel)
        ann = sum(max(r["pick_p"], r["pick_p_cote"]) for r in sel) / len(sel)
        cal.append({"bucket": f"{lo*100:.0f}-{hi*100:.0f}%", "n": len(sel),
                    "annonce": ann, "wr_reel": wr, "gap_pp": (wr - ann) * 100})
        print(f"  [{lo*100:3.0f}-{hi*100:3.0f}%) n={len(sel):5d}  annonce {ann*100:5.1f}%  "
              f"WR reel {wr*100:5.1f}%  gap {(wr-ann)*100:+5.1f}pp")
    out["calib_1x2_gate"] = cal
    wr_all = sum(r["pick"] == r["outcome"] for r in recs) / n
    out["pick_1x2_wr_global"] = wr_all
    print(f"\n  WR global picks V5 : {wr_all*100:.2f}% (n={n})")

    dest = ROOT / "exports" / "wf5_score_v2_audit.json"
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResultats -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
