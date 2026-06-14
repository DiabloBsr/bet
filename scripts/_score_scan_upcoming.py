"""Scan TOUS les matchs à venir (8035) et classe par fiabilité de SCORE EXACT.
Métrique = consensus entre 4 modèles (V5, V2 ensemble, inversion-sim, chaînage)
+ concentration (prob Top1 V2, somme Top3). Le plafond empirique reste
Top1 ~11.6% / Top3 ~30% — on remonte les matchs où les modèles s'accordent."""
from __future__ import annotations
import sys, json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
from sqlalchemy import create_engine
from scraper.config import load_settings
from scraper.predictor_v5 import fit_model_v5, predict_match_v5
from scraper.score_predictor_v2 import ScorePredictorV2
from scraper.journee_inference import infer_current_journee
from scraper.market_inversion import (
    invert_markets, apply_sim_deviations, grid_predictions,
)

MG = timezone(timedelta(hours=3))

def _load_json(name):
    try:
        return json.load(open(Path(__file__).resolve().parents[1] / "exports" / name, encoding="utf-8"))
    except Exception:
        return None
def _band(v, bands):
    for lo, hi, lbl in bands:
        if lo <= v < hi:
            return lbl
    return None
def _narrow_lookup(nt, lam_tot, lam_diff, p_btts):
    if not nt or p_btts is None: return None
    b = nt["_bands"]
    tl = _band(lam_tot, b["tot"]); dl = _band(lam_diff, b["diff"]); bl = _band(p_btts, b["btts"])
    if not (tl and dl and bl): return None
    return nt["cells"].get(f"{tl}|{dl}|{bl}")

def main():
    s = load_settings(); engine = create_engine(s.db_url)
    history = pd.read_sql("""
        SELECT e.team_a, e.team_b, o.odds_home, o.odds_draw, o.odds_away,
               r.score_a, r.score_b, r.ht_score_a, r.ht_score_b
        FROM events e JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
        JOIN results r ON r.event_id=e.id
        WHERE r.ht_score_a IS NOT NULL AND e.competition='InstantLeague-8035'
    """, engine)
    model_v5 = fit_model_v5(history, ht_history=history.copy(), engine=engine, form_alpha=0.0)
    score_v2 = ScorePredictorV2(engine)
    narrow_tab = _load_json("narrow_table.json")

    now = datetime.now(timezone.utc)
    up = pd.read_sql("""
        SELECT e.team_a, e.team_b, e.expected_start, e.round_info,
               o.odds_home, o.odds_draw, o.odds_away, o.extra_markets, e.id ev_id
        FROM events e JOIN odds_snapshots o ON o.id=(SELECT MAX(id) FROM odds_snapshots WHERE event_id=e.id)
        LEFT JOIN results r ON r.event_id=e.id
        WHERE r.id IS NULL AND e.expected_start IS NOT NULL AND e.competition='InstantLeague-8035'
    """, engine)
    up["expected_start"] = pd.to_datetime(up.expected_start, utc=True)
    up = up[up.expected_start > now - pd.Timedelta(minutes=3)].copy()
    up["local"] = up.expected_start.dt.tz_convert(MG).dt.strftime("%H:%M")
    up["has_round"] = up.round_info.fillna("0").astype(str).ne("0")
    up = up.sort_values(["has_round", "ev_id"], ascending=False).drop_duplicates(["team_a", "team_b", "local"])
    up = up.sort_values("expected_start")
    print(f"now Mada {now.astimezone(MG):%H:%M:%S} — {len(up)} matchs à venir, {up.local.nunique()} rounds\n")

    rows = []
    for _, m in up.iterrows():
        try:
            p5 = predict_match_v5(model_v5, m.team_a, m.team_b, m.odds_home, m.odds_draw, m.odds_away,
                                  extra_markets=m.extra_markets)
            top5 = p5.get("top5_scores_enriched") or []
            if not top5: continue
            v5_grid = {sc: p for sc, p in top5}
            for sc in ["0-0","1-0","0-1","1-1","2-0","0-2","2-1","1-2","2-2","3-0","0-3","3-1","1-3"]:
                v5_grid.setdefault(sc, 0.01)
            tt = sum(v5_grid.values()); v5_grid = {k: v/tt for k, v in v5_grid.items()}
            try:
                jrn = int(m.round_info) if str(m.round_info).isdigit() and m.round_info != "0" else (infer_current_journee(engine, m.expected_start) or 8)
            except Exception:
                jrn = 8
            v2 = score_v2.predict(m.team_a, m.team_b, jrn, v5_score_grid=v5_grid,
                                  extra_markets=m.extra_markets, odds_h=m.odds_home, odds_a=m.odds_away, top_n=5)
            inv = invert_markets(float(m.odds_home), float(m.odds_draw), float(m.odds_away), m.extra_markets)
            gp = grid_predictions(apply_sim_deviations(inv.lam_h, inv.lam_a, "cells"), top_k=3)
            sim_top = gp["top_scores"][0][0]
            nb = _narrow_lookup(narrow_tab, inv.lam_h + inv.lam_a, inv.lam_h - inv.lam_a, gp["btts_oui"])
            chain = nb["score"] if nb else None

            v5_top1 = top5[0][0]
            v2_top1, v2_top1_p = v2[0][0], v2[0][1]
            v2_top3 = v2[:3]
            v2_top3_sum = sum(p for _, p, _ in v2_top3)
            # consensus : combien des 4 modèles pointent le score V2-top1
            votes = [v5_top1, v2_top1, sim_top, chain]
            consensus = sum(1 for x in votes if x == v2_top1)
            rows.append({
                "local": m.local, "match": f"{m.team_a} vs {m.team_b}",
                "cotes": f"{m.odds_home:.2f}/{m.odds_draw:.2f}/{m.odds_away:.2f}",
                "best": v2_top1, "p1": v2_top1_p,
                "top3": " · ".join(f"{sc}({p*100:.0f}%)" for sc, p, _ in v2_top3),
                "p3": v2_top3_sum, "consensus": consensus,
                "v5": v5_top1, "sim": sim_top, "chain": chain or "—",
            })
        except Exception as ex:
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        print("aucun match exploitable"); return 0
    # score de fiabilité = consensus (poids fort) + prob Top1 V2
    df["rank"] = df.consensus * 1.0 + df.p1
    df = df.sort_values(["rank", "p3"], ascending=False)

    print("="*118)
    print(f"{'#':>2} {'heure':>5} {'match':<30} {'SCORE le+prob':>13} {'Top1':>5} {'cons':>5}  {'TOP-3 (couverture)':<34}")
    print("="*118)
    for i, r in enumerate(df.itertuples(), 1):
        flag = "🟢🟢" if r.consensus >= 3 else ("🟢" if r.consensus == 2 else "  ")
        print(f"{i:>2} {r.local:>5} {r.match:<30} {r.best:>9} {flag} {r.p1*100:>4.0f}% {r.consensus}/4  {r.top3} = {r.p3*100:.0f}%")
    print("="*118)
    print("\nLecture : 'cons' = nb de modèles (V5/V2/inversion-sim/chaînage) d'accord sur le score.")
    print("3-4/4 = consensus fort → score le plus fiable. Top1 ≈ accuracy attendue d'1 score ;")
    print("la colonne TOP-3 = jouer les 3 scores (couverture cumulée, plafond ~30-36% sur matchs équilibrés).")
    # focus : les meilleurs consensus
    best = df[df.consensus >= 3]
    print(f"\n🎯 CONSENSUS FORT (≥3/4 modèles d'accord) : {len(best)} matchs")
    for r in best.itertuples():
        print(f"   • {r.local} {r.match:<28} → {r.best}  (Top1 {r.p1*100:.0f}%, {r.consensus}/4 ; V5={r.v5} sim={r.sim} chaîne={r.chain})")
    return 0

if __name__ == "__main__":
    sys.exit(main())
