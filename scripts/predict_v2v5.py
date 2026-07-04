"""Prédicteur CROISEMENT V2 × V5 — importable (fit + predict_round) + CLI.

V5 (Poisson FT + HT/FT + marché) et V2 (grille blendée, consomme la grille V5)
croisés en un CONSENSUS (moyenne 50/50 des distributions de score).

CLI : ./.venv/Scripts/python.exe scripts/predict_v2v5.py [HH:MM]   (heure Mada)
Import : from predict_v2v5 import fit, predict_round
"""
from __future__ import annotations
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sqlalchemy import create_engine
from scraper.config import load_settings
from scraper.predictor_v5 import fit_model_v5, predict_match_v5
from scraper.score_predictor_v2 import ScorePredictorV2
from scraper.journee_inference import infer_current_journee
from scraper.market_inversion import exact_invert_1x2, apply_sim_deviations

MADA = timezone(timedelta(hours=3))
LG = "InstantLeague-8035"
FILL = ["0-0", "1-0", "0-1", "1-1", "2-0", "0-2", "2-1", "1-2", "2-2",
        "3-0", "0-3", "3-1", "1-3", "2-3", "3-2", "4-0", "0-4"]

# Calibration des scores (corrige la sous-prédiction de l'Over ~3pp : booste 3-0/3-1/4-0)
_CALIB = None
try:
    import json as _json
    _cp = Path(__file__).resolve().parents[1] / "data" / "vfoot_ml" / "score_calibration.json"
    if _cp.exists():
        _CALIB = _json.loads(_cp.read_text(encoding="utf-8"))["correction"]
except Exception:
    _CALIB = None


def _apply_calib(grid: dict) -> dict:
    """Multiplie chaque score par son facteur de calibration (réel/modèle), renormalise."""
    if not _CALIB:
        return grid
    adj = {}
    for sc, p in grid.items():
        try:
            h, a = map(int, sc.split("-"))
            f = _CALIB[h][a] if (0 <= h < 7 and 0 <= a < 7) else 1.0
        except Exception:
            f = 1.0
        adj[sc] = p * f
    tt = sum(adj.values()) or 1.0
    return {k: v/tt for k, v in adj.items()}


def load_hist(engine):
    return pd.read_sql(f"""SELECT e.team_a,e.team_b,o.odds_home,o.odds_draw,o.odds_away,
        r.score_a,r.score_b,r.ht_score_a,r.ht_score_b FROM events e
        JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
        JOIN results r ON r.event_id=e.id
        WHERE r.ht_score_a IS NOT NULL AND e.competition='{LG}'""", engine)


def fit(engine):
    """Fit V5 + V2. Retourne (m5, v2, n_hist)."""
    hist = load_hist(engine)
    m5 = fit_model_v5(hist, ht_history=hist.copy(), engine=engine, form_alpha=0.0)
    v2 = ScorePredictorV2(engine)
    return m5, v2, len(hist)


def _x12(grid: dict):
    ph = pd_ = pa = 0.0
    for sc, p in grid.items():
        h, a = map(int, sc.split("-"))
        if h > a: ph += p
        elif h == a: pd_ += p
        else: pa += p
    t = ph + pd_ + pa
    return (ph/t, pd_/t, pa/t) if t else (0.0, 0.0, 0.0)


def predict_one(engine, m5, v2, team_a, team_b, oh, od, oa, extra_markets, round_info, es) -> dict:
    """Croisement V2×V5 pour UN match -> dict complet."""
    oh, od, oa = float(oh), float(od), float(oa)
    p5 = {}
    try:
        p5 = predict_match_v5(m5, team_a, team_b, oh, od, oa, extra_markets=extra_markets)
        top5 = p5.get("top5_scores_enriched") or []
    except Exception:
        top5 = []
    v5g = {sc: p for sc, p in top5}
    v2list = []
    try:
        grid = dict(v5g)
        for sc in FILL:
            grid.setdefault(sc, 0.01)
        tt = sum(grid.values()); grid = {k: v/tt for k, v in grid.items()}
        jrn = int(round_info) if str(round_info).isdigit() and str(round_info) != "0" \
            else (infer_current_journee(engine, es) or 8)
        v2t = v2.predict(team_a, team_b, jrn, v5_score_grid=grid,
                         extra_markets=extra_markets, odds_h=oh, odds_a=oa, top_n=5)
        v2list = [(sc, p) for sc, p, _ in v2t]
    except Exception:
        v2list = []
    v2g = {sc: p for sc, p in v2list}
    # consensus = moyenne 50/50 (union), renormalisé, PUIS calibré (corrige l'Over)
    cons = {}
    for sc, p in v5g.items(): cons[sc] = cons.get(sc, 0) + 0.5 * p
    for sc, p in v2g.items(): cons[sc] = cons.get(sc, 0) + 0.5 * p
    tt = sum(cons.values()) or 1
    cons = {k: v/tt for k, v in cons.items()}
    cons = _apply_calib(cons)
    ctop = sorted(cons.items(), key=lambda kv: -kv[1])[:3]
    # Over 2.5 CALIBRÉ depuis la grille complète (odds -> sim -> calib) = totaux fiables
    try:
        lh, la = exact_invert_1x2(oh, od, oa)
        fg = np.asarray(apply_sim_deviations(lh, la, "cells"), float)[:7, :7]; fg /= fg.sum()
        if _CALIB:
            fg = fg * np.asarray(_CALIB); fg /= fg.sum()
        p_over25 = float(sum(fg[h, a] for h in range(7) for a in range(7) if h + a > 2.5))
    except Exception:
        p_over25 = sum(p for sc, p in cons.items() if sum(map(int, sc.split("-"))) > 2.5)
    ph = p5.get("p_h_blend") or _x12(cons)[0]
    pd_ = p5.get("p_d_blend") or _x12(cons)[1]
    pa = p5.get("p_a_blend") or _x12(cons)[2]
    accord = bool(top5 and v2g and top5[0][0] == max(v2g, key=v2g.get))
    return {"match": f"{team_a} v {team_b}", "team_a": team_a, "team_b": team_b,
            "cotes": [oh, od, oa], "x12": [round(ph, 3), round(pd_, 3), round(pa, 3)],
            "over25_pct": round(100 * p_over25, 1),
            "v5_top3": [(s, round(p, 3)) for s, p in top5[:3]],
            "v2_top3": [(s, round(p, 3)) for s, p in v2list[:3]],
            "consensus_top3": [(s, round(p, 3)) for s, p in ctop],
            "accord": accord}


def predict_round(engine, m5, v2, target_local=None) -> dict:
    """Prédit un round (heure Mada HH:MM) ou le prochain. Retourne {target,rounds,matches}."""
    now = datetime.now(timezone.utc)
    up = pd.read_sql(f"""SELECT e.team_a,e.team_b,e.expected_start,e.round_info,o.odds_home oh,
        o.odds_draw od,o.odds_away oa,o.extra_markets,e.id ev FROM events e
        JOIN odds_snapshots o ON o.id=(SELECT MAX(id) FROM odds_snapshots WHERE event_id=e.id)
        LEFT JOIN results r ON r.event_id=e.id
        WHERE r.id IS NULL AND e.expected_start IS NOT NULL AND e.competition='{LG}'""", engine)
    if not len(up):
        return {"target": None, "rounds": [], "matches": []}
    up["es"] = pd.to_datetime(up.expected_start, utc=True)
    up = up[up.es > now - pd.Timedelta(minutes=3)]
    up["local"] = up.es.dt.tz_convert(MADA).dt.strftime("%H:%M")
    up = up.sort_values(["es", "ev"]).drop_duplicates(["team_a", "team_b", "local"])
    rounds = sorted(up.local.unique())
    if not len(rounds):
        return {"target": None, "rounds": [], "matches": []}
    target = target_local if (target_local and target_local in rounds) else rounds[0]
    ms = up[up.local == target]
    matches = []
    for r in ms.itertuples():
        if float(r.oh) <= 1 or float(r.oa) <= 1:
            continue
        matches.append(predict_one(engine, m5, v2, r.team_a, r.team_b, r.oh, r.od, r.oa,
                                    r.extra_markets, r.round_info, r.es))
    return {"target": target, "rounds": rounds, "matches": matches}


def main():
    e = create_engine(load_settings().db_url)
    print("fit V5 + V2…")
    m5, v2, n = fit(e)
    tgt = sys.argv[1] if len(sys.argv) > 1 else None
    res = predict_round(e, m5, v2, tgt)
    if not res["matches"]:
        print(f"Aucun match. Rounds dispo : {res['rounds'][:8]}"); return
    print(f"\nROUND {res['target']} Mada — CROISEMENT V2×V5 (fit {n} matchs)\n")
    print(f"  {'match':<28}{'1X2 cons.':<16}{'V5 Top-3':<20}{'V2 Top-3':<20}{'CONSENSUS':<20}acc")
    print("  " + "-" * 108)
    for m in res["matches"]:
        ph, pd_, pa = m["x12"]
        x = f"1:{ph*100:.0f} X:{pd_*100:.0f} 2:{pa*100:.0f}"
        f = lambda lst: " ".join(f"{s}({p*100:.0f})" for s, p in lst)
        print(f"  {m['match'][:27]:<28}{x:<16}{f(m['v5_top3']):<20}{f(m['v2_top3']):<20}"
              f"{f(m['consensus_top3']):<20}{'🤝' if m['accord'] else '🔀'}")


if __name__ == "__main__":
    main()
