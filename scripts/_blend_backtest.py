"""Backtest du MÉLANGE : quels poids V2/V5/MARCHÉ maximisent l'accuracy OOS ?

Split chrono 70/30. Fit V2 + V5 sur TRAIN. Sur TEST, pour chaque match on
construit les 3 distributions (V2 blendée, V5 enrichie, marché devigé) puis on
évalue TOUTES les combinaisons de poids × calibration on/off :
  Top-1 / Top-3 score exact + proba moyenne du score réalisé (qualité).

But : remplacer les poids conventionnels (50/50, 1/3) par des poids MESURÉS.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sqlalchemy import create_engine
from scraper.config import load_settings
from scraper.predictor_v2 import (fit_model_v2, predict_match_v2, blended_score_grid,
                                  grid_top_k_scores, market_score_grid)
from scraper.predictor_v5 import fit_model_v5, predict_match_v5

LG = "InstantLeague-8035"
COMBOS = [  # (w_v2, w_v5, w_marché, label)
    (1.0, 0.0, 0.0, "V2 seul"),
    (0.0, 1.0, 0.0, "V5 seul"),
    (0.0, 0.0, 1.0, "MARCHÉ seul"),
    (0.5, 0.5, 0.0, "V2+V5 50/50 (dashboard)"),
    (1/3, 1/3, 1/3, "TRIO égal (app clone)"),
    (0.25, 0.25, 0.5, "marché-lourd 25/25/50"),
    (0.5, 0.0, 0.5, "V2+marché 50/50"),
    (0.0, 0.5, 0.5, "V5+marché 50/50"),
]

_CALIB = None
try:
    _cp = Path(__file__).resolve().parents[1] / "data" / "vfoot_ml" / "score_calibration.json"
    _CALIB = np.asarray(json.loads(_cp.read_text(encoding="utf-8"))["correction"], float)
except Exception:
    pass


def calib_dict(d):
    if _CALIB is None:
        return d
    out = {}
    for sc, p in d.items():
        try:
            h, a = map(int, sc.split("-"))
            f = _CALIB[h][a] if (0 <= h < 7 and 0 <= a < 7) else 1.0
        except Exception:
            f = 1.0
        out[sc] = p * f
    tt = sum(out.values()) or 1.0
    return {k: v / tt for k, v in out.items()}


def main():
    eng = create_engine(load_settings().db_url)
    df = pd.read_sql(f"""SELECT e.team_a,e.team_b,o.odds_home,o.odds_draw,o.odds_away,
        o.extra_markets,r.score_a,r.score_b,r.ht_score_a,r.ht_score_b
        FROM events e JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
        JOIN results r ON r.event_id=e.id
        WHERE r.ht_score_a IS NOT NULL AND e.competition='{LG}'
          AND o.odds_home>1 AND o.odds_draw>1 AND o.odds_away>1
        ORDER BY e.expected_start""", eng)
    cut = int(len(df) * 0.7)
    tr, te = df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)
    print(f"train={len(tr)} test={len(te)} (chrono)")

    print("fit V2 + V5 sur TRAIN…")
    v2m = fit_model_v2(tr)
    m5 = fit_model_v5(tr, ht_history=tr.copy(), engine=eng, form_alpha=0.0)

    # stats par combo x calib : [top1, top3, sum_p_actual, n]
    stats = {(i, c): [0, 0, 0.0, 0] for i in range(len(COMBOS)) for c in (False, True)}
    n_skip = 0
    for k, r in enumerate(te.itertuples()):
        try:
            xm = json.loads(r.extra_markets) if isinstance(r.extra_markets, str) else (r.extra_markets or {})
        except Exception:
            xm = {}
        sem = xm.get("Score exact") if isinstance(xm, dict) else None
        actual = f"{min(int(r.score_a),6)}-{min(int(r.score_b),6)}"
        # --- les 3 distributions ---
        g_v2 = {}
        try:
            p2 = predict_match_v2(v2m, r.team_a, r.team_b, r.odds_home, r.odds_draw, r.odds_away, sem)
            if p2.get("lam_h"):
                g = blended_score_grid(p2["lam_h"], p2["lam_a"], v2m.rho, sem, v2m.score_market_weight)
                g_v2 = dict(grid_top_k_scores(g, 20))
        except Exception:
            pass
        g_v5 = {}
        try:
            p5 = predict_match_v5(m5, r.team_a, r.team_b, r.odds_home, r.odds_draw, r.odds_away,
                                  extra_markets=r.extra_markets)
            g_v5 = dict(p5.get("top5_scores_enriched") or [])
        except Exception:
            pass
        g_mk = {}
        try:
            gm = market_score_grid(sem)
            if gm is not None:
                g_mk = dict(grid_top_k_scores(gm, 20))
        except Exception:
            pass
        if not (g_v2 or g_v5 or g_mk):
            n_skip += 1
            continue
        for i, (w2, w5, wm, _) in enumerate(COMBOS):
            src = [(w2, g_v2), (w5, g_v5), (wm, g_mk)]
            present = [(w, g) for w, g in src if g and w > 0]
            if not present:
                continue
            wt = sum(w for w, _ in present)
            blend = {}
            for w, g in present:
                for sc, p in g.items():
                    blend[sc] = blend.get(sc, 0.0) + (w / wt) * p
            tt = sum(blend.values()) or 1.0
            blend = {kk: v / tt for kk, v in blend.items()}
            for cal in (False, True):
                b = calib_dict(blend) if cal else blend
                top = sorted(b.items(), key=lambda kv: -kv[1])
                s = stats[(i, cal)]
                s[0] += (top[0][0] == actual)
                s[1] += (actual in [t[0] for t in top[:3]])
                s[2] += b.get(actual, 0.0)
                s[3] += 1
        if (k + 1) % 2000 == 0:
            print(f"  {k+1}/{len(te)}…")

    print(f"\nskippés (aucune distribution) : {n_skip}")
    print(f"\n{'combo':<28}{'calib':<7}{'Top-1':>8}{'Top-3':>8}{'p(réel)':>9}{'n':>7}")
    print("-" * 70)
    rows = []
    for i, (_, _, _, label) in enumerate(COMBOS):
        for cal in (False, True):
            t1, t3, sp, n = stats[(i, cal)]
            if n == 0:
                continue
            rows.append((label, cal, 100*t1/n, 100*t3/n, 100*sp/n, n))
    rows.sort(key=lambda x: -x[3])
    for label, cal, t1, t3, sp, n in rows:
        print(f"{label:<28}{'oui' if cal else 'non':<7}{t1:>7.2f}%{t3:>7.2f}%{sp:>8.2f}%{n:>7}")
    best = rows[0]
    print(f"\n>>> MEILLEUR (Top-3 OOS) : {best[0]} | calib={'oui' if best[1] else 'non'} "
          f"| Top-1 {best[2]:.2f}% Top-3 {best[3]:.2f}%")


if __name__ == "__main__":
    main()
