"""Prédicteur V2 SEUL — importable (fit + predict_round) + CLI.

Utilise uniquement predictor_v2 (Poisson + Dixon-Coles + blend marché Score-exact).
Sort : 1X2, score exact top-3, Over 2.5 calibré, + le pari notable à grosse cote
(l'issue à cote >= 2.6 que V2 juge la plus value).

CLI : ./.venv/Scripts/python.exe scripts/predict_v2.py [HH:MM]   (heure Mada)
"""
from __future__ import annotations
import sys, json
from datetime import datetime, timedelta, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sqlalchemy import create_engine
from scraper.config import load_settings
from scraper.predictor_v2 import fit_model_v2, predict_match_v2, poisson_score_grid

MADA = timezone(timedelta(hours=3))
LG = "InstantLeague-8035"
BIG_ODDS = 2.6          # seuil "grosse cote"
EV_FLAG = -0.06         # EV mini pour signaler une grosse cote notable

_CALIB = None
try:
    _cp = Path(__file__).resolve().parents[1] / "data" / "vfoot_ml" / "score_calibration.json"
    if _cp.exists():
        _CALIB = np.asarray(json.loads(_cp.read_text(encoding="utf-8"))["correction"], float)
except Exception:
    _CALIB = None


def load_hist(engine):
    return pd.read_sql(f"""SELECT e.team_a,e.team_b,o.odds_home,o.odds_draw,o.odds_away,r.score_a,r.score_b
        FROM events e JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
        JOIN results r ON r.event_id=e.id
        WHERE r.score_a IS NOT NULL AND e.competition='{LG}'""", engine)


def fit(engine):
    """Fit V2. Retourne (model, n_hist)."""
    hist = load_hist(engine)
    return fit_model_v2(hist), len(hist)


def _over25(model, lh, la) -> float:
    g = poisson_score_grid(lh, la, model.rho)[:7, :7]
    g = g / g.sum()
    if _CALIB is not None:
        g = g * _CALIB; g = g / g.sum()
    return float(100 * sum(g[h, a] for h in range(7) for a in range(7) if h + a > 2.5))


def _score_exact(extra_markets):
    if isinstance(extra_markets, str):
        try: extra_markets = json.loads(extra_markets)
        except Exception: return None
    if isinstance(extra_markets, dict):
        return extra_markets.get("Score exact")
    return None


def predict_one(engine, model, team_a, team_b, oh, od, oa, extra_markets=None) -> dict:
    oh, od, oa = float(oh), float(od), float(oa)
    p = predict_match_v2(model, team_a, team_b, oh, od, oa, _score_exact(extra_markets))
    ph = p.get("p_h_bl", p.get("p_h_pois", 0.0))
    pd_ = p.get("p_d_bl", p.get("p_d_pois", 0.0))
    pa = p.get("p_a_bl", p.get("p_a_pois", 0.0))
    t3 = p.get("top3_blend") or p.get("top3_pois") or []
    over = _over25(model, p.get("lam_h", 1.4), p.get("lam_a", 1.2)) if "lam_h" in p else None
    # grosse cote notable : issue à cote >= BIG_ODDS de meilleure EV
    evs = [("1", ph, oh), ("X", pd_, od), ("2", pa, oa)]
    best = max(evs, key=lambda x: x[1] * x[2] - 1)
    ev = best[1] * best[2] - 1
    notable = None
    if best[2] >= BIG_ODDS and ev > EV_FLAG:
        notable = {"pari": best[0], "cote": round(best[2], 2),
                   "proba": round(100 * best[1], 1), "ev": round(100 * ev, 1)}
    return {"match": f"{team_a} v {team_b}", "team_a": team_a, "team_b": team_b,
            "cotes": [oh, od, oa], "x12": [round(ph, 3), round(pd_, 3), round(pa, 3)],
            "over25_pct": round(over, 1) if over is not None else None,
            "score_top3": [(s, round(pp, 3)) for s, pp in t3[:3]],
            "notable": notable}


def predict_round(engine, model, target_local=None) -> dict:
    now = datetime.now(timezone.utc)
    up = pd.read_sql(f"""SELECT e.team_a,e.team_b,e.expected_start,o.odds_home oh,o.odds_draw od,
        o.odds_away oa,o.extra_markets,e.id ev FROM events e
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
    matches = [predict_one(engine, model, r.team_a, r.team_b, r.oh, r.od, r.oa, r.extra_markets)
               for r in ms.itertuples() if float(r.oh) > 1 and float(r.oa) > 1]
    return {"target": target, "rounds": rounds, "matches": matches}


def main():
    e = create_engine(load_settings().db_url)
    print("fit V2…")
    model, n = fit(e)
    tgt = sys.argv[1] if len(sys.argv) > 1 else None
    res = predict_round(e, model, tgt)
    if not res["matches"]:
        print(f"Aucun match. Rounds : {res['rounds'][:8]}"); return
    print(f"\nROUND {res['target']} Mada — PRÉDICTION V2 (fit {n} matchs)\n")
    print(f"  {'match':<27}{'1X2 (V2)':<17}{'score top-3':<22}{'Ov2.5':>6}  grosse cote")
    print("  " + "-" * 90)
    notables = []
    for m in res["matches"]:
        ph, pd_, pa = m["x12"]
        x = f"1:{ph*100:.0f} X:{pd_*100:.0f} 2:{pa*100:.0f}"
        ts = " ".join(f"{s}({p*100:.0f})" for s, p in m["score_top3"])
        nb = ""
        if m["notable"]:
            n_ = m["notable"]; nb = f"{n_['pari']}@{n_['cote']} (EV{n_['ev']:+.0f}%)"
            notables.append((m["match"], n_))
        ov = f"{m['over25_pct']:.0f}%" if m["over25_pct"] is not None else "-"
        print(f"  {m['match'][:26]:<27}{x:<17}{ts:<22}{ov:>6}  {nb}")
    if notables:
        print("\n  💰 Grosses cotes notables :")
        for mt, nb in sorted(notables, key=lambda z: -z[1]["ev"]):
            print(f"    {mt:<28} -> {nb['pari']} @{nb['cote']} (p={nb['proba']}%, EV{nb['ev']:+.0f}%)")


if __name__ == "__main__":
    main()
