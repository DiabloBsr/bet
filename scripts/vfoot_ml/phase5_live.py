"""VFoot-ML — PHASE 5 : Prédiction en temps réel.

Entrée : un match à venir (équipes + cotes 1X2). Sortie : probas 1X2, score exact
le plus probable, Over/Under, BTTS, détection de value, mise Kelly, niveau de
confiance + alerte si confiance faible.

Décision d'architecture (fondée sur la Phase 3) : les features équipe/séquence/RNG
n'apportant AUCUNE valeur OOS, le modèle live est STATELESS — il ne dépend que des
cotes. Robuste, instantané, performance identique au modèle complet.

Modèles : LogReg (1X2) + 2 régressions de Poisson (buts dom/ext -> grille de score).
Sauvegardés avec joblib ; ré-entraînés si absents.

Usage :
    python phase5_live.py --match "Liverpool" "Everton" 1.45 4.5 6.2
    python phase5_live.py --upcoming        # prédit les rounds live du scraper
"""
from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from scipy.stats import poisson
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # racine projet (pour `scraper`)
from phase3_models import load, MAP
from scraper.market_inversion import exact_invert_1x2, apply_sim_deviations   # modèle RÉALISÉ

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
logger = logging.getLogger("vfoot.live")

MODEL_DIR = Path("data/vfoot_ml/models"); MODEL_DIR.mkdir(parents=True, exist_ok=True)
ODDS_FEATS = ["imp_1", "imp_x", "imp_2", "odds_ratio_1_2", "odds_spread", "fav_strength", "lambda_tot_impl"]
CLASSES = ["1", "X", "2"]
KELLY_FRAC = 0.25
VALUE_THR = 0.02            # EV minimal pour signaler une "value"
OPEN_THR = 0.42            # P(total>=4) mini pour proposer un "score si ouvert" (sinon rien)


def odds_features(o1, ox, o2):
    """Construit le vecteur de features (cotes seules) pour UN match."""
    inv = 1 / o1 + 1 / ox + 1 / o2
    imp1, impx, imp2 = (1 / o1) / inv, (1 / ox) / inv, (1 / o2) / inv
    return {"imp_1": imp1, "imp_x": impx, "imp_2": imp2,
            "odds_ratio_1_2": o1 / o2, "odds_spread": max(o1, ox, o2) - min(o1, ox, o2),
            "fav_strength": max(imp1, imp2), "lambda_tot_impl": -np.log(max(impx, 1e-6))}


# ---------------------------------------------------------------------- #
def train_and_save():
    """Entraîne LogReg (1X2) + Poisson (buts) sur tout l'historique, sauvegarde."""
    df = load()
    clf = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=600))])
    clf.fit(df[ODDS_FEATS], df.y)
    ph = PoissonRegressor(max_iter=400).fit(df[ODDS_FEATS], df.y_home_goals)
    pa = PoissonRegressor(max_iter=400).fit(df[ODDS_FEATS], df.y_away_goals)
    joblib.dump({"clf": clf, "ph": ph, "pa": pa, "classes": CLASSES, "feats": ODDS_FEATS},
                MODEL_DIR / "live_models.joblib")
    logger.info("modèles entraînés et sauvegardés (%d matchs)", len(df))


class LivePredictor:
    """Prédicteur live : charge les modèles, prédit un match ou les rounds à venir."""

    def __init__(self):
        path = MODEL_DIR / "live_models.joblib"
        if not path.exists():
            train_and_save()
        self.m = joblib.load(path)
        # table de calibration des scores (7x7) — corrige les biais par cellule (ex. 3-1)
        self.calib = None
        cp = MODEL_DIR.parent / "score_calibration.json"
        if cp.exists():
            try:
                self.calib = np.asarray(json.loads(cp.read_text(encoding="utf-8"))["correction"], float)
            except Exception:
                self.calib = None

    def _score_grid(self, lh, la, maxg=8):
        gh, ga = poisson.pmf(np.arange(maxg), lh), poisson.pmf(np.arange(maxg), la)
        grid = np.outer(gh, ga); return grid / grid.sum()

    @staticmethod
    def _over(grid, line):
        N = grid.shape[0]
        return float(sum(grid[h, a] for h in range(N) for a in range(N) if h + a > line))

    @staticmethod
    def _parse_offered(raw) -> dict:
        """Dévigge les marchés annexes (extra_markets) -> probabilités offertes."""
        try:
            xm = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            return {}
        if not isinstance(xm, dict):
            return {}

        def gm(name):
            if name in xm:
                return xm[name]
            for k, v in xm.items():
                if k.startswith(name):
                    return v
            return None

        def devig(d):
            items = {k: v for k, v in d.items() if isinstance(v, (int, float)) and v > 1}
            z = sum(1 / v for v in items.values())
            return {k: round(100 * (1 / v) / z, 1) for k, v in items.items()} if z > 0 else {}

        out = {}
        for key, name in [("ht_ft", "HT/FT"), ("mi_temps_1x2", "Mi-tps 1X2"), ("btts", "G/NG")]:
            m = gm(name)
            if isinstance(m, dict):
                out[key] = devig(m)
        mtcs = gm("Mi-tps CS")
        if isinstance(mtcs, dict):
            dv = devig(mtcs)
            out["mi_temps_score_top3"] = sorted(dv.items(), key=lambda kv: -kv[1])[:3]
        totb = gm("Total de buts")
        if isinstance(totb, dict):
            out["total_buts"] = devig({k: v for k, v in totb.items() if k.isdigit()})
        pm = gm("+/-")
        if isinstance(pm, dict):
            out["over_under_3_5"] = devig(pm)
        se = gm("Score exact")
        if isinstance(se, dict):
            out["score_exact_odds"] = {k.replace(":", "-").replace(" ", ""): v
                                       for k, v in se.items() if isinstance(v, (int, float)) and v > 1}
        return out

    def predict_match(self, home, away, o1, ox, o2, extra_markets=None) -> dict:
        """Prédit UN match. Retourne un rapport complet (dict).
        extra_markets : JSON brut des marchés annexes (pour l'info complète)."""
        try:
            o1, ox, o2 = float(o1), float(ox), float(o2)
            if not (o1 > 1 and ox > 1 and o2 > 1):
                raise ValueError("cotes invalides (doivent être >1)")
            X = pd.DataFrame([odds_features(o1, ox, o2)])[self.m["feats"]]
            proba = self.m["clf"].predict_proba(X)[0]                    # [p1,pX,p2]
            # GRILLE = modèle RÉALISÉ (Poisson + Dixon-Coles + boosts) : plus varié + plus juste
            try:
                lh, la = exact_invert_1x2(o1, ox, o2)
                lh, la = float(np.clip(lh, 0.05, 6)), float(np.clip(la, 0.05, 6))
                grid = np.asarray(apply_sim_deviations(lh, la, "cells"), float)[:7, :7]
                grid = grid / grid.sum()
            except Exception:    # repli : Poisson pur via régression
                lh = float(np.clip(self.m["ph"].predict(X)[0], 0.05, 6))
                la = float(np.clip(self.m["pa"].predict(X)[0], 0.05, 6))
                grid = self._score_grid(lh, la)
            # ----- CALIBRATION par cellule (aligne sur la fréquence réelle) -----
            if self.calib is not None and grid.shape == self.calib.shape:
                grid = grid * self.calib
                grid = grid / grid.sum()
            N = grid.shape[0]
            flat = grid.ravel(); order = np.argsort(-flat)

            def _sc(i):
                return f"{i // N}-{i % N}"
            top_scores = [(_sc(i), round(float(flat[i]), 3)) for i in order[:3]]
            over25 = self._over(grid, 2.5)
            btts = float(sum(grid[h, a] for h in range(1, N) for a in range(1, N)))
            # "score si match ouvert" : proposé UNIQUEMENT si le match est vraiment ouvert
            # (P(total>=4) >= OPEN_THR). Sinon -> None (on ne propose rien).
            p_open = self._over(grid, 3.5)        # P(total >= 4 buts)
            hi = [i for i in order if (i // N + i % N) >= 4]
            high_score = ((_sc(hi[0]), round(float(flat[hi[0]]), 3))
                          if (hi and p_open >= OPEN_THR) else None)

            # accord des moteurs : réalisé-calibré (primaire) vs Poisson pur
            try:
                gp = np.outer(poisson.pmf(np.arange(N), lh), poisson.pmf(np.arange(N), la)); gp = gp / gp.sum()
                poisson_top = _sc(int(gp.argmax()))
            except Exception:
                poisson_top = None

            # ----- INFO COMPLÈTE -----
            details = {"engines": {"realise_calibre": top_scores[0][0], "poisson": poisson_top,
                                   "accord": bool(poisson_top == top_scores[0][0])},
                       "model": {
                "over": {str(l): round(100 * self._over(grid, l), 1) for l in (0.5, 1.5, 2.5, 3.5, 4.5)},
                "btts_pct": round(100 * btts, 1),
                "exact_top5": [(_sc(i), round(float(flat[i]), 3)) for i in order[:7]],
                "high_score": high_score,
                "open_pct": round(100 * p_open, 1),
                "total_buts": {str(k): round(100 * float(sum(grid[h, a] for h in range(N)
                               for a in range(N) if h + a == k)), 1) for k in range(0, 7)},
                "lambda": [round(lh, 2), round(la, 2)]}}
            off = {}
            if extra_markets:
                off = self._parse_offered(extra_markets)
                if off:
                    details["offered"] = off

            # ----- GROS COTES PROBABLES : outcomes à cote >=3, classés par proba modèle -----
            vbets = []
            for k, lab, od_ in [(0, "1", o1), (1, "X", ox), (2, "2", o2)]:
                if od_ >= 3.0:
                    pr = float(proba[k])
                    vbets.append({"marche": "1X2", "pari": lab, "cote": round(od_, 2),
                                  "proba": round(100 * pr, 1), "ev": round(100 * (pr * od_ - 1), 1)})
            se_odds = off.get("score_exact_odds", {})
            for sc, pr in details["model"]["exact_top5"]:
                od_ = se_odds.get(sc)
                if od_ and od_ >= 3.0:
                    vbets.append({"marche": "score", "pari": sc, "cote": round(od_, 2),
                                  "proba": round(100 * pr, 1), "ev": round(100 * (pr * od_ - 1), 1)})
            details["value_bets"] = sorted(vbets, key=lambda b: -b["proba"])

            odds = np.array([o1, ox, o2])
            ev = proba * odds - 1.0
            best = int(ev.argmax())
            confidence = float(proba.max())
            level = "HAUTE" if confidence >= 0.60 else ("MOYENNE" if confidence >= 0.45 else "FAIBLE")
            kelly = max(0.0, KELLY_FRAC * ev[best] / (odds[best] - 1.0)) if ev[best] > 0 else 0.0

            return {
                "match": f"{home} vs {away}", "cotes": [o1, ox, o2],
                "proba_1x2": {CLASSES[i]: round(float(proba[i]), 4) for i in range(3)},
                "issue_probable": CLASSES[int(proba.argmax())],
                "score_exact_top3": top_scores,
                "over_2_5_pct": round(100 * over25, 1), "btts_pct": round(100 * btts, 1),
                "lambda": [round(lh, 2), round(la, 2)],
                "value": {"issue": CLASSES[best], "ev_pct": round(100 * float(ev[best]), 2),
                          "detectee": bool(ev[best] > VALUE_THR)},
                "mise_kelly_pct_bankroll": round(100 * kelly, 2),
                "confiance": level, "confiance_val": round(confidence, 3),
                "alerte": ("⚠ CONFIANCE FAIBLE — match indécis" if level == "FAIBLE" else None),
                "details": details,
            }
        except Exception as exc:
            return {"match": f"{home} vs {away}", "error": str(exc)}

    def predict_upcoming(self, league="InstantLeague-8035", limit=10):
        """Lit les rounds à venir (cotes, sans résultat) du scraper et les prédit."""
        from sqlalchemy import create_engine, text
        from scraper.config import load_settings
        eng = create_engine(load_settings().db_url)
        # rounds VRAIMENT à venir = expected_start >= MAINTENANT (UTC réel) - 2 min
        # (évite les vieux orphelins cotes-sans-résultat de la nuit)
        now_min = (pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(minutes=2)).strftime("%Y-%m-%d %H:%M:%S")
        sql = text("""
            SELECT e.expected_start ts, e.team_a home, e.team_b away,
                   o.odds_home oh, o.odds_draw od, o.odds_away oa, o.extra_markets xm
            FROM events e
            JOIN odds_snapshots o ON o.id=(SELECT MAX(id) FROM odds_snapshots WHERE event_id=e.id)
            LEFT JOIN results r ON r.event_id=e.id
            WHERE r.id IS NULL AND e.competition=:lg AND o.odds_home>1
              AND e.expected_start >= :now_min
            ORDER BY e.expected_start LIMIT :lim""")
        df = pd.read_sql(sql, eng, params={"lg": league, "lim": int(limit), "now_min": now_min})
        reports = []
        for r in df.itertuples():
            rep = self.predict_match(r.home, r.away, r.oh, r.od, r.oa, extra_markets=r.xm)
            rep["ts_utc"] = str(r.ts)
            try:    # heure MADAGASCAR (UTC+3, sans DST)
                rep["heure_locale"] = (pd.to_datetime(r.ts) + pd.Timedelta(hours=3)).strftime("%d/%m %H:%M")
            except Exception:
                rep["heure_locale"] = None
            reports.append(rep)
        return reports, df

    def predict_round(self, ts_utc_prefix, league="InstantLeague-8035"):
        """Prédit les matchs d'un round à un instant DONNÉ (préfixe UTC 'YYYY-MM-DD HH:MM').
        Inclut le résultat réel s'il existe déjà (pour vérifier après coup)."""
        from sqlalchemy import create_engine, text
        from scraper.config import load_settings
        eng = create_engine(load_settings().db_url)
        sql = text("""
            SELECT e.expected_start ts, e.team_a home, e.team_b away,
                   o.odds_home oh, o.odds_draw od, o.odds_away oa, o.extra_markets xm,
                   r.score_a sa, r.score_b sb
            FROM events e
            JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
            LEFT JOIN results r ON r.event_id=e.id
            WHERE e.competition=:lg AND e.expected_start LIKE :pref AND o.odds_home>1
            ORDER BY e.id""")
        df = pd.read_sql(sql, eng, params={"lg": league, "pref": ts_utc_prefix + "%"})
        reports = []
        for r in df.itertuples():
            rep = self.predict_match(r.home, r.away, r.oh, r.od, r.oa, extra_markets=r.xm)
            rep["ts_utc"] = str(r.ts)
            try:
                rep["heure_locale"] = (pd.to_datetime(r.ts) + pd.Timedelta(hours=3)).strftime("%d/%m %H:%M")
            except Exception:
                rep["heure_locale"] = None
            if r.sa is not None and not pd.isna(r.sa):
                rep["actual"] = f"{int(r.sa)}-{int(r.sb)}"
            reports.append(rep)
        return reports, df


def _print_report(r):
    if r.get("error"):
        print(f"  {r['match']} : ERREUR {r['error']}"); return
    p = r["proba_1x2"]
    heure = f"🕐 {r['heure_locale']}  " if r.get("heure_locale") else ""
    print(f"\n  ┌─ {heure}{r['match']}  (cotes {r['cotes'][0]}/{r['cotes'][1]}/{r['cotes'][2]})")
    print(f"  │ 1X2 modèle : 1={100*p['1']:.1f}%  X={100*p['X']:.1f}%  2={100*p['2']:.1f}%  "
          f"-> issue probable : {r['issue_probable']}")
    sc = "  ".join(f"{s}({100*pr:.0f}%)" for s, pr in r['score_exact_top3'])
    print(f"  │ Score exact top3 : {sc}   (λ {r['lambda'][0]}-{r['lambda'][1]})")
    print(f"  │ Over 2.5 : {r['over_2_5_pct']}%   |  BTTS : {r['btts_pct']}%")
    v = r["value"]
    vtag = f"✅ VALUE {v['issue']} (+{v['ev_pct']}%)" if v["detectee"] else f"pas de value (best EV {v['ev_pct']}%)"
    print(f"  │ Value : {vtag}   |  Mise Kelly : {r['mise_kelly_pct_bankroll']}% bankroll")
    print(f"  │ Confiance : {r['confiance']} ({r['confiance_val']}){'  '+r['alerte'] if r['alerte'] else ''}")
    print(f"  └─")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", nargs=5, metavar=("HOME", "AWAY", "O1", "OX", "O2"))
    ap.add_argument("--upcoming", action="store_true")
    ap.add_argument("--retrain", action="store_true")
    a = ap.parse_args()
    if a.retrain:
        train_and_save()
    pred = LivePredictor()
    print("=" * 70); print("  VFoot-ML — PRÉDICTION TEMPS RÉEL (modèle stateless cotes)"); print("=" * 70)
    if a.match:
        _print_report(pred.predict_match(a.match[0], a.match[1], *a.match[2:]))
    elif a.upcoming:
        reports, df = pred.predict_upcoming()
        if not reports:
            print("  Aucun round à venir capté (le scraper doit tourner).")
        for r in reports:
            _print_report(r)
    else:
        # démo
        for m in [("Liverpool", "Everton", 1.45, 4.5, 6.2),
                  ("Sunderland", "Man Blue", 7.5, 4.6, 1.4),
                  ("Brighton", "Fulham", 2.3, 3.3, 3.0)]:
            _print_report(pred.predict_match(*m))
    print("\n  ⚠ Rappel : sur ce RNG calibré, 'value' = bruit anti-sélectif (cf. Phase 4). "
          "Le moteur n'est GO que si le moniteur de dérive ouvre une fenêtre.")


if __name__ == "__main__":
    main()
