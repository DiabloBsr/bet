"""VFoot-ML — PHASE 3 : Construction & comparaison des 7 modèles.

Tous les modèles sont évalués sur le MÊME split chronologique (70% train / 30%
test OOS) avec les MÊMES métriques (accuracy, log-loss, Brier multiclasse).

Le BENCHMARK À BATTRE = les probabilités implicites des cotes (le book lui-même).
Un modèle n'a de valeur QUE s'il bat le log-loss du book en OOS.

Modèles : 1 LogReg · 2 RandomForest(Grid) · 3 XGBoost(Optuna) · 4 LightGBM(Optuna)
          5 Chaîne de Markov · 6 Poisson · 7 Stacking.

Test clé : ODDS-ONLY vs ALL-FEATURES -> les features équipe/séquence/RNG
apportent-elles de la valeur incrémentale OOS ?
"""
from __future__ import annotations
import json, logging, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import log_loss, accuracy_score

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
logger = logging.getLogger("vfoot.models")

ODDS = ["imp_1", "imp_x", "imp_2", "odds_ratio_1_2", "odds_spread", "fav_strength", "lambda_tot_impl"]
TEAM = ["home_winrate_hist", "home_gf_avg", "home_ga_avg", "home_form5", "home_venue_winrate",
        "away_winrate_hist", "away_gf_avg", "away_ga_avg", "away_form5", "away_venue_winrate",
        "winrate_diff", "form_diff"]
SEQ = ["seq_home_wins_5", "seq_home_wins_10", "seq_no_draw_streak", "seq_over25_streak", "seq_gap_since_draw"]
TEMP = ["hour", "dayofweek", "match_pos_in_day", "minutes_since_prev"]
RNG = ["rng_entropy_10", "rng_runs_z_20", "rng_score_repeat"]
ALL = ODDS + TEAM + SEQ + TEMP + RNG
LABELS = [0, 1, 2]
MAP = {"1": 0, "X": 1, "2": 2}


def brier_multi(y, proba):
    oh = np.zeros_like(proba); oh[np.arange(len(y)), y] = 1
    return float(((proba - oh) ** 2).sum(axis=1).mean())


def metrics(y, proba):
    proba = np.clip(proba, 1e-6, 1); proba = proba / proba.sum(axis=1, keepdims=True)
    return {"accuracy": round(float(accuracy_score(y, proba.argmax(1))), 4),
            "log_loss": round(float(log_loss(y, proba, labels=LABELS)), 4),
            "brier": round(brier_multi(y, proba), 4)}


def load():
    p = Path("data/vfoot_ml/features.parquet")
    df = pd.read_parquet(p) if p.exists() else pd.read_csv("data/vfoot_ml/features.csv")
    df["y"] = df.y_1x2.map(MAP)
    return df


def split(df, frac=0.7):
    n = int(len(df) * frac)
    return df.iloc[:n].copy(), df.iloc[n:].copy()


# ---------------------------------------------------------------------- #
def m_logreg(tr, te, cols):
    pipe = Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=600, C=1.0))])
    pipe.fit(tr[cols], tr.y)
    return pipe.predict_proba(te[cols]), pipe


def m_rf(tr, te, cols):
    grid = GridSearchCV(
        RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1),
        {"max_depth": [5, 8], "min_samples_leaf": [20, 50]},
        cv=TimeSeriesSplit(3), scoring="neg_log_loss", n_jobs=-1)
    grid.fit(tr[cols], tr.y)
    return grid.predict_proba(te[cols]), grid.best_params_


def _optuna_gbm(tr, cols, kind, n_trials=15):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    tss = TimeSeriesSplit(3)
    X, y = tr[cols].to_numpy(), tr.y.to_numpy()

    def objective(t):
        if kind == "xgb":
            import xgboost as xgb
            params = dict(max_depth=t.suggest_int("max_depth", 2, 6),
                          learning_rate=t.suggest_float("lr", 0.02, 0.3, log=True),
                          n_estimators=t.suggest_int("n_estimators", 100, 400),
                          subsample=t.suggest_float("subsample", 0.6, 1.0),
                          colsample_bytree=t.suggest_float("cs", 0.6, 1.0),
                          objective="multi:softprob", num_class=3, tree_method="hist",
                          verbosity=0)
            mk = lambda: xgb.XGBClassifier(**params)
        else:
            import lightgbm as lgb
            params = dict(max_depth=t.suggest_int("max_depth", 2, 8),
                          learning_rate=t.suggest_float("lr", 0.02, 0.3, log=True),
                          n_estimators=t.suggest_int("n_estimators", 100, 400),
                          num_leaves=t.suggest_int("num_leaves", 8, 64),
                          subsample=t.suggest_float("subsample", 0.6, 1.0),
                          verbose=-1)
            mk = lambda: lgb.LGBMClassifier(**params)
        ll = []
        for a, b in tss.split(X):
            m = mk(); m.fit(X[a], y[a])
            ll.append(log_loss(y[b], m.predict_proba(X[b]), labels=LABELS))
        return float(np.mean(ll))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def m_xgb(tr, te, cols, n_trials=15):
    import xgboost as xgb
    bp = _optuna_gbm(tr, cols, "xgb", n_trials)
    m = xgb.XGBClassifier(max_depth=bp["max_depth"], learning_rate=bp["lr"],
                          n_estimators=bp["n_estimators"], subsample=bp["subsample"],
                          colsample_bytree=bp["cs"], objective="multi:softprob",
                          num_class=3, tree_method="hist", verbosity=0)
    m.fit(tr[cols], tr.y)
    return m.predict_proba(te[cols]), m, bp


def m_lgbm(tr, te, cols, n_trials=15):
    import lightgbm as lgb
    bp = _optuna_gbm(tr, cols, "lgbm", n_trials)
    m = lgb.LGBMClassifier(max_depth=bp["max_depth"], learning_rate=bp["lr"],
                           n_estimators=bp["n_estimators"], num_leaves=bp["num_leaves"],
                           subsample=bp["subsample"], verbose=-1)
    m.fit(tr[cols], tr.y)
    return m.predict_proba(te[cols]), m, bp


def m_markov(df, tr, te):
    """Transition result_{t-1} -> result_t depuis le train ; prédit avec le résultat
    du match PRÉCÉDENT (pur signal de séquence : doit ≈ taux de base si memoryless)."""
    prev = df.y_1x2.shift(1)
    trans = pd.crosstab(prev.iloc[tr.index], df.y_1x2.iloc[tr.index], normalize="index")
    trans = trans.reindex(index=["1", "X", "2"], columns=["1", "X", "2"]).fillna(1 / 3)
    pv = prev.iloc[te.index].fillna("1")
    proba = np.vstack([trans.loc[p].to_numpy() for p in pv])
    return proba


def m_poisson(tr, te, cols):
    """λ_home, λ_away par régression de Poisson sur les features -> grille indépendante."""
    ph = PoissonRegressor(max_iter=300).fit(tr[cols], tr.y_home_goals)
    pa = PoissonRegressor(max_iter=300).fit(tr[cols], tr.y_away_goals)
    lh = np.clip(ph.predict(te[cols]), 0.05, 6); la = np.clip(pa.predict(te[cols]), 0.05, 6)
    ks = np.arange(0, 8)
    out = np.zeros((len(te), 3))
    for i in range(len(te)):
        gh = poisson.pmf(ks, lh[i]); ga = poisson.pmf(ks, la[i])
        grid = np.outer(gh, ga); grid /= grid.sum()
        p1 = np.tril(grid, -1).sum(); px = np.trace(grid); p2 = np.triu(grid, 1).sum()
        out[i] = [p1, px, p2]
    return out


def m_stacking(tr, te, cols):
    import xgboost as xgb, lightgbm as lgb
    base = [("lr", Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=600))])),
            ("rf", RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=30, n_jobs=-1)),
            ("xgb", xgb.XGBClassifier(max_depth=4, n_estimators=200, tree_method="hist", verbosity=0)),
            ("lgbm", lgb.LGBMClassifier(max_depth=5, n_estimators=200, verbose=-1))]
    st = StackingClassifier(base, final_estimator=LogisticRegression(max_iter=600),
                            cv=TimeSeriesSplit(3), n_jobs=-1)
    st.fit(tr[cols], tr.y)
    return st.predict_proba(te[cols])


# ---------------------------------------------------------------------- #
def main():
    df = load(); tr, te = split(df)
    yte = te.y.to_numpy()
    logger.info("train=%d test=%d (OOS chronologique)", len(tr), len(te))
    res = {}

    # BENCHMARK : probas implicites du book
    book = te[["imp_1", "imp_x", "imp_2"]].to_numpy()
    res["BENCHMARK_cotes"] = metrics(yte, book)

    try: res["1_LogReg"] = metrics(yte, m_logreg(tr, te, ALL)[0]); logger.info("LogReg ok")
    except Exception as e: logger.warning("LogReg: %s", e)
    try:
        pr, bp = m_rf(tr, te, ALL); res["2_RandomForest"] = metrics(yte, pr); logger.info("RF ok %s", bp)
    except Exception as e: logger.warning("RF: %s", e)
    try:
        pr, mxgb, bp = m_xgb(tr, te, ALL); res["3_XGBoost"] = metrics(yte, pr); logger.info("XGB ok")
    except Exception as e: logger.warning("XGB: %s", e); mxgb = None
    try: res["4_LightGBM"] = metrics(yte, m_lgbm(tr, te, ALL)[0]); logger.info("LGBM ok")
    except Exception as e: logger.warning("LGBM: %s", e)
    try: res["5_Markov"] = metrics(yte, m_markov(df, tr, te)); logger.info("Markov ok")
    except Exception as e: logger.warning("Markov: %s", e)
    try: res["6_Poisson"] = metrics(yte, m_poisson(tr, te, ALL)); logger.info("Poisson ok")
    except Exception as e: logger.warning("Poisson: %s", e)
    try: res["7_Stacking"] = metrics(yte, m_stacking(tr, te, ALL)); logger.info("Stacking ok")
    except Exception as e: logger.warning("Stacking: %s", e)

    # TEST INCRÉMENTAL : odds-only vs all (sur XGBoost)
    incr = {}
    try:
        incr["xgb_odds_only"] = metrics(yte, m_xgb(tr, te, ODDS, 12)[0])
        incr["xgb_all_features"] = res.get("3_XGBoost")
    except Exception as e: logger.warning("incr: %s", e)

    # importance de features (XGBoost)
    imp = {}
    if mxgb is not None:
        try:
            fi = dict(zip(ALL, mxgb.feature_importances_))
            imp = dict(sorted(fi.items(), key=lambda kv: -kv[1])[:15])
            imp = {k: round(float(v), 4) for k, v in imp.items()}
        except Exception: pass

    out = {"n_train": len(tr), "n_test": len(te), "models": res,
           "incremental_value": incr, "feature_importance_xgb_top15": imp}
    Path("data/vfoot_ml/phase3_results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n" + "=" * 66)
    print("  PHASE 3 — COMPARAISON OOS (test = 30% le plus récent)")
    print("=" * 66)
    print(f"  {'modèle':<20}{'accuracy':>10}{'log_loss':>11}{'brier':>9}")
    print("  " + "-" * 50)
    for k, v in res.items():
        tag = "  <- BENCHMARK" if "BENCH" in k else ""
        print(f"  {k:<20}{v['accuracy']:>10}{v['log_loss']:>11}{v['brier']:>9}{tag}")
    if incr:
        print("\n  TEST INCRÉMENTAL (les features non-cotes servent-elles ?) :")
        for k, v in incr.items():
            print(f"    {k:<20} log_loss={v['log_loss']}  acc={v['accuracy']}")
    if imp:
        print("\n  Importance XGBoost (top) :")
        for k, v in list(imp.items())[:8]:
            print(f"    {k:<24}{v}")
    print("=" * 66)


if __name__ == "__main__":
    main()
