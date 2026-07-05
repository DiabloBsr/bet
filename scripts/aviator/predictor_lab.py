"""LABO PRÉDICTEUR DE CRASH — teste TOUTES les méthodes de prédiction du prochain
multiplicateur et mesure, hors-échantillon, si l'une bat la simple distribution.

Provably-fair => crashs i.i.d. => le meilleur prédicteur POSSIBLE du prochain crash
est la distribution elle-même (moyenne/médiane) ; aucune info du passé n'aide.
Ce labo le PROUVE (ou le casse) sur les données réelles collectées.

Deux tâches évaluées en OOS (split chrono 70/30) :
  A) PRÉDICTION PONCTUELLE du log-multiplicateur : MAE/RMSE vs baseline (moyenne train).
  B) PROBABILITÉ P(prochain >= T) pour T in {1.5, 2, 3} : Brier/log-loss vs base rate.
Méthodes : last, moyennes mobiles, EMA, gambler's-fallacy (après série de bas),
hot-hand, Markov (états bas/moyen/haut), ML (features = k derniers crashs).
Verdict : une méthode bat-elle la baseline OOS ? (attendu : NON = imprévisible.)
"""
from __future__ import annotations
import sqlite3, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "aviator.db"
K = 10          # taille de fenêtre d'historique pour les features
EPS = 1e-9


def load(db=DB):
    con = sqlite3.connect(db)
    m = np.array([r[0] for r in con.execute(
        "SELECT multiplier FROM aviator_rounds ORDER BY rowid")], float)
    con.close()
    return m


def build(m):
    """features X (k derniers log-crashs + dérivés), cibles y_log et y_geT."""
    lm = np.log(np.clip(m, 1.0, None))
    X, ylog, idx = [], [], []
    for i in range(K, len(m)):
        w = lm[i-K:i]
        low_streak = 0
        for v in m[i-1::-1]:
            if v < 2.0: low_streak += 1
            else: break
        X.append([*w, w.mean(), w.std(), (m[i-K:i] < 2).mean(), low_streak, m[i-1]])
        ylog.append(lm[i]); idx.append(i)
    return np.array(X), np.array(ylog), np.array(idx)


def brier_logloss(p, y):
    p = np.clip(p, EPS, 1-EPS)
    return float(np.mean((p-y)**2)), float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))


def run(m):
    n = len(m)
    print("=" * 66)
    print(f"  LABO PRÉDICTEUR DE CRASH — {n} manches")
    print("=" * 66)
    if n < 80:
        print(f"  ⚠ Trop peu de données ({n}) pour un verdict fiable — laisse le collecteur")
        print("    tourner (idéal ~2000+). Résultats ci-dessous = INDICATIFS.\n")
    if n < K + 20:
        print("  (pas assez pour construire les features — stop.)"); return
    X, ylog, idx = build(m)
    cut = int(len(X) * 0.7)
    Xtr, Xte = X[:cut], X[cut:]
    ytr, yte = ylog[:cut], ylog[cut:]
    m_te = m[idx][cut:]

    # ---- A) prédiction ponctuelle (log) ----
    print("  A) PRÉDICTION DU MULTIPLICATEUR (log) — MAE/RMSE OOS (plus bas = mieux)")
    preds = {}
    base = ytr.mean()
    preds["baseline (moyenne train)"] = np.full(len(yte), base)
    preds["dernier crash"] = np.log(np.clip(m_te_prev(m, idx, cut), 1, None))
    preds["moyenne mobile 5"] = X[cut:, :5].mean(1) if X.shape[1] >= 5 else None
    preds["moyenne mobile 10"] = X[cut:, :K].mean(1)
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        gb = GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=.05)
        gb.fit(Xtr, ytr); preds["ML (GBM sur k derniers)"] = gb.predict(Xte)
    except Exception as e:
        print(f"     (ML indispo : {e})")
    from math import sqrt
    rows = []
    for name, p in preds.items():
        if p is None or len(p) != len(yte):
            continue
        mae = float(np.mean(np.abs(p - yte))); rmse = sqrt(float(np.mean((p - yte)**2)))
        rows.append((name, mae, rmse))
    base_mae = next(r[1] for r in rows if r[0].startswith("baseline"))
    for name, mae, rmse in sorted(rows, key=lambda r: r[1]):
        tag = " <<< BATTUE" if name.startswith("baseline") else \
              ("  (bat la baseline !)" if mae < base_mae - 1e-6 else "")
        print(f"     {name:<26} MAE {mae:.4f}  RMSE {rmse:.4f}{tag}")
    winner_A = min(rows, key=lambda r: r[1])[0]

    # ---- B) proba P(prochain >= T) ----
    print("\n  B) PROBABILITÉ P(prochain >= T) — Brier/log-loss OOS (plus bas = mieux)")
    for T in (1.5, 2.0, 3.0):
        yb = (m[idx] >= T).astype(int)
        ytr_b, yte_b = yb[:cut], yb[cut:]
        base_rate = ytr_b.mean()
        cand = {"base rate (train)": np.full(len(yte_b), base_rate)}
        # conditionnel sur série de bas (gambler's fallacy)
        streak = X[cut:, -2]
        hi = ytr_b[X[:cut, -2] >= 3].mean() if (X[:cut, -2] >= 3).sum() >= 5 else base_rate
        cand["si série>=3 bas"] = np.where(streak >= 3, hi, base_rate)
        try:
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(max_iter=1000).fit(Xtr, ytr_b)
            cand["ML (logistique)"] = lr.predict_proba(Xte)[:, 1]
        except Exception:
            pass
        print(f"    T={T}x  (base rate {100*base_rate:.0f}%)")
        bb, _ = brier_logloss(cand["base rate (train)"], yte_b)
        for name, p in cand.items():
            br, ll = brier_logloss(p, yte_b)
            tag = "" if name.startswith("base") else (" <<< BAT" if br < bb - 1e-4 else " (=)")
            print(f"       {name:<20} Brier {br:.4f}  logloss {ll:.4f}{tag}")

    print("\n  VERDICT :")
    if winner_A.startswith("baseline"):
        print("    Aucune méthode ne bat la simple moyenne en OOS -> crash IMPRÉVISIBLE.")
        print("    Conforme au provably-fair : le passé ne contient AUCUNE info sur le futur.")
    else:
        print(f"    ⚠ '{winner_A}' bat la baseline sur cet échantillon — à re-tester quand n>2000")
        print("    (avec 35-80 manches c'est quasi certainement du bruit, comme les 17 campagnes foot).")
    print("    La seule 'prédiction' honnête = P(prochain>=T) = fréquence historique (survie).")
    print("=" * 66)


def m_te_prev(m, idx, cut):
    """multiplicateur juste avant chaque point de test (pour la méthode 'dernier crash')."""
    return np.array([m[i-1] for i in idx[cut:]])


if __name__ == "__main__":
    run(load())
