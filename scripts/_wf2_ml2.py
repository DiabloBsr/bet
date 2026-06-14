# -*- coding: utf-8 -*-
"""WF2 - suite de _wf2_ml.py :
   1) bootstrap CI du ROI des picks 'desaccord GBM vs favori marche'
   2) logistic parcimonieuse (cotes + top fondamentales) vs logistic cote-only
   Reutilise la construction de dataset de _wf2_ml en l'important comme module ?
   Non: _wf2_ml execute tout a l'import -> on re-execute la partie dataset ici (copie minimale)
   via exec du bloc, plus simple: on relance le pipeline en important les fonctions n'existe pas.
   => on refait tourner la construction en subprocess ? Non : on duplique la logique de
   construction en appelant _wf2_ml comme script n'est pas isolable. Solution retenue :
   _wf2_ml.py a ete concu d'un bloc; ici on recopie UNIQUEMENT le chargement+features
   en important le fichier avec un garde ? Le plus robuste : exec le fichier jusqu'au
   DataFrame puis travailler dessus.
"""
import sys, runpy, io
from contextlib import redirect_stdout

sys.path.insert(0, '.')
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss

RNG = 42

# --- recuperer le dataset en executant _wf2_ml.py avec un flag d'arret avant les modeles ---
src = open("scripts/_wf2_ml.py", encoding="utf-8").read()
cut_marker = "# ============================================================ 4. MODELES + EVAL"
src_dataset = src.split(cut_marker)[0]
g = {}
buf = io.StringIO()
with redirect_stdout(buf):
    exec(compile(src_dataset, "_wf2_ml_dataset", "exec"), g)
df = g["df"]
ODDS_FEATS = ["p_h", "p_d", "p_a", "overround"]
FUND_FEATS = [c for c in df.columns if c not in
              ("event_id", "sid", "round", "ts", "team_h", "team_a",
               "odds_h", "odds_d", "odds_a", "y") and c not in ODDS_FEATS]
ALL_FEATS = ODDS_FEATS + FUND_FEATS
print(f"dataset recharge: {len(df)} matchs")


def make_logistic():
    return Pipeline([("imp", SimpleImputer(strategy="mean")),
                     ("sc", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=3000, C=1.0, random_state=RNG))])


def make_gbm():
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
        random_state=RNG)


def boot_ci(pnl, n_boot=4000, seed=RNG):
    rng = np.random.default_rng(seed)
    n = len(pnl)
    means = np.array([pnl[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return np.percentile(means, [2.5, 97.5])


for frac in (0.70, 0.50):
    print("=" * 78)
    print(f"SPLIT {int(frac*100)}/{int((1-frac)*100)}")
    cut = int(len(df) * frac)
    tr, te = df.iloc[:cut], df.iloc[cut:]
    y_tr, y_te = tr["y"].values, te["y"].values
    odds_mat = te[["odds_h", "odds_d", "odds_a"]].values
    p_devig = te[["p_h", "p_d", "p_a"]].values
    fav = p_devig.argmax(axis=1)

    # ---------- 1) bootstrap des picks desaccord GBM ----------
    gbm = make_gbm().fit(tr[ALL_FEATS], y_tr)
    proba = gbm.predict_proba(te[ALL_FEATS])
    pred = proba.argmax(axis=1)
    dis = pred != fav
    nd = int(dis.sum())
    o = odds_mat[dis, pred[dis]]
    win = (pred[dis] == y_te[dis])
    pnl = np.where(win, o - 1.0, -1.0)
    lo, hi = boot_ci(pnl)
    print(f"picks desaccord GBM: n={nd}  wr={win.mean():.3f}  cote_moy={o.mean():.2f}  "
          f"ROI={pnl.mean()*100:+.2f}%  CI95=[{lo*100:+.2f}%, {hi*100:+.2f}%]")

    # repartition des desaccords par type de pick
    import collections
    print("  type de pick desaccord:", dict(collections.Counter(
        ['H' if p == 0 else ('D' if p == 1 else 'A') for p in pred[dis]])))

    # ---------- 2) logistic parcimonieuse ----------
    PARS = ODDS_FEATS + ["wr_alltime_h", "wr_alltime_a", "h2h_wr_h", "h2h_wr_a", "h2h_exact_wr_h"]
    base = make_logistic().fit(tr[ODDS_FEATS], y_tr)
    pars = make_logistic().fit(tr[PARS], y_tr)
    pb = base.predict_proba(te[ODDS_FEATS])
    pp = pars.predict_proba(te[PARS])
    print(f"logit cote-only : acc={accuracy_score(y_te, pb.argmax(1)):.4f}  "
          f"ll={log_loss(y_te, pb, labels=[0,1,2]):.4f}")
    print(f"logit parcimon. : acc={accuracy_score(y_te, pp.argmax(1)):.4f}  "
          f"ll={log_loss(y_te, pp, labels=[0,1,2]):.4f}  (feats={PARS})")

    # ---------- 3) GBM sans cotes : top-decile de confiance (fondamental pur) ----------
    gbm_f = make_gbm().fit(tr[FUND_FEATS], y_tr)
    proba_f = gbm_f.predict_proba(te[FUND_FEATS])
    conf = proba_f.max(axis=1)
    thr = np.quantile(conf, 0.9)
    sub = conf >= thr
    pred_f = proba_f.argmax(axis=1)
    win_f = pred_f[sub] == y_te[sub]
    o_f = odds_mat[sub, pred_f[sub]]
    pnl_f = np.where(win_f, o_f - 1.0, -1.0)
    fav_acc_sub = accuracy_score(y_te[sub], fav[sub])
    print(f"GBM SANS cotes top-decile: n={int(sub.sum())}  acc={win_f.mean():.3f}  "
          f"ROI={pnl_f.mean()*100:+.2f}%  | favori devig sur memes matchs acc={fav_acc_sub:.3f}")

print("\nFIN.")
