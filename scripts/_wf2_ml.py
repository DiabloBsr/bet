# -*- coding: utf-8 -*-
"""WF2 - MODELE ML INTEGRE toutes-features vs baseline cote-only.

Protocole:
  - dataset: 1 ligne par match (dedup, round!=0, result + cotes ouverture valides)
  - features AVANT-match uniquement (etat mis a jour APRES extraction, groupe par kickoff)
  - modeles: (a) logistic cote-only [BASELINE], (b) logistic toutes features,
             (c) HistGradientBoosting toutes features, (d) HGB SANS cotes
  - walk-forward temporel: split 70/30 + split 50/50
  - metriques: accuracy, log-loss, ROI de strategies de picks (edge>=0.05, top-decile)
  - permutation importance du GBM sur l'OOS
"""
import sys, json
from collections import defaultdict, Counter
from datetime import datetime

sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from scraper.config import load_settings
from sqlalchemy import create_engine, text
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
from sklearn.inspection import permutation_importance

RNG = 42
SEP = "=" * 78


def parse_dt(s):
    return datetime.fromisoformat(str(s).replace('Z', ''))


# ============================================================ 1. LOAD + DEDUP
eng = create_engine(load_settings().db_url)
with eng.connect() as c:
    rows = c.execute(text(
        "select e.id, cast(e.round_info as int) rd, e.team_a, e.team_b, e.expected_start, "
        "r.score_a, r.score_b, o.odds_home, o.odds_draw, o.odds_away "
        "from events e "
        "left join results r on r.event_id = e.id "
        "left join (select os.event_id, os.odds_home, os.odds_draw, os.odds_away "
        "           from odds_snapshots os "
        "           join (select event_id, min(id) mid from odds_snapshots group by event_id) m "
        "             on m.mid = os.id) o on o.event_id = e.id "
        "where e.round_info != '0' "
        "order by e.expected_start, e.id")).fetchall()

# dedup (team_a, team_b, expected_start) : garder celui AVEC result, sinon min id
seen = {}
for r in rows:
    k = (r[2], r[3], str(r[4]))
    if k not in seen or (seen[k][5] is None and r[5] is not None):
        seen[k] = r
evs = sorted(seen.values(), key=lambda r: (str(r[4]), r[0]))
print(f"events round!=0 dedup: {len(evs)} (avant {len(rows)})")

# ============================================================ 2. SAISONS ROBUSTES
# nouvelle saison si round redescend de >=5 OU gap temporel >45 min
seasons_of = {}
sid = 0
last_rd = None
last_t = None
for r in evs:
    rd = r[1]
    t = parse_dt(r[4])
    new = False
    if last_rd is not None:
        if rd < last_rd - 4:
            new = True
        if last_t is not None and (t - last_t).total_seconds() > 45 * 60:
            new = True
    if new:
        sid += 1
        last_rd = rd
    else:
        last_rd = rd if last_rd is None else max(last_rd, rd)
    last_t = t
    seasons_of[r[0]] = sid
print(f"saisons reconstruites: {sid + 1}")

# ============================================================ 3. FEATURES AVANT-MATCH
# Etats (mis a jour APRES extraction, groupe par kickoff identique):
h2h = defaultdict(lambda: [0, 0, 0])          # (host, visitor) -> [wins_host, draws, wins_visitor]
alltime = defaultdict(lambda: [0, 0, 0])      # team -> [w, d, l] toutes saisons confondues
stab = defaultdict(dict)                      # sid -> team -> {pts,gf,ga,gp,w,res:[('W',gf,ga),...]}


def team_state(s, team):
    if team not in stab[s]:
        stab[s][team] = {"pts": 0, "gf": 0, "ga": 0, "gp": 0, "w": 0, "res": []}
    return stab[s][team]


def season_position(s, team):
    tbl = stab[s]
    if team not in tbl or tbl[team]["gp"] == 0:
        return np.nan
    order = sorted(tbl.items(),
                   key=lambda kv: (-kv[1]["pts"], -(kv[1]["gf"] - kv[1]["ga"]), -kv[1]["gf"]))
    for i, (t, _) in enumerate(order, 1):
        if t == team:
            return float(i)
    return np.nan


def streak_signed(res):
    """+k = k victoires consecutives, -k = k defaites consecutives, 0 sinon."""
    if not res:
        return np.nan
    last = res[-1][0]
    if last == 'D':
        return 0.0
    k = 0
    for o, _, _ in reversed(res):
        if o == last:
            k += 1
        else:
            break
    return float(k) if last == 'W' else -float(k)


def f5(res, idx):
    lastn = res[-5:]
    if not lastn:
        return np.nan
    return float(np.mean([x[idx] for x in lastn]))


def form5_pts(res):
    lastn = res[-5:]
    if not lastn:
        return np.nan
    return float(sum(3 if x[0] == 'W' else (1 if x[0] == 'D' else 0) for x in lastn))


recs = []
i = 0
while i < len(evs):
    # groupe de matchs au meme kickoff -> features pour tous AVANT update
    j = i
    group = []
    t0 = str(evs[i][4])
    while j < len(evs) and str(evs[j][4]) == t0:
        group.append(evs[j])
        j += 1

    for r in group:
        eid, rd, th, ta, est, sa, sb, oh, od, oa = r
        s = seasons_of[eid]
        valid_odds = (oh and od and oa and oh > 1.0 and od > 1.0 and oa > 1.0)
        if sa is not None and valid_odds:
            inv = 1.0 / oh + 1.0 / od + 1.0 / oa
            p_h, p_d, p_a = (1.0 / oh) / inv, (1.0 / od) / inv, (1.0 / oa) / inv

            # H2H exact (meme orientation) + toutes confrontations
            ex = h2h[(th, ta)]
            rv = h2h[(ta, th)]
            n_ex = sum(ex)
            any_w_h = ex[0] + rv[2]
            any_d = ex[1] + rv[1]
            any_w_a = ex[2] + rv[0]
            n_any = any_w_h + any_d + any_w_a

            sh = team_state(s, th)
            sv = team_state(s, ta)
            ath = alltime[th]
            atv = alltime[ta]
            n_at_h = sum(ath)
            n_at_a = sum(atv)
            wr_at_h = ath[0] / n_at_h if n_at_h >= 10 else np.nan
            wr_at_a = atv[0] / n_at_a if n_at_a >= 10 else np.nan
            wr_se_h = sh["w"] / sh["gp"] if sh["gp"] > 0 else np.nan
            wr_se_a = sv["w"] / sv["gp"] if sv["gp"] > 0 else np.nan
            pos_h = season_position(s, th)
            pos_a = season_position(s, ta)
            ppg_h = sh["pts"] / sh["gp"] if sh["gp"] > 0 else np.nan
            ppg_a = sv["pts"] / sv["gp"] if sv["gp"] > 0 else np.nan

            recs.append({
                "event_id": eid, "sid": s, "round": rd, "ts": t0,
                "team_h": th, "team_a": ta,
                "odds_h": oh, "odds_d": od, "odds_a": oa,
                "y": 0 if sa > sb else (1 if sa == sb else 2),
                # --- bloc cotes ---
                "p_h": p_h, "p_d": p_d, "p_a": p_a, "overround": inv,
                # --- H2H ---
                "h2h_n": float(n_any),
                "h2h_wr_h": any_w_h / n_any if n_any > 0 else np.nan,
                "h2h_dr": any_d / n_any if n_any > 0 else np.nan,
                "h2h_wr_a": any_w_a / n_any if n_any > 0 else np.nan,
                "h2h_exact_n": float(n_ex),
                "h2h_exact_wr_h": ex[0] / n_ex if n_ex > 0 else np.nan,
                # --- series / forme intra-saison ---
                "streak_h": streak_signed(sh["res"]),
                "streak_a": streak_signed(sv["res"]),
                "form5_pts_h": form5_pts(sh["res"]),
                "form5_pts_a": form5_pts(sv["res"]),
                "gf5_h": f5(sh["res"], 1), "ga5_h": f5(sh["res"], 2),
                "gf5_a": f5(sv["res"], 1), "ga5_a": f5(sv["res"], 2),
                # --- classement intra-saison ---
                "gp_h": float(sh["gp"]), "gp_a": float(sv["gp"]),
                "pts_h": float(sh["pts"]), "pts_a": float(sv["pts"]),
                "ppg_h": ppg_h, "ppg_a": ppg_a,
                "pos_h": pos_h, "pos_a": pos_a,
                "pos_diff": pos_h - pos_a if not (np.isnan(pos_h) or np.isnan(pos_a)) else np.nan,
                "ppg_diff": ppg_h - ppg_a if not (np.isnan(ppg_h) or np.isnan(ppg_a)) else np.nan,
                # --- sur-regime ---
                "wr_season_h": wr_se_h, "wr_season_a": wr_se_a,
                "wr_alltime_h": wr_at_h, "wr_alltime_a": wr_at_a,
                "surregime_h": wr_se_h - wr_at_h if not (np.isnan(wr_se_h) or np.isnan(wr_at_h)) else np.nan,
                "surregime_a": wr_se_a - wr_at_a if not (np.isnan(wr_se_a) or np.isnan(wr_at_a)) else np.nan,
                # --- journee / segment ---
                "rd": float(rd),
                "seg_early": 1.0 if rd <= 13 else 0.0,
                "seg_mid": 1.0 if 14 <= rd <= 26 else 0.0,
                "seg_late": 1.0 if rd >= 27 else 0.0,
            })

    # update des etats avec les results du groupe (APRES extraction)
    for r in group:
        eid, rd, th, ta, est, sa, sb, oh, od, oa = r
        if sa is None:
            continue
        s = seasons_of[eid]
        if sa > sb:
            h2h[(th, ta)][0] += 1
            alltime[th][0] += 1; alltime[ta][2] += 1
            oh_, oa_ = 'W', 'L'
            ph_, pa_ = 3, 0
        elif sa == sb:
            h2h[(th, ta)][1] += 1
            alltime[th][1] += 1; alltime[ta][1] += 1
            oh_, oa_ = 'D', 'D'
            ph_, pa_ = 1, 1
        else:
            h2h[(th, ta)][2] += 1
            alltime[th][2] += 1; alltime[ta][0] += 1
            oh_, oa_ = 'L', 'W'
            ph_, pa_ = 0, 3
        sh = team_state(s, th)
        sv = team_state(s, ta)
        sh["pts"] += ph_; sh["gf"] += sa; sh["ga"] += sb; sh["gp"] += 1
        sh["w"] += 1 if oh_ == 'W' else 0
        sh["res"].append((oh_, sa, sb))
        sv["pts"] += pa_; sv["gf"] += sb; sv["ga"] += sa; sv["gp"] += 1
        sv["w"] += 1 if oa_ == 'W' else 0
        sv["res"].append((oa_, sb, sa))
    i = j

df = pd.DataFrame(recs).sort_values(["ts", "event_id"]).reset_index(drop=True)
print(f"dataset final: {len(df)} matchs  | repartition y: {df['y'].value_counts().to_dict()}")

ODDS_FEATS = ["p_h", "p_d", "p_a", "overround"]
FUND_FEATS = [c for c in df.columns if c not in
              ("event_id", "sid", "round", "ts", "team_h", "team_a",
               "odds_h", "odds_d", "odds_a", "y") and c not in ODDS_FEATS]
ALL_FEATS = ODDS_FEATS + FUND_FEATS
print(f"features cotes: {ODDS_FEATS}")
print(f"features fondamentales ({len(FUND_FEATS)}): {FUND_FEATS}")
nan_share = df[ALL_FEATS].isna().mean().sort_values(ascending=False)
print("part de NaN par feature (top 10):")
print(nan_share.head(10).to_string())


# ============================================================ 4. MODELES + EVAL
def make_logistic():
    return Pipeline([
        ("imp", SimpleImputer(strategy="mean")),
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(max_iter=3000, C=1.0, random_state=RNG)),
    ])


def make_gbm():
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
        random_state=RNG)


def roi_of(picks_outcome, y, odds_mat):
    """picks_outcome: array d'indices 0/1/2 (ou -1 = pas de pick)."""
    mask = picks_outcome >= 0
    n = int(mask.sum())
    if n == 0:
        return n, np.nan, np.nan
    win = (picks_outcome[mask] == y[mask])
    o = odds_mat[mask, picks_outcome[mask]]
    pnl = np.where(win, o - 1.0, -1.0)
    return n, float(win.mean()), float(pnl.mean())


def eval_split(df, frac, do_importance=False):
    print(); print(SEP)
    print(f"SPLIT TEMPOREL {int(frac*100)}/{int((1-frac)*100)}")
    print(SEP)
    cut = int(len(df) * frac)
    tr, te = df.iloc[:cut], df.iloc[cut:]
    print(f"train: {len(tr)} matchs ({tr['ts'].iloc[0][:16]} -> {tr['ts'].iloc[-1][:16]})")
    print(f"OOS  : {len(te)} matchs ({te['ts'].iloc[0][:16]} -> {te['ts'].iloc[-1][:16]})")

    y_tr, y_te = tr["y"].values, te["y"].values
    odds_mat = te[["odds_h", "odds_d", "odds_a"]].values
    p_devig = te[["p_h", "p_d", "p_a"]].values

    out = {"n_oos": len(te)}

    # --- baseline 0 : favori devig brut ---
    fav = p_devig.argmax(axis=1)
    acc_fav = accuracy_score(y_te, fav)
    ll_fav = log_loss(y_te, p_devig, labels=[0, 1, 2])
    n, wr, roi = roi_of(fav, y_te, odds_mat)
    print(f"\n[BASE-fav ] favori devig brut      acc={acc_fav:.4f}  logloss={ll_fav:.4f}  "
          f"ROI(all-in favori)={roi*100:+.2f}% (n={n}, wr={wr:.3f})")
    out["fav"] = dict(acc=acc_fav, ll=ll_fav, roi=roi, wr=wr, n=n)

    models = {}
    specs = [("a_logit_cotes", make_logistic(), ODDS_FEATS),
             ("b_logit_all", make_logistic(), ALL_FEATS),
             ("c_gbm_all", make_gbm(), ALL_FEATS),
             ("d_gbm_sans_cotes", make_gbm(), FUND_FEATS)]
    for name, mdl, feats in specs:
        mdl.fit(tr[feats], y_tr)
        proba = mdl.predict_proba(te[feats])
        pred = proba.argmax(axis=1)
        acc = accuracy_score(y_te, pred)
        ll = log_loss(y_te, proba, labels=[0, 1, 2])
        n, wr, roi = roi_of(pred, y_te, odds_mat)
        print(f"[{name:<16}] acc={acc:.4f}  logloss={ll:.4f}  ROI(pick argmax)={roi*100:+.2f}%")
        models[name] = (mdl, feats, proba)
        out[name] = dict(acc=acc, ll=ll, roi=roi)

    # --- strategies de picks sur (b) et (c) ---
    print("\n--- strategies de picks (OOS) ---")
    for name in ("b_logit_all", "c_gbm_all"):
        proba = models[name][2]
        edge = proba - p_devig
        best = edge.argmax(axis=1)
        for thr in (0.03, 0.05, 0.08):
            picks = np.where(edge[np.arange(len(te)), best] >= thr, best, -1)
            n, wr, roi = roi_of(picks, y_te, odds_mat)
            avg_o = float(odds_mat[picks >= 0, picks[picks >= 0]].mean()) if n else np.nan
            print(f"  [{name}] edge>={thr:.2f}: n={n:>4}  wr={wr if n else float('nan'):.3f}  "
                  f"ROI={roi*100:+.2f}%  cote_moy={avg_o:.2f}" if n else
                  f"  [{name}] edge>={thr:.2f}: n=0")
            out[f"{name}_edge{thr}"] = dict(n=n, wr=wr, roi=roi, avg_o=avg_o if n else np.nan)
        # top-decile de confiance
        conf = proba.max(axis=1)
        thr_conf = np.quantile(conf, 0.9)
        picks = np.where(conf >= thr_conf, proba.argmax(axis=1), -1)
        n, wr, roi = roi_of(picks, y_te, odds_mat)
        avg_o = float(odds_mat[picks >= 0, picks[picks >= 0]].mean()) if n else np.nan
        # baseline sur le MEME sous-ensemble : favori devig
        sub = picks >= 0
        acc_fav_sub = accuracy_score(y_te[sub], fav[sub]) if n else np.nan
        nf, wrf, roif = roi_of(np.where(sub, fav, -1), y_te, odds_mat)
        print(f"  [{name}] top-decile conf (>= {thr_conf:.3f}): n={n}  wr={wr:.3f}  ROI={roi*100:+.2f}%  "
              f"cote_moy={avg_o:.2f} | favori devig sur memes matchs: wr={wrf:.3f} ROI={roif*100:+.2f}%")
        out[f"{name}_top10"] = dict(n=n, wr=wr, roi=roi, fav_wr=wrf, fav_roi=roif, avg_o=avg_o)

    # --- desaccords modele vs favori marche ---
    proba_c = models["c_gbm_all"][2]
    pred_c = proba_c.argmax(axis=1)
    dis = pred_c != fav
    nd = int(dis.sum())
    if nd > 0:
        acc_model_dis = accuracy_score(y_te[dis], pred_c[dis])
        acc_fav_dis = accuracy_score(y_te[dis], fav[dis])
        n, wr, roi = roi_of(np.where(dis, pred_c, -1), y_te, odds_mat)
        print(f"\n  desaccords GBM vs favori marche: n={nd} ({100*nd/len(te):.1f}%) | "
              f"acc GBM={acc_model_dis:.3f} vs acc favori={acc_fav_dis:.3f} | ROI picks GBM={roi*100:+.2f}%")
        out["disagree"] = dict(n=nd, acc_model=acc_model_dis, acc_fav=acc_fav_dis, roi=roi)

    # --- permutation importance (split principal seulement) ---
    if do_importance:
        print("\n--- permutation importance GBM toutes-features (OOS, neg_log_loss, 5 repeats) ---")
        mdl, feats, _ = models["c_gbm_all"]
        r = permutation_importance(mdl, te[feats], y_te, scoring="neg_log_loss",
                                   n_repeats=5, random_state=RNG)
        order = np.argsort(-r.importances_mean)
        for k in order[:18]:
            print(f"  {feats[k]:<16} {r.importances_mean[k]:+.5f} +/- {r.importances_std[k]:.5f}")
        out["importances"] = {feats[k]: float(r.importances_mean[k]) for k in order}

    return out


res70 = eval_split(df, 0.70, do_importance=True)
res50 = eval_split(df, 0.50, do_importance=False)

# ============================================================ 5. VERDICT
print(); print(SEP); print("VERDICT CHIFFRE (split 70/30)"); print(SEP)
base = res70["a_logit_cotes"]
for name in ("b_logit_all", "c_gbm_all", "d_gbm_sans_cotes"):
    m = res70[name]
    d_acc = (m["acc"] - base["acc"]) * 100
    d_ll = m["ll"] - base["ll"]
    print(f"{name:<18} d_acc={d_acc:+.2f}pp  d_logloss={d_ll:+.4f}  "
          f"(seuils: >=+0.5pp ou <=-0.01)")
print(f"favori devig brut: acc={res70['fav']['acc']:.4f}")
print(json.dumps({"res70": {k: v for k, v in res70.items() if k != 'importances'},
                  "res50": res50}, indent=1, default=str)[:1])
print("\nFIN.")
