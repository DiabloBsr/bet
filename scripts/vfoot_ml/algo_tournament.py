"""TOURNOI D'ALGORITHMES PAR MARCHÉ — qui prédit le mieux quoi ?

9 marchés × jusqu'à 12 algos, MÊMES données (split chrono 70/30), MÊME métrique
(log-loss OOS + hit@1). Champion par marché -> data/vfoot_ml/algo_champions.json.

Moteurs EXISTANTS : devig (baseline), grid (Poisson inversé), grid_sim (+déviations),
grid_sim_cal (+calibration 7x7 train-only), v5 (team-strength+HT), ml (LightGBM).
Moteurs NOUVEAUX (jamais utilisés) :
  knn        grille EMPIRIQUE des ~400 matchs du passé aux cotes les plus proches
  bipois     Poisson BIVARIÉ (choc commun lambda3, corrélation vraie ; c fit sur train)
  negbin     binomiale négative sur le total (sur-dispersion fit sur train)
  isotonic   repolissage isotonique des probas dévigées (apprend les micro-gaps)
  mlp        réseau de neurones (sklearn MLP)
  blend_opt  mélange devig x grid_sim_cal au poids OPTIMISÉ sur train
"""
from __future__ import annotations
import json, os, sys, warnings
from pathlib import Path
_R = Path(__file__).resolve().parents[2]
os.chdir(_R)                          # db_url relative -> CWD projet obligatoire
sys.path.insert(0, str(_R))
if sys.stdout is None:                # pythonw / Task Scheduler
    (_R / "data" / "logs").mkdir(parents=True, exist_ok=True)
    _lg = open(_R / "data" / "logs" / "tournament.log", "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = _lg
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import poisson as _poi, nbinom
from sqlalchemy import create_engine, text
from scraper.config import load_settings
from scraper.market_inversion import exact_invert_1x2, apply_sim_deviations
from scraper.predictor_v5 import fit_model_v5, predict_match_v5

ROOT = Path(__file__).resolve().parents[2]
LG = "InstantLeague-8035"
EPS = 1e-6

eng = create_engine(load_settings().db_url)
df = pd.read_sql(text(f"""
    SELECT e.expected_start ts, e.team_a, e.team_b,
           o.odds_home oh, o.odds_draw od, o.odds_away oa, o.extra_markets xm,
           r.score_a sa, r.score_b sb, r.ht_score_a ha, r.ht_score_b hb
    FROM events e
    JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
    JOIN results r ON r.event_id=e.id
    WHERE r.ht_score_a IS NOT NULL AND e.competition='{LG}'
      AND o.odds_home>1 AND o.odds_draw>1 AND o.odds_away>1
    ORDER BY e.expected_start"""), eng)
df = df.drop_duplicates(["ts", "team_a", "team_b"]).reset_index(drop=True)
cut = int(len(df) * 0.7)
N = len(df)
print(f"{N} matchs | train {cut} / test {N-cut}", flush=True)

# ---------- dévig ----------
def parse_xm(raw):
    try: return json.loads(raw) if isinstance(raw, str) else (raw or {})
    except Exception: return {}

def gm(xm, pref):
    for k, v in xm.items():
        if k.replace("\x82", "é").replace("\xe9", "é").startswith(pref):
            return v
    return None

def devig(sels):
    v = {s: float(o) for s, o in (sels or {}).items()
         if isinstance(o, (int, float)) and 1 < o < 99.99}
    ti = sum(1/o for o in v.values())
    return ({s: (1/o)/ti for s, o in v.items()} if ti >= 0.95 else None)

XM = [parse_xm(x) for x in df.xm]

def devig_probs(pref, classes):
    out = np.full((N, len(classes)), np.nan)
    for i, xm in enumerate(XM):
        d = devig(gm(xm, pref))
        if d:
            for j, c in enumerate(classes):
                out[i, j] = d.get(c, np.nan)
    return out

inv3 = 1/df.oh + 1/df.od + 1/df.oa
DVG1 = np.stack([(1/df.oh)/inv3, (1/df.od)/inv3, (1/df.oa)/inv3], axis=1)

# ---------- grilles ----------
print("grilles Poisson / sim / calibrée…", flush=True)
lam_h = np.zeros(N); lam_a = np.zeros(N)
G_base = np.zeros((N, 7, 7)); G_sim = np.zeros((N, 7, 7)); ok = np.zeros(N, bool)
for i, r in enumerate(df.itertuples()):
    try:
        lh, la = exact_invert_1x2(r.oh, r.od, r.oa)
        lam_h[i], lam_a[i] = lh, la
        g = np.outer(_poi.pmf(np.arange(7), lh), _poi.pmf(np.arange(7), la))
        G_base[i] = g / g.sum()
        gs = np.asarray(apply_sim_deviations(lh, la, "cells"), float)[:7, :7]
        G_sim[i] = gs / gs.sum()
        ok[i] = True
    except Exception:
        pass

sa6 = df.sa.clip(0, 6).astype(int).values; sb6 = df.sb.clip(0, 6).astype(int).values
emp = np.zeros((7, 7))
for i in range(cut):
    emp[sa6[i], sb6[i]] += 1
emp /= emp.sum()
CAL = np.clip(emp / np.clip(G_sim[:cut][ok[:cut]].mean(0), 1e-5, None), 0.4, 2.5)
G_cal = G_sim * CAL[None]; G_cal /= G_cal.sum((1, 2), keepdims=True) + 1e-12

# mi-temps (part des buts 1re MT sur train)
share = (df.ha + df.hb).iloc[:cut].sum() / max((df.sa + df.sb).iloc[:cut].sum(), 1)
GH = np.zeros((N, 7, 7)); G2 = np.zeros((N, 7, 7))
for i in range(N):
    if not ok[i]: continue
    g = np.outer(_poi.pmf(np.arange(7), share*lam_h[i]), _poi.pmf(np.arange(7), share*lam_a[i]))
    GH[i] = g / g.sum()
    g2 = np.outer(_poi.pmf(np.arange(7), (1-share)*lam_h[i]), _poi.pmf(np.arange(7), (1-share)*lam_a[i]))
    G2[i] = g2 / g2.sum()
print(f"part 1re mi-temps (train) : {share:.3f}", flush=True)

# ---------- NOUVEAU : Poisson BIVARIÉ (choc commun) ----------
def bipois_grid(c):
    G = np.zeros((N, 7, 7))
    l3 = c * np.minimum(lam_h, lam_a)
    l1 = np.maximum(lam_h - l3, 1e-4); l2 = np.maximum(lam_a - l3, 1e-4)
    ks = np.arange(7)
    for i in range(N):
        if not ok[i]: continue
        p1 = _poi.pmf(ks, l1[i]); p2 = _poi.pmf(ks, l2[i]); p3 = _poi.pmf(ks, l3[i])
        g = np.zeros((7, 7))
        for k in range(7):
            g[k:, k:] += p3[k] * np.outer(p1[:7-k], p2[:7-k])
        G[i] = g / g.sum()
    return G

M1 = np.triu(np.ones((7, 7)), 1).T.astype(bool)  # i>j
MX = np.eye(7, dtype=bool)
def p1x2(G): return np.stack([G[:, M1].sum(1), G[:, MX].sum(1),
                              1 - G[:, M1].sum(1) - G[:, MX].sum(1)], axis=1)
def ll_train(P, y):
    p = np.clip(P[:cut][ok[:cut]], EPS, 1); p /= p.sum(1, keepdims=True)
    return -np.log(p[np.arange(len(p)), y[:cut][ok[:cut]]]).mean()

y_1x2 = np.where(df.sa > df.sb, 0, np.where(df.sa == df.sb, 1, 2)).astype(int)
print("fit bivarié (c)…", flush=True)
best_c, best_ll = 0.0, 1e9
for c in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3):
    ll = ll_train(p1x2(bipois_grid(c)), y_1x2)
    if ll < best_ll: best_c, best_ll = c, ll
G_bp = bipois_grid(best_c)
print(f"  c*={best_c} (logloss train {best_ll:.5f})", flush=True)

# ---------- NOUVEAU : NegBin sur le TOTAL ----------
tot_tr = (df.sa + df.sb).iloc[:cut]
mu, var = tot_tr.mean(), tot_tr.var()
print(f"total train : mean {mu:.3f} var {var:.3f} -> {'SUR' if var>mu else 'SOUS'}-dispersé", flush=True)
NB_OK = var > mu * 1.01
if NB_OK:
    rdisp = mu**2 / (var - mu)
    lt = lam_h + lam_a
    NB_TOT = np.stack([nbinom.pmf(k, rdisp, rdisp/(rdisp+lt)) if k < 6
                       else 1 - nbinom.cdf(5, rdisp, rdisp/(rdisp+lt)) for k in range(7)], axis=1)
    NB_TOT /= NB_TOT.sum(1, keepdims=True) + 1e-12

# ---------- NOUVEAU : kNN EMPIRIQUE (matchs aux cotes proches) ----------
print("kNN empirique…", flush=True)
from scipy.spatial import cKDTree
pts = DVG1[:, :2]                                 # (imp_h, imp_d) suffisent
tree = cKDTree(pts[:cut])
_, KNN = tree.query(pts, k=400)

def knn_probs(y, n_classes, alpha=30.0):
    prior = np.bincount(y[:cut], minlength=n_classes) / cut
    onehot = np.zeros((cut, n_classes)); onehot[np.arange(cut), y[:cut]] = 1
    counts = onehot[KNN].sum(axis=1)              # (N, n_classes)
    return (counts + alpha * prior) / (counts.sum(1, keepdims=True) + alpha)

# ---------- NOUVEAU : isotonic sur dévig ----------
from sklearn.isotonic import IsotonicRegression
def isotonic_probs(DV, y, n_classes):
    out = np.full_like(DV, np.nan)
    tr_ok = ~np.isnan(DV[:cut]).any(1)
    for cl in range(n_classes):
        iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip")
        iso.fit(DV[:cut][tr_ok][:, cl], (y[:cut][tr_ok] == cl).astype(float))
        m = ~np.isnan(DV[:, cl])
        out[m, cl] = iso.predict(DV[m, cl])
    return out / (np.nansum(out, 1, keepdims=True) + 1e-12)

# ---------- ML + MLP ----------
feat = np.column_stack([df.oh, df.od, df.oa, DVG1, lam_h, lam_a, lam_h+lam_a, lam_h-lam_a])
def fit_ml(y, n_classes, kind="lgb"):
    Xtr, ytr = feat[:cut], y[:cut]
    try:
        if kind == "lgb":
            import lightgbm as lgb
            m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, verbose=-1)
        else:
            from sklearn.neural_network import MLPClassifier
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            m = make_pipeline(StandardScaler(),
                              MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=300,
                                            random_state=0))
        m.fit(Xtr, ytr)
        P = m.predict_proba(feat)
    except Exception:
        from sklearn.linear_model import LogisticRegression
        P = LogisticRegression(max_iter=1000).fit(Xtr, ytr).predict_proba(feat)
    return P

# ---------- NOUVEAU : blend optimisé (poids appris sur train) ----------
def blend_opt(DV, P2, y):
    best_w, best = 0.5, 1e9
    m_tr = ~np.isnan(DV[:cut]).any(1)
    for w in np.arange(0, 1.01, 0.1):
        P = w*DV[:cut][m_tr] + (1-w)*P2[:cut][m_tr]
        p = np.clip(P, EPS, 1); p /= p.sum(1, keepdims=True)
        ll = -np.log(p[np.arange(len(p)), y[:cut][m_tr]]).mean()
        if ll < best: best, best_w = ll, w
    return best_w * DV + (1-best_w) * P2, best_w

# ---------- cibles ----------
y_o35 = (df.sa + df.sb > 3.5).astype(int).values
y_o25 = (df.sa + df.sb > 2.5).astype(int).values
y_gng = ((df.sa > 0) & (df.sb > 0)).astype(int).values
y_tot = np.minimum(df.sa + df.sb, 6).astype(int).values
y_ht = np.where(df.ha > df.hb, 0, np.where(df.ha == df.hb, 1, 2)).astype(int)
HTFT_ORDER = ["1/1", "1/X", "1/2", "X/1", "X/X", "X/2", "2/1", "2/X", "2/2"]
_r1 = np.where(df.ha > df.hb, "1", np.where(df.ha == df.hb, "X", "2"))
_r2 = np.where(df.sa > df.sb, "1", np.where(df.sa == df.sb, "X", "2"))
y_htft = np.array([HTFT_ORDER.index(f"{a}/{b}") for a, b in zip(_r1, _r2)])
y_b1h = ((df.ha > 0) & (df.hb > 0)).astype(int).values

# ordre des colonnes binaires = [classe 0 (négatif), classe 1 (positif)] pour
# coller aux cibles y (1 = Over / les-deux-marquent) et à sklearn.predict_proba
DVG = {"OU35": devig_probs("+/-", ["< 3.5", "> 3.5"]),
       "GNG": devig_probs("G/NG", ["Non", "Oui"]),
       "TOTAL7": devig_probs("Total de buts", [str(k) for k in range(7)]),
       "HT1X2": devig_probs("Mi-tps 1X2", ["1", "X", "2"]),
       "HTFT": devig_probs("HT/FT", HTFT_ORDER),
       "BTTS1H": devig_probs("Les deux", ["Non", "Oui"])}

# ---------- V5 ----------
print("fit V5 + prédictions test…", flush=True)
tr = df.iloc[:cut].rename(columns={"sa": "score_a", "sb": "score_b",
                                   "ha": "ht_score_a", "hb": "ht_score_b",
                                   "oh": "odds_home", "od": "odds_draw", "oa": "odds_away"})
m5 = fit_model_v5(tr, ht_history=tr.copy(), engine=eng, form_alpha=0.0)
V5_1, V5_H, V5_HF = (np.full((N, 3), np.nan), np.full((N, 3), np.nan), np.full((N, 9), np.nan))
for i in range(cut, N):
    r = df.iloc[i]
    try:
        p = predict_match_v5(m5, r.team_a, r.team_b, r.oh, r.od, r.oa, extra_markets=r.xm)
        if p.get("p_h_blend"): V5_1[i] = [p["p_h_blend"], p["p_d_blend"], p["p_a_blend"]]
        if p.get("p_h_ht") is not None: V5_H[i] = [p["p_h_ht"], p["p_d_ht"], p["p_a_ht"]]
        hf = p.get("ht_ft_probs") or {}
        v = np.array([hf.get(k, hf.get(k.replace("/", "-"), 0.0)) for k in HTFT_ORDER], float)
        if v.sum() > 0.5: V5_HF[i] = v / v.sum()
    except Exception:
        pass

# ---------- HTFT modèle 2 mi-temps ----------
def htft_halves():
    d1 = np.zeros((N, 13)); d2 = np.zeros((N, 13))
    for a in range(7):
        for b in range(7):
            d1[:, a-b+6] += GH[:, a, b]; d2[:, a-b+6] += G2[:, a, b]
    out = np.zeros((N, 9))
    combos = {(r1, r2): k for k, (r1, r2) in enumerate(
        [(1, 1), (1, 0), (1, -1), (0, 1), (0, 0), (0, -1), (-1, 1), (-1, 0), (-1, -1)])}
    for x in range(13):
        s1 = int(np.sign(x-6))
        for y in range(13):
            out[:, combos[(s1, int(np.sign(x-6+y-6)))]] += d1[:, x]*d2[:, y]
    return out

# ---------- évaluation ----------
def two(p_pos): return np.stack([1-p_pos, p_pos], axis=1)   # classe 1 = positif
M_O35 = np.add.outer(np.arange(7), np.arange(7)) > 3.5
M_O25 = np.add.outer(np.arange(7), np.arange(7)) > 2.5
M_GG = np.outer(np.arange(7) > 0, np.arange(7) > 0)
M_TOT = [np.add.outer(np.arange(7), np.arange(7)) == k if k < 6
         else np.add.outer(np.arange(7), np.arange(7)) >= 6 for k in range(7)]
def pj(G, m): return G[:, m].sum(1)
def pjt(G): return np.stack([pj(G, m) for m in M_TOT], axis=1)

results = {}
def contest(name, y, algos):
    te = np.arange(cut, N)
    valid = te[np.all([~np.isnan(P[te]).any(1) & (P[te].sum(1) > 0.5) for P in algos.values()], 0)]
    rows = {}
    for a, P in algos.items():
        p = np.clip(P[valid], EPS, 1); p /= p.sum(1, keepdims=True)
        rows[a] = {"logloss": round(float(-np.log(p[np.arange(len(valid)), y[valid]]).mean()), 5),
                   "hit1": round(float((p.argmax(1) == y[valid]).mean()), 4)}
    champ = min(rows, key=lambda a: rows[a]["logloss"])
    results[name] = {"n_test": int(len(valid)), "algos": rows, "champion": champ}
    print(f"\n  {name} (n={len(valid)})", flush=True)
    for a, m in sorted(rows.items(), key=lambda kv: kv[1]["logloss"]):
        print(f"    {a:<14} logloss {m['logloss']:.5f}  hit@1 {100*m['hit1']:5.1f}%"
              f"{'  <<< CHAMPION' if a == champ else ''}")

print("\n" + "="*62 + "\n  TOURNOI (log-loss OOS — plus bas = meilleur)\n" + "="*62, flush=True)
c1 = p1x2(G_cal)
bo1, w1 = blend_opt(DVG1, c1, y_1x2)
contest("1X2", y_1x2, {"devig": DVG1, "grid": p1x2(G_base), "grid_sim": p1x2(G_sim),
                       "grid_sim_cal": c1, "bipois": p1x2(G_bp), "v5": V5_1,
                       "knn": knn_probs(y_1x2, 3), "isotonic": isotonic_probs(DVG1, y_1x2, 3),
                       "ml": fit_ml(y_1x2, 3), "mlp": fit_ml(y_1x2, 3, "mlp"),
                       f"blend_opt(w={w1:.1f})": bo1})
co = two(pj(G_cal, M_O35)); dv = DVG["OU35"]
boo, wo = blend_opt(dv, co, y_o35)
contest("OU35", y_o35, {"devig": dv, "grid": two(pj(G_base, M_O35)), "grid_sim": two(pj(G_sim, M_O35)),
                        "grid_sim_cal": co, "bipois": two(pj(G_bp, M_O35)),
                        "knn": knn_probs(y_o35, 2), "isotonic": isotonic_probs(dv, y_o35, 2),
                        "ml": fit_ml(y_o35, 2), "mlp": fit_ml(y_o35, 2, "mlp"),
                        f"blend_opt(w={wo:.1f})": boo}
        | ({"negbin": two(1 - NB_TOT[:, :4].sum(1) + NB_TOT[:, 3]*0)} if NB_OK else {}))
contest("OU25", y_o25, {"grid": two(pj(G_base, M_O25)), "grid_sim": two(pj(G_sim, M_O25)),
                        "grid_sim_cal": two(pj(G_cal, M_O25)), "bipois": two(pj(G_bp, M_O25)),
                        "knn": knn_probs(y_o25, 2), "ml": fit_ml(y_o25, 2),
                        "mlp": fit_ml(y_o25, 2, "mlp")})
cg = two(pj(G_cal, M_GG)); dg = DVG["GNG"]
bog, wg = blend_opt(dg, cg, y_gng)
contest("GNG", y_gng, {"devig": dg, "grid": two(pj(G_base, M_GG)), "grid_sim": two(pj(G_sim, M_GG)),
                       "grid_sim_cal": cg, "bipois": two(pj(G_bp, M_GG)),
                       "knn": knn_probs(y_gng, 2), "isotonic": isotonic_probs(dg, y_gng, 2),
                       "ml": fit_ml(y_gng, 2), "mlp": fit_ml(y_gng, 2, "mlp"),
                       f"blend_opt(w={wg:.1f})": bog})
ct = pjt(G_cal); dt = DVG["TOTAL7"]
bot, wt = blend_opt(dt, ct, y_tot)
tot_algos = {"devig": dt, "grid": pjt(G_base), "grid_sim": pjt(G_sim), "grid_sim_cal": ct,
             "bipois": pjt(G_bp), "knn": knn_probs(y_tot, 7),
             "isotonic": isotonic_probs(dt, y_tot, 7), "ml": fit_ml(y_tot, 7),
             f"blend_opt(w={wt:.1f})": bot}
if NB_OK: tot_algos["negbin"] = NB_TOT
contest("TOTAL7", y_tot, tot_algos)
dh = DVG["HT1X2"]
contest("HT1X2", y_ht, {"devig": dh, "grid_ht": p1x2(GH), "v5": V5_H,
                        "knn": knn_probs(y_ht, 3), "isotonic": isotonic_probs(dh, y_ht, 3),
                        "ml": fit_ml(y_ht, 3), "mlp": fit_ml(y_ht, 3, "mlp")})
dhf = DVG["HTFT"]
contest("HTFT", y_htft, {"devig": dhf, "halves": htft_halves(), "v5": V5_HF,
                         "knn": knn_probs(y_htft, 9), "isotonic": isotonic_probs(dhf, y_htft, 9),
                         "ml": fit_ml(y_htft, 9)})
db = DVG["BTTS1H"]
contest("BTTS1H", y_b1h, {"devig": db, "grid_ht": two(pj(GH, M_GG)),
                          "knn": knn_probs(y_b1h, 2), "isotonic": isotonic_probs(db, y_b1h, 2),
                          "ml": fit_ml(y_b1h, 2)})

out = ROOT / "data" / "vfoot_ml" / "algo_champions.json"
champs = {k: v["champion"] for k, v in results.items()}
# SENTINELLE : une BASCULE de champion = signature d'un changement de RNG/pricing
prev = None
try:
    prev = {k: v["champion"] for k, v in
            json.loads(out.read_text(encoding="utf-8")).items()}
except Exception:
    pass
out.write_text(json.dumps(results, indent=1), encoding="utf-8")
from datetime import datetime, timezone
hist = ROOT / "data" / "vfoot_ml" / "algo_champions_history.jsonl"
with hist.open("a", encoding="utf-8") as f:
    f.write(json.dumps({"run_utc": datetime.now(timezone.utc).isoformat(),
                        "champions": champs}) + "\n")
FLAG = ROOT / "data" / "vfoot_ml" / "champion_switch.flag"
def _norm(c):  # blend_opt(w=0.9) et devig = même régime marché
    return "marche" if (c or "").startswith(("devig", "blend_opt", "grid")) else c
switched = {k: (prev.get(k), champs[k]) for k in champs
            if prev and k in prev and _norm(prev.get(k)) != _norm(champs[k])}
if switched:
    FLAG.write_text(json.dumps({"run_utc": datetime.now(timezone.utc).isoformat(),
                                "switched": {k: f"{a} -> {b}" for k, (a, b) in switched.items()}},
                               ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n*** BASCULE DE CHAMPION detectee : {switched} -> {FLAG.name} ***", flush=True)
elif FLAG.exists():
    FLAG.unlink()
print(f"\n-> {out}", flush=True)
print("CHAMPIONS :", champs, flush=True)
