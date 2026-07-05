"""TESTEUR D'IDÉES FOLLES — hypothèses de RETOUR À LA MOYENNE / martingale du parieur.

Chaque idée = "après une série X, l'issue Y devient plus/moins probable que ce que
disent les cotes". On teste par LOTS, avec la discipline anti-mirage :
  résidu = taux RÉEL - proba IMPLICITE (dévigée) ; split chrono 70/30 ;
  cellule retenue si TRAIN n>=150 & |résidu|>=0.02 ; survit si TEST même signe +
  binomtest + BH-FDR sur toutes les cellules du lot ; + compte des faux positifs attendus.

Idées de l'utilisateur incluses :
  - pas de nul depuis k -> le nul est-il "dû" ?
  - série de victoires -> nul/défaite plus probable ? (reversion)
  - série de défaites -> victoire "due" ?
  - a marqué >=3 en série -> marque-t-il MOINS ce match ?
  - série de matchs pauvres (<=1 but) -> explosion ?  + BTTS, etc.
"""
from __future__ import annotations
import json, sys
from math import lgamma, log, exp
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np, pandas as pd
from sqlalchemy import create_engine, text
from scraper.config import load_settings

LG = "InstantLeague-8035"
eng = create_engine(load_settings().db_url)
df = pd.read_sql(text(f"""
    SELECT e.expected_start ts, e.team_a, e.team_b, o.odds_home oh, o.odds_draw od,
           o.odds_away oa, o.extra_markets xm, r.score_a sa, r.score_b sb
    FROM events e JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
    JOIN results r ON r.event_id=e.id
    WHERE r.score_a IS NOT NULL AND e.competition='{LG}' AND o.odds_home>1 AND o.odds_draw>1 AND o.odds_away>1
    ORDER BY e.expected_start"""), eng).drop_duplicates(["ts", "team_a", "team_b"]).reset_index(drop=True)
print(f"{len(df)} matchs", flush=True)

# implicites dévigés
inv = 1/df.oh + 1/df.od + 1/df.oa
df["imp_h"], df["imp_d"], df["imp_a"] = (1/df.oh)/inv, (1/df.od)/inv, (1/df.oa)/inv
df["total"] = df.sa + df.sb
df["over25"] = (df.total > 2.5).astype(int)
df["btts"] = ((df.sa > 0) & (df.sb > 0)).astype(int)

# P(over2.5) et P(btts) implicites depuis les marchés
def gm(xm, pref):
    for k, v in (xm or {}).items():
        if k.replace("\x82", "é").replace("\xe9", "é").startswith(pref):
            return v
    return None
def devig(sels, keys):
    v = {k: 1/sels[k] for k in keys if isinstance((sels or {}).get(k), (int, float)) and 1 < sels[k] < 99.99}
    s = sum(v.values()); return {k: v[k]/s for k in v} if s and len(v) == len(keys) else None
imp_o25, imp_btts = [], []
for raw in df.xm:
    try: xm = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except Exception: xm = {}
    tt = devig(gm(xm, "Total de buts"), [str(k) for k in range(7)])
    imp_o25.append(sum(tt[str(k)] for k in range(3, 7)) if tt else np.nan)
    gg = devig(gm(xm, "G/NG"), ["Oui", "Non"])
    imp_btts.append(gg["Oui"] if gg else np.nan)
df["imp_o25"], df["imp_btts"] = imp_o25, imp_btts

# perspective ÉQUIPE (2 lignes/match) avec séries
rows = []
for side in ("H", "A"):
    t = pd.DataFrame({
        "ts": df.ts, "team": df.team_a if side == "H" else df.team_b,
        "res": np.where(df.sa == df.sb, "D", np.where((df.sa > df.sb) == (side == "H"), "W", "L")),
        "gf": df.sa if side == "H" else df.sb,
        "win": ((df.sa > df.sb) if side == "H" else (df.sb > df.sa)).astype(int),
        "draw": (df.sa == df.sb).astype(int),
        "imp_win": df.imp_h if side == "H" else df.imp_a, "imp_draw": df.imp_d,
        "over25": df.over25, "imp_o25": df.imp_o25, "btts": df.btts, "imp_btts": df.imp_btts})
    rows.append(t)
L = pd.concat(rows).sort_values(["team", "ts"]).reset_index(drop=True)
g = L.groupby("team")

def streak_len(cond_series):
    """longueur de la série de True se terminant JUSTE AVANT chaque ligne (shift 1)."""
    out = np.zeros(len(cond_series), int); c = 0
    vals = cond_series.values
    for i in range(len(vals)):
        out[i] = c
        c = c + 1 if vals[i] else 0
    return out

L["win_streak"] = g["res"].transform(lambda s: streak_len(s == "W"))
L["loss_streak"] = g["res"].transform(lambda s: streak_len(s == "L"))
L["nodraw_streak"] = g["res"].transform(lambda s: streak_len(s != "D"))
L["unbeaten_streak"] = g["res"].transform(lambda s: streak_len(s != "L"))
L["winless_streak"] = g["res"].transform(lambda s: streak_len(s != "W"))
L["hi_score_streak"] = g["gf"].transform(lambda s: streak_len(s >= 3))
L["lo_score_streak"] = g["gf"].transform(lambda s: streak_len(s <= 1))
L["over_streak"] = g["over25"].transform(lambda s: streak_len(s == 1))
L["under_streak"] = g["over25"].transform(lambda s: streak_len(s == 0))
L["btts_streak"] = g["btts"].transform(lambda s: streak_len(s == 1))
L = L.dropna(subset=["imp_o25", "imp_btts"]).reset_index(drop=True)
cut = L.ts.iloc[len(L)//2]

def binom_p(k, n, p):
    """binomtest bilatéral exact (log-gamma), sans scipy."""
    if n == 0: return 1.0
    def lpmf(x): return lgamma(n+1)-lgamma(x+1)-lgamma(n-x+1)+x*log(max(p,1e-12))+(n-x)*log(max(1-p,1e-12))
    obs = lpmf(k); tot = 0.0
    for x in range(n+1):
        if lpmf(x) <= obs + 1e-9: tot += exp(lpmf(x))
    return min(1.0, tot)

def test_lot(name, cells):
    """cells = list of (label, mask, outcome_col, imp_col). Applique la discipline."""
    tr = L[L.ts < cut]; te = L[L.ts >= cut]
    cands = []
    for label, mfn, outc, impc in cells:
        mtr = mfn(tr)
        if mtr.sum() < 150: continue
        resid = tr[outc][mtr].mean() - tr[impc][mtr].mean()
        if abs(resid) >= 0.02:
            cands.append((label, mfn, outc, impc, resid))
    results = []
    for label, mfn, outc, impc, resid_tr in cands:
        mte = mfn(te)
        if mte.sum() < 30: continue
        rt = te[outc][mte].mean() - te[impc][mte].mean()
        k, n = int(te[outc][mte].sum()), int(mte.sum())
        p = binom_p(k, n, te[impc][mte].mean())
        results.append({"lot": name, "cell": label, "n_tr": int(mfn(tr).sum()),
                        "resid_tr": round(resid_tr, 3), "resid_te": round(rt, 3),
                        "same_sign": np.sign(resid_tr) == np.sign(rt), "p": p, "n_te": n})
    return results, len(cands)

# ---- définition des lots (idées folles) ----
def band(col, lo, hi=99): return lambda d, c=col, l=lo, h=hi: (d[c] >= l) & (d[c] < h)
LOTS = {
 "nul_du (pas de nul depuis k -> nul?)": [
    (f"nodraw>={k}", band("nodraw_streak", k), "draw", "imp_draw") for k in (2,3,4,5,6)],
 "victoires_serie (W streak -> reversion?)": [
    (f"win>={k} -> pas victoire", band("win_streak", k), "win", "imp_win") for k in (2,3,4)] +
    [(f"win>={k} -> nul", band("win_streak", k), "draw", "imp_draw") for k in (2,3,4)],
 "defaites_serie (L streak -> victoire due?)": [
    (f"loss>={k} -> victoire", band("loss_streak", k), "win", "imp_win") for k in (2,3,4)],
 "buts_hauts (>=3 en serie -> marque moins?)": [
    (f"hi_score>={k} -> over2.5", band("hi_score_streak", k), "over25", "imp_o25") for k in (2,3)] +
    [(f"over>={k} -> over2.5", band("over_streak", k), "over25", "imp_o25") for k in (2,3,4)],
 "buts_bas (<=1 en serie -> explosion?)": [
    (f"lo_score>={k} -> over2.5", band("lo_score_streak", k), "over25", "imp_o25") for k in (2,3)] +
    [(f"under>={k} -> over2.5", band("under_streak", k), "over25", "imp_o25") for k in (2,3,4)],
 "invaincu/sans_victoire": [
    (f"unbeaten>={k} -> victoire", band("unbeaten_streak", k), "win", "imp_win") for k in (3,4,5)] +
    [(f"winless>={k} -> victoire", band("winless_streak", k), "win", "imp_win") for k in (3,4,5)],
 "btts_serie": [
    (f"btts>={k} -> btts", band("btts_streak", k), "btts", "imp_btts") for k in (2,3,4)],
}

all_res, total_cells = [], 0
for name, cells in LOTS.items():
    res, nc = test_lot(name, cells); all_res += res; total_cells += nc
    print(f"\n=== LOT: {name} ({nc} cellules testées) ===", flush=True)
    for r in sorted(res, key=lambda x: x["p"]):
        print(f"   {r['cell']:<26} resid tr {r['resid_tr']:+.3f} te {r['resid_te']:+.3f} "
              f"| p={r['p']:.3f} | {'même signe' if r['same_sign'] else 'INVERSÉ'} (n_te={r['n_te']})")

# BH-FDR global sur toutes les cellules testées OOS
ps = sorted([r["p"] for r in all_res])
m = len(ps)
surv = 0
if m:
    crit = [(i+1)/m*0.05 for i in range(m)]
    kmax = max([i for i in range(m) if ps[i] <= crit[i]], default=-1)
    thr = ps[kmax] if kmax >= 0 else 0
    surv = sum(1 for r in all_res if r["p"] <= thr and r["same_sign"])
print("\n" + "="*60)
print(f"  BILAN : {total_cells} cellules candidates TRAIN, {m} testées OOS")
print(f"  Faux positifs attendus par HASARD (5%) : ~{0.05*m:.0f}")
print(f"  SURVIVANTS OOS + BH-FDR + même signe : {surv}")
print("  -> " + ("⚠️ un survivant ! à vérifier en adversarial (probablement bruit)"
                 if surv else "AUCUN. Le 'retour à la moyenne' est un mirage : le RNG n'a pas de mémoire."))
print("="*60)
