"""TEST DÉCISIF — les RÉSULTATS PASSÉS ajoutent-ils quoi que ce soit aux cotes ?
On fabrique, sans fuite (strictement antérieur), des features d'historique :
forme N-derniers, buts pour/contre récents, taux Over récent, série (streak),
hot-hand (dernier match), head-to-head. Puis :
 (1) corrélation de chaque feature avec le RÉSIDU (réel - implicite par les cotes) ;
 (2) test OOS décisif : cotes-seules VS cotes+historique (log-loss / RMSE / Brier) ;
 (3) hot-hand / retour à la moyenne explicite sur les totaux.
Si tout ≈ 0 et que cotes+histo ne bat PAS cotes-seules en OOS -> le passé n'aide pas.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sqlalchemy import create_engine
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import log_loss, brier_score_loss, mean_squared_error
from scraper.config import load_settings
from scraper.market_inversion import exact_invert_1x2, devig, _fast_grid, total_distribution
warnings.filterwarnings("ignore")

e = create_engine(load_settings().db_url)
df = pd.read_sql("""SELECT e.team_a,e.team_b,e.expected_start,o.odds_home oh,o.odds_draw od,
  o.odds_away oa, r.score_a sa, r.score_b sb FROM events e
  JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
  JOIN results r ON r.event_id=e.id
  WHERE r.score_a IS NOT NULL AND e.competition='InstantLeague-8035'""", e)
df = df[(df.oh > 1) & (df.od > 1) & (df.oa > 1)].copy()
df["es"] = pd.to_datetime(df.expected_start, utc=True, errors="coerce")
df = df.dropna(subset=["es"]).sort_values("es").reset_index(drop=True)
df["tot"] = df.sa + df.sb
df["home_win"] = (df.sa > df.sb).astype(int)
df["over25"] = (df.tot >= 3).astype(int)
df["btts"] = ((df.sa >= 1) & (df.sb >= 1)).astype(int)
n = len(df)
print(f"n matchs = {n}\n")

# ---- implicites par les cotes (le baseline à battre) ----
inv = 1/df.oh + 1/df.od + 1/df.oa
df["imp_home"] = (1/df.oh)/inv
df["imp_away"] = (1/df.oa)/inv
# lam + over25 implicite : inversion par triplet de cotes UNIQUE (rapide)
uniq = df[["oh", "od", "oa"]].round(2).drop_duplicates()
cache = {}
for r in uniq.itertuples(index=False):
    lh, la = exact_invert_1x2(r.oh, r.od, r.oa)
    g = _fast_grid(lh, la, 0.0)
    td = total_distribution(g)
    cache[(r.oh, r.od, r.oa)] = (lh + la, float(td[3:].sum()))
key = list(zip(df.oh.round(2), df.od.round(2), df.oa.round(2)))
df["lam_tot"] = [cache[k][0] for k in key]
df["imp_over"] = [cache[k][1] for k in key]
print(f"triplets de cotes uniques inversés : {len(uniq)}")

# ---- features d'historique (long format, strictement antérieur) ----
home = df[["es", "team_a", "sa", "sb"]].rename(columns={"team_a": "team", "sa": "gf", "sb": "ga"})
away = df[["es", "team_b", "sb", "sa"]].rename(columns={"team_b": "team", "sb": "gf", "sa": "ga"})
home["idx"] = df.index; away["idx"] = df.index
lng = pd.concat([home, away]).sort_values("es").reset_index(drop=True)
lng["pts"] = np.where(lng.gf > lng.ga, 3, np.where(lng.gf == lng.ga, 1, 0))
lng["tot"] = lng.gf + lng.ga
lng["over"] = (lng.tot >= 3).astype(int)
lng["won"] = (lng.gf > lng.ga).astype(int)
lng["scored"] = (lng.gf >= 1).astype(int)

def roll_prev(s, w):  # moyenne des w matchs PRÉCÉDENTS (shift 1 = pas de fuite)
    return s.shift(1).rolling(w, min_periods=2).mean()

g = lng.groupby("team", sort=False)
lng["form5"] = g.pts.transform(lambda s: roll_prev(s, 5))
lng["gf5"]   = g.gf.transform(lambda s: roll_prev(s, 5))
lng["ga5"]   = g.ga.transform(lambda s: roll_prev(s, 5))
lng["over5"] = g.over.transform(lambda s: roll_prev(s, 5))
lng["win10"] = g.won.transform(lambda s: roll_prev(s, 10))
lng["last_tot"]    = g.tot.transform(lambda s: s.shift(1))
lng["last_scored"] = g.scored.transform(lambda s: s.shift(1))
# streak de victoires AVANT ce match
def winstreak(s):
    out, c = [], 0
    for v in s.shift(1).fillna(0):
        out.append(c); c = c + 1 if v == 1 else 0
    return pd.Series(out, index=s.index)
lng["wstreak"] = g.won.transform(winstreak)

feat_cols = ["form5", "gf5", "ga5", "over5", "win10", "last_tot", "last_scored", "wstreak"]
HF = lng[lng.idx.isin(home.idx)].drop_duplicates("idx").set_index("idx")
# reconstruire home/away features par idx
h_long = lng.iloc[:len(df)] if False else None  # placeholder
# map back : pour chaque match, features du home (team_a) et away (team_b)
lng_h = lng.merge(df[["team_a"]].rename_axis("idx").reset_index(), left_on=["idx", "team"], right_on=["idx", "team_a"])
lng_a = lng.merge(df[["team_b"]].rename_axis("idx").reset_index(), left_on=["idx", "team"], right_on=["idx", "team_b"])
for c in feat_cols:
    df["h_" + c] = lng_h.set_index("idx")[c]
    df["a_" + c] = lng_a.set_index("idx")[c]

# head-to-head : total moyen des rencontres antérieures de la même paire
df["pair"] = df.team_a + "|" + df.team_b
df["h2h_tot"] = df.groupby("pair").tot.transform(lambda s: s.shift(1).expanding().mean())

ALL_FEATS = [f"{s}_{c}" for c in feat_cols for s in ("h", "a")] + ["h2h_tot"]

# ---- résidus (réel - implicite) ----
df["r_home"] = df.home_win - df.imp_home
df["r_tot"]  = df.tot - df.lam_tot
df["r_over"] = df.over25 - df.imp_over

print("\n" + "=" * 78)
print("(1) CORRÉLATION feature d'historique × RÉSIDU (réel - implicite par cote)")
print("    ≈0 partout = le passé n'explique RIEN que les cotes ne contiennent déjà")
print("=" * 78)
print(f"{'feature':<16}{'corr r_home':>13}{'corr r_tot':>12}{'corr r_over':>13}")
for f in ALL_FEATS:
    sub = df[[f, "r_home", "r_tot", "r_over"]].dropna()
    if len(sub) < 500:
        continue
    ch = sub[f].corr(sub.r_home); ct = sub[f].corr(sub.r_tot); co = sub[f].corr(sub.r_over)
    print(f"{f:<16}{ch:>+13.3f}{ct:>+12.3f}{co:>+13.3f}")
print("  (|corr|>0.05 = piste ; au-delà de 0.10 = signal réel)")

# ---- (2) TEST OOS DÉCISIF : cotes-seules VS cotes+historique ----
dfm = df.dropna(subset=ALL_FEATS).reset_index(drop=True)
cut = int(len(dfm) * 0.7); tr, te = dfm.iloc[:cut], dfm.iloc[cut:]
print("\n" + "=" * 78)
print(f"(2) TEST OOS — cotes-seules VS cotes+historique (train {len(tr)} / test {len(te)})")
print("    si 'cotes+histo' ne bat PAS 'cotes-seules' en OOS -> le passé n'aide pas")
print("=" * 78)

def cmp_clf(target, base_col, name):
    Xb_tr, Xb_te = tr[[base_col]].values, te[[base_col]].values
    Xf_tr = tr[[base_col] + ALL_FEATS].values; Xf_te = te[[base_col] + ALL_FEATS].values
    yt, ye = tr[target].values, te[target].values
    A = LogisticRegression(max_iter=2000).fit(Xb_tr, yt)
    B = LogisticRegression(max_iter=2000, C=0.5).fit(Xf_tr, yt)
    pa, pb = A.predict_proba(Xb_te)[:, 1], B.predict_proba(Xf_te)[:, 1]
    print(f"\n  [{name}]  (base rate test {ye.mean()*100:.1f}%)")
    print(f"    cotes-seules : logloss {log_loss(ye,pa):.4f} | brier {brier_score_loss(ye,pa):.4f} | acc {( (pa>0.5).astype(int)==ye ).mean()*100:.1f}%")
    print(f"    cotes+histo  : logloss {log_loss(ye,pb):.4f} | brier {brier_score_loss(ye,pb):.4f} | acc {( (pb>0.5).astype(int)==ye ).mean()*100:.1f}%")
    d = log_loss(ye, pa) - log_loss(ye, pb)
    print(f"    -> {'✅ histo AIDE' if d>0.001 else '❌ histo n_aide pas'} (Δlogloss {d:+.4f}, + = mieux avec histo)")

cmp_clf("home_win", "imp_home", "Victoire à domicile")
cmp_clf("over25", "imp_over", "Over 2.5")

# total (régression) RMSE
yt, ye = tr.tot.values, te.tot.values
A = LinearRegression().fit(tr[["lam_tot"]].values, yt)
B = LinearRegression().fit(tr[["lam_tot"] + ALL_FEATS].values, yt)
ra = mean_squared_error(ye, A.predict(te[["lam_tot"]].values)) ** 0.5
rb = mean_squared_error(ye, B.predict(te[["lam_tot"] + ALL_FEATS].values)) ** 0.5
print(f"\n  [Total de buts]  RMSE cotes-seules {ra:.4f} | cotes+histo {rb:.4f}")
print(f"    -> {'✅ histo AIDE' if (ra-rb)>0.005 else '❌ histo n_aide pas'} (ΔRMSE {ra-rb:+.4f}, + = mieux avec histo)")

# ---- (3) hot-hand / retour à la moyenne explicite ----
print("\n" + "=" * 78)
print("(3) HOT-HAND / retour à la moyenne — le passé immédiat 'lance' une tendance ?")
print("=" * 78)
base_over = df.over25.mean()
m = df.dropna(subset=["h_last_tot"])
hot = m[m.h_last_tot >= 3]; cold = m[m.h_last_tot <= 1]
print(f"  base Over2.5 global : {base_over*100:.1f}%")
print(f"  dom a fait Over au dernier match (last_tot>=3) -> Over now : {hot.over25.mean()*100:.1f}% (n={len(hot)})")
print(f"  dom a fait Under au dernier (last_tot<=1)      -> Over now : {cold.over25.mean()*100:.1f}% (n={len(cold)})")
ws = df.dropna(subset=["h_wstreak"])
for k in [0, 1, 2, 3]:
    s = ws[ws.h_wstreak == k]
    if len(s) > 200:
        print(f"  série dom = {k} victoires -> win réel {s.home_win.mean()*100:.1f}% vs implicite {s.imp_home.mean()*100:.1f}% (n={len(s)})")
print("\n  -> si 'Over now' ≈ base quel que soit le passé, et win≈implicite quelle que soit la série :")
print("     AUCUNE mémoire. Le passé immédiat ne lance aucune tendance (RNG sans mémoire).")
