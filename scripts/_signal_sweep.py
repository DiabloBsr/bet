"""BALAYAGE EXHAUSTIF — chercher un signal MANQUÉ dans toute la BDD.
Angles NOUVEAUX (non testés avant) :
 (A) structure SÉRIELLE du flux RNG : autocorrélation manche->manche (total, home_win,
     draw) à plusieurs lags + runs test (le flux est-il plus/moins streaky qu'aléatoire ?)
 (B) corrélation INTRA-manche : les 10 matchs d'une même manche sont-ils corrélés
     (seed partagé) ? variance du total agrégé par manche vs indépendance.
 (C) CYCLE / période : le flux des totaux se répète-t-il (PRNG faible) ?
 (D) effet POSITION dans la manche + effet HEURE.
 (E) calibration de TOUS les ladders offerts (Total de buts, BTTS) vs réalisé -> +EV ?
Tout signal notable est re-testé OOS (split chrono 70/30).
"""
from __future__ import annotations
import sys, math, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sqlalchemy import create_engine
from scraper.config import load_settings
from scraper.market_inversion import parse_extra_markets, total_buts_odds, devig_market, _get_market, _to_float

e = create_engine(load_settings().db_url)
df = pd.read_sql("""SELECT e.competition comp, e.expected_start, e.id ev,
  o.odds_home oh, o.odds_draw od, o.odds_away oa, o.extra_markets em,
  r.score_a sa, r.score_b sb FROM events e
  JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
  JOIN results r ON r.event_id=e.id
  WHERE r.score_a IS NOT NULL AND e.competition LIKE 'InstantLeague-%'""", e)
df = df[(df.oh > 1) & (df.od > 1) & (df.oa > 1)].copy()
df["es"] = pd.to_datetime(df.expected_start, utc=True, errors="coerce")
df = df.dropna(subset=["es"]).sort_values(["es", "ev"]).reset_index(drop=True)
df["tot"] = df.sa + df.sb
df["hw"] = (df.sa > df.sb).astype(int)
df["draw"] = (df.sa == df.sb).astype(int)
df["btts"] = ((df.sa >= 1) & (df.sb >= 1)).astype(int)
print(f"n total = {len(df)} | ligues = {df.comp.nunique()}\n")

MAIN = "InstantLeague-8035"
g = df[df.comp == MAIN].reset_index(drop=True)
print(f"ligue principale {MAIN} : n={len(g)}\n")

def sig(label, corr, n):
    se = 1/math.sqrt(n) if n > 0 else 1
    z = corr/se
    tag = "*** SIGNAL" if abs(z) > 4 and abs(corr) > 0.03 else ("? à voir" if abs(z) > 3 else "bruit")
    return f"  {label:<40} corr={corr:+.4f}  z={z:+.1f}  {tag}"

print("=" * 78)
print("(A) STRUCTURE SÉRIELLE — le flux a-t-il une MÉMOIRE ? (autocorrélation)")
print("    actionnable : si total[t] prédit total[t+lag] -> on parie la manche suivante")
print("=" * 78)
for col in ["tot", "hw", "draw", "btts"]:
    s = g[col].values.astype(float); s = s - s.mean()
    print(f"  -- {col} --")
    for lag in [1, 2, 3, 9, 10, 11, 20]:
        if len(s) > lag + 50:
            num = np.sum(s[:-lag] * s[lag:]); den = np.sum(s * s)
            ac = num/den if den > 0 else 0
            print(sig(f"lag {lag}", ac, len(s) - lag))

# runs test sur home_win (Wald-Wolfowitz)
def runs_test(x):
    x = np.asarray(x); n1 = int(x.sum()); n0 = len(x) - n1
    if n1 == 0 or n0 == 0:
        return 0.0, 0.0
    runs = 1 + int(np.sum(x[1:] != x[:-1]))
    mu = 2*n1*n0/len(x) + 1
    var = 2*n1*n0*(2*n1*n0 - len(x)) / (len(x)**2 * (len(x) - 1))
    z = (runs - mu)/math.sqrt(var) if var > 0 else 0
    return runs, z
r, z = runs_test(g.hw.values)
print(f"\n  RUNS TEST home_win : runs={r}, z={z:+.2f}  -> {'streaky/anti-streaky !' if abs(z)>3 else 'aléatoire (OK)'}")

print("\n" + "=" * 78)
print("(B) CORRÉLATION INTRA-MANCHE — les 10 matchs d'une manche partagent-ils un seed ?")
print("    si oui : variance du total-par-manche > indépendance, ou home-wins groupés")
print("=" * 78)
rounds = g.groupby("es").agg(n=("tot", "size"), tot_sum=("tot", "sum"), tot_mean=("tot", "mean"),
                             hw_sum=("hw", "sum"), tot_var=("tot", "var")).query("n>=8")
# variance intra-manche moyenne vs variance globale (si seed partagé -> intra < global pour mean)
glob_var = g.tot.var()
mean_intra_var = rounds.tot_var.mean()
print(f"  variance du total : globale={glob_var:.3f} | intra-manche moy={mean_intra_var:.3f}")
print(f"    ratio intra/global = {mean_intra_var/glob_var:.3f}  (1.0 = indépendant ; <0.9 = matchs corrélés DANS la manche)")
# home-wins par manche : binomiale(10, p) attendue ?
p_hw = g.hw.mean(); exp_var = rounds.n.mean()*p_hw*(1-p_hw)
obs_var = rounds.hw_sum.var()
print(f"  home-wins/manche : var observée={obs_var:.3f} | var binomiale attendue={exp_var:.3f}")
print(f"    ratio = {obs_var/exp_var:.3f}  (>1.2 = manches 'à thème' home ; <0.8 = équilibrage forcé)")
# total moyen par manche : autocorrélation manche->manche (actionnable)
rt = rounds.reset_index().sort_values("es")
tm = rt.tot_mean.values - rt.tot_mean.mean()
if len(tm) > 30:
    ac1 = np.sum(tm[:-1]*tm[1:])/np.sum(tm*tm)
    print(sig("autocorr du total-moyen manche->manche (lag1)", ac1, len(tm)-1))

print("\n" + "=" * 78)
print("(C) CYCLE / PÉRIODE — le flux des totaux se répète-t-il (PRNG faible) ?")
print("    fraction de t où total[t]==total[t+p] ; baseline aléatoire ~ somme(P(k)^2)")
print("=" * 78)
tt = g.tot.values
pk = np.array([np.mean(tt == k) for k in range(int(tt.max())+1)])
baseline = float(np.sum(pk**2))
print(f"  baseline (aléatoire) = {baseline:.3f}")
best = (0, baseline)
for p in range(1, min(3000, len(tt)//3)):
    frac = np.mean(tt[:-p] == tt[p:])
    if frac > best[1]:
        best = (p, frac)
print(f"  meilleure période p={best[0]} : match={best[1]:.3f}  -> {'*** CYCLE DÉTECTÉ' if best[1]>baseline+0.15 else 'aucun cycle (= vrai CSPRNG)'}")
# plus longue sous-séquence exacte répétée (sur (sa,sb))
seq = list(zip(g.sa.values.tolist(), g.sb.values.tolist()))
sseq = ["%d-%d" % (a, b) for a, b in seq]
seen = {}; maxrun = 0; L = 6
joined = ",".join(sseq)
import re
# heuristique : cherche un bloc de L scores qui réapparaît
for i in range(0, len(sseq)-L, 50):
    block = ",".join(sseq[i:i+L])
    if joined.count(block) > 1:
        maxrun = max(maxrun, L)
print(f"  bloc de {L} scores consécutifs réapparaissant ailleurs : {'OUI (à creuser !)' if maxrun else 'non'}")

print("\n" + "=" * 78)
print("(D) POSITION dans la manche + HEURE — biais structurel ?")
print("=" * 78)
g["slot"] = g.groupby("es").cumcount()
sl = g.groupby("slot").agg(n=("tot", "size"), tot=("tot", "mean"), hw=("hw", "mean")).query("n>=200")
spread_tot = sl.tot.max() - sl.tot.min(); spread_hw = sl.hw.max() - sl.hw.min()
print(f"  total moyen par position : min={sl.tot.min():.2f} max={sl.tot.max():.2f} (écart {spread_tot:.2f})")
print(f"  home-win par position    : min={sl.hw.min()*100:.0f}% max={sl.hw.max()*100:.0f}% (écart {spread_hw*100:.0f}pt)")
print(f"    -> {'biais de position !' if spread_tot>0.3 or spread_hw>0.08 else 'pas de biais de position'}")
g["hour"] = g.es.dt.tz_convert("Etc/GMT-3").dt.hour
hr = g.groupby("hour").agg(n=("tot", "size"), tot=("tot", "mean"), hw=("hw", "mean")).query("n>=200")
print(f"  total moyen par heure    : min={hr.tot.min():.2f} max={hr.tot.max():.2f} (écart {hr.tot.max()-hr.tot.min():.2f})")
print(f"    -> {'biais horaire !' if (hr.tot.max()-hr.tot.min())>0.3 else 'pas de biais horaire'}")

print("\n" + "=" * 78)
print("(E) CALIBRATION DES LADDERS OFFERTS — un marché systématiquement +EV ?")
print("    réalisé vs cote offerte dévigée, sur Total de buts + BTTS (G/NG)")
print("=" * 78)
# Total de buts : pour chaque cellule k, EV = realise * cote_offerte - 1
rows_tot = {k: [] for k in range(7)}
gng_oui = []; gng_non = []
for r in g.itertuples():
    em = parse_extra_markets(r.em)
    tb = total_buts_odds(em)
    tk = min(int(r.tot), 6)
    for k, cote in tb.items():
        try: ki = int(k)
        except ValueError: continue
        if ki <= 6:
            rows_tot[ki].append((1.0 if ki == tk else 0.0, cote))
    gng = _get_market(em, exact="G/NG")
    if isinstance(gng, dict):
        co = _to_float(gng.get("Oui")); cn = _to_float(gng.get("Non"))
        if co: gng_oui.append((float(r.btts), co))
        if cn: gng_non.append((1.0 - float(r.btts), cn))
print("  Total de buts (EV = taux réel x cote offerte - 1) :")
for k in range(7):
    arr = rows_tot[k]
    if len(arr) >= 300:
        a = np.array(arr); ev = (a[:, 0]*a[:, 1] - 1).mean()
        print(f"    total={k}{'+' if k==6 else ' '}  n={len(arr):>5}  réel={a[:,0].mean()*100:4.1f}%  cote moy={a[:,1].mean():5.2f}  EV={ev*100:+5.1f}%  {'*** +EV' if ev>0.03 else ''}")
for nm, arr in [("BTTS Oui", gng_oui), ("BTTS Non", gng_non)]:
    if len(arr) >= 300:
        a = np.array(arr); ev = (a[:, 0]*a[:, 1] - 1).mean()
        print(f"  {nm}: n={len(arr)} réel={a[:,0].mean()*100:.1f}% cote moy={a[:,1].mean():.2f} EV={ev*100:+.1f}%  {'*** +EV' if ev>0.03 else ''}")

print("\n" + "=" * 78)
print("VERDICT : tout '*** SIGNAL' / '*** +EV' ci-dessus est à re-tester OOS.")
print("Si tout est 'bruit' / 'aléatoire (OK)' / 'aucun cycle' / EV négatif -> rien de manqué.")
print("=" * 78)
