"""AUDIT FINAL — réfutation de la conclusion négative.
(1) PUISSANCE / MDE par famille  (2) DÉRIVE temporelle (CUSUM + rolling + confirmation)
(4) QUEUES de distribution (favorite-longshot bias, extrêmes).
Manches propres (==10). r_fav = I(favori gagne) - p_fav_devig.
"""
from __future__ import annotations
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy.stats import norm
from sqlalchemy import create_engine
from scraper.config import load_settings
from scraper.market_inversion import parse_extra_markets, total_buts_odds, devig_market, _get_market, _to_float

e = create_engine(load_settings().db_url)
raw = pd.read_sql("""SELECT e.competition comp, e.expected_start, e.id ev,
  o.odds_home oh, o.odds_draw od, o.odds_away oa, o.extra_markets em,
  r.score_a sa, r.score_b sb FROM events e
  JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
  JOIN results r ON r.event_id=e.id
  WHERE r.score_a IS NOT NULL AND e.competition LIKE 'InstantLeague-%'""", e)
raw = raw[(raw.oh > 1) & (raw.od > 1) & (raw.oa > 1)].copy()
raw["es"] = pd.to_datetime(raw.expected_start, utc=True, errors="coerce"); raw = raw.dropna(subset=["es"])
raw["rsize"] = raw.groupby(["comp", "es"]).ev.transform("size")
d = raw[raw.rsize == 10].sort_values(["es", "comp", "ev"]).reset_index(drop=True)
inv = 1/d.oh + 1/d.od + 1/d.oa
d["imp_home"] = (1/d.oh)/inv; d["imp_draw"] = (1/d.od)/inv; d["imp_away"] = (1/d.oa)/inv
d["fav_home"] = d.imp_home > d.imp_away
d["p_fav"] = d[["imp_home", "imp_away"]].max(axis=1); d["p_dog"] = d[["imp_home", "imp_away"]].min(axis=1)
d["tot"] = d.sa + d.sb
d["fav_won"] = np.where(d.fav_home, d.sa > d.sb, d.sb > d.sa).astype(float)
d["over25"] = (d.tot >= 3).astype(float)
d["r_fav"] = d.fav_won - d.p_fav
d["league"] = d.comp.str.replace("InstantLeague-", "", regex=False)
# over ladder
io = []
for r in d.itertuples():
    em = parse_extra_markets(r.em); tb = total_buts_odds(em)
    q = devig_market(tb) if len(tb) >= 4 else {}
    io.append(sum(v for k, v in q.items() if k.isdigit() and int(k) >= 3) if q else np.nan)
d["imp_over"] = io; d["r_over"] = d.over25 - d.imp_over
n = len(d); print(f"manches propres : n={n}")

# ============ (1) PUISSANCE / MDE ============
print("\n" + "="*86)
print("(1) PUISSANCE — taille d'effet minimale détectable (MDE) par famille")
print("    MDE_pp = (z_alpha + z_power) * sigma / sqrt(n) * 100")
print("="*86)
sig_fav = d.r_fav.std()
print(f"  sigma(r_fav) observé = {sig_fav:.3f}  (marge book ~6% sur 1X2 -> edge exploitable si > ~3-4 pp)")
fam = [
    ("A cotes (5 bins)", 5, n*0.6/5),
    ("B transition (~22 cells)", 22, 1000),
    ("C surprise x cote (~12)", 12, 1300),
    ("D position (10 slots)", 10, n*0.6/10),
    ("E ligue x cote (~45)", 45, 600),
    ("F runs / bandes", 6, 2400),
]
def mde(ncell, ntyp, k_family, sigma, power):
    za_unc = norm.ppf(1-0.05/2)
    za_bon = norm.ppf(1-0.05/(2*k_family))
    zb = norm.ppf(power)
    return ((za_unc+zb)*sigma/math.sqrt(ntyp)*100, (za_bon+zb)*sigma/math.sqrt(ntyp)*100)
print(f"\n  {'famille':<26}{'n/cell':>8}{'MDE80 brut':>12}{'MDE80 corr':>12}{'MDE90 corr':>12}")
for name, k, ntyp in fam:
    m80u, m80c = mde(k, ntyp, k, sig_fav, 0.80)
    _, m90c = mde(k, ntyp, k, sig_fav, 0.90)
    print(f"  {name:<26}{int(ntyp):>8}{m80u:>11.1f}{m80c:>11.1f}{m90c:>11.1f}  pp")
print("\n  Lecture : MDE80 corr = plus petit edge qu'on aurait vu (puissance 80%, alpha Bonferroni).")
print("  => tout edge SUPÉRIEUR à cette valeur aurait été détecté ; en-dessous = invisible.")

# ============ (2) DÉRIVE TEMPORELLE ============
print("\n" + "="*86)
print("(2) DÉRIVE — CUSUM + rolling windows (calibration r_fav) + confirmation")
print("="*86)
def cusum_stat(res):
    r = res.values - res.mean(); S = np.cumsum(r); sd = res.std()
    # bridge : max|S_k - (k/n)S_n| / (sd*sqrt(n))
    k = np.arange(1, len(r)+1); bridge = S - (k/len(r))*S[-1]
    return float(np.max(np.abs(bridge))/(sd*math.sqrt(len(r)))) if sd > 0 else 0.0
cs = cusum_stat(d.r_fav)
print(f"  CUSUM standardisé (global r_fav) = {cs:.3f}  | seuil 5% (pont brownien) ~1.36  -> {'RUPTURE' if cs>1.36 else 'pas de rupture'}")
# rolling non-chevauchant W=1500, z par fenêtre + confirmation par la suivante
W = 1500; flags = []
zs = []
for i in range(0, n - W, W):
    w = d.r_fav.iloc[i:i+W]; z = w.mean()/(w.std()/math.sqrt(len(w)))
    zs.append((i//W, z, w.mean()))
zs = pd.DataFrame(zs, columns=["win", "z", "m"])
kwin = len(zs); bonf = norm.ppf(1-0.05/(2*kwin))
print(f"  {kwin} fenêtres de {W} matchs | seuil Bonferroni |z|>{bonf:.2f}")
print(f"    max|z| fenêtre = {zs.z.abs().max():.2f}  -> {'fenêtre suspecte' if zs.z.abs().max()>bonf else 'aucune fenêtre hors normale'}")
# confirmation : une fenêtre flaggée est-elle confirmée (même signe |z|>1) par la suivante ?
conf = 0
for i in range(len(zs)-1):
    if abs(zs.z.iloc[i]) > bonf and np.sign(zs.z.iloc[i]) == np.sign(zs.z.iloc[i+1]) and abs(zs.z.iloc[i+1]) > 1:
        conf += 1
print(f"    fenêtres suspectes confirmées par la suivante : {conf}")
# par ligue : CUSUM
print("  CUSUM par ligue (r_fav) :")
for lg, g in d.groupby("league"):
    if len(g) > 1500:
        print(f"    {lg}: CUSUM={cusum_stat(g.r_fav):.2f} (n={len(g)}) {'<- RUPTURE' if cusum_stat(g.r_fav)>1.36 else ''}")
# dérive sur r_over
ov = d.dropna(subset=["r_over"]).reset_index(drop=True)
print(f"  CUSUM r_over (global) = {cusum_stat(ov.r_over):.2f} (n={len(ov)}) {'<- RUPTURE' if cusum_stat(ov.r_over)>1.36 else ''}")

# ============ (4) QUEUES / FAVORITE-LONGSHOT BIAS ============
print("\n" + "="*86)
print("(4) QUEUES — favorite-longshot bias & extrêmes (réalisé vs implicite, IC95, OOS)")
print("="*86)
cut = int(n*0.7); tr, te = d.iloc[:cut], d.iloc[cut:]
print("  Calibration du FAVORI par bande fine de p_fav (réel - implicite) :")
print(f"  {'bande p_fav':<14}{'n':>7}{'réel%':>8}{'impl%':>8}{'résidu pp':>11}{'IC95 pp':>16}{'test pp':>9}")
edges = [.40, .45, .50, .55, .60, .65, .70, .75, .80, .85, 1.01]
for lo, hi in zip(edges[:-1], edges[1:]):
    s = d[(d.p_fav >= lo) & (d.p_fav < hi)]
    if len(s) < 150: continue
    rr = (s.fav_won.mean() - s.p_fav.mean())*100
    se = s.r_fav.std()/math.sqrt(len(s))*100
    st = te[(te.p_fav >= lo) & (te.p_fav < hi)]
    rt = (st.fav_won.mean() - st.p_fav.mean())*100 if len(st) > 50 else float("nan")
    print(f"  [{lo:.2f},{hi:.2f})  {len(s):>7}{s.fav_won.mean()*100:>7.1f}{s.p_fav.mean()*100:>8.1f}{rr:>+11.1f}{('['+format(rr-1.96*se,'+.1f')+','+format(rr+1.96*se,'+.1f')+']'):>16}{rt:>+9.1f}")
print("  Outsider extrême (p_dog très bas = très grosse cote) :")
for lo, hi in [(0.0, 0.08), (0.08, 0.12), (0.12, 0.18)]:
    s = d[(d.p_dog >= lo) & (d.p_dog < hi)]
    if len(s) < 150: continue
    dog_won = (1 - d.loc[s.index].fav_won) - 0  # dog gagne quand favori ne gagne pas... approx: use draw-excluded
    # réalisé: l'outsider gagne ?
    s2 = s.copy(); s2["dog_won"] = np.where(s2.fav_home, s2.sa < s2.sb, s2.sb < s2.sa).astype(float)
    rr = (s2.dog_won.mean() - s2.p_dog.mean())*100; se = (s2.dog_won - s2.p_dog).std()/math.sqrt(len(s2))*100
    print(f"  p_dog[{lo:.2f},{hi:.2f}) n={len(s2):>5} dog réel={s2.dog_won.mean()*100:.1f}% impl={s2.p_dog.mean()*100:.1f}% résidu={rr:+.1f}pp IC95=[{rr-1.96*se:+.1f},{rr+1.96*se:+.1f}]")
print("  Matchs très équilibrés (p_fav < 0.42) — taux de nul :")
s = d[d.p_fav < 0.42]
if len(s) > 150:
    rr = (s.fav_won.mean()-s.p_fav.mean())*100; se = s.r_fav.std()/math.sqrt(len(s))*100
    print(f"    n={len(s)} fav réel={s.fav_won.mean()*100:.1f}% impl={s.p_fav.mean()*100:.1f}% résidu={rr:+.1f}pp IC95=[{rr-1.96*se:+.1f},{rr+1.96*se:+.1f}]")
print("\n  -> si tous les résidus ~0 avec IC95 traversant 0 et test cohérent : pas de biais de queue.")
