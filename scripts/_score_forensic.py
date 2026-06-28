"""FORENSIC SCORES EXACTS — la distribution fine est-elle Poisson ? exploitable ?
(1) distribution brute vs Poisson indép. (calibré sur lambdas implicites 1X2) + chi2
(2) mécanique RNG : sur-dispersion, corr H-A résiduelle, Dixon-Coles, Skellam, parité, troncature, grille lambda
(3) par tranche de cote favori : chi2 + scores déviants
(4) DÉCISIF : calibration réalisé vs COTES SCORE EXACT offertes -> EV + BH-FDR + OOS
(5) EV/Kelly. Manches: toutes 8035 (l'historique sert à la distribution).
"""
from __future__ import annotations
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy.stats import poisson, skellam, norm, chi2 as chi2dist
from sqlalchemy import create_engine
from scraper.config import load_settings
from scraper.market_inversion import exact_invert_1x2, _fast_grid, devig_market, score_exact_odds, parse_extra_markets

e = create_engine(load_settings().db_url)
d = pd.read_sql("""SELECT e.expected_start, o.odds_home oh, o.odds_draw od, o.odds_away oa, o.extra_markets em,
  r.score_a sa, r.score_b sb FROM events e
  JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
  JOIN results r ON r.event_id=e.id
  WHERE r.score_a IS NOT NULL AND e.competition='InstantLeague-8035'""", e)
d = d[(d.oh > 1) & (d.od > 1) & (d.oa > 1)].copy()
d["es"] = pd.to_datetime(d.expected_start, utc=True, errors="coerce")
d = d.dropna(subset=["es"]).sort_values("es").reset_index(drop=True)
d = d[(d.sa <= 9) & (d.sb <= 9)]
n = len(d)
d["sc"] = d.sa.astype(int).astype(str) + "-" + d.sb.astype(int).astype(str)
d["tot"] = d.sa + d.sb
print(f"n matchs 8035 = {n}")

# lambdas implicites par match (inversion 1X2) — cache par triplet
uniq = d[["oh", "od", "oa"]].round(2).drop_duplicates()
cache = {}
for r in uniq.itertuples(index=False):
    cache[(r.oh, r.od, r.oa)] = exact_invert_1x2(r.oh, r.od, r.oa)
key = list(zip(d.oh.round(2), d.od.round(2), d.oa.round(2)))
d["lh"] = [cache[k][0] for k in key]; d["la"] = [cache[k][1] for k in key]

# ===== (1) DISTRIBUTION BRUTE vs POISSON INDÉPENDANT =====
CAP = 6
exp_pois = np.zeros((CAP+1, CAP+1))
for lh, la in zip(d.lh.values, d.la.values):
    g = _fast_grid(lh, la, 0.0)  # Poisson indép pur
    gg = g[:CAP+1, :CAP+1].copy()
    gg[CAP, :] += g[CAP+1:, :CAP+1].sum(axis=0); gg[:, CAP] += g[:CAP+1, CAP+1:].sum(axis=1)
    exp_pois += gg
obs = np.zeros((CAP+1, CAP+1))
for sa, sb in zip(d.sa.values, d.sb.values):
    obs[min(int(sa), CAP), min(int(sb), CAP)] += 1
print("\n" + "="*84)
print("(1) SCORES qui dévient le plus de POISSON INDÉPENDANT (calibré sur lambdas 1X2)")
print("    z = (obs - exp)/sqrt(exp) ; + = sort trop souvent, - = pas assez")
print("="*84)
cells = []
for h in range(CAP+1):
    for a in range(CAP+1):
        o, ex = obs[h, a], exp_pois[h, a]
        if ex >= 20:
            z = (o-ex)/math.sqrt(ex)
            cells.append((f"{h}-{a}{'+' if h==CAP or a==CAP else ''}", int(o), ex, o-ex, z, o/n*100, ex/n*100))
cells.sort(key=lambda x: -abs(x[4]))
print(f"  {'score':<8}{'obs':>6}{'exp':>9}{'obs%':>7}{'exp%':>7}{'résidu':>9}{'z':>8}")
for sc, o, ex, r, z, op, ep in cells[:16]:
    print(f"  {sc:<8}{o:>6}{ex:>9.0f}{op:>6.1f}%{ep:>6.1f}%{r:>+9.0f}{z:>+8.1f}")
chi2_stat = sum((o-ex)**2/ex for _, o, ex, _, _, _, _ in cells)
dfree = len(cells)-1
print(f"\n  chi2 global (vs Poisson indép) = {chi2_stat:.0f}, df={dfree}, p={1-chi2dist.cdf(chi2_stat,dfree):.1e}")
print(f"  -> {'REJET de Poisson indépendant (distribution déviée)' if chi2_stat>chi2dist.ppf(0.999,dfree) else 'compatible Poisson'}")

# ===== (2) MÉCANIQUE RNG =====
print("\n" + "="*84)
print("(2) MÉCANIQUE DU RNG")
print("="*84)
# sur-dispersion conditionnelle : résidu standardisé vs lambda
d["rh"] = d.sa - d.lh; d["ra"] = d.sb - d.la
od_h = d.rh.var()/d.lh.mean(); od_a = d.ra.var()/d.la.mean()
print(f"  Sur-dispersion (var_résidu/lambda) : Home={od_h:.3f}  Away={od_a:.3f}  (1.0=Poisson, <1=sous-dispersé)")
# corrélation H-A résiduelle (contrôlée pour lambda)
corr = np.corrcoef(d.rh, d.ra)[0, 1]
zc = corr*math.sqrt(n)
print(f"  Corrélation H-A résiduelle = {corr:+.4f} (z={zc:+.1f})  {'<0 = défensif/DC' if corr<-0.02 else ('>0 = offensif' if corr>0.02 else '~0 indépendant')}")
# Dixon-Coles : ratio obs/poisson sur les 4 cellules basses
print("  Dixon-Coles (obs/exp sur cellules basses) :")
for h, a in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    print(f"    {h}-{a}: obs/exp = {obs[h,a]/exp_pois[h,a]:.3f}")
# négative binomiale ? var/mean marginal par équipe
print(f"  Marginal var/mean : Home={d.sa.var()/d.sa.mean():.3f}  Away={d.sb.var()/d.sb.mean():.3f} (>1=NB, =1=Poisson ; mixture gonfle)")
# parité
tot_even = (d.tot % 2 == 0).mean()
print(f"  Parité total : pair={tot_even*100:.1f}% impair={(1-tot_even)*100:.1f}%")
# goal diff vs Skellam
d["gd"] = d.sa - d.sb
lh_bar, la_bar = d.lh.mean(), d.la.mean()
print(f"  Écart de buts (Home-Away) obs vs Skellam(lam_h_bar,lam_a_bar) :")
for k in range(-3, 4):
    obs_gd = (d.gd == k).mean(); exp_gd = skellam.pmf(k, lh_bar, la_bar)
    print(f"    diff={k:+d}: obs={obs_gd*100:>4.1f}%  Skellam={exp_gd*100:>4.1f}%  écart={obs_gd*100-exp_gd*100:+.1f}pt")
# troncature
print(f"  Score max observé : Home={int(d.sa.max())} Away={int(d.sb.max())} total={int(d.tot.max())} (cap apparent ?)")
# grille lambda ?
lt = (d.lh+d.la).round(2)
print(f"  lambda_tot : {lt.nunique()} valeurs uniques sur {n} matchs -> {'continu' if lt.nunique()>n*0.3 else 'GRILLE (discrétisé)'}")

# ===== (3) PAR TRANCHE DE COTE FAVORI =====
print("\n" + "="*84)
print("(3) DÉVIATIONS vs Poisson PAR TRANCHE DE COTE FAVORI (chi2 + pire score)")
print("="*84)
d["favc"] = d[["oh", "oa"]].min(axis=1)
for lo, hi in [(1.0, 1.3), (1.3, 1.5), (1.5, 1.8), (1.8, 2.2), (2.2, 2.8), (2.8, 99)]:
    s = d[(d.favc >= lo) & (d.favc < hi)]
    if len(s) < 300: continue
    ep = np.zeros((CAP+1, CAP+1))
    for lh, la in zip(s.lh.values, s.la.values):
        g = _fast_grid(lh, la, 0.0); gg = g[:CAP+1, :CAP+1].copy()
        gg[CAP, :] += g[CAP+1:, :CAP+1].sum(axis=0); gg[:, CAP] += g[:CAP+1, CAP+1:].sum(axis=1); ep += gg
    ob = np.zeros((CAP+1, CAP+1))
    for sa, sb in zip(s.sa.values, s.sb.values): ob[min(int(sa), CAP), min(int(sb), CAP)] += 1
    chi = sum((ob[h, a]-ep[h, a])**2/ep[h, a] for h in range(CAP+1) for a in range(CAP+1) if ep[h, a] >= 15)
    worst = max(((f"{h}-{a}", (ob[h, a]-ep[h, a])/math.sqrt(ep[h, a])) for h in range(CAP+1) for a in range(CAP+1) if ep[h, a] >= 15), key=lambda x: abs(x[1]))
    print(f"  cote fav [{lo},{hi}) n={len(s):>5}  chi2={chi:>5.0f}  pire score: {worst[0]} (z={worst[1]:+.1f})")

# ===== (4) DÉCISIF : CALIBRATION vs COTES SCORE EXACT OFFERTES (EV + FDR + OOS) =====
print("\n" + "="*84)
print("(4) EXPLOITABILITÉ — réalisé vs COTE SCORE EXACT offerte (EV, BH-FDR, OOS)")
print("="*84)
cut1, cut2 = int(n*0.6), int(n*0.8)
d["split"] = np.where(d.index < cut1, "tr", np.where(d.index < cut2, "va", "te"))
# accumuler par score : (indicateur réalisé, cote offerte, implicite devigé)
from collections import defaultdict
rec = defaultdict(lambda: {"tr": [], "va": [], "te": []})
for r in d.itertuples():
    em = parse_extra_markets(r.em); so = score_exact_odds(em)
    if len(so) < 8: continue
    q = devig_market(so)
    if not q: continue
    for sc, cote in so.items():
        if sc in q:
            rec[sc][r.split].append((1.0 if sc == r.sc else 0.0, cote, q[sc]))
rows = []
for sc, sp in rec.items():
    if len(sp["tr"]) < 200: continue
    def stats(arr):
        if len(arr) < 50: return (len(arr), np.nan, np.nan, np.nan, np.nan)
        a = np.array(arr); real = a[:, 0].mean(); imp = a[:, 2].mean(); ev = (a[:, 0]*a[:, 1]-1).mean()
        z = (real-imp)/math.sqrt(imp*(1-imp)/len(a)) if 0 < imp < 1 else 0
        return (len(a), real, imp, ev, z)
    ntr, rtr, itr, evtr, ztr = stats(sp["tr"]); nva, rva, iva, evva, zva = stats(sp["va"]); nte, rte, ite, evte, zte = stats(sp["te"])
    p = 2*(1-norm.cdf(abs(ztr)))
    rows.append(dict(sc=sc, ntr=ntr, rtr=rtr, itr=itr, evtr=evtr, ztr=ztr, p=p,
                     evva=evva, evte=evte, ztr2=ztr, zva=zva, zte=zte))
R = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
m = len(R); R["bh"] = (R.index+1)/m*0.10; R["bh_pass"] = R.p <= R.bh
R["sign_ok"] = (np.sign(R.evtr) == np.sign(R.evva)) & (np.sign(R.evtr) == np.sign(R.evte)) & (R.evtr != 0)
print(f"  scores testés (n_tr>=200) : {m}")
print(f"  {'score':<7}{'n_tr':>6}{'réel%':>7}{'impl%':>7}{'EV_tr':>7}{'EV_va':>7}{'EV_te':>7}{'z_tr':>6}{'FDR':>5}{'signe':>6}")
top = R.reindex(R.ztr.abs().sort_values(ascending=False).index).head(14)
for r in top.itertuples():
    print(f"  {r.sc:<7}{r.ntr:>6}{r.rtr*100:>6.1f}%{r.itr*100:>6.1f}%{r.evtr*100:>+6.0f}%{r.evva*100:>+6.0f}%{r.evte*100:>+6.0f}%{r.ztr:>+6.1f}{'pass' if r.bh_pass else '-':>5}{'OK' if r.sign_ok else 'flip':>6}")
robust = R[(R.bh_pass) & (R.sign_ok) & (R.evtr > 0) & (R.evva > 0) & (R.evte > 0)]
print(f"\n  Scores +EV ROBUSTES (FDR + signe stable + EV>0 sur tr/va/te) : {len(robust)}")
for r in robust.itertuples():
    print(f"    {r.sc}: EV tr/va/te = {r.evtr*100:+.0f}/{r.evva*100:+.0f}/{r.evte*100:+.0f}%")
# marge moyenne du marché score exact
allcote = []
for r in d.itertuples():
    so = score_exact_odds(parse_extra_markets(r.em))
    if len(so) >= 8:
        ov = sum(1/c for c in so.values())
        if ov > 0: allcote.append(ov)
if allcote:
    print(f"\n  Marge moyenne marché Score exact (overround) = {(np.mean(allcote)-1)*100:.0f}%")
print("\nVERDICT (4): si 0 score +EV robuste ET marge énorme -> marché score exact efficient, inexploitable.")
