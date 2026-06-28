"""LAG-10 : signal réel ou artefact des fixtures qui se répètent ?
Le test décisif : l'autocorrélation à lag 10 survit-elle APRÈS avoir retiré les cotes ?
 - si l'autocorr du RÉSIDU (réel - implicite) à lag 10 ~ 0  -> c'est juste les cotes/fixtures
   qui se répètent (NON exploitable).
 - si elle survit (z>4)  -> vraie mémoire du RNG (exploitable), à confirmer OOS.
Plus : variance home-wins/manche vs Poisson-binomial (avec p implicite par match),
et test OOS actionnable (lag10 améliore-t-il la prédiction vs cotes seules ?).
"""
from __future__ import annotations
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sqlalchemy import create_engine
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import log_loss, mean_squared_error
from scraper.config import load_settings
from scraper.market_inversion import parse_extra_markets, total_buts_odds, devig_market, _get_market, _to_float

e = create_engine(load_settings().db_url)
g = pd.read_sql("""SELECT e.expected_start, e.id ev, o.odds_home oh, o.odds_draw od, o.odds_away oa,
  o.extra_markets em, r.score_a sa, r.score_b sb FROM events e
  JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
  JOIN results r ON r.event_id=e.id
  WHERE r.score_a IS NOT NULL AND e.competition='InstantLeague-8035'""", e)
g = g[(g.oh > 1) & (g.od > 1) & (g.oa > 1)].copy()
g["es"] = pd.to_datetime(g.expected_start, utc=True, errors="coerce")
g = g.dropna(subset=["es"]).sort_values(["es", "ev"]).reset_index(drop=True)
g["tot"] = g.sa + g.sb; g["hw"] = (g.sa > g.sb).astype(int)
g["draw"] = (g.sa == g.sb).astype(int); g["btts"] = ((g.sa >= 1) & (g.sb >= 1)).astype(int)
g["over25"] = (g.tot >= 3).astype(int)
inv = 1/g.oh + 1/g.od + 1/g.oa
g["imp_home"] = (1/g.oh)/inv; g["imp_draw"] = (1/g.od)/inv

# implicites total + btts depuis les ladders offerts
imp_tot_mean = []; imp_over = []; imp_btts = []
for r in g.itertuples():
    em = parse_extra_markets(r.em)
    tb = total_buts_odds(em)
    q = devig_market(tb) if len(tb) >= 4 else {}
    if q:
        mean = sum(int(k)*v for k, v in q.items() if k.isdigit())
        over = sum(v for k, v in q.items() if k.isdigit() and int(k) >= 3)
        imp_tot_mean.append(mean); imp_over.append(over)
    else:
        imp_tot_mean.append(np.nan); imp_over.append(np.nan)
    gng = _get_market(em, exact="G/NG")
    if isinstance(gng, dict):
        d = devig_market({k: gng.get(k) for k in ("Oui", "Non") if gng.get(k)})
        imp_btts.append(d.get("Oui", np.nan) if d else np.nan)
    else:
        imp_btts.append(np.nan)
g["imp_tot"] = imp_tot_mean; g["imp_over"] = imp_over; g["imp_btts"] = imp_btts

def ac_lag(series, lag=10):
    s = series.values.astype(float); s = s - np.nanmean(s)
    m = ~np.isnan(s)
    # garder paires valides
    a = s[:-lag]; b = s[lag:]; mm = ~np.isnan(a) & ~np.isnan(b)
    a, b = a[mm], b[mm]
    if len(a) < 100: return 0.0, 0, 0
    corr = np.sum(a*b)/math.sqrt(np.sum(a*a)*np.sum(b*b))
    z = corr*math.sqrt(len(a))
    return corr, z, len(a)

print("="*82)
print("(1) AUTOCORR à lag 10 — BRUT vs RÉSIDU (réel - implicite par les cotes)")
print("    si BRUT significatif mais RÉSIDU ~0 => c'est les COTES/fixtures, pas le RNG")
print("="*82)
print(f"{'métrique':<14}{'brut corr':>11}{'z':>6}{'  | résidu corr':>16}{'z':>6}   verdict")
specs = [("hw", g.hw, g.hw - g.imp_home), ("draw", g.draw, g.draw - g.imp_draw),
         ("over25", g.over25, g.over25 - g.imp_over), ("btts", g.btts, g.btts - g.imp_btts),
         ("tot", g.tot, g.tot - g.imp_tot)]
for name, raw, resid in specs:
    cr, zr, _ = ac_lag(raw); cd, zd, n = ac_lag(resid)
    verdict = "RNG réel !" if abs(zd) > 4 else ("cotes/fixtures" if abs(zr) > 4 else "rien")
    print(f"{name:<14}{cr:>+11.4f}{zr:>+6.1f}{cd:>+16.4f}{zd:>+6.1f}   {verdict}")

print("\n" + "="*82)
print("(2) Les fixtures se répètent-elles ? autocorr des COTES à lag 10")
print("    si imp_home[t] corrèle fort avec imp_home[t+10] => même fixture/cote => confond")
print("="*82)
for name, col in [("imp_home", g.imp_home), ("imp_tot", g.imp_tot), ("imp_btts", g.imp_btts)]:
    cr, zr, n = ac_lag(col)
    print(f"  {name:<10} autocorr lag10 = {cr:+.3f} (z={zr:+.1f})  {'<- fixtures répétées' if abs(zr)>4 else ''}")

print("\n" + "="*82)
print("(3) Variance home-wins/manche vs Poisson-binomial (p implicite par match)")
print("    Poisson-binomial = somme p_i(1-p_i) ; si obs ~ PB => expliqué par les cotes")
print("="*82)
g["slot"] = g.groupby("es").cumcount()
rk = g.groupby("es").agg(n=("hw", "size"), hw_sum=("hw", "sum"),
                         pb_var=("imp_home", lambda s: np.sum(s*(1-s)))).query("n>=8")
print(f"  var(home-wins/manche) observée = {rk.hw_sum.var():.3f}")
print(f"  var Poisson-binomiale (cotes)  = {rk.pb_var.mean():.3f}")
ratio = rk.hw_sum.var()/rk.pb_var.mean()
print(f"  ratio = {ratio:.3f}  -> {'sur-dispersion RÉELLE (au-delà des cotes) !' if ratio>1.2 else 'expliqué par les cotes (OK)'}")

print("\n" + "="*82)
print("(4) TEST OOS ACTIONNABLE — lag10 améliore-t-il la prédiction vs cotes seules ?")
print("    feature = résultat 10 matchs avant (même slot, manche précédente)")
print("="*82)
g["hw_l10"] = g.hw.shift(10); g["tot_l10"] = g.tot.shift(10); g["btts_l10"] = g.btts.shift(10)
d = g.dropna(subset=["hw_l10", "tot_l10", "btts_l10", "imp_home", "imp_tot", "imp_btts", "imp_over"]).reset_index(drop=True)
cut = int(len(d)*0.7); tr, te = d.iloc[:cut], d.iloc[cut:]
def clf_cmp(target, base, extra, name):
    A = LogisticRegression(max_iter=2000).fit(tr[[base]], tr[target])
    B = LogisticRegression(max_iter=2000).fit(tr[[base, extra]], tr[target])
    pa = A.predict_proba(te[[base]])[:, 1]; pb = B.predict_proba(te[[base, extra]])[:, 1]
    la, lb = log_loss(te[target], pa), log_loss(te[target], pb)
    print(f"  [{name}] cotes={la:.4f}  cotes+lag10={lb:.4f}  -> {'AIDE (Δ %+.4f)'%(la-lb) if (la-lb)>0.0005 else 'aucun gain'}")
clf_cmp("hw", "imp_home", "hw_l10", "home win")
clf_cmp("btts", "imp_btts", "btts_l10", "BTTS")
A = LinearRegression().fit(tr[["imp_tot"]], tr.tot); B = LinearRegression().fit(tr[["imp_tot", "tot_l10"]], tr.tot)
ra = mean_squared_error(te.tot, A.predict(te[["imp_tot"]]))**.5
rb = mean_squared_error(te.tot, B.predict(te[["imp_tot", "tot_l10"]]))**.5
print(f"  [total] RMSE cotes={ra:.4f}  cotes+lag10={rb:.4f}  -> {'AIDE' if (ra-rb)>0.003 else 'aucun gain'}")
print("\n  Si (1) résidu~0, (2) cotes répétées, (3) ratio expliqué, (4) aucun gain OOS")
print("  => le lag-10 est 100% les fixtures qui se répètent. Rien d'exploitable.")
