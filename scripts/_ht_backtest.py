"""Backtest du modele mi-temps : accuracy (HT 1X2 / score HT / HT-FT) vs plafond+naif,
et EV vs cotes offertes (Mi-tps 1X2, Mi-tps CS, HT/FT). Forward 70/30 chrono.

Question cle : le nul HT (Mi-tps X) sous-price par la grille (41,2% reel vs 37,9% grille)
est-il +EV vs la cote offerte ? (stratifie par lam_tot).
Usage: ./.venv/Scripts/python.exe scripts/_ht_backtest.py
"""
import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from scraper.config import load_settings
from scraper.analysis_utils import load_corrupted_ids
from scraper.market_inversion import exact_invert_1x2, parse_extra_markets, devig_market, _get_market
from scraper.halftime_model import ht_grid, ht_predictions, _grid_1x2

e = create_engine(load_settings().db_url)
corr = load_corrupted_ids()
SQL = """
    SELECT e.id, e.expected_start, o.odds_home oh, o.odds_draw od, o.odds_away oa, o.extra_markets,
           r.ht_score_a, r.ht_score_b, r.score_a, r.score_b
    FROM events e
    JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
    JOIN results r ON r.event_id=e.id
    WHERE r.ht_score_a IS NOT NULL AND e.competition='InstantLeague-8035'
      AND o.odds_home IS NOT NULL
"""
rows = []
for chunk in pd.read_sql(SQL, e, chunksize=2000):
    chunk = chunk[~chunk.id.isin(corr)]
    for r in chunk.itertuples():
        if r.ht_score_a > r.score_a or r.ht_score_b > r.score_b:
            continue
        em = parse_extra_markets(r.extra_markets)
        lh, la = exact_invert_1x2(r.oh, r.od, r.oa)
        # offered Mi-tps 1X2
        m1 = _get_market(em, exact="Mi-tps 1X2")
        ht_odds = devig_market({k: m1.get(k) for k in ("1", "X", "2") if isinstance(m1, dict) and m1.get(k)}) if isinstance(m1, dict) else {}
        x_cote = m1.get("X") if isinstance(m1, dict) else None
        rows.append(dict(id=r.id, es=r.expected_start, lh=lh, la=la,
                         hh=int(r.ht_score_a), ha=int(r.ht_score_b),
                         x_cote=float(x_cote) if x_cote and float(x_cote) > 1 else np.nan))
df = pd.DataFrame(rows)
df["es"] = pd.to_datetime(df.es, utc=True, errors="coerce")
df = df.dropna(subset=["es"]).sort_values("es").reset_index(drop=True)
df["ht_res"] = np.where(df.hh > df.ha, "1", np.where(df.hh == df.ha, "X", "2"))
df["ht_score"] = df.hh.astype(str) + "-" + df.ha.astype(str)
split = int(len(df) * 0.70)
te = df.iloc[split:].copy()
print(f"n={len(df)} test={len(te)}")

# predictions HT par event (test)
preds = [ht_predictions(r.lh, r.la) for r in te.itertuples()]
te["ht_pick"] = [p["ht_pick"] for p in preds]
te["ht_top1"] = [p["ht_top_scores"][0][0] for p in preds]
te["ht_top3"] = [[s for s, _ in p["ht_top_scores"]] for p in preds]
te["pX"] = [p["ht_1x2"]["X"] for p in preds]

# accuracy HT 1X2
acc_1x2 = (te.ht_pick == te.ht_res).mean()
# naif (grille Poisson rho=0) pour comparaison
from scraper.predictor_v2 import poisson_score_grid
def naive_ht_pick(lh, la):
    g = poisson_score_grid(lh*0.476, la*0.476, 0.0, 7)
    p1, pX, p2 = _grid_1x2(g)
    return max((("1",p1),("X",pX),("2",p2)), key=lambda kv: kv[1])[0]
te["naive_pick"] = [naive_ht_pick(r.lh, r.la) for r in te.itertuples()]
acc_naive = (te.naive_pick == te.ht_res).mean()
ceil_1x2 = te.ht_res.value_counts(normalize=True).iloc[0]  # toujours le modal
print(f"\n=== HT 1X2 accuracy ===")
print(f"  modele DC : {100*acc_1x2:.1f}%   naif Poisson : {100*acc_naive:.1f}%   (toujours-favori-HT ceil {100*ceil_1x2:.1f}%)")

# accuracy score HT
top1 = (te.ht_top1 == te.ht_score).mean()
top3 = te.apply(lambda r: r.ht_score in r.ht_top3, axis=1).mean()
ceil_s = te.ht_score.value_counts(normalize=True)
print(f"\n=== Score HT accuracy ===")
print(f"  modele Top1 {100*top1:.1f}% Top3 {100*top3:.1f}%   (plafond Top1 {100*ceil_s.iloc[0]:.1f}% Top3 {100*ceil_s.iloc[:3].sum():.1f}%)")

# EV Mi-tps X (nul HT) vs cote offerte
sub = te.dropna(subset=["x_cote"])
win = (sub.ht_res == "X").astype(float).values
profit = sub.x_cote.values * win - 1.0
print(f"\n=== EV nul HT (Mi-tps X) vs cote offerte ===  n={len(sub)} cote_med={np.median(sub.x_cote):.2f}")
print(f"  taux nul reel {100*win.mean():.1f}%  EV global {100*profit.mean():+.1f}%")
# stratifie par lam_tot (bas total -> + de nuls)
sub = sub.assign(lam_tot=sub.lh + sub.la)
for lo, hi in [(0,2.4),(2.4,2.8),(2.8,3.2),(3.2,9)]:
    g = sub[(sub.lam_tot>=lo)&(sub.lam_tot<hi)]
    if len(g)<100: continue
    w=(g.ht_res=="X").astype(float).values; pr=g.x_cote.values*w-1
    print(f"  lam_tot∈[{lo},{hi}) n={len(g):>4} nul {100*w.mean():.1f}% cote_med {np.median(g.x_cote):.2f} EV {100*pr.mean():+.1f}%")
