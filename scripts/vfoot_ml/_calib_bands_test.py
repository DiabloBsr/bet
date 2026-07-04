"""Calibration PAR BANDE DE COTE vs GLOBALE — la bande apporte-t-elle du Top-1 ?

3 bandes de force du favori (terciles train) × table 7x7, avec lissage vers la
table globale (alpha = n_b/(n_b+3000)). Évaluation OOS : log-loss score exact,
Top-1, Top-3 — adoption seulement si la bande bat la globale sur logloss ET Top-1.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import poisson as _poi
from sqlalchemy import create_engine, text
from scraper.config import load_settings
from scraper.market_inversion import exact_invert_1x2, apply_sim_deviations

eng = create_engine(load_settings().db_url)
df = pd.read_sql(text("""
    SELECT o.odds_home oh, o.odds_draw od, o.odds_away oa, r.score_a sa, r.score_b sb
    FROM events e
    JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
    JOIN results r ON r.event_id=e.id
    WHERE r.score_a IS NOT NULL AND e.competition='InstantLeague-8035'
      AND o.odds_home>1 AND o.odds_draw>1 AND o.odds_away>1
    ORDER BY e.expected_start"""), eng)
N = len(df); cut = int(N * 0.7)
print(f"{N} matchs | train {cut} / test {N-cut}")

G = np.zeros((N, 7, 7)); ok = np.zeros(N, bool)
fav = np.zeros(N)
inv = 1/df.oh + 1/df.od + 1/df.oa
fav = np.maximum((1/df.oh)/inv, (1/df.oa)/inv).values
for i, r in enumerate(df.itertuples()):
    try:
        lh, la = exact_invert_1x2(r.oh, r.od, r.oa)
        g = np.asarray(apply_sim_deviations(lh, la, "cells"), float)[:7, :7]
        G[i] = g / g.sum(); ok[i] = True
    except Exception:
        pass
sa6 = df.sa.clip(0, 6).astype(int).values; sb6 = df.sb.clip(0, 6).astype(int).values

def emp_of(idx):
    m = np.zeros((7, 7))
    for i in idx:
        m[sa6[i], sb6[i]] += 1
    return m

tr_idx = np.arange(cut)[ok[:cut]]
emp_g = emp_of(tr_idx); n_g = emp_g.sum(); emp_g = emp_g / n_g
mod_g = G[tr_idx].mean(0)
CAL_G = np.clip(emp_g / np.clip(mod_g, 1e-5, None), 0.4, 2.5)

edges = np.quantile(fav[:cut], [1/3, 2/3])
band = np.digitize(fav, edges)                      # 0/1/2
print(f"bandes favori (terciles train) : <{edges[0]:.3f} / <{edges[1]:.3f} / au-dela")
N0 = 3000.0
CAL_B = np.zeros((3, 7, 7))
for b in range(3):
    idx = tr_idx[band[tr_idx] == b]
    emp_b = emp_of(idx); n_b = emp_b.sum(); emp_b = emp_b / max(n_b, 1)
    mod_b = G[idx].mean(0)
    raw = np.clip(emp_b / np.clip(mod_b, 1e-5, None), 0.3, 3.0)
    a = n_b / (n_b + N0)
    CAL_B[b] = np.clip(a * raw + (1 - a) * CAL_G, 0.4, 2.5)
    print(f"  bande {b}: n={int(n_b)} alpha={a:.2f}")

def evaluate(cal_for):
    te = np.arange(cut, N)[ok[cut:]]
    ll = hits1 = hits3 = 0
    for i in te:
        g = G[i] * cal_for(i); g = g / g.sum()
        p = max(g[sa6[i], sb6[i]], 1e-9)
        ll += -np.log(p)
        flat = g.ravel(); order = np.argsort(-flat)[:3]
        cell = sa6[i] * 7 + sb6[i]
        hits1 += int(order[0] == cell); hits3 += int(cell in order)
    n = len(te)
    return ll / n, hits1 / n, hits3 / n

ll0, h10, h30 = evaluate(lambda i: 1.0)
llg, h1g, h3g = evaluate(lambda i: CAL_G)
llb, h1b, h3b = evaluate(lambda i: CAL_B[band[i]])
print(f"\n{'variante':<22}{'logloss':>9}{'Top-1':>8}{'Top-3':>8}")
print(f"{'sans calibration':<22}{ll0:>9.5f}{100*h10:>7.2f}%{100*h30:>7.2f}%")
print(f"{'calibration GLOBALE':<22}{llg:>9.5f}{100*h1g:>7.2f}%{100*h3g:>7.2f}%")
print(f"{'calibration PAR BANDE':<22}{llb:>9.5f}{100*h1b:>7.2f}%{100*h3b:>7.2f}%")
win = llb < llg and h1b >= h1g
print(f"\nVERDICT : {'ADOPTER la calibration par bande' if win else 'GARDER la globale'} "
      f"(delta logloss {llb-llg:+.5f}, delta Top-1 {100*(h1b-h1g):+.2f}pp)")
