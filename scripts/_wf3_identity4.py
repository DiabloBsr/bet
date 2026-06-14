# -*- coding: utf-8 -*-
"""
WF3 IDENTITY - PART 4 : EXPLOITATION DU JITTER COSMETIQUE
La proba 'vraie' d'un matchup = base de paire (stable, estimable a 0.002 pres).
La cote du jour = base + jitter cosmetique (sd~0.008).
=> backer l'issue quand jitter rend la cote > fair value de la base.
Walk-forward ONLINE STRICT: base = moyenne des probas implicites des matchs ANTERIEURS
de la meme paire (aucun lookahead). Evaluation: full online + fenetre OOS finale 30%.
"""
import sys, math
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import create_engine, text
from scraper.config import load_settings

pd.set_option('display.width', 220)
eng = create_engine(load_settings().db_url)

with eng.connect() as c:
    ev = pd.read_sql(text("""
        SELECT e.id, e.team_a, e.team_b, e.expected_start, r.score_a, r.score_b
        FROM events e JOIN results r ON r.event_id = e.id
        WHERE e.round_info != '0' AND r.score_a IS NOT NULL
    """), c)
    od = pd.read_sql(text("""
        SELECT id, event_id, odds_home, odds_draw, odds_away
        FROM odds_snapshots WHERE odds_home IS NOT NULL ORDER BY id
    """), c)
od_open = od.groupby('event_id', as_index=False).first()
df = ev.merge(od_open[['event_id', 'odds_home', 'odds_draw', 'odds_away']], left_on='id', right_on='event_id')
df['expected_start'] = pd.to_datetime(df['expected_start'])
df = df.sort_values('expected_start').drop_duplicates(
    subset=['team_a', 'team_b', 'expected_start'], keep='first').reset_index(drop=True)
inv = 1/df.odds_home + 1/df.odds_draw + 1/df.odds_away
df['pH'] = (1/df.odds_home)/inv
df['pD'] = (1/df.odds_draw)/inv
df['pA'] = (1/df.odds_away)/inv
df['res'] = np.where(df.score_a > df.score_b, 'H', np.where(df.score_a < df.score_b, 'A', 'D'))
n = len(df)
print(f"n={n}")

# calibration fine: proba implicite normalisee -> proba vraie (correction biais leger)
# (global: pH 0.4751 -> 0.4805 realise; pD 0.2255 -> 0.2197) : facteurs multiplicatifs
adjH = 0.4805/0.4751; adjD = 0.2197/0.2255; adjA = 0.2999/0.2994
print(f"facteurs calibration: H x{adjH:.4f} D x{adjD:.4f} A x{adjA:.4f}")

# ---------------- walk-forward online
hist = {}  # pair -> list of (pH,pD,pA)
bets = []  # (i, outcome, odds, won, EV_est, n_prior)
evs_all = []
oos_start = int(0.7*n)
MIN_PRIOR = 5
for i, r in df.iterrows():
    key = (r.team_a, r.team_b)
    prior = hist.get(key, [])
    if len(prior) >= MIN_PRIOR:
        arr = np.array(prior)
        base = arr.mean(axis=0)  # pH, pD, pA base
        for k, (out, ocol, adj) in enumerate([('H', 'odds_home', adjH), ('D', 'odds_draw', adjD), ('A', 'odds_away', adjA)]):
            ev_est = base[k]*adj*r[ocol]
            evs_all.append((i, out, ev_est))
            bets.append((i, out, r[ocol], r.res == out, ev_est, len(prior)))
    hist.setdefault(key, []).append((r.pH, r.pD, r.pA))

B = pd.DataFrame(bets, columns=['i', 'out', 'odds', 'won', 'ev', 'n_prior'])
print(f"\ncotes evaluees (online, n_prior>={MIN_PRIOR}): {len(B)} sur {n} matchs")
print(f"distribution EV estime: q50={B.ev.median():.3f} q90={B.ev.quantile(.9):.3f} "
      f"q97.5={B.ev.quantile(.975):.3f} max={B.ev.max():.3f}")
print(f"part des cotes avec EV_est>1.00 : {(B.ev > 1).mean()*100:.2f}%")

print("\n--- ROI par seuil d'EV estime (FULL online walk-forward) ---")
for th in [0.98, 1.00, 1.01, 1.02, 1.03]:
    s = B[B.ev > th]
    if len(s) == 0:
        continue
    pnl = np.where(s.won, s.odds-1, -1)
    se = pnl.std()/math.sqrt(len(s))
    print(f"EV>{th:.2f}: n={len(s)} ROI={pnl.mean()*100:+.2f}% (se {se*100:.1f}%) "
          f"WR={s.won.mean()*100:.1f}% avg_odds={s.odds.mean():.2f} "
          f"[H:{(s.out=='H').sum()} D:{(s.out=='D').sum()} A:{(s.out=='A').sum()}]")

print("\n--- ROI par seuil, fenetre OOS stricte (derniers 30%) ---")
Bo = B[B.i >= oos_start]
for th in [0.98, 1.00, 1.01, 1.02]:
    s = Bo[Bo.ev > th]
    if len(s) == 0:
        continue
    pnl = np.where(s.won, s.odds-1, -1)
    se = pnl.std()/math.sqrt(len(s))
    # p-value binomiale vs proba implicite brute moyenne
    p_imp = (1/s.odds).mean()/1.06
    pv = stats.binomtest(int(s.won.sum()), len(s), p_imp).pvalue
    print(f"EV>{th:.2f}: n={len(s)} ROI={pnl.mean()*100:+.2f}% (se {se*100:.1f}%) "
          f"WR={s.won.mean()*100:.1f}% avg_odds={s.odds.mean():.2f} p_binom_vs_implied={pv:.3f}")

# baseline pour contexte
pnl_all = []
for ocol, out in [('odds_home', 'H'), ('odds_draw', 'D'), ('odds_away', 'A')]:
    g = df.iloc[oos_start:]
    pnl_all.append(np.where(g.res == out, g[ocol]-1, -1).mean())
print(f"baselines OOS bet-all: H={pnl_all[0]*100:+.1f}% D={pnl_all[1]*100:+.1f}% A={pnl_all[2]*100:+.1f}%")

# ---------------- decomposition: d'ou vient l'EV>1 ? jitter vs bruit d'estimation
print("\n--- sanity: EV_est>1 est-il du jitter reel ou du bruit de base ? ---")
s = B[B.ev > 1.0]
# si jitter reel: la cote du match courant doit etre au-dessus de sa base FUTURE aussi
# proxy: dispersion attendue du bruit de base = sd_jitter/sqrt(n_prior)
print(f"n_prior moyen des bets EV>1: {s.n_prior.mean():.1f} (bruit base ~{0.0078/math.sqrt(s.n_prior.mean()):.4f} vs jitter 0.0078)")
# part de l'exces d'EV explicable par le bruit: ratio var
ratio = (1/s.n_prior.mean())/(1+1/s.n_prior.mean())
print(f"part de la variance du signal due au bruit d'estimation: {ratio*100:.0f}%")

# ---------------- meme test PAR issue (le jitter aide-t-il plus les outsiders ?)
print("\n--- EV>1.00 (full) par issue ---")
for out in ['H', 'D', 'A']:
    s = B[(B.ev > 1.0) & (B.out == out)]
    if len(s) < 10:
        print(f"{out}: n={len(s)} (trop peu)")
        continue
    pnl = np.where(s.won, s.odds-1, -1)
    se = pnl.std()/math.sqrt(len(s))
    print(f"{out}: n={len(s)} ROI={pnl.mean()*100:+.2f}% (se {se*100:.1f}%) WR={s.won.mean()*100:.1f}% odds={s.odds.mean():.2f}")

# ---------------- ceiling theorique
print("\n--- plafond theorique du jitter cosmetique ---")
# EV moyen conditionnel au seuil si base parfaite et jitter logit N(0, 0.035)
# approx par simulation sur les probas de la base
rng = np.random.default_rng(1)
sim_ev = []
bases = df[['pH', 'pD', 'pA']].values
for _ in range(3):
    for p in bases[rng.choice(n, 2000)]:
        for k in range(3):
            pb = p[k]
            lg = math.log(pb/(1-pb)) + rng.normal(0, 0.035)
            pj = 1/(1+math.exp(-lg))
            odds = 1/(pj*1.06)
            sim_ev.append(pb*odds)
sim_ev = np.array(sim_ev)
print(f"simulation: part EV>1 = {(sim_ev > 1).mean()*100:.2f}% ; EV moyen | EV>1 = {sim_ev[sim_ev > 1].mean():.4f}")
print(f"=> plafond ROI attendu sur les bets EV>1: {(sim_ev[sim_ev > 1].mean()-1)*100:+.2f}%")
print("\nDONE")
