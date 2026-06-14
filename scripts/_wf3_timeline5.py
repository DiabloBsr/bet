# -*- coding: utf-8 -*-
"""WF3 — TIMELINE part 5 : validation walk-forward de la règle FTTS '1' heavy-fav.

Règle primaire (choisie sur train uniquement, quintile 1 de cote) :
    BET FTTS '1' quand cote('1') <= 1.46
Validation OOS (30% final, ordre chronologique), settlement timeline ET clean-only.
+ stabilité temporelle du biais favorite-first (5 bins, CUSUM binomial).
"""
import sys, json
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import create_engine, text
from scraper.config import load_settings

eng = create_engine(load_settings().db_url)
q = """
SELECT e.id, e.team_a, e.team_b, e.round_info, e.expected_start,
       r.score_a, r.score_b, r.goals_json
FROM events e JOIN results r ON r.event_id = e.id
WHERE e.round_info != '0'
"""
df = pd.read_sql(text(q), eng)
df = df.sort_values('id').drop_duplicates(subset=['team_a', 'team_b', 'expected_start'], keep='first')
df = df.dropna(subset=['score_a', 'score_b']).reset_index(drop=True)
qo = """
SELECT o.event_id, o.odds_home, o.odds_draw, o.odds_away, o.extra_markets
FROM odds_snapshots o
JOIN (SELECT event_id, MIN(id) mid FROM odds_snapshots GROUP BY event_id) m ON m.mid = o.id
"""
od = pd.read_sql(text(qo), eng)
df = df.merge(od, left_on='id', right_on='event_id', how='left')
def parse_goals(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    try:
        g = json.loads(s) if isinstance(s, str) else s
        return g if isinstance(g, list) else None
    except Exception:
        return None
df['goals'] = df.goals_json.apply(parse_goals)
df['total'] = (df.score_a + df.score_b).astype(int)
df['em'] = df.extra_markets.apply(lambda s: json.loads(s) if isinstance(s, str) else s)
def get_ftts(em):
    if em and isinstance(em, dict):
        m = em.get('FTTS')
        if m and all(k in m for k in ['1', '2', 'Pas de but']):
            return m
    return None
df['ftts_odds'] = df.em.apply(get_ftts)
def ftts_outcome(row):
    if row.total == 0:
        return 'Pas de but'
    if row.goals is None or not row.goals:
        return None
    g0 = min(row.goals, key=lambda g: int(g['minute']))
    return '1' if g0['team'] == 'Home' else '2'
df['ftts_out'] = df.apply(ftts_outcome, axis=1)
df['tl_h'] = df.goals.apply(lambda gs: sum(1 for g in gs if g['team'] == 'Home') if gs else None)
df['tl_a'] = df.goals.apply(lambda gs: sum(1 for g in gs if g['team'] == 'Away') if gs else None)
df['clean'] = (df.tl_h == df.score_a) & (df.tl_a == df.score_b) | (df.total == 0)

S = df[df.ftts_odds.notna() & df.ftts_out.notna()].sort_values('expected_start').reset_index(drop=True)
S['c1'] = S.ftts_odds.apply(lambda d: d['1'])
S['hit1'] = (S.ftts_out == '1')
ntr = int(len(S) * 0.7)
TR, TE = S.iloc[:ntr], S.iloc[ntr:]
print(f"total n={len(S)} train={len(TR)} OOS={len(TE)}")
print(f"période train: {TR.expected_start.min()} -> {TR.expected_start.max()}")
print(f"période OOS  : {TE.expected_start.min()} -> {TE.expected_start.max()}")

print("\n--- REGLE PRIMAIRE: bet FTTS '1' si c1 <= 1.46 ---")
for nm, T in [('train', TR), ('OOS', TE)]:
    for variant, mask in [('all rows', T.c1 <= 1.46), ('clean-only', (T.c1 <= 1.46) & T.clean)]:
        g = T[mask]
        if len(g) == 0:
            continue
        roi = (g.hit1 * g.c1 - 1).mean()
        wr = g.hit1.mean()
        # test binomial: WR observé vs break-even 1/cote_mean
        be = float((1 / g.c1).mean())
        pb = stats.binomtest(int(g.hit1.sum()), len(g), be, alternative='greater').pvalue
        print(f"{nm:5s} {variant:10s}: n={len(g):4d} WR={wr:.4f} (break-even={be:.4f}) cote={g.c1.mean():.3f} ROI={roi*100:+.2f}% p(WR>BE)={pb:.4f}")

print("\n--- robustesse seuils (info, pas la règle primaire) ---")
for th in [1.35, 1.40, 1.45, 1.50, 1.55, 1.60]:
    gtr = TR[TR.c1 <= th]; gte = TE[TE.c1 <= th]
    if len(gtr) < 50 or len(gte) < 30:
        continue
    print(f"  c1<={th}: train n={len(gtr):4d} ROI={(gtr.hit1*gtr.c1-1).mean()*100:+6.2f}% | OOS n={len(gte):4d} ROI={(gte.hit1*gte.c1-1).mean()*100:+6.2f}%")

print("\n--- stabilité temporelle du biais favorite-first ---")
both = S[(S.score_a > 0) & (S.score_b > 0) & S.clean & S.odds_home.notna()].copy()
both['hfirst'] = both.ftts_out == '1'
both['exp_rand'] = both.score_a / (both.score_a + both.score_b)
both['bin'] = pd.qcut(np.arange(len(both)), 5, labels=False)
for b, g in both.groupby('bin'):
    z = (g.hfirst.sum() - g.exp_rand.sum()) / np.sqrt((g.exp_rand * (1 - g.exp_rand)).sum())
    print(f"  bin {b}: n={len(g):4d} P(hfirst)={g.hfirst.mean():.4f} attendu_random={g.exp_rand.mean():.4f} z={z:+.2f}")
# CUSUM sur hit1 des paris c1<=1.46 (dérive du WR dans le temps ?)
bets = S[S.c1 <= 1.46].reset_index(drop=True)
dev = (bets.hit1 - bets.hit1.mean()).cumsum()
print(f"\nCUSUM WR (c1<=1.46, n={len(bets)}): max|dev|={dev.abs().max():.1f}")
# seuil approx KS-style: 1.36*sqrt(n)*sd
sd = bets.hit1.std()
print(f"  seuil ~1.36*sd*sqrt(n)={1.36*sd*np.sqrt(len(bets)):.1f} -> {'DERIVE' if dev.abs().max() > 1.36*sd*np.sqrt(len(bets)) else 'stable'}")

# EV théorique du mécanisme : P(hfirst) modèle interleave+boost vs cote
print("\n--- modèle: quel boost le moteur applique-t-il au 1er buteur ? ---")
# fit logistic: logit P(hfirst) = logit(h/(h+a)) + beta * log(odds_away/odds_home)
from sklearn.linear_model import LogisticRegression
X = np.column_stack([
    np.log(both.exp_rand / (1 - both.exp_rand)),
    np.log(both.odds_away / both.odds_home),
])
lr = LogisticRegression(C=1e6)
lr.fit(X, both.hfirst)
print(f"logit(P hfirst) = {lr.intercept_[0]:+.3f} + {lr.coef_[0][0]:.3f}*logit(h/(h+a)) + {lr.coef_[0][1]:+.3f}*log(oddsA/oddsH)")
print("(coef 2 > 0 => boost du favori au 1er but, au-delà de la composition du score final)")
from scipy.stats import chi2 as chi2dist
# LRT for beta2
lr0 = LogisticRegression(C=1e6); lr0.fit(X[:, :1], both.hfirst)
ll1 = -len(both) * stats.entropy([1]) if False else None
from sklearn.metrics import log_loss
ll_full = -log_loss(both.hfirst, lr.predict_proba(X)[:, 1], normalize=False)
ll_red = -log_loss(both.hfirst, lr0.predict_proba(X[:, :1])[:, 1], normalize=False)
lrt = 2 * (ll_full - ll_red)
print(f"LRT beta2: stat={lrt:.2f} p={chi2dist.sf(lrt, 1):.3e}")
print("\nDONE")
