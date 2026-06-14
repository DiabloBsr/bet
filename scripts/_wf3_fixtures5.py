# -*- coding: utf-8 -*-
"""WF3 fixtures part 5 --
A. Walk-forward ameliore: k'=0 impose + regle home/away comme contrainte
   -> K minimal de rounds observes pour predire toute la saison (pairing + H/A)
B. S37 vs S45 (meme pivot Newcastle, meme c=2): calendriers identiques ?!
C. Permutation des positions entre saisons: points fixes (toutes paires de saisons),
   adjacentes vs eloignees ; uniformite pivot et c
"""
import sys, random, itertools
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from scipy import stats
from sqlalchemy import create_engine, text
from scraper.config import load_settings

random.seed(7)
INV2 = 10
eng = create_engine(load_settings().db_url)
ev = pd.read_sql(text("""
    SELECT e.id, e.team_a, e.team_b, e.round_info, e.expected_start
    FROM events e WHERE e.round_info != '0'
"""), eng)
ev['expected_start'] = pd.to_datetime(ev['expected_start'])
ev = ev.sort_values('id').drop_duplicates(['team_a', 'team_b', 'expected_start'], keep='first')
ev['round'] = ev['round_info'].astype(int)
ev = ev.sort_values(['expected_start', 'id']).reset_index(drop=True)
blk = ev.groupby('expected_start').agg(round=('round', 'first')).reset_index().sort_values('expected_start').reset_index(drop=True)
season, prev = 0, None
sids = []
for r in blk['round']:
    if prev is not None and r < prev:
        season += 1
    sids.append(season); prev = r
blk['season'] = sids
ev = ev.merge(blk[['expected_start', 'season']], on='expected_start')

def leg1_rows(g):
    """liste (r_leg1, home_leg1, away_leg1, r_observe) ; r>19 -> miroir."""
    out = []
    for _, row in g.iterrows():
        r = row['round']
        if r <= 19:
            out.append((r, row['team_a'], row['team_b']))
        else:
            out.append((r - 19, row['team_b'], row['team_a']))
    return out

def ha_ok(pos_h, base, r, is_pivot_home=None):
    """regle: r pair -> home en base+i (i 1..9) ; r impair -> home en base-i."""
    d = (pos_h - base) % 19
    if r % 2 == 0:
        return 1 <= d <= 9
    else:
        return 10 <= d <= 18

def fit_from_rounds(gtr, teams):
    """fit (pivot, c, pos) avec k'=0 et regle H/A imposees. retourne liste candidats."""
    rows = leg1_rows(gtr)
    cands = []
    for pivot in teams:
        # pivot H/A rule: pivot home ssi r impair
        ok_piv = True
        piv_rounds = {}
        for r, h, a in rows:
            if pivot == h or pivot == a:
                if (pivot == h) != (r % 2 == 1):
                    ok_piv = False; break
                piv_rounds[r] = a if pivot == h else h
        if not ok_piv:
            continue
        for c in range(19):
            pos = {}
            consistent = True
            # pivot opponents: pos = (r-1)+c*INV2
            for r, opp in piv_rounds.items():
                p = ((r - 1) + c * INV2) % 19
                if pos.get(opp, p) != p:
                    consistent = False; break
                pos[opp] = p
            if not consistent:
                continue
            # propagate pair constraints pos(h)+pos(a) = 2*base, base=(r-1)+c*INV2
            edges = [(h, a, r) for r, h, a in rows if pivot not in (h, a)]
            changed = True
            while changed and consistent:
                changed = False
                for h, a, r in edges:
                    base = ((r - 1) + c * INV2) % 19
                    s = (2 * base) % 19
                    if h in pos and a in pos:
                        if (pos[h] + pos[a]) % 19 != s:
                            consistent = False; break
                    elif h in pos:
                        pos[a] = (s - pos[h]) % 19; changed = True
                    elif a in pos:
                        pos[h] = (s - pos[a]) % 19; changed = True
            if not consistent:
                continue
            # H/A check on assigned pairs
            for h, a, r in edges:
                if h in pos:
                    base = ((r - 1) + c * INV2) % 19
                    if not ha_ok(pos[h], base, r):
                        consistent = False; break
            if not consistent:
                continue
            if len(pos) == 19 and len(set(pos.values())) == 19:
                cands.append((pivot, c, dict(pos)))
    return cands

def predict_full(pivot, c, pos):
    """genere les 380 matchs (round -> liste (home, away))."""
    inv = {v: t for t, v in pos.items()}
    sched = {}
    for r in range(1, 20):
        base = ((r - 1) + c * INV2) % 19
        ms = []
        # pivot: home si r impair
        opp = inv[base]
        ms.append((pivot, opp) if r % 2 == 1 else (opp, pivot))
        for i in range(1, 10):
            tp, tm = inv[(base + i) % 19], inv[(base - i) % 19]
            ms.append((tp, tm) if r % 2 == 0 else (tm, tp))
        sched[r] = ms
        sched[r + 19] = [(b, a) for a, b in ms]
    return sched

# ---------------------------------------------------------- A. walk-forward
print("================ A. WALK-FORWARD k'=0 + regle H/A ================")
seas_cov = ev.groupby('season').size()
good = [s for s in seas_cov.index if seas_cov[s] >= 120]
for K in (1, 2, 3):
    n_uniq = n_amb = 0
    ok_pair = tot_pair = ok_ha = 0
    for s in good:
        g = ev[ev['season'] == s]
        rounds_obs = sorted(g['round'].unique())
        if len(rounds_obs) < K + 3:
            continue
        train_r = rounds_obs[:K]
        test_r = [r for r in rounds_obs if r > max(train_r)]
        gtr = g[g['round'].isin(train_r)]
        teams = sorted(set(g['team_a']) | set(g['team_b']))
        if len(teams) != 20 or len(set(gtr['team_a']) | set(gtr['team_b'])) != 20:
            continue
        cands = fit_from_rounds(gtr, teams)
        if len(cands) != 1:
            n_amb += 1
            continue
        n_uniq += 1
        sched = predict_full(*cands[0])
        for r in test_r:
            rr = g[g['round'] == r]
            obs = set(zip(rr['team_a'], rr['team_b']))
            pred = set(sched[r])
            obs_unord = set(frozenset(x) for x in obs)
            pred_unord = set(frozenset(x) for x in pred)
            tot_pair += len(obs)
            ok_pair += len(obs_unord & pred_unord)
            ok_ha += len(obs & pred)
    print(f"K={K}: saisons resolues uniques={n_uniq}, ambigues={n_amb} | "
          f"pairing OOS {ok_pair}/{tot_pair} ({ok_pair/max(tot_pair,1):.4f}) | "
          f"pairing+H/A exact {ok_ha}/{tot_pair} ({ok_ha/max(tot_pair,1):.4f})")

# ---------------------------------------------------------- B. S37 vs S45
print("\n================ B. SAISONS AU MEME (pivot, c) : CALENDRIER IDENTIQUE ? ================")
fits = {}
for s in good:
    g = ev[ev['season'] == s]
    teams = sorted(set(g['team_a']) | set(g['team_b']))
    if len(teams) != 20:
        continue
    cands = fit_from_rounds(g, teams)
    if len(cands) == 1:
        fits[s] = cands[0]
print(f"fits exacts (toute la saison, H/A inclus): {len(fits)} saisons")
by_pc = defaultdict(list)
for s, (pivot, c, pos) in fits.items():
    by_pc[(pivot, c)].append(s)
for (pivot, c), ss in sorted(by_pc.items()):
    if len(ss) > 1:
        print(f"(pivot={pivot}, c={c}): saisons {ss}")
        # positions identiques ?
        for s1, s2 in itertools.combinations(ss, 2):
            p1, p2 = fits[s1][2], fits[s2][2]
            same = sum(1 for t in p1 if p1[t] == p2[t])
            print(f"   S{s1} vs S{s2}: positions identiques {same}/19 -> calendrier {'IDENTIQUE' if same==19 else 'different'}")

# ---------------------------------------------------------- C. permutations inter-saisons
print("\n================ C. CARACTERE ALEATOIRE DES PERMUTATIONS ================")
sl = sorted(fits)
# pivot inclus comme 'position 19'
def full_perm(s):
    pivot, c, pos = fits[s]
    d = dict(pos); d[pivot] = 19
    return d
fp_adj, fp_far = [], []
for i, j in itertools.combinations(range(len(sl)), 2):
    s1, s2 = sl[i], sl[j]
    d1, d2 = full_perm(s1), full_perm(s2)
    fx = sum(1 for t in d1 if d1[t] == d2[t])
    (fp_adj if j == i + 1 else fp_far).append(fx)
print(f"points fixes (20 slots, pivot inclus): adjacentes n={len(fp_adj)} mean={np.mean(fp_adj):.2f} | "
      f"eloignees n={len(fp_far)} mean={np.mean(fp_far):.2f} | attendu permutation uniforme = 1.0")
mw = stats.mannwhitneyu(fp_adj, fp_far, alternative='two-sided')
print(f"Mann-Whitney adj vs far: p={mw.pvalue:.4f}")
allfx = fp_adj + fp_far
# distribution vs Poisson(1)
cnt = Counter(allfx)
print("distribution points fixes:", dict(sorted(cnt.items())))
ks = [k for k in range(max(cnt) + 1)]
obs = np.array([cnt.get(k, 0) for k in ks], dtype=float)
exp = np.array([stats.poisson.pmf(k, 1) for k in ks]) * len(allfx)
exp[-1] += (1 - stats.poisson.cdf(max(cnt), 1)) * len(allfx)
mask = exp >= 1
chi2 = ((obs[mask] - exp[mask]) ** 2 / exp[mask]).sum()
print(f"chi2 vs Poisson(1): chi2={chi2:.2f} (dof~{mask.sum()-1}) p={1-stats.chi2.cdf(chi2, mask.sum()-1):.4f} "
      f"mean={np.mean(allfx):.3f} (theorie 1.0)")

# uniformite du pivot
pc = Counter(f[0] for f in fits.values())
teams20 = sorted(set(ev['team_a']) | set(ev['team_b']))
obs_p = np.array([pc.get(t, 0) for t in teams20])
chi2p, pp = stats.chisquare(obs_p)
print(f"\npivot sur {len(fits)} saisons: chi2 uniformite={chi2p:.2f} p={pp:.4f}")
cc = Counter(f[1] for f in fits.values())
obs_c = np.array([cc.get(k, 0) for k in range(19)])
chi2c, pcv = stats.chisquare(obs_c)
print(f"c sur {len(fits)} saisons: chi2 uniformite={chi2c:.2f} p={pcv:.4f}")
print("\nOK part5 done")
