# -*- coding: utf-8 -*-
"""WF3 fixtures part 4 -- fit Berger COMPLET avec constante de rotation c.

Modele: non-pivot pair (a,b) joue leg1 au round r (1..19) ssi
        p(a)+p(b) = 2(r-1) + c   (mod 19)
        pivot joue au round r l'equipe avec p = (r-1) + c*INV2 + k' (verifier k'=0)
        leg2 = round r+19, home/away inverse.
Gauge: p(ref)=0 -> (c, p) unique.
"""
import sys, random
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

def leg1(g):
    d = {}
    for _, row in g.iterrows():
        r = row['round']
        if r <= 19:
            d.setdefault(frozenset((row['team_a'], row['team_b'])), (r, row['team_a'], row['team_b']))
        else:
            d.setdefault(frozenset((row['team_a'], row['team_b'])), (r - 19, row['team_b'], row['team_a']))
    return d

def solve(d, teams, pivot, c):
    """resout p avec p(a)+p(b)=2(r-1)+c. retourne (pos, n_violations, n_resolu)"""
    others = [t for t in teams if t != pivot]
    edges = []
    for k, (r, h, a) in d.items():
        if pivot in k:
            continue
        x, y = tuple(k)
        edges.append((x, y, (2 * (r - 1) + c) % 19))
    pos = {others[0]: 0}
    changed = True
    while changed:
        changed = False
        for x, y, v in edges:
            if x in pos and y not in pos:
                pos[y] = (v - pos[x]) % 19; changed = True
            elif y in pos and x not in pos:
                pos[x] = (v - pos[y]) % 19; changed = True
    bad = sum(1 for x, y, v in edges if x in pos and y in pos and (pos[x] + pos[y]) % 19 != v)
    return pos, bad, len(pos)

def fit_season(g, verbose=False):
    teams = sorted(set(g['team_a']) | set(g['team_b']))
    if len(teams) != 20:
        return None
    d = leg1(g)
    best = None
    for pivot in teams:
        for c in range(19):
            pos, bad, nres = solve(d, teams, pivot, c)
            if nres == 19 and bad == 0:
                # positions must be a permutation of Z19
                if len(set(pos.values())) != 19:
                    continue
                # pivot offset
                ks = set()
                for k, (r, h, a) in d.items():
                    if pivot in k:
                        opp = [t for t in k if t != pivot][0]
                        ks.add((pos[opp] - (r - 1) - (c * INV2)) % 19)
                if len(ks) > 1:
                    continue
                cand = dict(pivot=pivot, c=c, pos=pos, kprime=(ks.pop() if ks else None))
                if best is None:
                    best = cand
                else:
                    return 'AMBIGU'
    return best

# ------------------------------------------------- fit toutes saisons bien couvertes
print("================ FIT EXACT PAR SAISON (pivot, c, k') ================")
seas_cov = ev.groupby('season').size()
good = [s for s in seas_cov.index if seas_cov[s] >= 120]
fits = {}
n_amb = 0
for s in good:
    g = ev[ev['season'] == s]
    f = fit_season(g)
    if f == 'AMBIGU':
        n_amb += 1
        print(f"S{s}: AMBIGU (plusieurs fits exacts)")
    elif f is None:
        print(f"S{s}: pas de fit exact (donnees bruitees ou saison mal segmentee)")
    else:
        # validation sur TOUS les matchs observes de la saison
        d = leg1(g)
        n_ok = 0
        for k, (r, h, a) in d.items():
            if f['pivot'] in k:
                opp = [t for t in k if t != f['pivot']][0]
                pred_r = (f['pos'][opp] - f['c'] * INV2 - f['kprime']) % 19 + 1
            else:
                x, y = tuple(k)
                pred_r = (INV2 * (f['pos'][x] + f['pos'][y] - f['c'])) % 19 + 1
            n_ok += (pred_r == r)
        fits[s] = f
        f['valid'] = n_ok / len(d)
        print(f"S{s}: pivot={f['pivot']:<16} c={f['c']:<2} k'={f['kprime']} validation={n_ok}/{len(d)} ({f['valid']:.3f}) n_matchs={len(g)}")
print(f"\nfits exacts: {len(fits)}/{len(good)} saisons testees, ambigus: {n_amb}")
print("k' (offset pivot au-dela de c*INV2):", Counter(f['kprime'] for f in fits.values()).most_common())

# ------------------------------------------------- walk-forward K rounds
print("\n================ WALK-FORWARD : K premiers rounds observes ================")
def predict_wf(g, K, enforce_kprime=None):
    rounds_obs = sorted(g['round'].unique())
    if len(rounds_obs) < K + 3:
        return None
    train_r = rounds_obs[:K]
    test_r = [r for r in rounds_obs if r > max(train_r)]
    gtr = g[g['round'].isin(train_r)]
    teams = sorted(set(g['team_a']) | set(g['team_b']))
    if len(set(gtr['team_a']) | set(gtr['team_b'])) != 20 or len(teams) != 20:
        return None
    d = leg1(gtr)
    cands = []
    for pivot in teams:
        for c in range(19):
            pos, bad, nres = solve(d, teams, pivot, c)
            if bad > 0 or nres < 19 or len(set(pos.values())) != 19:
                continue
            ks = set()
            for k, (r, h, a) in d.items():
                if pivot in k:
                    opp = [t for t in k if t != pivot][0]
                    ks.add((pos[opp] - (r - 1) - c * INV2) % 19)
            if len(ks) > 1:
                continue
            kp = ks.pop() if ks else None
            if enforce_kprime is not None:
                if kp is not None and kp != enforce_kprime:
                    continue
                kp = enforce_kprime
            cands.append((pivot, c, pos, kp))
    if len(cands) != 1:
        return ('ambiguous', len(cands))
    pivot, c, pos, kp = cands[0]
    if kp is None:
        return ('no_kprime', 0)
    inv_pos = {v: t for t, v in pos.items()}
    n_ok = n_tot = 0
    for r in test_r:
        rr = g[g['round'] == r]
        obs = set(frozenset((a, b)) for a, b in zip(rr['team_a'], rr['team_b']))
        r1 = (r - 1) % 19
        base = (r1 + c * INV2 + kp) % 19  # position de l'adversaire du pivot
        pred = {frozenset((pivot, inv_pos[base]))}
        for i in range(1, 10):
            pred.add(frozenset((inv_pos[(base + i) % 19], inv_pos[(base - i) % 19])))
        n_tot += len(obs)
        n_ok += len(obs & pred)
    return ('ok', n_ok, n_tot)

for K in (1, 2, 3, 4):
    tot_ok = tot_n = n_seas = amb = nores = 0
    for s in good:
        g = ev[ev['season'] == s]
        res = predict_wf(g, K)
        if res is None:
            continue
        if res[0] == 'ok':
            n_seas += 1; tot_ok += res[1]; tot_n += res[2]
        elif res[0] == 'ambiguous':
            amb += 1
        else:
            nores += 1
    acc = tot_ok / max(tot_n, 1)
    print(f"K={K}: saisons predites={n_seas} (ambigues={amb}, sans pivot vu={nores}) "
          f"accuracy pairings OOS={tot_ok}/{tot_n} ({acc:.4f})")

# ------------------------------------------------- HOME/AWAY rule
print("\n================ REGLE HOME/AWAY (leg1, fits valides) ================")
rule = defaultdict(Counter)
piv_rule = Counter()
for s, f in fits.items():
    if f['valid'] < 0.995:
        continue
    g = ev[ev['season'] == s]
    pos, pivot, c, kp = f['pos'], f['pivot'], f['c'], f['kprime']
    for _, row in g[g['round'] <= 19].iterrows():
        r, h, a = row['round'], row['team_a'], row['team_b']
        base = ((r - 1) + c * INV2 + kp) % 19
        if pivot == h:
            piv_rule[('pivot_HOME', r % 2)] += 1; continue
        if pivot == a:
            piv_rule[('pivot_away', r % 2)] += 1; continue
        ih = (pos[h] - base) % 19
        i = ih if ih <= 9 else 19 - ih
        rule[(i, r % 2)]['home=+i' if ih == i else 'home=-i'] += 1
print("match du pivot (parite round):", dict(piv_rule))
rows = [(i, par, cc.get('home=+i', 0), cc.get('home=-i', 0)) for (i, par), cc in sorted(rule.items())]
print(pd.DataFrame(rows, columns=['i', 'r%2', 'home=+i', 'home=-i']).to_string(index=False))

# ------------------------------------------------- cross-saison
print("\n================ CROSS-SAISON ================")
ok_fits = {s: f for s, f in fits.items() if f['valid'] >= 0.995}
print(f"pivots ({len(ok_fits)} saisons):", Counter(f['pivot'] for f in ok_fits.values()).most_common())
print("c:", Counter(f['c'] for f in ok_fits.values()).most_common())
sl = sorted(ok_fits)
same_pos_tot = aff_tot = n_pairs_cmp = 0
for s1, s2 in zip(sl, sl[1:]):
    p1, p2 = ok_fits[s1]['pos'], ok_fits[s2]['pos']
    common = set(p1) & set(p2)
    same = sum(1 for t in common if p1[t] == p2[t])
    aff = 0
    for a_ in range(1, 19):
        for b_ in range(19):
            aff = max(aff, sum(1 for t in common if p2[t] == (a_ * p1[t] + b_) % 19))
    same_pos_tot += same; aff_tot += aff; n_pairs_cmp += len(common)
    print(f"S{s1}->S{s2}: pos identiques {same}/{len(common)}, meilleur affine {aff}/{len(common)}, "
          f"pivot {ok_fits[s1]['pivot']} -> {ok_fits[s2]['pivot']}")
exp_same = n_pairs_cmp / 19
print(f"\ntotal pos identiques: {same_pos_tot}/{n_pairs_cmp} (attendu hasard ~{exp_same:.1f})")
p_same = stats.binomtest(same_pos_tot, n_pairs_cmp, 1/19).pvalue
print(f"binomial p (uniforme 1/19) = {p_same:.4f}")

# feed order vs i
print("\n-- ordre du feed (1-10) vs indice i du cercle --")
feed_i = defaultdict(Counter)
for s, f in list(ok_fits.items())[:10]:
    g = ev[ev['season'] == s].copy()
    g['posfeed'] = g.groupby('round')['id'].rank(method='first').astype(int)
    g = g[g.groupby('round')['id'].transform('size') == 10]
    pos, pivot, c, kp = f['pos'], f['pivot'], f['c'], f['kprime']
    for _, row in g.iterrows():
        r = (row['round'] - 1) % 19 + 1
        base = ((r - 1) + c * INV2 + kp) % 19
        h, a = row['team_a'], row['team_b']
        if pivot in (h, a):
            feed_i[row['posfeed']]['P'] += 1
        else:
            ih = (pos[h] - base) % 19
            i = ih if ih <= 9 else 19 - ih
            feed_i[row['posfeed']][i] += 1
for pf in sorted(feed_i):
    print(f"feed#{pf}: {dict(sorted(feed_i[pf].items(), key=lambda kv: str(kv[0])))}")
print("\nOK part4 done")
