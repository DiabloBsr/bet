# -*- coding: utf-8 -*-
"""WF3 fixtures part 3 -- reconstruction des positions du cercle (Berger) et
PREDICTION des fixtures restantes d'une saison a partir des K premiers rounds.

Modele circle method (20 equipes, 19 positions + 1 pivot):
  - non-pivot pair (a,b) joue au round r (1-19) avec pos(a)+pos(b) = 2(r-1)+c (mod 19)
  - pivot joue au round r l'equipe en pos (r-1)+c' (mod 19)
  - retour au round r+19, home/away inverse
Etapes:
 1. par saison bien observee: identifier pivot (invariant quadruples) + resoudre positions
 2. valider: % des matchs observes conformes au modele
 3. WALK-FORWARD fixtures: K=2,3 premiers rounds observes -> predire pairings des rounds
    suivants; accuracy OOS
 4. regle home/away: fonction deterministe de (i, parite du round) ? pivot vs parite ?
 5. cross-saison: positions re-permutees ? pivot uniforme ? ordre du feed (1-10) vs i ?
"""
import sys, itertools, random
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from scipy import stats
from sqlalchemy import create_engine, text
from scraper.config import load_settings

random.seed(7)
INV2 = 10  # inverse de 2 mod 19

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
TEAMS_ALL = sorted(set(ev['team_a']) | set(ev['team_b']))

def leg1(g):
    """{frozenset(pair): (round, home, away)} pour rounds 1..19 (et miroir des rounds 20..38 ramenes)"""
    d = {}
    for _, row in g.iterrows():
        r = row['round']
        if r <= 19:
            d.setdefault(frozenset((row['team_a'], row['team_b'])), (r, row['team_a'], row['team_b']))
        else:
            # miroir: leg1 = (r-19) avec home/away inverse
            d.setdefault(frozenset((row['team_a'], row['team_b'])), (r - 19, row['team_b'], row['team_a']))
    return d

def find_pivot(d, teams):
    best_t, best_frac = None, -1
    for pivot in teams:
        others = [t for t in teams if t != pivot]
        ok = n = 0; trials = 0
        while trials < 3000 and n < 300:
            trials += 1
            a, b, c, dd = random.sample(others, 4)
            ks = [frozenset(x) for x in ((a, b), (c, dd), (a, c), (b, dd), (a, dd), (b, c))]
            if not all(k in d for k in ks):
                continue
            r = [d[k][0] - 1 for k in ks]
            n += 1
            if (r[0] + r[1]) % 19 == (r[2] + r[3]) % 19 == (r[4] + r[5]) % 19:
                ok += 1
        frac = ok / max(n, 1)
        if frac > best_frac:
            best_frac, best_t = frac, pivot
    return best_t, best_frac

def solve_positions(d, teams, pivot):
    """pos(a)+pos(b) = 2(r-1) mod 19 ; on fixe pos(ref)=0 puis on resout, et on
    verifie la coherence. Retourne pos dict ou None."""
    others = [t for t in teams if t != pivot]
    # graphe des contraintes pos(a)+pos(b)=s_ab
    s = {}
    for k, (r, h, a) in d.items():
        if pivot in k:
            continue
        x, y = tuple(k)
        s[(x, y)] = s[(y, x)] = (2 * (r - 1)) % 19
    pos = {others[0]: 0}
    # BFS
    changed = True
    while changed:
        changed = False
        for (x, y), v in s.items():
            if x in pos and y not in pos:
                pos[y] = (v - pos[x]) % 19
                changed = True
    if len(pos) < len(others):
        return None, None
    # verif coherence
    bad = sum(1 for (x, y), v in s.items() if (pos[x] + pos[y]) % 19 != v)
    return pos, bad // 2

# ------------------------------------------------- 1+2 : fit et validation par saison
print("================ 1-2. FIT BERGER PAR SAISON ================")
seas_cov = ev.groupby('season').size()
good_seasons = [s for s in seas_cov.index if seas_cov[s] >= 150]
fit = {}
for s in good_seasons:
    g = ev[ev['season'] == s]
    d = leg1(g)
    teams = sorted(set(g['team_a']) | set(g['team_b']))
    if len(teams) != 20:
        continue
    pivot, frac = find_pivot(d, teams)
    pos, bad = solve_positions(d, teams, pivot)
    if pos is None:
        print(f"S{s}: pivot={pivot} ({frac:.2f}) positions non resolues (donnees insuffisantes)")
        continue
    # offset pivot: pos(opp_pivot(r)) - (r-1) = const ?
    offs = []
    for k, (r, h, a) in d.items():
        if pivot in k:
            opp = [t for t in k if t != pivot][0]
            offs.append((pos[opp] - (r - 1)) % 19)
    off_c = Counter(offs)
    # validation complete: % de matchs observes conformes (pairing au bon round)
    n_ok = n_tot = 0
    for k, (r, h, a) in d.items():
        n_tot += 1
        if pivot in k:
            opp = [t for t in k if t != pivot][0]
            pred_r = None
            if len(off_c) == 1:
                pred_r = (pos[opp] - off_c.most_common(1)[0][0]) % 19 + 1
            n_ok += (pred_r == r)
        else:
            x, y = tuple(k)
            pred_r = (INV2 * (pos[x] + pos[y])) % 19 + 1
            n_ok += (pred_r == r)
    fit[s] = dict(pivot=pivot, pos=pos, off=off_c.most_common(1)[0][0], n_pairs=n_tot,
                  bad=bad, valid=n_ok / n_tot)
    print(f"S{s}: pivot={pivot:<16} contraintes violees={bad:<3} offset_pivot={dict(off_c)} "
          f"validation pairing={n_ok}/{n_tot} ({n_ok/n_tot:.3f})")

# ------------------------------------------------- 3. WALK-FORWARD fixtures
print("\n================ 3. WALK-FORWARD : K rounds observes -> predire le reste ================")
def predict_season(g, k_rounds):
    """utilise les k_rounds premiers rounds OBSERVES de la saison pour fitter,
    predit les pairings de tous les rounds > max(train rounds). Retourne (n_ok, n_tot)."""
    rounds_obs = sorted(g['round'].unique())
    train_r = rounds_obs[:k_rounds]
    test_r = [r for r in rounds_obs if r > max(train_r)]
    gtr = g[g['round'].isin(train_r)]
    teams = sorted(set(g['team_a']) | set(g['team_b']))
    if len(teams) != 20 or not test_r:
        return None
    d = leg1(gtr)
    # pivot: l'equipe impliquee dans une violation d'invariant... avec k petit, on
    # resout autrement: brute force sur les 20 pivots, garder ceux qui donnent un
    # systeme coherent ET complet
    cands = []
    for pivot in teams:
        pos, bad = solve_positions(d, teams, pivot)
        if pos is not None and bad == 0:
            # offset pivot coherent ?
            offs = set()
            for kk, (r, h, a) in d.items():
                if pivot in kk:
                    opp = [t for t in kk if t != pivot][0]
                    offs.add((pos[opp] - (r - 1)) % 19)
            if len(offs) <= 1:
                cands.append((pivot, pos, offs.pop() if offs else None))
    if len(cands) != 1:
        return ('ambiguous', len(cands), len(test_r))
    pivot, pos, off = cands[0]
    if off is None:
        return ('no_pivot_obs', 0, len(test_r))
    # predire: pour chaque round de test, generer les 10 pairings
    inv_pos = {v: k for k, v in pos.items()}
    n_ok = n_tot = 0
    for r in test_r:
        rr = g[g['round'] == r]
        obs_pairs = set(frozenset((a, b)) for a, b in zip(rr['team_a'], rr['team_b']))
        r1 = (r - 1) % 19  # round du leg correspondant (0-indexe)
        base = r1
        pred_pairs = set()
        opp_piv = inv_pos[(base + off) % 19]
        pred_pairs.add(frozenset((pivot, opp_piv)))
        for i in range(1, 10):
            t1 = inv_pos[(base + off + i) % 19]
            t2 = inv_pos[(base + off - i) % 19]
            pred_pairs.add(frozenset((t1, t2)))
        n_tot += len(obs_pairs)
        n_ok += len(obs_pairs & pred_pairs)
    return ('ok', n_ok, n_tot)

for K in (2, 3, 4):
    tot_ok = tot_n = 0; amb = 0; n_seas = 0
    for s in good_seasons:
        g = ev[ev['season'] == s]
        res = predict_season(g, K)
        if res is None:
            continue
        tag = res[0]
        if tag == 'ok':
            n_seas += 1; tot_ok += res[1]; tot_n += res[2]
        else:
            amb += 1
    print(f"K={K} rounds observes: saisons predites={n_seas}, ambigues/incompletes={amb}, "
          f"accuracy pairings OOS = {tot_ok}/{tot_n} ({tot_ok/max(tot_n,1):.4f})")

# ------------------------------------------------- 4. regle HOME/AWAY
print("\n================ 4. REGLE HOME/AWAY ================")
# avec fit complet: pour chaque match leg1 observe (round r, home h, away a):
#   i_h = (pos(h) - (r-1) - off) mod 19, i_a idem -> {i, 19-i}
# home = celui avec indice i dans 1..9 ? ou parite de r ?
rule_counts = defaultdict(Counter)
piv_counts = Counter()
for s, f in fit.items():
    if f['valid'] < 0.99:
        continue
    g = ev[ev['season'] == s]
    g1 = g[g['round'] <= 19]
    pos, pivot, off = f['pos'], f['pivot'], f['off']
    for _, row in g1.iterrows():
        r, h, a = row['round'], row['team_a'], row['team_b']
        if pivot in (h, a):
            piv_counts[('pivot_home' if h == pivot else 'pivot_away', r % 2)] += 1
            continue
        ih = (pos[h] - (r - 1) - off) % 19
        ia = (pos[a] - (r - 1) - off) % 19
        # i = min(ih, 19-ih) etc: la paire est (base+i, base-i) -> indices i et 19-i
        i = ih if ih <= 9 else 19 - ih
        home_is_plus = (ih == i)  # home est en base+i ?
        rule_counts[(i, r % 2)][home_is_plus] += 1
print("pivot home/away selon parite du round:", dict(piv_counts))
print("\npour paires non-pivot: home = equipe en position base+i ? (par i, parite round)")
rows = []
for (i, par), c in sorted(rule_counts.items()):
    rows.append((i, par, c.get(True, 0), c.get(False, 0)))
df_rule = pd.DataFrame(rows, columns=['i', 'round_pair', 'home=+i', 'home=-i'])
print(df_rule.to_string(index=False))

# ------------------------------------------------- 5. cross-saison
print("\n================ 5. CROSS-SAISON ================")
pivots = {s: f['pivot'] for s, f in fit.items() if f['valid'] >= 0.99}
print(f"pivots par saison ({len(pivots)} saisons):", Counter(pivots.values()).most_common())
# positions: meme equipe meme position d'une saison a l'autre ?
fits_ok = [s for s in sorted(fit) if fit[s]['valid'] >= 0.99]
agree = []
for s1, s2 in zip(fits_ok, fits_ok[1:]):
    p1, p2 = fit[s1]['pos'], fit[s2]['pos']
    common = set(p1) & set(p2)
    same = sum(1 for t in common if p1[t] == p2[t])
    # relation affine pos2 = a*pos1+b ?
    aff = 0
    for a_ in range(1, 19):
        for b_ in range(19):
            n_match = sum(1 for t in common if p2[t] == (a_ * p1[t] + b_) % 19)
            aff = max(aff, n_match)
    agree.append((s1, s2, same, len(common), aff))
for s1, s2, same, n, aff in agree[:10]:
    print(f"S{s1}->S{s2}: pos identiques {same}/{n}, meilleure relation affine {aff}/{n}")

# ordre du feed (1-10) dans le bloc vs structure du cercle
print("\n-- position dans le feed (ordre id) vs indice i du cercle --")
feed_i = defaultdict(Counter)
for s in fits_ok[:6]:
    f = fit[s]
    g = ev[ev['season'] == s].copy()
    g['posfeed'] = g.groupby('round')['id'].rank(method='first').astype(int)
    g = g[g.groupby('round')['id'].transform('size') == 10]
    pos, pivot, off = f['pos'], f['pivot'], f['off']
    for _, row in g.iterrows():
        r = (row['round'] - 1) % 19 + 1
        h, a = row['team_a'], row['team_b']
        if pivot in (h, a):
            feed_i[row['posfeed']]['pivot'] += 1
        else:
            ih = (pos[h] - (r - 1) - off) % 19
            i = ih if ih <= 9 else 19 - ih
            feed_i[row['posfeed']][i] += 1
for pf in sorted(feed_i):
    print(f"feed#{pf}: {dict(sorted(feed_i[pf].items(), key=lambda kv: str(kv[0])))}")
print("\nOK part3 done")
