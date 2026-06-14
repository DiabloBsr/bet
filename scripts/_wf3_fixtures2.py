# -*- coding: utf-8 -*-
"""WF3 fixtures part 2 -- structure round-robin + test Berger sur saisons PARTIELLES.

A. Verifs round-robin sur donnees partielles:
   - une equipe joue exactement 1 fois par round observe
   - paires vues 2x dans une saison : delta de rounds, home/away inverse ?
   - leg1 dans 1-19, leg2 dans 20-38 ?
B. Test Berger label-free (invariant des quadruples):
   circle method 20 equipes: pair non-pivot (a,b) -> round r avec a+b = 2(r-1) mod 19
   => pour 4 equipes non-pivot a,b,c,d :
      r(a,b)+r(c,d) = r(a,c)+r(b,d) = r(a,d)+r(b,c)  (mod 19)  [r 0-indexe]
   Teste chaque candidat pivot ; si Berger, un pivot donne 100% de quadruples OK.
C. Le calendrier d'une saison est-il un RELABELING (permutation d'equipes)
   du calendrier d'une autre saison ?
D. Grille temporelle : round_diff == gap/120 ? cycle saison = 37*120+300 = 4740s ?
"""
import sys, itertools, random
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from scipy import stats
from sqlalchemy import create_engine, text
from scraper.config import load_settings

random.seed(42)
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

# ---------------------------------------------------------- A. round-robin checks
print("================ A. ROUND-ROBIN (donnees partielles) ================")
viol_once = 0; tot_team_rounds = 0
for (s, r), g in ev.groupby(['season', 'round']):
    c = Counter(list(g['team_a']) + list(g['team_b']))
    tot_team_rounds += len(c)
    viol_once += sum(1 for v in c.values() if v > 1)
print(f"equipe jouant 2x dans le meme round: {viol_once}/{tot_team_rounds} apparitions")

deltas = Counter(); mirror = Counter(); leg_split = Counter(); same_ordered = 0
pair3 = 0
for s, g in ev.groupby('season'):
    seen = defaultdict(list)
    for _, row in g.iterrows():
        seen[frozenset((row['team_a'], row['team_b']))].append((row['round'], row['team_a'], row['team_b']))
    for k, v in seen.items():
        if len(v) >= 3:
            pair3 += 1
        if len(v) == 2:
            v.sort()
            (r1, a1, b1), (r2, a2, b2) = v
            deltas[r2 - r1] += 1
            mirror['inverse' if (a1, b1) == (b2, a2) else 'meme'] += 1
            leg_split[(r1 <= 19, r2 >= 20)] += 1
print(f"paires vues 3+ fois dans une saison: {pair3}")
print(f"delta rounds leg2-leg1: {dict(sorted(deltas.items()))}")
print(f"home/away leg2 vs leg1: {dict(mirror)}")
print(f"(leg1<=19, leg2>=20): {dict(leg_split)}")

# delta=19 exactement ? sinon distribution
n2 = sum(deltas.values())
print(f"part delta==19: {deltas.get(19,0)}/{n2}")

# ---------------------------------------------------------- B. Berger quadruple invariant
print("\n================ B. TEST BERGER (invariant quadruples, label-free) ================")
def first_leg_rounds(g):
    """round du 1er affrontement (round<=19) pour chaque paire non-ordonnee."""
    d = {}
    for _, row in g[g['round'] <= 19].iterrows():
        k = frozenset((row['team_a'], row['team_b']))
        if k not in d or row['round'] < d[k]:
            d[k] = row['round']
    return d

def berger_pivot_scan(d, teams, n_quad=4000):
    """pour chaque pivot candidat, fraction de quadruples (a,b,c,d) non-pivot
    satisfaisant r(ab)+r(cd) == r(ac)+r(bd) == r(ad)+r(bc) (mod 19), r 0-indexe."""
    res = {}
    for pivot in teams:
        others = [t for t in teams if t != pivot]
        ok = bad = 0
        trials = 0
        while trials < n_quad and (ok + bad) < 800:
            trials += 1
            a, b, c, dd = random.sample(others, 4)
            ks = [frozenset(x) for x in ((a, b), (c, dd), (a, c), (b, dd), (a, dd), (b, c))]
            if not all(k in d for k in ks):
                continue
            r = [d[k] - 1 for k in ks]
            s1, s2, s3 = (r[0] + r[1]) % 19, (r[2] + r[3]) % 19, (r[4] + r[5]) % 19
            if s1 == s2 == s3:
                ok += 1
            else:
                bad += 1
        res[pivot] = (ok, ok + bad)
    return res

# saisons les mieux observees
seas_cov = ev.groupby('season').size().sort_values(ascending=False)
best = [s for s in seas_cov.index if seas_cov[s] >= 300][:8]
print(f"saisons testees (>=300 matchs observes): {best}")
for s in best[:4]:
    g = ev[ev['season'] == s]
    d = first_leg_rounds(g)
    teams = sorted(set(g['team_a']) | set(g['team_b']))
    res = berger_pivot_scan(d, teams)
    top = sorted(res.items(), key=lambda kv: -(kv[1][0] / max(kv[1][1], 1)))
    line = ", ".join(f"{t}:{o}/{n}" for t, (o, n) in top[:3])
    worst = ", ".join(f"{t}:{o}/{n}" for t, (o, n) in top[-2:])
    print(f"S{s}: top pivots [{line}]  pires [{worst}]")

# baseline attendue au hasard
rand_ok = 0; rand_n = 20000
for _ in range(rand_n):
    r = [random.randrange(19) for _ in range(6)]
    if (r[0] + r[1]) % 19 == (r[2] + r[3]) % 19 == (r[4] + r[5]) % 19:
        rand_ok += 1
print(f"baseline aleatoire attendue: {rand_ok/rand_n:.4f} (~1/361={1/361:.4f})")

# ---------------------------------------------------------- B2. invariant SANS pivot (mod 19 sur les 38 rounds, sur tous quadruples)
print("\n-- variante: invariant teste sur les 20 equipes sans exclusion --")
for s in best[:2]:
    g = ev[ev['season'] == s]
    d = first_leg_rounds(g)
    teams = sorted(set(g['team_a']) | set(g['team_b']))
    ok = bad = 0; trials = 0
    while trials < 20000 and (ok + bad) < 2000:
        trials += 1
        a, b, c, dd = random.sample(teams, 4)
        ks = [frozenset(x) for x in ((a, b), (c, dd), (a, c), (b, dd), (a, dd), (b, c))]
        if not all(k in d for k in ks):
            continue
        r = [d[k] - 1 for k in ks]
        if (r[0] + r[1]) % 19 == (r[2] + r[3]) % 19 == (r[4] + r[5]) % 19:
            ok += 1
        else:
            bad += 1
    p_binom = stats.binomtest(ok, ok + bad, 1/361, alternative='greater').pvalue if ok + bad else 1
    print(f"S{s}: quadruples OK {ok}/{ok+bad} ({ok/max(ok+bad,1):.4f}) p_binom={p_binom:.2e}")

# ---------------------------------------------------------- C. relabeling entre saisons ?
print("\n================ C. SAISON = PERMUTATION D'EQUIPES D'UNE AUTRE ? ================")
# signature invariante par relabeling: multiset, pour chaque equipe, de la sequence
# 'home/away' (H/A) indexee par round -> si relabeling pur, les patterns H/A par
# round sont les memes a permutation d'equipes pres.
def ha_patterns(g):
    pats = {}
    for t in set(g['team_a']) | set(g['team_b']):
        seq = {}
        sub_h = g[g['team_a'] == t]; sub_a = g[g['team_b'] == t]
        for _, row in sub_h.iterrows():
            seq[row['round']] = 'H'
        for _, row in sub_a.iterrows():
            seq[row['round']] = 'A'
        pats[t] = seq
    return pats

s1, s2 = best[0], best[1]
p1, p2 = ha_patterns(ev[ev['season'] == s1]), ha_patterns(ev[ev['season'] == s2])
# compare sur rounds observes dans les 2 saisons
common_r = sorted(set(ev[ev['season'] == s1]['round']) & set(ev[ev['season'] == s2]['round']))
def pat_str(seq, rounds):
    return "".join(seq.get(r, '.') for r in rounds)
ps1 = sorted(pat_str(p1[t], common_r) for t in p1)
ps2 = sorted(pat_str(p2[t], common_r) for t in p2)
n_match = sum(1 for a, b in zip(ps1, ps2) if a == b)
print(f"S{s1} vs S{s2}: multisets de patterns H/A identiques sur {len(common_r)} rounds communs: {n_match}/20 equipes")
for a, b in list(zip(ps1, ps2))[:6]:
    print(f"   {a}\n   {b}\n   --")

# une equipe garde-t-elle le meme pattern H/A d'une saison a l'autre (slot fixe) ?
same_team_same_pat = sum(1 for t in p1 if t in p2 and pat_str(p1[t], common_r) == pat_str(p2[t], common_r))
print(f"meme equipe -> meme pattern H/A entre S{s1} et S{s2}: {same_team_same_pat}/20")

# ---------------------------------------------------------- D. grille temporelle
print("\n================ D. GRILLE TEMPORELLE ================")
blk2 = blk.sort_values('expected_start').reset_index(drop=True)
blk2['gap'] = blk2['expected_start'].diff().dt.total_seconds()
blk2['rdiff'] = blk2['round'].diff()
intra = blk2[blk2['season'] == blk2['season'].shift(1)].dropna(subset=['gap'])
ok_grid = (intra['gap'] == intra['rdiff'] * 120).sum()
print(f"intra-saison: gap == rdiff*120s pour {ok_grid}/{len(intra)} transitions")
bad_grid = intra[intra['gap'] != intra['rdiff'] * 120]
print("violations:", bad_grid[['expected_start', 'round', 'gap', 'rdiff']].head(10).to_string())

# saison -> saison : J38(t) -> J1(t+1) : 300s ? cycle complet 4740s ?
inter = blk2[blk2['season'] != blk2['season'].shift(1)].dropna(subset=['gap'])
# reconstruire le start theorique de J1 de chaque saison: t_block - (round-1)*120
blk2['j1_start'] = blk2['expected_start'] - pd.to_timedelta((blk2['round'] - 1) * 120, unit='s')
j1 = blk2.groupby('season')['j1_start'].agg(['min', 'max', 'nunique'])
print(f"\ncoherence J1 reconstruit par saison (nunique==1 attendu): {(j1['nunique']==1).sum()}/{len(j1)} saisons")
print("saisons incoherentes:", j1[j1['nunique'] > 1].head().to_string())
j1s = blk2.groupby('season')['j1_start'].min().sort_values()
dj1 = j1s.diff().dt.total_seconds().dropna()
print("\ndiff entre J1 de saisons consecutives (s):", Counter(dj1).most_common(8))
print(f"multiples exacts de 4740s: {sum(1 for x in dj1 if x % 4740 == 0)}/{len(dj1)}")
resid = Counter(x % 4740 for x in dj1)
print("residus mod 4740:", resid.most_common(8))
print("\nOK part2 done")
