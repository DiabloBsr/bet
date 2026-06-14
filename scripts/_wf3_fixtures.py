# -*- coding: utf-8 -*-
"""WF3 - Facette CALENDRIER : comment le moteur genere-t-il les fixtures ?

1. Reconstruction des saisons (round_info redescend)
2. Round-robin aller/retour ? (chaque paire 2x, 1x home / 1x away)
3. Calendrier identique chaque saison ?
4. Si variable : permutation aleatoire vs rotation deterministe (Berger)
5. Biais de position dans le round (ordre id 1-10)
6. Espacement temporel expected_start (pauses, nuit ?)
"""
import sys, json
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from scipy import stats
from sqlalchemy import create_engine, text
from scraper.config import load_settings

pd.set_option('display.width', 200)
eng = create_engine(load_settings().db_url)

# ---------------------------------------------------------------- load
ev = pd.read_sql(text("""
    SELECT e.id, e.team_a, e.team_b, e.round_info, e.expected_start,
           r.score_a, r.score_b, r.ht_score_a, r.ht_score_b
    FROM events e LEFT JOIN results r ON r.event_id = e.id
    WHERE e.round_info != '0'
"""), eng)
ev['expected_start'] = pd.to_datetime(ev['expected_start'])
# dedup (team_a, team_b, expected_start) -- keep lowest id
ev = ev.sort_values('id').drop_duplicates(['team_a', 'team_b', 'expected_start'], keep='first')
ev['round'] = ev['round_info'].astype(int)
ev = ev.sort_values(['expected_start', 'id']).reset_index(drop=True)
print(f"events dedup (round!=0): {len(ev)}, avec resultat: {ev['score_a'].notna().sum()}")
print(f"rounds distincts: {sorted(ev['round'].unique())}")

# ---------------------------------------------------------------- 1. seasons
# group rows into "rounds" = blocks sharing same expected_start
blocks = ev.groupby('expected_start')
blk = blocks.agg(round=('round', lambda s: s.iloc[0]),
                 n_round_vals=('round', 'nunique'),
                 n=('id', 'size')).reset_index().sort_values('expected_start').reset_index(drop=True)
print("\n-- blocs par expected_start --")
print("tailles de bloc:", Counter(blk['n']).most_common())
print("blocs avec >1 round_info distinct:", (blk['n_round_vals'] > 1).sum())

# season id: increments each time round number drops vs previous block
season = 0
prev_round = None
season_ids = []
for r in blk['round']:
    if prev_round is not None and r < prev_round:
        season += 1
    season_ids.append(season)
    prev_round = r
blk['season'] = season_ids
ev = ev.merge(blk[['expected_start', 'season']], on='expected_start', how='left')

print("\n-- saisons reconstruites --")
seas_sum = blk.groupby('season').agg(first=('expected_start', 'min'), last=('expected_start', 'max'),
                                     rmin=('round', 'min'), rmax=('round', 'max'),
                                     n_rounds=('round', 'nunique'), n_blocks=('round', 'size'))
print(seas_sum.to_string())

# complete seasons = rounds 1..38 all present with 10 matches each
complete = []
for s, g in ev.groupby('season'):
    rc = g.groupby('round').size()
    if len(rc) == 38 and (rc == 10).all():
        complete.append(s)
print(f"\nsaisons COMPLETES (38 rounds x 10 matchs): {complete}")

# ---------------------------------------------------------------- 2. round-robin ?
print("\n================ 2. ROUND-ROBIN ALLER/RETOUR ? ================")
for s in complete:
    g = ev[ev['season'] == s]
    ordered_pairs = Counter(zip(g['team_a'], g['team_b']))
    unordered = Counter(frozenset(p) for p in ordered_pairs)
    n_pairs = len(unordered)
    pair_counts = Counter(unordered.values())
    ordered_dup = sum(1 for v in ordered_pairs.values() if v != 1)
    # each team: 19 home, 19 away ?
    home_c = Counter(g['team_a']); away_c = Counter(g['team_b'])
    ha_ok = all(home_c[t] == 19 and away_c[t] == 19 for t in home_c)
    # within-season: a team appears exactly once per round
    once_per_round = all(
        Counter(list(rr['team_a']) + list(rr['team_b'])).most_common(1)[0][1] == 1
        for _, rr in g.groupby('round'))
    print(f"saison {s}: paires non-ordonnees={n_pairs} (attendu 190), "
          f"chaque paire jouee {dict(pair_counts)} fois, affiches ordonnees dupliquees={ordered_dup}, "
          f"19H/19A par equipe={ha_ok}, 1 match/equipe/round={once_per_round}")
    # aller/retour split: first leg in rounds 1-19, return in 20-38 ?
    legs_ok, mirror_ok = 0, 0
    first_leg = {}
    for _, row in g.sort_values('round').iterrows():
        key = frozenset((row['team_a'], row['team_b']))
        if key not in first_leg:
            first_leg[key] = (row['round'], row['team_a'], row['team_b'])
        else:
            r1, a1, b1 = first_leg[key]
            if r1 <= 19 < row['round']:
                legs_ok += 1
            if row['team_a'] == b1 and row['team_b'] == a1:
                mirror_ok += 1
    print(f"   retours en 2e moitie (r1<=19<r2): {legs_ok}/190 ; retour = home/away inverse: {mirror_ok}/190")
    # mirror structure: round k vs round k+19 same pairings ?
    same_pairings_shift = None
    for shift in [19]:
        match_cnt = 0
        for r in range(1, 20):
            p1 = set(frozenset((a, b)) for a, b in zip(g[g['round'] == r]['team_a'], g[g['round'] == r]['team_b']))
            p2 = set(frozenset((a, b)) for a, b in zip(g[g['round'] == r + shift]['team_a'], g[g['round'] == r + shift]['team_b']))
            if p1 == p2:
                match_cnt += 1
        print(f"   miroir round k <-> k+19 (memes 10 affiches): {match_cnt}/19 rounds")

# ---------------------------------------------------------------- 3. same calendar each season ?
print("\n================ 3. CALENDRIER IDENTIQUE ENTRE SAISONS ? ================")
def fixture_sig(g):
    """per round: frozenset of ordered pairs"""
    return {r: frozenset(zip(rr['team_a'], rr['team_b'])) for r, rr in g.groupby('round')}

sigs = {s: fixture_sig(ev[ev['season'] == s]) for s in complete}
cs = complete
for i in range(len(cs)):
    for j in range(i + 1, len(cs)):
        common_rounds = set(sigs[cs[i]]) & set(sigs[cs[j]])
        same = sum(1 for r in common_rounds if sigs[cs[i]][r] == sigs[cs[j]][r])
        # unordered too
        same_unord = sum(1 for r in common_rounds
                         if {frozenset(p) for p in sigs[cs[i]][r]} == {frozenset(p) for p in sigs[cs[j]][r]})
        print(f"saison {cs[i]} vs {cs[j]}: rounds identiques (ordonnes) {same}/{len(common_rounds)}, "
              f"(non-ordonnes) {same_unord}/{len(common_rounds)}")

# also include partial seasons: compare any two seasons on shared rounds
print("\n-- toutes saisons (y compris partielles), rounds partages --")
all_sigs = {s: fixture_sig(ev[ev['season'] == s]) for s in sorted(ev['season'].unique())}
sl = sorted(all_sigs)
id_mat = []
for i in range(len(sl)):
    for j in range(i + 1, len(sl)):
        common = [r for r in set(all_sigs[sl[i]]) & set(all_sigs[sl[j]])
                  if len(all_sigs[sl[i]][r]) == 10 and len(all_sigs[sl[j]][r]) == 10]
        if not common:
            continue
        same = sum(1 for r in common if all_sigs[sl[i]][r] == all_sigs[sl[j]][r])
        id_mat.append((sl[i], sl[j], same, len(common)))
for a, b, s_, n_ in id_mat:
    print(f"  S{a} vs S{b}: {s_}/{n_} rounds identiques")

# is the same SET of round-pairings reused but shuffled across rounds ?
print("\n-- meme affiche-set global, ordre des journees permute ? --")
for i in range(len(cs)):
    for j in range(i + 1, len(cs)):
        rounds_i = {r: sigs[cs[i]][r] for r in sigs[cs[i]]}
        rounds_j_set = {v: k for k, v in sigs[cs[j]].items()}
        matched = {r: rounds_j_set.get(rounds_i[r]) for r in rounds_i}
        n_found = sum(1 for v in matched.values() if v is not None)
        print(f"saison {cs[i]} rounds retrouves tels quels dans saison {cs[j]}: {n_found}/38")
        if n_found > 25:
            print("   mapping round_i -> round_j:", {k: matched[k] for k in sorted(matched) if matched[k]})

# ---------------------------------------------------------------- 4. Berger / circle method ?
print("\n================ 4. STRUCTURE BERGER (methode du cercle) ? ================")
# circle method: fix team 0, rotate others. Test: for each complete season, can we
# find a labeling consistent with circle rotation? Simpler diagnostic: opponent
# sequence of each team -- in Berger, opponent index increases by 1 (mod 19) each round.
for s in cs[:3]:
    g = ev[ev['season'] == s]
    teams = sorted(set(g['team_a']) | set(g['team_b']))
    # opponent map per round for one team
    t0 = teams[0]
    opp_seq = []
    for r in range(1, 20):
        rr = g[g['round'] == r]
        row = rr[(rr['team_a'] == t0) | (rr['team_b'] == t0)].iloc[0]
        opp = row['team_b'] if row['team_a'] == t0 else row['team_a']
        opp_seq.append(opp)
    print(f"S{s} adversaires de {t0} r1..r19: {opp_seq}")

# cross-season: does each team meet opponents in the SAME ORDER (rotation of rounds)?
print("\n-- sequence d'adversaires par equipe : identique entre saisons (a rotation pres) ? --")
def opp_cycle(g, team):
    seq = []
    for r in range(1, 39):
        rr = g[g['round'] == r]
        row = rr[(rr['team_a'] == team) | (rr['team_b'] == team)]
        if len(row) == 0:
            seq.append(None); continue
        row = row.iloc[0]
        seq.append(row['team_b'] if row['team_a'] == team else row['team_a'])
    return seq
if len(cs) >= 2:
    t0 = sorted(set(ev['team_a']))[0]
    seqs = {s: opp_cycle(ev[ev['season'] == s], t0) for s in cs}
    s1, s2 = cs[0], cs[1]
    a, b = seqs[s1], seqs[s2]
    rot_found = [k for k in range(38) if all(a[(i + k) % 38] == b[i] for i in range(38))]
    print(f"{t0}: S{s1} sequence == rotation de S{s2} ? shifts valides: {rot_found}")

# ---------------------------------------------------------------- 5. position in round bias
print("\n================ 5. BIAIS DE POSITION DANS LE ROUND (1-10) ================")
ev_r = ev[ev['score_a'].notna()].copy()
ev_r['pos'] = ev_r.groupby(['season', 'round'])['id'].rank(method='first').astype(int)
ev_r = ev_r[ev_r.groupby(['season', 'round'])['id'].transform('size') == 10]
ev_r['total_goals'] = ev_r['score_a'] + ev_r['score_b']
ev_r['home_win'] = (ev_r['score_a'] > ev_r['score_b']).astype(int)
ev_r['draw'] = (ev_r['score_a'] == ev_r['score_b']).astype(int)
agg = ev_r.groupby('pos').agg(n=('id', 'size'), goals=('total_goals', 'mean'),
                              hw=('home_win', 'mean'), draw=('draw', 'mean'))
print(agg.to_string(float_format=lambda x: f"{x:.3f}"))
# chi2 on home/draw/away by position
ct = pd.crosstab(ev_r['pos'], np.select([ev_r['score_a'] > ev_r['score_b'], ev_r['score_a'] == ev_r['score_b']], ['H', 'D'], 'A'))
chi2, p, dof, _ = stats.chi2_contingency(ct)
print(f"chi2 1X2 x position: chi2={chi2:.2f} dof={dof} p={p:.4f}")
# ANOVA on goals by position
groups = [grp['total_goals'].values for _, grp in ev_r.groupby('pos')]
f, p_a = stats.f_oneway(*groups)
print(f"ANOVA buts ~ position: F={f:.2f} p={p_a:.4f}")
# kruskal
h, p_k = stats.kruskal(*groups)
print(f"Kruskal buts ~ position: H={h:.2f} p={p_k:.4f}")

# position stable pour une equipe donnee ? (ex: le match 1 = toujours certaines equipes ?)
print("\n-- la position 1 est-elle occupee par les memes equipes ? --")
pos1 = ev_r[ev_r['pos'] == 1]
print("home team en pos 1:", Counter(pos1['team_a']).most_common(5))

# ---------------------------------------------------------------- 6. timing expected_start
print("\n================ 6. TIMING expected_start ================")
times = blk['expected_start'].sort_values().reset_index(drop=True)
gaps = times.diff().dt.total_seconds().dropna()
print("gaps entre blocs successifs (s):", Counter(gaps).most_common(10))
big = gaps[gaps > gaps.mode().iloc[0] * 1.5]
print(f"\ngaps anormaux (> 1.5x mode): {len(big)}")
for idx in big.index[:30]:
    print(f"   {times[idx-1]} -> {times[idx]}  gap={gaps[idx]:.0f}s")
# inter-season gap vs intra-season
blk2 = blk.sort_values('expected_start').reset_index(drop=True)
blk2['gap_prev'] = blk2['expected_start'].diff().dt.total_seconds()
intra = blk2[blk2['season'] == blk2['season'].shift(1)]['gap_prev']
inter = blk2[blk2['season'] != blk2['season'].shift(1)]['gap_prev'].dropna()
print(f"\ngap intra-saison: mode={intra.mode().iloc[0]:.0f}s, min={intra.min():.0f}, max={intra.max():.0f}")
print(f"gap inter-saison (J38 -> J1 suivante): {sorted(inter.values)}")
# heures de la journee : moteur s'arrete la nuit ?
hours = times.dt.hour
print("\nrepartition par heure UTC:", dict(sorted(Counter(hours).items())))
# secondes/minutes pattern
print("secondes des starts:", Counter(times.dt.second).most_common(5))
print("minutes des starts (mod):", Counter(times.dt.minute % 4).most_common())
print("\nOK _wf3_fixtures done")
