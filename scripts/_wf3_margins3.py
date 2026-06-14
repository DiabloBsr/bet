# -*- coding: utf-8 -*-
"""WF3 iter 3 - mecanisme minutes home/away, fit grilles HT/2T, capped-mass check,
walk-forward regle combinee FTTS favori (+ worst-case NA)."""
import sys, json, math
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from collections import Counter
from scipy import stats
from sqlalchemy import create_engine, text
from scraper.config import load_settings

pd.set_option('display.width', 250)
eng = create_engine(load_settings().db_url)
Q = """
SELECT e.id, e.team_a, e.team_b, e.expected_start,
       r.score_a, r.score_b, r.ht_score_a, r.ht_score_b, r.goals_json,
       o.odds_home, o.odds_draw, o.odds_away, o.extra_markets
FROM events e
JOIN results r ON r.event_id = e.id
JOIN odds_snapshots o ON o.event_id = e.id
JOIN (SELECT event_id, MIN(id) mid FROM odds_snapshots GROUP BY event_id) m ON m.mid = o.id
WHERE e.round_info != '0' AND r.score_a IS NOT NULL
"""
with eng.connect() as c:
    df = pd.read_sql(text(Q), c)
df = df.drop_duplicates(subset=['team_a', 'team_b', 'expected_start'])
df = df.sort_values('expected_start').reset_index(drop=True)
CAP = 99.99

def parse_em(row):
    em = row['extra_markets']
    if em is None: return None
    return json.loads(em) if isinstance(em, str) else em

def goals(row):
    gj = row['goals_json']
    if gj is None or (isinstance(gj, float) and math.isnan(gj)): return None
    g = json.loads(gj) if isinstance(gj, str) else gj
    if not g: return None
    return sorted(g, key=lambda x: int(x['minute']))

# ================================================== 1. minutes des buts home vs away
print("=" * 90)
print("1. MINUTES DES BUTS : home vs away, histogramme global")
print("=" * 90)
mh, ma, firsts = [], [], []
for _, row in df.iterrows():
    g = goals(row)
    if not g: continue
    # coherence: dernier homeScore/awayScore = score final ?
    for ev in g:
        m = int(ev['minute'])
        (mh if ev['team'] == 'Home' else ma).append(m)
    firsts.append((int(g[0]['minute']), g[0]['team']))
mh, ma = np.array(mh), np.array(ma)
print(f"buts Home n={len(mh)} minute moyenne={mh.mean():.2f} | Away n={len(ma)} moyenne={ma.mean():.2f}")
u, p = stats.mannwhitneyu(mh, ma, alternative='two-sided')
print(f"Mann-Whitney home vs away minutes: p={p:.3e}")
allm = np.concatenate([mh, ma])
bins = [1, 16, 31, 46, 61, 76, 91]
hist, _ = np.histogram(allm, bins=bins)
print(f"histogramme 15-min (tous buts): {hist}  proportions={np.round(hist/hist.sum(), 4)}")
chi2, pu = stats.chisquare(hist)
print(f"chi2 vs uniforme (6 bins): chi2={chi2:.1f} p={pu:.3e}")
hh, _ = np.histogram(mh, bins=bins); ha, _ = np.histogram(ma, bins=bins)
print(f"home par bin: {np.round(hh/hh.sum(),4)}")
print(f"away par bin: {np.round(ha/ha.sum(),4)}")
c2, ph = stats.chi2_contingency(np.vstack([hh, ha]))[:2]
print(f"chi2 contingence home vs away par bin: chi2={c2:.1f} p={ph:.3e}")
fh = sum(1 for _, t in firsts if t == 'Home')
print(f"1er but = Home: {fh}/{len(firsts)} = {fh/len(firsts):.4f}")

# mi-temps : buts 1-45 vs 46-90
print(f"\nbuts 1ere MT: {np.sum(allm<=45)} ({np.mean(allm<=45):.4f})  2nde MT: {np.sum(allm>45)}")

# ================================================== 2. masse cappee Score exact
print("\n" + "=" * 90)
print("2. MASSE CAPPEE (Score exact) : prediction 1-somme(p_model) vs taux realise")
print("=" * 90)
n_ev, hit_cap, pmass = 0, 0, []
for _, row in df.iterrows():
    em = parse_em(row)
    if not em or 'Score exact' not in em: continue
    se = em['Score exact']
    n_ev += 1
    s = 0.0
    for k, o in se.items():
        o = float(o)
        if o <= CAP: s += (1 / o) / 1.12
    pmass.append(1 - s)
    key = f"{int(row['score_a'])}-{int(row['score_b'])}"
    if key in se and float(se[key]) > CAP: hit_cap += 1
print(f"masse cappee predite (1-somme p_model, moyenne) = {np.mean(pmass):.4f}")
print(f"taux realise de scores tombant sur cellule cappee = {hit_cap}/{n_ev} = {hit_cap/n_ev:.4f}")
pv = stats.binomtest(hit_cap, n_ev, float(np.mean(pmass))).pvalue
print(f"binomial: p={pv:.3f}")

# ================================================== 3. fit grilles Mi-tps CS et 2eme MT CS
print("\n" + "=" * 90)
print("3. FIT DES GRILLES HT / 2nde MT (profile du vig v, chi2 min)")
print("=" * 90)
for MK, get_real in [
    ('Mi-tps CS', lambda r: (int(r['ht_score_a']), int(r['ht_score_b']))
        if r['ht_score_a'] is not None and not (isinstance(r['ht_score_a'], float) and math.isnan(r['ht_score_a'])) else None),
    ('2ème mi-tps - CS', lambda r: (int(r['score_a']) - int(r['ht_score_a']), int(r['score_b']) - int(r['ht_score_b']))
        if r['ht_score_a'] is not None and not (isinstance(r['ht_score_a'], float) and math.isnan(r['ht_score_a']))
        else None)]:
    E_inv = Counter(); O = Counter(); n_used = 0; out_grid = 0
    inv_rows = []
    for _, row in df.iterrows():
        em = parse_em(row)
        if not em or MK not in em: continue
        real = get_real(row)
        if real is None: continue
        key = f"{real[0]}-{real[1]}"
        sels = {k: float(o) for k, o in em[MK].items()}
        noncap = {k: o for k, o in sels.items() if o <= CAP}
        if key not in sels:
            out_grid += 1
            continue
        if key not in noncap:
            out_grid += 0  # realise sur cellule cappee : comptabilise a part
        n_used += 1
        O[key] += 1
        for k, o in noncap.items():
            E_inv[k] += 1 / o
        inv_rows.append(sum(1/o for o in noncap.values()))
    keys = sorted(E_inv)
    best = None
    for v in np.arange(0.08, 0.25, 0.005):
        chi = sum((O[k] - E_inv[k] / (1 + v)) ** 2 / (E_inv[k] / (1 + v)) for k in keys)
        if best is None or chi < best[1]: best = (v, chi)
    v, chi = best
    ddl = len(keys) - 2
    print(f"{MK}: n={n_used} hors-grille={out_grid} ({out_grid/(n_used+out_grid)*100:.2f}%) "
          f"v_fit={v:.3f} chi2={chi:.1f} ddl~{ddl} p={1-stats.chi2.cdf(chi, ddl):.3f} "
          f"overround_brut_moyen={np.mean(inv_rows)-1:.4f}")
    # detail par cellule au v fitte
    det = pd.DataFrame([{'cell': k, 'E': E_inv[k]/(1+v), 'O': O[k],
                         'ratio': O[k]/(E_inv[k]/(1+v))} for k in keys])
    print(det.sort_values('cell').to_string(index=False, float_format=lambda x: f"{x:.1f}"))

# ================================================== 4. regle combinee FTTS favori
print("\n" + "=" * 90)
print("4. WALK-FORWARD FINAL : FTTS favori (cote du favori <= 1.50), '1' ou '2'")
print("=" * 90)
rows = []
for _, row in df.iterrows():
    em = parse_em(row)
    if not em or 'FTTS' not in em: continue
    try:
        o1, o2 = float(em['FTTS']['1']), float(em['FTTS']['2'])
    except (KeyError, TypeError, ValueError):
        continue
    a, b = int(row['score_a']), int(row['score_b'])
    if a == 0 and b == 0: ftm = None
    elif a > 0 and b == 0: ftm = 'Home'
    elif b > 0 and a == 0: ftm = 'Away'
    else:
        g = goals(row)
        ftm = ('Home' if g[0]['team'] == 'Home' else 'Away') if g else 'NA'
    rows.append({'ts': row['expected_start'], 'o1': o1, 'o2': o2, 'ft': ftm})
fd = pd.DataFrame(rows).sort_values('ts').reset_index(drop=True)
cut = int(len(fd) * 0.7)

def eval_rule(d, na_as_loss=False):
    bets = []
    for _, r in d.iterrows():
        if r['o1'] <= 1.50: side, o = 'Home', r['o1']
        elif r['o2'] <= 1.50: side, o = 'Away', r['o2']
        else: continue
        if r['ft'] == 'NA':
            if na_as_loss: bets.append((0, o))
            continue
        bets.append((int(r['ft'] == side), o))
    if not bets: return 0, np.nan, np.nan, np.nan
    w = np.array([b[0] for b in bets]); o = np.array([b[1] for b in bets])
    return len(bets), w.mean(), o.mean(), (w * o).mean() - 1

for nm, d in [('train', fd.iloc[:cut]), ('OOS', fd.iloc[cut:]), ('FULL', fd)]:
    n, wr, ao, roi = eval_rule(d)
    n2, wr2, ao2, roi2 = eval_rule(d, na_as_loss=True)
    print(f"{nm:5s}: n={n} wr={wr:.4f} avg_odds={ao:.3f} ROI={roi:+.4f}   "
          f"[worst-case NA=perdu: n={n2} ROI={roi2:+.4f}]")
# binomial OOS
d = fd.iloc[cut:]
bets = [(int(r['ft'] == ('Home' if r['o1'] <= 1.5 else 'Away')), (r['o1'] if r['o1'] <= 1.5 else r['o2']))
        for _, r in d.iterrows() if (r['o1'] <= 1.5 or r['o2'] <= 1.5) and r['ft'] != 'NA']
w = np.array([b[0] for b in bets]); o = np.array([b[1] for b in bets])
p_be = (1 / o).mean()
print(f"OOS: binomial wins={w.sum()}/{len(w)} vs break-even p={p_be:.4f}: "
      f"p={stats.binomtest(int(w.sum()), len(w), p_be, alternative='greater').pvalue:.4f}")

print("\nDONE")
