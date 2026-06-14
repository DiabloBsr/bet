# -*- coding: utf-8 -*-
"""WF3 - FACETTE HT/FT : le moteur tire-t-il le score FT puis repartit-il les buts
binomialement entre les deux mi-temps ?

A. Test binomial but-par-but : P(HT=(i,j) | FT=(h,a)) = Bin(h,p)xBin(a,p) ?
B. p constant ? (home vs away, total de buts, cote, equipe, temps calendaire)
C. Coherence des grilles du marche : Score exact + binomiale => HT/FT, Mi-tps 1X2,
   Mi-tps CS, 2eme mi-tps CS. Deviations marche vs modele vs reel.
D. Transitions par segment (DS HT1->FT1, FS HTX->FT1) : effet segment reel ou mix de cotes ?
E. Selections mal pricees -> validation walk-forward 70/30.
"""
import sys, json, math
from collections import Counter, defaultdict
sys.path.insert(0, '.')
from scraper.config import load_settings
from sqlalchemy import create_engine, text
import numpy as np
from scipy import stats

eng = create_engine(load_settings().db_url)
SEP = "=" * 78


# ---------------------------------------------------------------- load
def load():
    q = text('''
        SELECT e.id, e.round_info, e.expected_start, e.team_a, e.team_b,
               o.odds_home, o.odds_draw, o.odds_away, o.extra_markets,
               r.score_a, r.score_b, r.ht_score_a, r.ht_score_b
        FROM events e
        JOIN results r ON r.event_id = e.id
        JOIN odds_snapshots o ON o.event_id = e.id
         AND o.id = (SELECT MIN(id) FROM odds_snapshots WHERE event_id = e.id)
        WHERE e.round_info != '0'
          AND r.ht_score_a IS NOT NULL AND r.ht_score_b IS NOT NULL
          AND r.score_a IS NOT NULL AND r.score_b IS NOT NULL
        ORDER BY e.expected_start ASC, e.id ASC
    ''')
    seen = set()
    out = []
    with eng.connect() as c:
        for r in c.execute(q):
            key = (r[3], r[4], r[2])
            if key in seen:
                continue
            seen.add(key)
            em = r[8]
            if isinstance(em, str):
                try:
                    em = json.loads(em)
                except Exception:
                    em = None
            h, a = int(r[9]), int(r[10])
            hh, ha = int(r[11]), int(r[12])
            if hh > h or ha > a:   # HT > FT impossible -> corrompu
                continue
            try:
                rnd = int(r[1])
            except (TypeError, ValueError):
                continue
            out.append(dict(id=r[0], rnd=rnd, start=r[2], ta=r[3], tb=r[4],
                            oh=r[5], od=r[6], oa=r[7], em=em,
                            h=h, a=a, hh=hh, ha=ha))
    return out

M = load()
print(f"matches charges (dedup, HT+FT valides) : {len(M)}")

# ---------------------------------------------------------------- A. binomial split
print(SEP); print("A. MODELE BINOMIAL BUT-PAR-BUT"); print(SEP)

tot_home = sum(m['h'] for m in M); h1_home = sum(m['hh'] for m in M)
tot_away = sum(m['a'] for m in M); h1_away = sum(m['ha'] for m in M)
p_home = h1_home / tot_home
p_away = h1_away / tot_away
p_all = (h1_home + h1_away) / (tot_home + tot_away)
print(f"p(but en MT1) global = {p_all:.5f}   ({h1_home+h1_away}/{tot_home+tot_away})")
print(f"  p_home = {p_home:.5f} ({h1_home}/{tot_home})   p_away = {p_away:.5f} ({h1_away}/{tot_away})")
# 2-proportions
z = (p_home - p_away) / math.sqrt(p_all*(1-p_all)*(1/tot_home + 1/tot_away))
print(f"  z(home vs away) = {z:.3f}  p-value = {2*(1-stats.norm.cdf(abs(z))):.4f}")

def binom_pmf(k, n, p):
    return math.comb(n, k) * p**k * (1-p)**(n-k)

# chi2 GOF par score FT
print("\n--- chi2 par score FT : HT observe vs Bin(h,p)xBin(a,p), p global ---")
ft_groups = defaultdict(list)
for m in M:
    ft_groups[(m['h'], m['a'])].append((m['hh'], m['ha']))

total_chi2, total_dof = 0.0, 0
rows_rep = []
for (h, a), lst in sorted(ft_groups.items(), key=lambda kv: -len(kv[1])):
    n = len(lst)
    if h + a == 0 or n < 30:
        continue
    obs = Counter(lst)
    cells = [(i, j) for i in range(h+1) for j in range(a+1)]
    exp = {c: n * binom_pmf(c[0], h, p_all) * binom_pmf(c[1], a, p_all) for c in cells}
    # pool cellules attendues < 5
    big = [c for c in cells if exp[c] >= 5]
    small = [c for c in cells if exp[c] < 5]
    o_list = [obs.get(c, 0) for c in big]
    e_list = [exp[c] for c in big]
    if small:
        o_list.append(sum(obs.get(c, 0) for c in small))
        e_list.append(sum(exp[c] for c in small))
    if len(o_list) < 2:
        continue
    # renormalise expected pour matcher n exactement (pooling)
    e_arr = np.array(e_list) * (n / sum(e_list))
    chi2 = float(((np.array(o_list) - e_arr)**2 / e_arr).sum())
    dof = len(o_list) - 1
    pv = 1 - stats.chi2.cdf(chi2, dof)
    total_chi2 += chi2; total_dof += dof
    rows_rep.append((h, a, n, chi2, dof, pv))

for h, a, n, chi2, dof, pv in rows_rep:
    flag = '  <<<' if pv < 0.01 else ''
    print(f"  FT {h}-{a}  n={n:5d}  chi2={chi2:7.2f} dof={dof:2d}  p={pv:.4f}{flag}")
pv_tot = 1 - stats.chi2.cdf(total_chi2, total_dof - 1)  # -1 pour p estime
print(f"  GLOBAL: chi2={total_chi2:.1f} dof={total_dof-1}  p={pv_tot:.4f}")

# independance home/away dans le split (FT 1-1)
print("\n--- independance des splits home/away (FT 1-1) ---")
ll = ft_groups.get((1, 1), [])
tab = np.zeros((2, 2))
for hh, ha in ll:
    tab[hh, ha] += 1
chi2, pv, dof, _ = stats.chi2_contingency(tab)
print(f"  n={len(ll)}  table={tab.tolist()}  chi2={chi2:.3f} p={pv:.4f}")

# surdispersion : FT avec h>=2 -> var du nb de buts home en MT1 vs binomiale
print("\n--- surdispersion intra-equipe (h>=2, buts home en MT1) ---")
for hgoals in (2, 3, 4):
    ks = [hh for (h, a), lst in ft_groups.items() if h == hgoals for (hh, ha) in lst]
    if len(ks) < 50:
        continue
    n = len(ks)
    mean = np.mean(ks); var = np.var(ks, ddof=1)
    p_hat = mean / hgoals
    var_bin = hgoals * p_hat * (1 - p_hat)
    # test de dispersion: sum (k-mean)^2/var_bin ~ chi2(n-1)
    disp = (n - 1) * var / var_bin
    pv = 2 * min(stats.chi2.cdf(disp, n-1), 1 - stats.chi2.cdf(disp, n-1))
    print(f"  h={hgoals}: n={n} mean={mean:.3f} var={var:.3f} var_bin={var_bin:.3f} "
          f"ratio={var/var_bin:.3f} p={pv:.4f}")

# ---------------------------------------------------------------- B. p constant ?
print(SEP); print("B. p EST-IL CONSTANT ?"); print(SEP)

def p_test_groups(groups, label):
    """groups: dict name -> (h1, tot). chi2 d'homogeneite."""
    print(f"\n--- {label} ---")
    names = sorted(groups)
    tab = np.array([[groups[g][0], groups[g][1] - groups[g][0]] for g in names], dtype=float)
    keep = tab.sum(axis=1) >= 30
    tab = tab[keep]; names = [g for g, k in zip(names, keep) if k]
    for g, row in zip(names, tab):
        tot = row.sum()
        print(f"  {g:<22} p={row[0]/tot:.4f}  (n_buts={int(tot)})")
    if len(tab) >= 2:
        chi2, pv, dof, _ = stats.chi2_contingency(tab)
        print(f"  chi2={chi2:.2f} dof={dof} p={pv:.4f}" + ("  <<< NON CONSTANT" if pv < 0.01 else "  (constant)"))

# par total de buts FT
g = defaultdict(lambda: [0, 0])
for m in M:
    t = m['h'] + m['a']
    key = f"T={t}" if t <= 6 else "T>=7"
    g[key][0] += m['hh'] + m['ha']; g[key][1] += m['h'] + m['a']
p_test_groups({k: tuple(v) for k, v in g.items()}, "p par TOTAL de buts FT")

# par cote home (bucket)
g = defaultdict(lambda: [0, 0])
for m in M:
    if m['oh'] is None:
        continue
    oh = float(m['oh'])
    b = ("H<=1.45" if oh <= 1.45 else "H1.45-2" if oh <= 2 else
         "H2-3" if oh <= 3 else "H>3")
    g[b][0] += m['hh'] + m['ha']; g[b][1] += m['h'] + m['a']
p_test_groups({k: tuple(v) for k, v in g.items()}, "p par cote HOME")

# par equipe qui marque
g = defaultdict(lambda: [0, 0])
for m in M:
    g[m['ta']][0] += m['hh']; g[m['ta']][1] += m['h']
    g[m['tb']][0] += m['ha']; g[m['tb']][1] += m['a']
p_test_groups({k: tuple(v) for k, v in g.items()}, "p par EQUIPE (buteuse)")

# par segment de saison
SEGS = [(1, 3, 'DS'), (4, 12, 'MS_early'), (13, 25, 'MS_mid'), (26, 33, 'MS_late'), (34, 38, 'FS')]
def seg_of(rnd):
    for lo, hi, nm in SEGS:
        if lo <= rnd <= hi:
            return nm
    return None
g = defaultdict(lambda: [0, 0])
for m in M:
    s = seg_of(m['rnd'])
    if s:
        g[s][0] += m['hh'] + m['ha']; g[s][1] += m['h'] + m['a']
p_test_groups({k: tuple(v) for k, v in g.items()}, "p par SEGMENT de saison")

# par moitie chronologique du dataset
half = len(M) // 2
g = {'1ere moitie data': [0, 0], '2eme moitie data': [0, 0]}
for i, m in enumerate(M):
    k = '1ere moitie data' if i < half else '2eme moitie data'
    g[k][0] += m['hh'] + m['ha']; g[k][1] += m['h'] + m['a']
p_test_groups({k: tuple(v) for k, v in g.items()}, "p par PERIODE (drift temporel)")
