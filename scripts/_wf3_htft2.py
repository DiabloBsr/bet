# -*- coding: utf-8 -*-
"""WF3 - FACETTE HT/FT, partie 2 : MECANISME = DEUX MI-TEMPS INDEPENDANTES ?

Decouvertes partie 1 (_wf3_htft.py) :
  - split binomial but-par-but REJETE (chi2=1089, dof=106, p~0)
  - correlation POSITIVE inter-equipes (FT 1-1 : les 2 buts dans la meme MT)
  - SOUS-dispersion intra-equipe (h=4 : ratio var 0.54)
  - p croit avec le total (0.42 -> 0.48)
  - support : total MT1 <= 3 TOUJOURS (cap dur), grilles marche par MT = 10 cellules

Hypothese H2 : le moteur tire MT1 ~ G1 (grille 'Mi-tps CS' devigee) et
MT2 ~ G2 ('2eme mi-tps - CS' devigee), INDEPENDAMMENT. FT = MT1 + MT2.
Tests :
  0. integrite ht_score vs goals_json (les 47 anomalies MT2>=4)
  1. support exact
  2. calibration G1 vs HT observe (chi2 cellule par cellule)
  3. calibration G2 vs MT2 observe
  4. independance MT1 (vs) MT2 en CONTROLANT les grilles par match
  5. convolution G1xG2 vs grille 'Score exact' (coherence interne du pricing)
     + laquelle des deux la realite suit quand elles divergent
  6. 'HT/FT' et 'Mi-tps 1X2' market vs derivation depuis G1,G2
  7. minutes : bursts meme minute, distribution
  8. transitions par segment (DS HT1->FT1, FS HTX->FT1) : mix de cotes ou effet reel ?
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
CELLS = [(i, j) for i in range(4) for j in range(4) if i + j <= 3]   # 10 cellules
FT_CELLS = sorted({(a[0]+b[0], a[1]+b[1]) for a in CELLS for b in CELLS})


def parse_cell(s):
    a, b = s.split('-')
    return int(a), int(b)


def devig_grid(d):
    """grille {'i-j': cote} -> (probas normalisees {cell: p}, marge)."""
    imp = {}
    for k, v in d.items():
        try:
            c = parse_cell(k)
        except Exception:
            return None, None
        if v is None or float(v) < 1.01:
            return None, None
        imp[c] = 1.0 / float(v)
    s = sum(imp.values())
    return {c: p / s for c, p in imp.items()}, s


def res_of(h, a):
    return '1' if h > a else ('2' if h < a else 'X')


# ---------------------------------------------------------------- load
def load():
    q = text('''
        SELECT e.id, e.round_info, e.expected_start, e.team_a, e.team_b,
               o.odds_home, o.odds_draw, o.odds_away, o.extra_markets,
               r.score_a, r.score_b, r.ht_score_a, r.ht_score_b, r.goals_json
        FROM events e
        JOIN results r ON r.event_id = e.id
        JOIN odds_snapshots o ON o.event_id = e.id
         AND o.id = (SELECT MIN(id) FROM odds_snapshots WHERE event_id = e.id)
        WHERE e.round_info != '0'
          AND r.ht_score_a IS NOT NULL AND r.ht_score_b IS NOT NULL
          AND r.score_a IS NOT NULL AND r.score_b IS NOT NULL
        ORDER BY e.expected_start ASC, e.id ASC
    ''')
    seen, out = set(), []
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
                    em = {}
            em = em or {}
            h, a, hh, ha = int(r[9]), int(r[10]), int(r[11]), int(r[12])
            gj = r[13]
            goals = []
            if gj:
                try:
                    goals = json.loads(gj) if isinstance(gj, str) else gj
                except Exception:
                    goals = []
            try:
                rnd = int(r[1])
            except (TypeError, ValueError):
                continue
            out.append(dict(id=r[0], rnd=rnd, start=r[2], ta=r[3], tb=r[4],
                            oh=r[5], od=r[6], oa=r[7], em=em,
                            h=h, a=a, hh=hh, ha=ha, goals=goals))
    return out

RAW = load()
print(f"matchs charges (dedup) : {len(RAW)}")

# ---------------------------------------------------------------- 0. integrite
print(SEP); print("0. INTEGRITE ht_score vs goals_json"); print(SEP)
ok, fixed, dropped, nogj = 0, 0, 0, 0
M = []
for m in RAW:
    h, a, hh, ha = m['h'], m['a'], m['hh'], m['ha']
    if hh > h or ha > a:
        dropped += 1
        continue
    if m['goals'] and len(m['goals']) == h + a:
        jh = sum(1 for g in m['goals'] if g['minute'] <= 45 and g['team'] == 'Home')
        ja = sum(1 for g in m['goals'] if g['minute'] <= 45 and g['team'] == 'Away')
        if (jh, ja) == (hh, ha):
            ok += 1
        else:
            # ht_score colonne contredit le fil des buts -> corrige depuis goals_json
            m['hh'], m['ha'] = jh, ja
            fixed += 1
    else:
        if h + a > 0:
            nogj += 1
        else:
            ok += 1
    M.append(m)
print(f"coherents={ok}  corriges_depuis_goals_json={fixed}  sans_goals_json={nogj}  drop(ht>ft)={dropped}")

bad_h2 = [m for m in M if (m['h']-m['hh']) + (m['a']-m['ha']) > 3 or m['hh']+m['ha'] > 3]
print(f"apres correction : matchs avec total MT1>3 ou MT2>3 : {len(bad_h2)}")
for m in bad_h2[:8]:
    print(f"   id={m['id']} FT {m['h']}-{m['a']} HT {m['hh']}-{m['ha']} goals={[(g['minute'],g['team']) for g in m['goals']]}")
M = [m for m in M if (m['h']-m['hh']) + (m['a']-m['ha']) <= 3 and m['hh']+m['ha'] <= 3]
print(f"echantillon final : {len(M)}")

# attach H2
for m in M:
    m['h2h'], m['h2a'] = m['h'] - m['hh'], m['a'] - m['ha']

# ---------------------------------------------------------------- 1. support
print(SEP); print("1. SUPPORT DES SCORES PAR MI-TEMPS"); print(SEP)
c1 = Counter((m['hh'], m['ha']) for m in M)
c2 = Counter((m['h2h'], m['h2a']) for m in M)
cf = Counter((m['h'], m['a']) for m in M)
print("MT1 :", {f"{i}-{j}": c1.get((i, j), 0) for (i, j) in CELLS}, " hors-support:", sum(v for k, v in c1.items() if k not in CELLS))
print("MT2 :", {f"{i}-{j}": c2.get((i, j), 0) for (i, j) in CELLS}, " hors-support:", sum(v for k, v in c2.items() if k not in CELLS))
print("FT hors conv-support :", sum(v for k, v in cf.items() if k not in FT_CELLS))

# ---------------------------------------------------------------- grilles par match
G = []     # matchs avec G1, G2 devigees
for m in M:
    em = m['em']
    g1d = em.get('Mi-tps CS'); g2d = em.get('2ème mi-tps - CS') or em.get('2eme mi-tps - CS')
    if not (isinstance(g1d, dict) and isinstance(g2d, dict)):
        continue
    if set(map(parse_cell, g1d.keys())) != set(CELLS) or set(map(parse_cell, g2d.keys())) != set(CELLS):
        continue
    g1, mg1 = devig_grid(g1d)
    g2, mg2 = devig_grid(g2d)
    if g1 is None or g2 is None:
        continue
    m['g1'], m['g2'], m['mg1'], m['mg2'] = g1, g2, mg1, mg2
    se = em.get('Score exact')
    m['se'] = None
    if isinstance(se, dict):
        sed, mse = devig_grid(se)
        if sed is not None:
            m['se'], m['mse'] = sed, mse
    G.append(m)
print(f"\nmatchs avec grilles MT1+MT2 completes : {len(G)} "
      f"(marge mediane MT1={np.median([m['mg1'] for m in G]):.4f}, MT2={np.median([m['mg2'] for m in G]):.4f})")

# ---------------------------------------------------------------- 2-3. calibration
def calib(label, get_obs, get_grid):
    print(f"\n--- calibration {label} ---")
    obs = Counter(get_obs(m) for m in G)
    exp = defaultdict(float)
    for m in G:
        for c, p in get_grid(m).items():
            exp[c] += p
    chi2 = 0.0
    for c in CELLS:
        o, e = obs.get(c, 0), exp[c]
        z = (o - e) / math.sqrt(max(e * (1 - e / len(G)), 1e-9))
        chi2 += (o - e) ** 2 / e
        flag = '  <<<' if abs(z) > 3 else ''
        print(f"  {c[0]}-{c[1]}  obs={o:5d}  exp={e:8.1f}  z={z:+6.2f}{flag}")
    pv = 1 - stats.chi2.cdf(chi2, len(CELLS) - 1)
    print(f"  chi2={chi2:.2f} dof={len(CELLS)-1} p={pv:.4f}" + ("  CALIBRE" if pv > 0.01 else "  <<< PAS CALIBRE"))
    return pv

calib("G1 ('Mi-tps CS' devig) vs score MT1 observe", lambda m: (m['hh'], m['ha']), lambda m: m['g1'])
calib("G2 ('2eme mi-tps CS' devig) vs score MT2 observe", lambda m: (m['h2h'], m['h2a']), lambda m: m['g2'])

# ---------------------------------------------------------------- 4. independance MT1/MT2
print(SEP); print("4. INDEPENDANCE MT1 (vs) MT2, grilles controlees"); print(SEP)
# test joint 10x10 : E[c1,c2] = sum_m g1[c1]*g2[c2]  (heterogeneite geree)
obs_j = Counter(((m['hh'], m['ha']), (m['h2h'], m['h2a'])) for m in G)
exp_j = defaultdict(float)
for m in G:
    for c1_, p1 in m['g1'].items():
        for c2_, p2 in m['g2'].items():
            exp_j[(c1_, c2_)] += p1 * p2
pairs = [k for k in exp_j if exp_j[k] >= 5]
small_e = sum(exp_j[k] for k in exp_j if exp_j[k] < 5)
small_o = sum(obs_j.get(k, 0) for k in obs_j if k not in pairs)
o_list = [obs_j.get(k, 0) for k in pairs] + [small_o]
e_list = [exp_j[k] for k in pairs] + [small_e]
e_arr = np.array(e_list) * (len(G) / sum(e_list))
chi2 = float(((np.array(o_list) - e_arr) ** 2 / e_arr).sum())
dof = len(o_list) - 1
print(f"joint (cellules exp>=5 poolees) : chi2={chi2:.1f} dof={dof} p={1-stats.chi2.cdf(chi2,dof):.4f}")

# version 3x3 resultats : HT result vs H2 result
print("\n3x3 resultat MT1 x resultat MT2 (obs / exp modele indep):")
res1 = lambda m: res_of(m['hh'], m['ha'])
res2 = lambda m: res_of(m['h2h'], m['h2a'])
obs33 = Counter((res1(m), res2(m)) for m in G)
exp33 = defaultdict(float)
for m in G:
    p1 = {'1': 0.0, 'X': 0.0, '2': 0.0}; p2 = {'1': 0.0, 'X': 0.0, '2': 0.0}
    for c, p in m['g1'].items():
        p1[res_of(*c)] += p
    for c, p in m['g2'].items():
        p2[res_of(*c)] += p
    for r1_ in '1X2':
        for r2_ in '1X2':
            exp33[(r1_, r2_)] += p1[r1_] * p2[r2_]
chi33 = 0.0
for r1_ in '1X2':
    line = []
    for r2_ in '1X2':
        o, e = obs33.get((r1_, r2_), 0), exp33[(r1_, r2_)]
        chi33 += (o - e) ** 2 / e
        line.append(f"{r1_}/{r2_}: {o:4d}/{e:7.1f}")
    print("   " + "   ".join(line))
print(f"chi2={chi33:.2f} dof=8 p={1-stats.chi2.cdf(chi33,8):.4f}")

# ---------------------------------------------------------------- 5. convolution vs Score exact
print(SEP); print("5. CONVOLUTION G1xG2 vs GRILLE 'Score exact'"); print(SEP)

def convolve(g1, g2):
    out = defaultdict(float)
    for (i, j), p1 in g1.items():
        for (k, l), p2 in g2.items():
            out[(i + k, j + l)] += p1 * p2
    return out

GSE = [m for m in G if m['se'] is not None]
print(f"matchs avec Score exact complet : {len(GSE)}")
diffs = defaultdict(list)
for m in GSE:
    cv = convolve(m['g1'], m['g2'])
    for c in FT_CELLS:
        diffs[c].append(cv.get(c, 0.0) - m['se'].get(c, 0.0))
print("cellule  conv-SE moy   |conv-SE| moy   (probas)")
tot_abs = 0.0
for c in FT_CELLS:
    d = np.array(diffs[c])
    tot_abs += np.abs(d).mean()
    if abs(d.mean()) > 0.004 or np.abs(d).mean() > 0.006:
        print(f"  {c[0]}-{c[1]}   {d.mean():+8.4f}      {np.abs(d).mean():.4f}")
print(f"somme des |ecarts| moyens sur la grille FT : {tot_abs:.4f} (0 = pricing parfaitement coherent)")

# laquelle la realite suit ? log-likelihood par match du score FT observe
ll_conv = sum(math.log(max(convolve(m['g1'], m['g2']).get((m['h'], m['a']), 1e-9), 1e-9)) for m in GSE)
ll_se = sum(math.log(max(m['se'].get((m['h'], m['a']), 1e-9), 1e-9)) for m in GSE)
print(f"\nlog-vraisemblance du FT observe :  conv(G1,G2)={ll_conv:.1f}   Score-exact={ll_se:.1f}   "
      f"delta={(ll_conv-ll_se):+.1f}  ({'CONV gagne' if ll_conv > ll_se else 'SE gagne'})")
# test par paires (per-match diff de log-vrais)
dll = [math.log(max(convolve(m['g1'], m['g2']).get((m['h'], m['a']), 1e-9), 1e-9)) -
       math.log(max(m['se'].get((m['h'], m['a']), 1e-9), 1e-9)) for m in GSE]
t, pv = stats.ttest_1samp(dll, 0)
print(f"t-test apparie sur dLL/match : mean={np.mean(dll):+.5f} t={t:.2f} p={pv:.4f}")

# ---------------------------------------------------------------- 6. HT/FT & Mi-tps 1X2 vs modele
print(SEP); print("6. MARCHES 'HT/FT' et 'Mi-tps 1X2' vs DERIVATION G1,G2"); print(SEP)

def model_htft(m):
    out = defaultdict(float)
    for c1_, p1 in m['g1'].items():
        r1_ = res_of(*c1_)
        for c2_, p2 in m['g2'].items():
            rf = res_of(c1_[0] + c2_[0], c1_[1] + c2_[1])
            out[f"{r1_}/{rf}"] += p1 * p2
    return out

COMBOS = ['1/1', '1/X', '1/2', 'X/1', 'X/X', 'X/2', '2/1', '2/X', '2/2']
GH = [m for m in G if isinstance(m['em'].get('HT/FT'), dict) and set(m['em']['HT/FT'].keys()) == set(COMBOS)]
print(f"matchs avec HT/FT complet : {len(GH)}")
# devig HT/FT
for m in GH:
    imp = {k: 1.0 / float(v) for k, v in m['em']['HT/FT'].items() if float(v) >= 1.01}
    s = sum(imp.values())
    m['htft_mkt'] = {k: v / s for k, v in imp.items()} if len(imp) == 9 else None
    m['htft_margin'] = s
GH = [m for m in GH if m['htft_mkt']]
print(f"marge mediane HT/FT : {np.median([m['htft_margin'] for m in GH]):.4f}")

obs_c = Counter(f"{res_of(m['hh'], m['ha'])}/{res_of(m['h'], m['a'])}" for m in GH)
exp_model = defaultdict(float); exp_mkt = defaultdict(float)
for m in GH:
    mm = model_htft(m)
    for k in COMBOS:
        exp_model[k] += mm[k]
        exp_mkt[k] += m['htft_mkt'][k]
print("combo   obs    exp_modele(G1xG2)   exp_marche(devig)    z_mod   z_mkt")
chi_mod = chi_mkt = 0.0
for k in COMBOS:
    o, em_, ek = obs_c.get(k, 0), exp_model[k], exp_mkt[k]
    zm = (o - em_) / math.sqrt(em_)
    zk = (o - ek) / math.sqrt(ek)
    chi_mod += (o - em_) ** 2 / em_; chi_mkt += (o - ek) ** 2 / ek
    print(f"  {k:4s} {o:5d}   {em_:9.1f}          {ek:9.1f}         {zm:+6.2f}  {zk:+6.2f}")
print(f"chi2 modele={chi_mod:.1f} (p={1-stats.chi2.cdf(chi_mod,8):.4f})   chi2 marche={chi_mkt:.1f} (p={1-stats.chi2.cdf(chi_mkt,8):.4f})")

# ecart systematique marche vs modele par combo (ou est la marge ?)
print("\nratio cote_marche_implicite / proba_modele par combo (median):")
for k in COMBOS:
    r = [m['htft_mkt'][k] / max(model_htft(m)[k], 1e-9) for m in GH[:1500]]
    print(f"  {k:4s} median={np.median(r):.4f}")

# Mi-tps 1X2
GM = [m for m in G if isinstance(m['em'].get('Mi-tps 1X2'), dict) and set(m['em']['Mi-tps 1X2'].keys()) == {'1', 'X', '2'}]
print(f"\nmatchs avec Mi-tps 1X2 : {len(GM)}")
print("issue   obs    exp_G1     exp_marche   z_G1   z_mkt")
for r_ in '1X2':
    o = sum(1 for m in GM if res_of(m['hh'], m['ha']) == r_)
    eg = sum(sum(p for c, p in m['g1'].items() if res_of(*c) == r_) for m in GM)
    imp_tot = []
    for m in GM:
        imp = {k: 1.0 / float(v) for k, v in m['em']['Mi-tps 1X2'].items()}
        s = sum(imp.values())
        imp_tot.append(imp[r_] / s)
    ek = sum(imp_tot)
    print(f"  {r_}   {o:5d}   {eg:8.1f}   {ek:8.1f}   {(o-eg)/math.sqrt(eg):+5.2f}  {(o-ek)/math.sqrt(ek):+5.2f}")

# ---------------------------------------------------------------- 7. minutes
print(SEP); print("7. MINUTES : bursts et distribution"); print(SEP)
allmin = []
pair_same = pair_tot = 0
pair_same_cross = 0
for m in M:
    if not m['goals'] or len(m['goals']) != m['h'] + m['a']:
        continue
    mins = [(g['minute'], g['team']) for g in m['goals']]
    allmin.extend(x[0] for x in mins)
    for i in range(len(mins)):
        for j in range(i + 1, len(mins)):
            pair_tot += 1
            if mins[i][0] == mins[j][0]:
                pair_same += 1
                if mins[i][1] != mins[j][1]:
                    pair_same_cross += 1
allmin = np.array(allmin)
f = Counter(allmin)
n_all = len(allmin)
p_same_exp = sum((v / n_all) ** 2 for v in f.values())
exp_same = pair_tot * p_same_exp
print(f"buts={n_all}  paires intra-match={pair_tot}  meme minute obs={pair_same} exp(indep)={exp_same:.1f} "
      f"ratio={pair_same/exp_same:.2f}")
print(f"  dont equipes opposees : {pair_same_cross}")
pv = stats.poisson.sf(pair_same - 1, exp_same)
print(f"  p-value (Poisson approx, sur-representation) : {pv:.2e}")
print(f"frac buts minute<=45 : {np.mean(allmin <= 45):.4f}")
hist = Counter((mn - 1) // 15 for mn in allmin)
print("histogramme par quart d'heure (1-15...76-90+):", [hist.get(i, 0) for i in range(6)],
      " minute>90:", int(np.sum(allmin > 90)), " minute 45 exact:", int(np.sum(allmin == 45)), " minute 90 exact:", int(np.sum(allmin == 90)))

# ---------------------------------------------------------------- 8. transitions par segment
print(SEP); print("8. TRANSITIONS HT->FT PAR SEGMENT : mix de cotes ou effet segment ?"); print(SEP)
SEGS = [(1, 3, 'DS'), (4, 12, 'MS_early'), (13, 25, 'MS_mid'), (26, 33, 'MS_late'), (34, 38, 'FS')]
def seg_of(rnd):
    for lo, hi, nm in SEGS:
        if lo <= rnd <= hi:
            return nm
    return None

def p_ft_given_ht(m, target):
    """P(FT result = target | HT observe) via G2."""
    hh, ha = m['hh'], m['ha']
    return sum(p for (k, l), p in m['g2'].items() if res_of(hh + k, ha + l) == target)

print("transition      segment    n     obs%     exp%(G2|HT)      z")
for ht_state, ft_target, label in [('1', '1', 'HT1->FT1'), ('X', '1', 'HTX->FT1'),
                                   ('X', '2', 'HTX->FT2'), ('2', '2', 'HT2->FT2'),
                                   ('X', 'X', 'HTX->FTX')]:
    for lo, hi, nm in SEGS:
        sub = [m for m in G if seg_of(m['rnd']) == nm and res_of(m['hh'], m['ha']) == ht_state]
        if len(sub) < 30:
            continue
        obs = sum(1 for m in sub if res_of(m['h'], m['a']) == ft_target)
        ps = [p_ft_given_ht(m, ft_target) for m in sub]
        e, v = sum(ps), sum(p * (1 - p) for p in ps)
        z = (obs - e) / math.sqrt(v)
        flag = '  <<<' if abs(z) > 3 else ''
        print(f"  {label:10s}  {nm:8s} {len(sub):5d}  {obs/len(sub):6.1%}   {e/len(sub):6.1%}      {z:+6.2f}{flag}")
    print()
