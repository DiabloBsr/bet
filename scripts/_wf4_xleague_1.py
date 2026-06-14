# -*- coding: utf-8 -*-
"""
WF4 — UNIVERSALITE CROSS-LIGUES DU MOTEUR
But: comparer par bucket de cote (1X2 et totals) les frequences reelles de
resultats entre les 9 ligues (chi2, 8035 = reference). Verdict poolabilite.

Etapes:
 1. Charge tous les matchs finis avec cote d'ouverture (MIN snap id) des 9 ligues.
 2. Nettoie: ids corrompus (8035), garde-fou maison (HT>FT, goals_json incoherent), dedup.
 3. Stats par ligue: marge 1X2, taux H/D/A, buts moyens, calibration.
 4. Chi2 par bucket de prob home devig (1X2) : ligue vs 8035.
 5. Chi2 par bucket de cote Over 3.5 (totals) : ligue vs 8035 (outcome O/U + distribution 0..6+).
 6. Inversion lambda Poisson -> biais offensif (reel - mu price) par ligue.
 7. Deviation Dixon-Coles 2-1/1-2 vs grille par groupe de ligues.
 8. Groupes: championnats nouveaux pooles, coupes poolees, vs 8035 et entre eux.
Sortie: exports/wf4_xleague.json
"""
import sys, json, math
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from scipy import stats
from scraper.config import load_settings
from sqlalchemy import create_engine

eng = create_engine(load_settings().db_url)

LEAGUES = ['InstantLeague-8035', 'InstantLeague-8036', 'InstantLeague-8037',
           'InstantLeague-8042', 'InstantLeague-8043', 'InstantLeague-8044',
           'InstantLeague-8056', 'InstantLeague-8060', 'InstantLeague-8065']
CHAMPS_NEW = ['InstantLeague-8036', 'InstantLeague-8037', 'InstantLeague-8042',
              'InstantLeague-8043', 'InstantLeague-8044']
CUPS = ['InstantLeague-8056', 'InstantLeague-8060', 'InstantLeague-8065']
REF = 'InstantLeague-8035'

n_tests = 0  # compteur global de tests statistiques scannes
RESULTS = {}

# ---------- 1. LOAD ----------
Q = """
SELECT e.id AS event_id, e.competition, e.round_info, e.team_a, e.team_b, e.expected_start,
       os.id AS snap_id, os.odds_home, os.odds_draw, os.odds_away, os.extra_markets,
       r.score_a, r.score_b, r.ht_score_a, r.ht_score_b, r.goals_json
FROM events e
JOIN (SELECT event_id, MIN(id) AS mid FROM odds_snapshots GROUP BY event_id) m ON m.event_id = e.id
JOIN odds_snapshots os ON os.id = m.mid
JOIN results r ON r.event_id = e.id
WHERE r.score_a IS NOT NULL AND r.score_b IS NOT NULL
"""
df = pd.read_sql(Q, eng)
df = df[df['competition'].isin(LEAGUES)].copy()
print(f"raw rows: {len(df)}")

# exclusion corrompus (couvre 8035 uniquement)
with open('exports/corrupted_events.json', 'r', encoding='utf-8') as f:
    corr = json.load(f)
bad_ids = set(int(k) for k in corr['events'].keys())
df = df[~df['event_id'].isin(bad_ids)].copy()
print(f"after corrupted excl: {len(df)}")

# garde-fou maison pour TOUTES les ligues (les nouvelles n'ont pas ete auditees):
# (a) HT > FT impossible ; (b) goals_json parseable mais len != total de buts
def guard(row):
    sa, sb = int(row['score_a']), int(row['score_b'])
    if row['ht_score_a'] is not None and row['ht_score_b'] is not None:
        if int(row['ht_score_a']) > sa or int(row['ht_score_b']) > sb:
            return False
    gj = row['goals_json']
    if gj:
        try:
            g = json.loads(gj)
            if isinstance(g, list) and len(g) > 0 and len(g) != sa + sb:
                return False
        except Exception:
            pass
    return True

mask = df.apply(guard, axis=1)
print(f"guard removed: {(~mask).sum()} ({(~mask).groupby(df['competition']).sum().to_dict()})")
df = df[mask].copy()

# dedup (meme paire+meme heure+meme ligue = meme match reel)
df = df.sort_values('snap_id').drop_duplicates(['competition', 'team_a', 'team_b', 'expected_start'], keep='first')
df = df.reset_index(drop=True)
print(f"after dedup: {len(df)}  par ligue: {df['competition'].value_counts().to_dict()}")

# variables de base
df['total'] = df['score_a'] + df['score_b']
df['out'] = np.where(df['score_a'] > df['score_b'], 'H', np.where(df['score_a'] < df['score_b'], 'A', 'D'))
inv = 1/df['odds_home'] + 1/df['odds_draw'] + 1/df['odds_away']
df['margin'] = inv - 1
df['ph'] = (1/df['odds_home']) / inv
df['pd'] = (1/df['odds_draw']) / inv
df['pa'] = (1/df['odds_away']) / inv

# cote over 3.5 d'ouverture depuis extra_markets "+/-"
def get_ou(em):
    try:
        d = json.loads(em)
        m = d.get('+/-')
        if m:
            return m.get('> 3.5'), m.get('< 3.5')
    except Exception:
        pass
    return None, None
ou = df['extra_markets'].apply(get_ou)
df['o_over'] = [x[0] for x in ou]
df['o_under'] = [x[1] for x in ou]

# ---------- 2. STATS PAR LIGUE ----------
per_league = {}
for lg, g in df.groupby('competition'):
    per_league[lg] = {
        'n': len(g),
        'margin_1x2_mean': round(float(g['margin'].mean()), 5),
        'margin_1x2_std': round(float(g['margin'].std()), 5),
        'rate_H': round(float((g['out'] == 'H').mean()), 4),
        'rate_D': round(float((g['out'] == 'D').mean()), 4),
        'rate_A': round(float((g['out'] == 'A').mean()), 4),
        'implied_H': round(float(g['ph'].mean()), 4),
        'implied_D': round(float(g['pd'].mean()), 4),
        'implied_A': round(float(g['pa'].mean()), 4),
        'mean_total_goals': round(float(g['total'].mean()), 3),
        'over35_rate': round(float((g['total'] >= 4).mean()), 4),
        'ou_odds_avail': int(g['o_over'].notna().sum()),
        'odds_home_med': round(float(g['odds_home'].median()), 3),
        'ph_min': round(float(g['ph'].min()), 4), 'ph_max': round(float(g['ph'].max()), 4),
    }
RESULTS['per_league'] = per_league
print(json.dumps(per_league, indent=1))

# ---------- 3. CHI2 1X2 PAR BUCKET DE PROB HOME DEVIG ----------
BUCKETS_PH = [0.0, 0.20, 0.35, 0.50, 0.65, 0.80, 1.01]
df['bucket_ph'] = pd.cut(df['ph'], BUCKETS_PH, right=False, labels=[f"[{a:.2f}-{b:.2f})" for a, b in zip(BUCKETS_PH[:-1], BUCKETS_PH[1:])])

def chi2_outcome_vs_ref(sub_ref, sub_lg):
    """table 2 x 3 (ref/ligue x H/D/A)"""
    t = np.array([[ (sub_ref['out'] == o).sum() for o in 'HDA'],
                  [ (sub_lg['out'] == o).sum() for o in 'HDA']], dtype=float)
    # retire les colonnes vides
    t = t[:, t.sum(axis=0) > 0]
    if t.shape[1] < 2 or t.sum(axis=1).min() < 20:
        return None, None
    chi2, p, dof, _ = stats.chi2_contingency(t)
    return float(chi2), float(p)

ref_df = df[df['competition'] == REF]
x2_1x2 = {}
for lg in LEAGUES:
    if lg == REF:
        continue
    gl = df[df['competition'] == lg]
    rows, chi_sum, dof_sum = {}, 0.0, 0
    for b in df['bucket_ph'].cat.categories:
        sr, sl = ref_df[ref_df['bucket_ph'] == b], gl[gl['bucket_ph'] == b]
        c, p = chi2_outcome_vs_ref(sr, sl)
        if c is not None:
            n_tests += 1
            rows[str(b)] = {'n_ref': len(sr), 'n_lg': len(sl), 'chi2': round(c, 3), 'p': round(p, 5),
                            'lg_HDA': [int((sl['out'] == o).sum()) for o in 'HDA'],
                            'ref_rates': [round(float((sr['out'] == o).mean()), 4) for o in 'HDA']}
            chi_sum += c; dof_sum += 2
    p_glob = float(1 - stats.chi2.cdf(chi_sum, dof_sum)) if dof_sum else None
    n_tests += 1
    x2_1x2[lg] = {'buckets': rows, 'global_chi2': round(chi_sum, 2), 'global_dof': dof_sum,
                  'global_p': round(p_glob, 6) if p_glob is not None else None}
RESULTS['chi2_1x2_vs_8035'] = x2_1x2

# groupes pooles
def grp(name, lgs):
    global n_tests
    gl = df[df['competition'].isin(lgs)]
    rows, chi_sum, dof_sum = {}, 0.0, 0
    for b in df['bucket_ph'].cat.categories:
        sr, sl = ref_df[ref_df['bucket_ph'] == b], gl[gl['bucket_ph'] == b]
        c, p = chi2_outcome_vs_ref(sr, sl)
        if c is not None:
            n_tests += 1
            rows[str(b)] = {'n_ref': len(sr), 'n_lg': len(sl), 'chi2': round(c, 3), 'p': round(p, 5),
                            'lg_rates': [round(float((sl['out'] == o).mean()), 4) for o in 'HDA'],
                            'ref_rates': [round(float((sr['out'] == o).mean()), 4) for o in 'HDA']}
            chi_sum += c; dof_sum += 2
    n_tests += 1
    return {'buckets': rows, 'global_chi2': round(chi_sum, 2), 'global_dof': dof_sum,
            'global_p': round(float(1 - stats.chi2.cdf(chi_sum, dof_sum)), 6) if dof_sum else None}

RESULTS['chi2_1x2_groups'] = {
    'champs_new_vs_8035': grp('champs', CHAMPS_NEW),
    'cups_vs_8035': grp('cups', CUPS),
}
# champs_new vs cups directement
ch, cu = df[df['competition'].isin(CHAMPS_NEW)], df[df['competition'].isin(CUPS)]
rows, chi_sum, dof_sum = {}, 0.0, 0
for b in df['bucket_ph'].cat.categories:
    sr, sl = ch[ch['bucket_ph'] == b], cu[cu['bucket_ph'] == b]
    c, p = chi2_outcome_vs_ref(sr, sl)
    if c is not None:
        n_tests += 1
        rows[str(b)] = {'n_champs': len(sr), 'n_cups': len(sl), 'chi2': round(c, 3), 'p': round(p, 5)}
        chi_sum += c; dof_sum += 2
n_tests += 1
RESULTS['chi2_1x2_groups']['champs_vs_cups'] = {'buckets': rows, 'global_chi2': round(chi_sum, 2),
    'global_dof': dof_sum, 'global_p': round(float(1 - stats.chi2.cdf(chi_sum, dof_sum)), 6) if dof_sum else None}

# ---------- 4. CALIBRATION GLOBALE PAR LIGUE (actual - implied, par issue) ----------
calib = {}
for lg, g in df.groupby('competition'):
    e = {}
    for o, pcol in [('H', 'ph'), ('D', 'pd'), ('A', 'pa')]:
        act = float((g['out'] == o).mean()); imp = float(g[pcol].mean())
        se = math.sqrt(max(imp * (1 - imp), 1e-9) / len(g))
        z = (act - imp) / se
        n_tests += 1
        e[o] = {'actual': round(act, 4), 'implied': round(imp, 4), 'z': round(z, 2),
                'p': round(float(2 * (1 - stats.norm.cdf(abs(z)))), 5)}
    calib[lg] = e
RESULTS['calibration'] = calib

# ---------- 5. TOTALS: O/U 3.5 PAR BUCKET DE COTE OVER ----------
dfo = df[df['o_over'].notna() & (df['o_over'] > 1.0)].copy()
BUCKETS_OU = [1.0, 1.5, 1.85, 2.2, 3.0, 100.0]
dfo['bucket_ou'] = pd.cut(dfo['o_over'], BUCKETS_OU, right=False,
                          labels=[f"[{a}-{b})" for a, b in zip(BUCKETS_OU[:-1], BUCKETS_OU[1:])])
dfo['is_over'] = (dfo['total'] >= 4).astype(int)
ref_o = dfo[dfo['competition'] == REF]
x2_ou = {}
for lg in LEAGUES:
    if lg == REF:
        continue
    gl = dfo[dfo['competition'] == lg]
    rows, chi_sum, dof_sum = {}, 0.0, 0
    for b in dfo['bucket_ou'].cat.categories:
        sr, sl = ref_o[ref_o['bucket_ou'] == b], gl[gl['bucket_ou'] == b]
        if len(sr) < 20 or len(sl) < 20:
            continue
        t = np.array([[sr['is_over'].sum(), len(sr) - sr['is_over'].sum()],
                      [sl['is_over'].sum(), len(sl) - sl['is_over'].sum()]], dtype=float)
        chi2, p, dof, _ = stats.chi2_contingency(t)
        n_tests += 1
        rows[str(b)] = {'n_ref': len(sr), 'n_lg': len(sl), 'chi2': round(float(chi2), 3), 'p': round(float(p), 5),
                        'over_rate_ref': round(float(sr['is_over'].mean()), 4),
                        'over_rate_lg': round(float(sl['is_over'].mean()), 4)}
        chi_sum += float(chi2); dof_sum += 1
    n_tests += 1
    x2_ou[lg] = {'buckets': rows, 'global_chi2': round(chi_sum, 2), 'global_dof': dof_sum,
                 'global_p': round(float(1 - stats.chi2.cdf(chi_sum, dof_sum)), 6) if dof_sum else None}
RESULTS['chi2_over35_vs_8035'] = x2_ou

# distribution du total exact (0..6+) par bucket OU, ligue groupee vs 8035
def chi2_total_dist(sub_ref, sub_lg):
    tr = np.array([ (sub_ref['total'].clip(upper=6) == k).sum() for k in range(7)], dtype=float)
    tl = np.array([ (sub_lg['total'].clip(upper=6) == k).sum() for k in range(7)], dtype=float)
    keep = (tr + tl) > 0
    t = np.vstack([tr[keep], tl[keep]])
    if t.shape[1] < 2 or t.sum(axis=1).min() < 30:
        return None, None
    chi2, p, dof, _ = stats.chi2_contingency(t)
    return float(chi2), float(p)

dist_tot = {}
for gname, lgs in [('champs_new', CHAMPS_NEW), ('cups', CUPS)]:
    gl = dfo[dfo['competition'].isin(lgs)]
    rows, chi_sum, dof_sum = {}, 0.0, 0
    for b in dfo['bucket_ou'].cat.categories:
        sr, sl = ref_o[ref_o['bucket_ou'] == b], gl[gl['bucket_ou'] == b]
        c, p = chi2_total_dist(sr, sl)
        if c is not None:
            n_tests += 1
            rows[str(b)] = {'n_ref': len(sr), 'n_lg': len(sl), 'chi2': round(c, 3), 'p': round(p, 5)}
            chi_sum += c; dof_sum += 6
    n_tests += 1
    dist_tot[gname] = {'buckets': rows, 'global_chi2': round(chi_sum, 2), 'global_dof': dof_sum,
                       'global_p': round(float(1 - stats.chi2.cdf(chi_sum, dof_sum)), 6) if dof_sum else None}
RESULTS['chi2_totaldist_groups'] = dist_tot

# ---------- 6. INVERSION LAMBDA + BIAIS OFFENSIF PAR LIGUE ----------
GMAX = 13
ks = np.arange(GMAX)

def grid_probs(lh, la):
    """vectorise: lh, la (N,) -> p_home, p_draw, mu"""
    from scipy.stats import poisson
    ph_ = poisson.pmf(ks[None, :], lh[:, None])   # (N, GMAX)
    pa_ = poisson.pmf(ks[None, :], la[:, None])
    M = ph_[:, :, None] * pa_[:, None, :]          # (N, GMAX, GMAX)
    tri_h = np.tril(np.ones((GMAX, GMAX)), -1)     # h > a
    p_home = (M * tri_h[None]).sum(axis=(1, 2))
    p_draw = (M * np.eye(GMAX)[None]).sum(axis=(1, 2))
    return p_home, p_draw

def invert_lambdas(ph_t, pd_t, iters=40):
    N = len(ph_t)
    lh = np.full(N, 1.6); la = np.full(N, 1.2)
    for _ in range(iters):
        f1, f2 = grid_probs(lh, la)
        eps = 1e-4
        f1h, f2h = grid_probs(lh + eps, la)
        f1a, f2a = grid_probs(lh, la + eps)
        J11 = (f1h - f1) / eps; J12 = (f1a - f1) / eps
        J21 = (f2h - f2) / eps; J22 = (f2a - f2) / eps
        det = J11 * J22 - J12 * J21
        det = np.where(np.abs(det) < 1e-12, 1e-12, det)
        r1 = f1 - ph_t; r2 = f2 - pd_t
        dlh = (J22 * r1 - J12 * r2) / det
        dla = (-J21 * r1 + J11 * r2) / det
        lh = np.clip(lh - np.clip(dlh, -0.5, 0.5), 0.05, 6.0)
        la = np.clip(la - np.clip(dla, -0.5, 0.5), 0.05, 6.0)
    return lh, la

lh, la = invert_lambdas(df['ph'].values, df['pd'].values)
f1, f2 = grid_probs(lh, la)
fit_err = np.abs(f1 - df['ph'].values) + np.abs(f2 - df['pd'].values)
df['lh'], df['la'], df['mu'] = lh, la, lh + la
df['fit_ok'] = fit_err < 0.002
print(f"lambda fit ok: {df['fit_ok'].mean():.4f}")

bias = {}
for lg, g in df[df['fit_ok']].groupby('competition'):
    b = float((g['total'] - g['mu']).mean())
    se = float(g['total'].std() / math.sqrt(len(g)))
    n_tests += 1
    bias[lg] = {'n': len(g), 'mean_mu_price': round(float(g['mu'].mean()), 3),
                'mean_total_real': round(float(g['total'].mean()), 3),
                'bias_goals': round(b, 4), 'se': round(se, 4), 'z_vs_0': round(b / se, 2)}
# difference de biais ligue vs 8035
b0 = bias[REF]
for lg in LEAGUES:
    if lg == REF:
        continue
    d = bias[lg]['bias_goals'] - b0['bias_goals']
    se = math.sqrt(bias[lg]['se']**2 + b0['se']**2)
    n_tests += 1
    bias[lg]['diff_vs_8035'] = round(d, 4)
    bias[lg]['z_vs_8035'] = round(d / se, 2)
    bias[lg]['p_vs_8035'] = round(float(2 * (1 - stats.norm.cdf(abs(d / se)))), 5)
RESULTS['offensive_bias'] = bias

# ---------- 7. DEVIATION DC 2-1 / 1-2 / 3-3 vs GRILLE PAR GROUPE ----------
def score_dev(sub, scores=((2, 1), (1, 2), (3, 3), (1, 1), (2, 2))):
    from scipy.stats import poisson
    out = {}
    lh_, la_ = sub['lh'].values, sub['la'].values
    for (h, a) in scores:
        pexp = poisson.pmf(h, lh_) * poisson.pmf(a, la_)
        exp_n = float(pexp.sum())
        obs = int(((sub['score_a'] == h) & (sub['score_b'] == a)).sum())
        var = float((pexp * (1 - pexp)).sum())
        z = (obs - exp_n) / math.sqrt(max(var, 1e-9))
        out[f"{h}-{a}"] = {'obs': obs, 'exp': round(exp_n, 1), 'ratio': round(obs / exp_n, 3) if exp_n > 0 else None,
                           'z': round(z, 2), 'p': round(float(2 * (1 - stats.norm.cdf(abs(z)))), 5)}
    return out

dc = {}
for gname, lgs in [('8035', [REF]), ('champs_new', CHAMPS_NEW), ('cups', CUPS)]:
    sub = df[df['fit_ok'] & df['competition'].isin(lgs)]
    dc[gname] = {'n': len(sub), 'scores': score_dev(sub)}
    n_tests += 5
RESULTS['dc_scores_vs_grid'] = dc

# ---------- 8. ROUND 0 (8035) check homogeneite interne ----------
g0 = df[(df['competition'] == REF) & (df['round_info'] == '0')]
g1 = df[(df['competition'] == REF) & (df['round_info'] != '0')]
rows, chi_sum, dof_sum = {}, 0.0, 0
for b in df['bucket_ph'].cat.categories:
    c, p = chi2_outcome_vs_ref(g1[g1['bucket_ph'] == b], g0[g0['bucket_ph'] == b])
    if c is not None:
        n_tests += 1
        rows[str(b)] = {'chi2': round(c, 3), 'p': round(p, 5)}
        chi_sum += c; dof_sum += 2
n_tests += 1
RESULTS['round0_vs_rest_8035'] = {'n_r0': len(g0), 'n_rest': len(g1), 'buckets': rows,
    'global_p': round(float(1 - stats.chi2.cdf(chi_sum, dof_sum)), 6) if dof_sum else None}

RESULTS['n_tests_scanned'] = n_tests
RESULTS['n_total_events'] = len(df)

with open('exports/wf4_xleague.json', 'w', encoding='utf-8') as f:
    json.dump(RESULTS, f, ensure_ascii=False, indent=1)
print(f"\nn_tests_scanned: {n_tests}")
print("saved exports/wf4_xleague.json")

# resume console
print("\n=== GLOBAL P (ligue vs 8035, 1X2 par bucket) ===")
for lg, v in x2_1x2.items():
    print(f"{lg}: chi2={v['global_chi2']} dof={v['global_dof']} p={v['global_p']}")
print("\n=== GROUPES 1X2 ===")
for k, v in RESULTS['chi2_1x2_groups'].items():
    print(f"{k}: chi2={v['global_chi2']} dof={v['global_dof']} p={v['global_p']}")
print("\n=== O/U 3.5 vs 8035 ===")
for lg, v in x2_ou.items():
    print(f"{lg}: chi2={v['global_chi2']} dof={v['global_dof']} p={v['global_p']}")
print("\n=== BIAIS OFFENSIF ===")
for lg, v in bias.items():
    print(f"{lg}: mu={v['mean_mu_price']} real={v['mean_total_real']} bias={v['bias_goals']} z0={v['z_vs_0']} "
          f"z_vs8035={v.get('z_vs_8035', '-')}")
print("\n=== DC SCORES ===")
for gname, v in dc.items():
    print(gname, {k: (s['ratio'], s['z']) for k, s in v['scores'].items()})
