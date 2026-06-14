# -*- coding: utf-8 -*-
"""
WF3 — RNG & TEMPS, partie 2 : le biais de buts (obs 2.95 < CS 3.07) est-il réel et exploitable ?
A) Distribution totale de buts obs vs mixture CS devig vs marché 'Total de buts'
B) Cases de score exact : obs vs attendu (où est la déviation ?)
C) Artefact du cap odds=100 : mu recalculé sans les lignes cappées
D) Marché '+/-' : calibration + ROI flat bet Under/Over (cotes réelles)
E) Walk-forward : edge Under exploitable ? (train 70% / OOS 30%)
F) Minutes de but : hazard temporel du moteur, split MT, marché 'Minute du premier but'
"""
import sys, json, math
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import create_engine, text
from scraper.config import load_settings

rng = np.random.default_rng(7)
eng = create_engine(load_settings().db_url)

SQL = """
SELECT e.id AS event_id, e.round_info, e.team_a, e.team_b, e.expected_start,
       o.odds_home, o.odds_draw, o.odds_away, o.extra_markets,
       r.score_a, r.score_b, r.ht_score_a, r.ht_score_b, r.goals_json
FROM events e
JOIN (SELECT event_id, MIN(id) AS first_snap FROM odds_snapshots GROUP BY event_id) f
     ON f.event_id = e.id
JOIN odds_snapshots o ON o.id = f.first_snap
JOIN results r ON r.event_id = e.id
WHERE e.round_info != '0'
"""
with eng.connect() as c:
    df = pd.read_sql(text(SQL), c)
df = df.drop_duplicates(subset=['team_a', 'team_b', 'expected_start'], keep='first')
df['ts'] = pd.to_datetime(df['expected_start'])
df = df.sort_values(['ts', 'event_id']).reset_index(drop=True)
df['total'] = df['score_a'] + df['score_b']
df['em'] = [json.loads(x) if isinstance(x, str) else x for x in df['extra_markets']]
print(f"Matchs finis dédupliqués: {len(df)}")

# ----- CS devig (avec et sans lignes cappées à 100) -----
def cs_devig(em, drop_capped=False):
    cs = em.get('Score exact') if em else None
    if not cs: return None
    items = []
    for k, v in cs.items():
        try:
            a, b = k.split('-'); v = float(v)
            if drop_capped and v >= 99.99: continue
            items.append(((int(a), int(b)), 1.0 / v))
        except Exception: continue
    if not items: return None
    tot = sum(p for _, p in items)
    return {s: p / tot for s, p in items}

df['cs'] = [cs_devig(em) for em in df['em']]
df['cs_nc'] = [cs_devig(em, drop_capped=True) for em in df['em']]
def mu_of(d): return sum((a + b) * p for (a, b), p in d.items()) if d else np.nan
df['mu_cs'] = [mu_of(d) for d in df['cs']]
df['mu_cs_nc'] = [mu_of(d) for d in df['cs_nc']]

# ----- Marché 'Total de buts' devig (bucket max = 6 -> '6+') -----
def tg_devig(em):
    tg = em.get('Total de buts') if em else None
    if not tg: return None
    items = {int(k): 1.0 / float(v) for k, v in tg.items()}
    tot = sum(items.values())
    return {k: v / tot for k, v in items.items()}
df['tg'] = [tg_devig(em) for em in df['em']]

print("\n" + "=" * 80)
print("A — DISTRIBUTION DU TOTAL DE BUTS : OBS vs CS-devig vs marché 'Total de buts'")
print("=" * 80)
maxg = 9
obs = np.bincount(df['total'].astype(int).clip(0, maxg), minlength=maxg + 1).astype(float)
exp_cs = np.zeros(maxg + 1)
for d in df['cs']:
    for (a, b), p in d.items(): exp_cs[min(a + b, maxg)] += p
exp_tg = np.zeros(7)   # bucket 6 = '6+'
n_tg = 0
for d in df['tg']:
    if d is None: continue
    n_tg += 1
    for k, p in d.items(): exp_tg[min(k, 6)] += p
obs_tg = np.bincount(df['total'].astype(int).clip(0, 6), minlength=7).astype(float)
print(f"{'buts':>4} {'obs':>6} {'exp_CS':>8} {'z_CS':>7} {'exp_TG(6=6+)':>12} {'z_TG':>7}")
for k in range(maxg + 1):
    z_cs = (obs[k] - exp_cs[k]) / math.sqrt(max(exp_cs[k] * (1 - exp_cs[k] / len(df)), 1e-9))
    if k <= 6:
        o2 = obs_tg[k]
        z_tg = (o2 - exp_tg[k]) / math.sqrt(max(exp_tg[k] * (1 - exp_tg[k] / n_tg), 1e-9))
        print(f"{k:>4} {int(obs[k]):>6} {exp_cs[k]:>8.1f} {z_cs:>+7.2f} {exp_tg[k]:>12.1f} {z_tg:>+7.2f}")
    else:
        print(f"{k:>4} {int(obs[k]):>6} {exp_cs[k]:>8.1f} {z_cs:>+7.2f}")
chi_cs = (((obs - exp_cs) ** 2) / np.maximum(exp_cs, 1e-9))[exp_cs > 5].sum()
chi_tg = (((obs_tg - exp_tg) ** 2) / np.maximum(exp_tg, 1e-9))[exp_tg > 5].sum()
print(f"chi2 vs CS: {chi_cs:.1f} (dof~{(exp_cs > 5).sum() - 1}, p={1 - stats.chi2.cdf(chi_cs, (exp_cs > 5).sum() - 1):.2e})")
print(f"chi2 vs TG: {chi_tg:.1f} (dof~{(exp_tg > 5).sum() - 1}, p={1 - stats.chi2.cdf(chi_tg, (exp_tg > 5).sum() - 1):.2e})")
mu_tg = sum(sum(min(k, 6) * p for k, p in d.items()) for d in df['tg'] if d) / n_tg
print(f"E[buts]: obs={df['total'].mean():.3f} | CS={df['mu_cs'].mean():.3f} | CS sans cap={df['mu_cs_nc'].mean():.3f} "
      f"| TG (6+~6)={mu_tg:.3f}")

print("\n" + "=" * 80)
print("B — CASES SCORE EXACT : TOP DÉVIATIONS obs vs CS-devig")
print("=" * 80)
cells_obs, cells_exp = {}, {}
for _, row in df.iterrows():
    sc = (int(row['score_a']), int(row['score_b']))
    cells_obs[sc] = cells_obs.get(sc, 0) + 1
    for s, p in row['cs'].items():
        cells_exp[s] = cells_exp.get(s, 0.0) + p
rows = []
for s in sorted(set(cells_obs) | set(cells_exp)):
    o = cells_obs.get(s, 0); e = cells_exp.get(s, 0.0)
    if e < 3: continue
    z = (o - e) / math.sqrt(e)
    rows.append((s, o, e, z))
rows.sort(key=lambda r: -abs(r[3]))
print(f"{'score':>6} {'obs':>5} {'exp':>7} {'z':>6}")
for s, o, e, z in rows[:15]:
    print(f"{s[0]}-{s[1]:>4} {o:>5} {e:>7.1f} {z:>+6.2f}")
chi_cells = sum(((o - e) ** 2) / e for _, o, e, _ in rows)
print(f"chi2 global cases (e>=3): {chi_cells:.1f} dof~{len(rows) - 1} "
      f"p={1 - stats.chi2.cdf(chi_cells, len(rows) - 1):.2e}")

print("\n" + "=" * 80)
print("D — MARCHÉ '+/-' : CALIBRATION ET ROI FLAT (cotes réelles, sans devig)")
print("=" * 80)
recs = []
for _, row in df.iterrows():
    pm = row['em'].get('+/-') if row['em'] else None
    if not pm: continue
    line, o_over, o_under = None, None, None
    for k, v in pm.items():
        if k.startswith('>'): line = float(k.replace('>', '').strip()); o_over = float(v)
        elif k.startswith('<'): o_under = float(v)
    if line is None or o_over is None or o_under is None: continue
    recs.append((row['ts'], line, o_over, o_under, row['total'], row['mu_cs']))
pm_df = pd.DataFrame(recs, columns=['ts', 'line', 'o_over', 'o_under', 'total', 'mu_cs'])
print(f"Matchs avec marché +/-: {len(pm_df)} | lignes: {sorted(pm_df['line'].unique())}")
for line, g in pm_df.groupby('line'):
    if len(g) < 50: continue
    over = (g['total'] > line).astype(int)
    p_over_dev = (1 / g['o_over']) / (1 / g['o_over'] + 1 / g['o_under'])
    roi_over = (over * g['o_over'] - 1).mean()
    roi_under = ((1 - over) * g['o_under'] - 1).mean()
    print(f"ligne {line}: n={len(g)}  P(over) obs={over.mean():.4f} devig={p_over_dev.mean():.4f} "
          f"| ROI over={roi_over * 100:+.2f}%  ROI under={roi_under * 100:+.2f}% "
          f"| cote moy over={g['o_over'].mean():.2f} under={g['o_under'].mean():.2f}")

print("\n" + "=" * 80)
print("E — WALK-FORWARD : EDGE UNDER ? (train 70% temporel / OOS 30%)")
print("=" * 80)
cut = pm_df['ts'].quantile(0.7)
tr, te = pm_df[pm_df['ts'] <= cut], pm_df[pm_df['ts'] > cut]
print(f"train={len(tr)} (<= {cut})  oos={len(te)}")
picks = []
for line, g in tr.groupby('line'):
    if len(g) < 100: continue
    p_under = (g['total'] < line).mean()
    ev = p_under * g['o_under'].mean() - 1
    print(f"  train ligne {line}: n={len(g)} P(under)={p_under:.4f} cote_moy={g['o_under'].mean():.2f} EV={ev * 100:+.2f}%")
    if ev > 0.02: picks.append(line)
print(f"Lignes sélectionnées (EV train > +2%): {picks}")
if picks:
    sel = te[te['line'].isin(picks)]
    win = (sel['total'] < sel['line']).astype(int)
    pnl = win * sel['o_under'] - 1
    roi = pnl.mean(); se = pnl.std(ddof=1) / math.sqrt(len(pnl))
    bt = stats.binomtest(int(win.sum()), len(sel), (1 / sel['o_under']).mean())
    print(f"OOS: n={len(sel)} WR={win.mean():.4f} ROI={roi * 100:+.2f}% (±{se * 100:.2f}% SE) "
          f"cote moy={sel['o_under'].mean():.3f} | binom p={bt.pvalue:.4f}")
else:
    print("Aucune ligne avec EV>+2% sur train -> pas d'edge Under exploitable au flat.")

# Variante conditionnelle : parier Under seulement quand mu_cs (gonflé) est élevé ?
print("\nVariante: Under conditionné au gap (mu_cs - ligne)")
for thr in [-0.5, -0.25, 0.0, 0.25]:
    g = tr[tr['mu_cs'] - tr['line'] > thr]
    if len(g) < 80: continue
    p_under = (g['total'] < g['line']).mean()
    ev = (np.where(g['total'] < g['line'], g['o_under'], 0) - 1).mean()
    print(f"  train gap>{thr:+.2f}: n={len(g)} P(under)={p_under:.4f} EV={ev * 100:+.2f}%")

print("\n" + "=" * 80)
print("F — MINUTES DE BUT : HAZARD TEMPOREL DU MOTEUR")
print("=" * 80)
allmin, first_min = [], []
n_gj = 0
ht_check_ok = ht_check_bad = 0
for _, row in df.iterrows():
    gj = row['goals_json']
    if not gj: continue
    try: goals = json.loads(gj) if isinstance(gj, str) else gj
    except Exception: continue
    if not isinstance(goals, list): continue
    n_gj += 1
    ms = sorted(int(g['minute']) for g in goals)
    allmin.extend(ms)
    if ms: first_min.append(ms[0])
    # cohérence HT
    if pd.notna(row['ht_score_a']):
        nht = sum(1 for g in goals if int(g['minute']) <= 45)
        if nht == row['ht_score_a'] + row['ht_score_b']: ht_check_ok += 1
        else: ht_check_bad += 1
allmin = np.array(allmin); first_min = np.array(first_min)
print(f"Matchs avec goals_json: {n_gj} | buts: {len(allmin)} | cohérence HT(<=45): ok={ht_check_ok} bad={ht_check_bad}")
print(f"Minutes: min={allmin.min()} max={allmin.max()} | mean={allmin.mean():.2f} (45.5 si uniforme 1-90)")
hist, _ = np.histogram(allmin, bins=np.arange(0.5, 91.6, 5))
print("Histogramme par tranche de 5 min (1-90):", hist.tolist())
# KS vs uniforme discret 1..90
ks = stats.kstest((allmin - 0.5) / 90.0, 'uniform')
print(f"KS vs uniforme[1,90]: D={ks.statistic:.4f} p={ks.pvalue:.2e}")
h1 = (allmin <= 45).sum(); h2 = (allmin > 45).sum()
bt = stats.binomtest(h1, h1 + h2, 0.5)
print(f"1ère MT: {h1} buts vs 2ème MT: {h2} (p binomial 50/50 = {bt.pvalue:.4f})")
# minute par minute: spikes ?
mhist = np.bincount(allmin, minlength=91)[1:91]
top = np.argsort(mhist)[::-1][:5] + 1
print(f"Top 5 minutes les plus fréquentes: {[(int(m), int(mhist[m - 1])) for m in top]}")
flat = mhist.mean()
chi_min = ((mhist - flat) ** 2 / flat).sum()
print(f"chi2 uniformité minute (1-90): {chi_min:.1f} dof=89 p={1 - stats.chi2.cdf(chi_min, 89):.2e}")
# pente du hazard: régression du compte par minute
x = np.arange(1, 91)
sl, ic, rv, pv, se = stats.linregress(x, mhist)
print(f"Pente buts/minute: {sl:+.3f}/min (p={pv:.2e}) -> hazard {'croissant' if sl > 0 else 'décroissant/plat'}")

# Marché 'Minute du premier but' : calibration
print("\nMarché 'Minute du premier but':")
buckets_obs, buckets_exp, nfb = {}, {}, 0
sample_keys = None
for _, row in df.iterrows():
    m = row['em'].get('Minute du premier but') if row['em'] else None
    gj = row['goals_json']
    if not m or not gj: continue
    try: goals = json.loads(gj) if isinstance(gj, str) else gj
    except Exception: continue
    inv = {k: 1.0 / float(v) for k, v in m.items()}
    tot = sum(inv.values())
    if sample_keys is None: sample_keys = list(m.keys())
    fmin = min((int(g['minute']) for g in goals), default=None)
    nfb += 1
    for k, p in inv.items():
        buckets_exp[k] = buckets_exp.get(k, 0.0) + p / tot
    def in_bucket(k, fm):
        k = k.strip()
        if 'aucun' in k.lower() or 'no goal' in k.lower(): return fm is None
        k2 = k.replace('+', '-200')
        try:
            lo, hi = k2.split('-'); return fm is not None and int(lo) <= fm <= int(hi)
        except Exception: return False
    for k in inv:
        if in_bucket(k, fmin):
            buckets_obs[k] = buckets_obs.get(k, 0) + 1
print(f"n={nfb} | buckets: {sample_keys}")
for k in (sample_keys or []):
    o = buckets_obs.get(k, 0); e = buckets_exp.get(k, 0.0)
    if e > 0:
        z = (o - e) / math.sqrt(e * (1 - e / nfb))
        print(f"  {k:>12}: obs={o:>5} exp={e:>7.1f} z={z:+.2f}")

print("\nFIN _wf3_rng2.py")
