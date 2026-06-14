# -*- coding: utf-8 -*-
"""
WF3 — RNG & TEMPS, partie 6 (mécanisme des minutes + FTTS final)
Q) Premier but de la MT par nb de buts k : tirage dédié vs premier-de-k-iid ?
   + gaps entre buts consécutifs
R) FTTS '1' : validation OOS avec seuil défini sur train (quantile 20% des cotes)
"""
import sys, json, math
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import create_engine, text
from scraper.config import load_settings

rng = np.random.default_rng(19)
eng = create_engine(load_settings().db_url)

SQL = """
SELECT e.id AS event_id, e.team_a, e.team_b, e.expected_start,
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
df['em'] = [json.loads(x) if isinstance(x, str) else x for x in df['extra_markets']]
df['total'] = df['score_a'] + df['score_b']
def parse_goals(gj):
    if gj is None or (isinstance(gj, float) and pd.isna(gj)): return None
    try: g = json.loads(gj) if isinstance(gj, str) else gj
    except Exception: return None
    return g if isinstance(g, list) else None
df['goals'] = [parse_goals(g) for g in df['goals_json']]

print("=" * 80)
print("Q — MÉCANISME DES MINUTES")
print("=" * 80)
# matchs cohérents uniquement
first1 = {1: [], 2: [], 3: []}   # 1er but MT1 par k
first2 = {1: [], 2: [], 3: []}   # 1er but MT2 par k (minute-45)
gaps1 = []
all1 = {1: [], 2: [], 3: []}
for _, row in df.iterrows():
    g = row['goals']
    if not g or pd.isna(row['ht_score_a']): continue
    if len(g) != row['total']: continue
    m1 = sorted(int(x['minute']) for x in g if int(x['minute']) <= 45)
    m2 = sorted(int(x['minute']) - 45 for x in g if int(x['minute']) > 45)
    if len(m1) != row['ht_score_a'] + row['ht_score_b']: continue
    if len(m1) in first1:
        first1[len(m1)].append(m1[0]); all1[len(m1)].extend(m1)
        if len(m1) >= 2: gaps1.extend(np.diff(m1))
    if len(m2) in first2:
        first2[len(m2)].append(m2[0])
print("Premier but MT1 par k:", {k: f"mean={np.mean(v):.2f} n={len(v)}" for k, v in first1.items()})
print("Premier but MT2 par k:", {k: f"mean={np.mean(v):.2f} n={len(v)}" for k, v in first2.items()})
ks12 = stats.ks_2samp(first1[1], first1[2]); ks13 = stats.ks_2samp(first1[1], first1[3])
print(f"KS 1er but MT1, k=1 vs k=2: D={ks12.statistic:.3f} p={ks12.pvalue:.2e} | "
      f"k=1 vs k=3: D={ks13.statistic:.3f} p={ks13.pvalue:.2e}")
# Sous iid-f (f = minutes des halves k=1), premier-de-k devrait être plus tôt :
f1 = np.array(first1[1])
sim_first2 = np.minimum(rng.choice(f1, 100000), rng.choice(f1, 100000))
sim_first3 = np.minimum(sim_first2, rng.choice(f1, 100000))
print(f"Si iid depuis f(k=1): E[min de 2]={sim_first2.mean():.2f} E[min de 3]={sim_first3.mean():.2f} "
      f"vs obs k=2: {np.mean(first1[2]):.2f} k=3: {np.mean(first1[3]):.2f}")
# 2e but: position relative au 1er
gaps1 = np.array(gaps1)
print(f"Gaps entre buts MT1 consécutifs: mean={gaps1.mean():.2f} median={np.median(gaps1):.0f} "
      f"P(gap=0)={np.mean(gaps1 == 0):.4f} P(gap<=2)={np.mean(gaps1 <= 2):.4f}")
h = np.bincount(gaps1, minlength=45)
print("Histogramme gaps 0-14:", h[:15].tolist())
# hazard du premier but du match (toutes minutes 1-90) vs hazard global
fb_all, atrisk = np.zeros(91), 0
nofirst = 0
for _, row in df.iterrows():
    g = row['goals']
    if row['total'] == 0: nofirst += 1; continue
    if not g or len(g) != row['total']: continue
    fb_all[min(int(x['minute']) for x in g)] += 1
n_m = fb_all.sum() + nofirst
surv = n_m - np.cumsum(fb_all)
haz = fb_all[1:91] / np.maximum(n_m - np.cumsum(fb_all)[:90] + fb_all[1:91], 1)
print("\nHazard 1er but par tranche de 5 min (proba conditionnelle x100):")
for i in range(0, 90, 15):
    print(f"  min {i + 1}-{i + 15}: " + " ".join(f"{100 * haz[j]:.1f}" for j in range(i, i + 15)))

print("\n" + "=" * 80)
print("R — FTTS '1' : OOS AVEC SEUILS DÉFINIS SUR TRAIN")
print("=" * 80)
recs = []
for _, row in df.iterrows():
    m = (row['em'] or {}).get('FTTS')
    if not m or '1' not in m: continue
    if row['total'] > 0 and not row['goals']: continue
    if row['total'] == 0: win = 0
    else:
        win = 1 if sorted(row['goals'], key=lambda g: int(g['minute']))[0]['team'] == 'Home' else 0
    recs.append((row['ts'], float(m['1']), win, row['odds_home']))
ft = pd.DataFrame(recs, columns=['ts', 'odds', 'win', 'oh'])
cut = ft['ts'].quantile(0.7)
tr, te = ft[ft['ts'] <= cut], ft[ft['ts'] > cut]
for label, col, q in [("cote FTTS1 <= q20 train", 'odds', 0.20), ("cote FTTS1 <= q30 train", 'odds', 0.30)]:
    th = tr[col].quantile(q)
    gtr = tr[tr[col] <= th]; gte = te[te[col] <= th]
    roi_tr = (gtr['win'] * gtr['odds'] - 1).mean()
    roi_te = (gte['win'] * gte['odds'] - 1).mean()
    se_te = (gte['win'] * gte['odds'] - 1).std(ddof=1) / math.sqrt(len(gte))
    bt = stats.binomtest(int(gte['win'].sum()), len(gte), float(np.mean(1 / gte['odds'])))
    print(f"{label} (th={th:.2f}): train n={len(gtr)} ROI={roi_tr * 100:+.2f}% | "
          f"OOS n={len(gte)} WR={gte['win'].mean():.4f} ROI={roi_te * 100:+.2f}% (SE {se_te * 100:.2f}%) "
          f"binom_p={bt.pvalue:.4f}")
# combiné: oh<1.6 ET cote FTTS1<=1.5
g = ft[(ft['oh'] < 1.6)]
gtr = g[g['ts'] <= cut]; gte = g[g['ts'] > cut]
roi_te = (gte['win'] * gte['odds'] - 1).mean()
print(f"oh<1.6: train ROI={((gtr['win'] * gtr['odds'] - 1).mean()) * 100:+.2f}% (n={len(gtr)}) | "
      f"OOS ROI={roi_te * 100:+.2f}% (n={len(gte)}) cote moy={gte['odds'].mean():.3f}")
# stat de l'ensemble du sample sur le segment (puissance)
gall = ft[ft['oh'] < 1.6]
pnl = gall['win'] * gall['odds'] - 1
print(f"Full-sample oh<1.6: n={len(gall)} ROI={pnl.mean() * 100:+.2f}% t={pnl.mean() / (pnl.std(ddof=1) / math.sqrt(len(pnl))):.2f}")

print("\nFIN _wf3_rng6.py")
