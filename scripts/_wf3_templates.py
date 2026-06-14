# -*- coding: utf-8 -*-
"""WF3 — FACETTE: LE CATALOGUE DE COTES (templates de match)
Q1: nb de triplets distincts, grille discrete?
Q2: draw = f(home, away) deterministe?
Q3: overround constant?
Q4: distribution resultats par template (n>=30), stabilite temporelle
Q5: meme matchup -> meme template? sinon, qu'est-ce qui le determine?
Q6: walk-forward (train 70% / OOS 30%) sur tout edge par-template
"""
import sys, json, math
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from collections import Counter
from scipy import stats
from sqlalchemy import create_engine, text
from scraper.config import load_settings

pd.set_option('display.width', 220)
eng = create_engine(load_settings().db_url)

# ---------- LOAD ----------
q = """
SELECT e.id, e.round_info, e.team_a, e.team_b, e.expected_start, e.competition,
       o.odds_home, o.odds_draw, o.odds_away, o.id AS snap_id,
       r.score_a, r.score_b, r.ht_score_a, r.ht_score_b
FROM events e
JOIN results r ON r.event_id = e.id
JOIN odds_snapshots o ON o.event_id = e.id
WHERE e.round_info != '0'
"""
df = pd.read_sql(q, eng)
print("rows raw (event x snapshot):", len(df))

# Do odds move between snapshots of the same event?
snap_per_event = df.groupby('id')['snap_id'].nunique()
multi = snap_per_event[snap_per_event > 1]
print(f"\n=== SNAPSHOTS: events avec >1 snapshot: {len(multi)} / {snap_per_event.shape[0]}")
if len(multi):
    sub = df[df['id'].isin(multi.index)]
    var = sub.groupby('id')[['odds_home','odds_draw','odds_away']].nunique()
    moved = var[(var > 1).any(axis=1)]
    print(f"  events dont les cotes 1X2 BOUGENT entre snapshots: {len(moved)} / {len(multi)}")

# opening odds = MIN(snap_id)
df = df.sort_values('snap_id').groupby('id', as_index=False).first()
before = len(df)
df = df.sort_values('id').drop_duplicates(['team_a','team_b','expected_start'], keep='first')
print(f"events finis dedup: {len(df)} (drop {before-len(df)} dups)")
df = df.dropna(subset=['odds_home','odds_draw','odds_away','score_a','score_b'])
df['expected_start'] = pd.to_datetime(df['expected_start'])
df = df.sort_values(['expected_start','id']).reset_index(drop=True)
print("events exploitables:", len(df))

H, D, A = df['odds_home'].values, df['odds_draw'].values, df['odds_away'].values

# ---------- Q1: triplets distincts + grille ----------
print("\n" + "="*70)
print("Q1 — TRIPLETS DISTINCTS & GRILLE")
trip = list(zip(H, D, A))
cnt = Counter(trip)
print(f"triplets distincts: {len(cnt)} sur {len(df)} matchs (ratio {len(cnt)/len(df):.3f})")
freq = cnt.most_common(20)
print("Top 20 triplets:")
for t, n in freq:
    ov_t = 1/t[0]+1/t[1]+1/t[2]
    print(f"  H={t[0]:>6} D={t[1]:>6} A={t[2]:>6}  n={n:>3}  overround={ov_t:.4f}")
sizes = np.array(sorted(cnt.values(), reverse=True))
print(f"distribution tailles: n>=30: {(sizes>=30).sum()} templates couvrant {sizes[sizes>=30].sum()} matchs "
      f"({sizes[sizes>=30].sum()/len(df)*100:.1f}%)")
print(f"  n>=10: {(sizes>=10).sum()} ({sizes[sizes>=10].sum()/len(df)*100:.1f}% des matchs); "
      f"n==1: {(sizes==1).sum()}")

for name, v in [('home', H), ('draw', D), ('away', A)]:
    vals = np.unique(v)
    cents = np.round(v*100).astype(int)
    last = cents % 10
    lc = Counter(last)
    print(f"  {name}: {len(vals)} valeurs distinctes, min={vals.min()}, max={vals.max()}")
    print(f"    last-digit cents: {dict(sorted(lc.items()))}")
for name, v in [('home', H), ('draw', D), ('away', A)]:
    u = np.unique(np.round(v*100).astype(int))
    gaps = Counter(np.diff(u))
    print(f"  {name} gaps (cents) entre valeurs consecutives: {dict(sorted(gaps.items())[:12])}")

# dimension du catalogue : famille a 1 parametre ?
tpl_arr = np.array(list(cnt.keys()))
r_ha = np.corrcoef(1/tpl_arr[:,0], 1/tpl_arr[:,2])[0,1]
print(f"  corr(1/H, 1/A) sur les templates uniques: {r_ha:.4f} (-1 => famille 1-parametre)")

# ---------- Q2: draw = f(H, A)? ----------
print("\n" + "="*70)
print("Q2 — DRAW DETERMINISTE EN FONCTION DE (H,A)?")
pair_cnt = df.groupby(['odds_home','odds_away'])['odds_draw'].nunique()
multi_draw = pair_cnt[pair_cnt > 1]
print(f"paires (H,A) distinctes: {len(pair_cnt)}; paires avec >1 valeur de draw: {len(multi_draw)} "
      f"({len(multi_draw)/len(pair_cnt)*100:.1f}%)")
if len(multi_draw):
    for (h,a), k in multi_draw.head(8).items():
        ds = sorted(df[(df.odds_home==h)&(df.odds_away==a)]['odds_draw'].unique())
        print(f"  H={h} A={a}: draws={ds}")
X = np.column_stack([1/H, 1/A, (1/H)**2, (1/A)**2, (1/H)*(1/A), np.ones(len(H))])
coef, *_ = np.linalg.lstsq(X, D, rcond=None)
pred = X @ coef
res = D - pred
print(f"fit D ~ poly2(1/H,1/A): RMSE={np.sqrt((res**2).mean()):.4f}, R2={1-res.var()/D.var():.5f}, "
      f"max|res|={np.abs(res).max():.3f}")
# draw implicite par overround constant: D = 1/(K - 1/H - 1/A)
ov_all = 1/H + 1/D + 1/A
K = np.median(ov_all)
D_pred = 1/(K - 1/H - 1/A)
res_d = D - D_pred
print(f"D_pred=1/(K-1/H-1/A), K={K:.4f}: RMSE={np.sqrt((res_d**2).mean()):.4f}, "
      f"max|res|={np.abs(res_d).max():.3f}, %|res|<=0.02: {(np.abs(res_d)<=0.021).mean()*100:.1f}%")
ratio = np.minimum(H, A) / np.maximum(H, A)
print(f"corr(ratio min/max des cotes, draw) = {np.corrcoef(ratio, D)[0,1]:.4f}")

# ---------- Q3: overround ----------
print("\n" + "="*70)
print("Q3 — OVERROUND 1X2")
ov = ov_all
print(f"overround: mean={ov.mean():.5f}, std={ov.std():.5f}, min={ov.min():.5f}, max={ov.max():.5f}")
qs = np.percentile(ov, [1,5,25,50,75,95,99])
print(f"  percentiles 1/5/25/50/75/95/99: {np.round(qs,4)}")
ovc = Counter(np.round(ov, 3))
print(f"  valeurs arrondies 3 dec les + frequentes: {ovc.most_common(8)}")
fav = np.minimum(H, A)
bins = pd.cut(fav, [1.0,1.3,1.6,2.0,2.5,3.5])
tmp = pd.DataFrame({'fav':bins, 'ov':ov}).groupby('fav', observed=True)['ov'].agg(['mean','std','count'])
print("  overround par niveau de cote favori:")
print(tmp.to_string())
sl, ic, r_, p_, se = stats.linregress(fav, ov)
print(f"  linregress ov ~ fav: pente={sl:.5f}, r={r_:.4f}, p={p_:.2e}")

# ---------- Q4: distribution resultats par template ----------
print("\n" + "="*70)
print("Q4 — RESULTATS PAR TEMPLATE (n>=30) + STABILITE TEMPORELLE")
df['trip'] = list(zip(H, D, A))
df['res'] = np.where(df.score_a > df.score_b, 'H', np.where(df.score_a < df.score_b, 'A', 'D'))
df['half'] = (np.arange(len(df)) >= len(df)//2).astype(int)  # time-sorted

big = [t for t, n in cnt.items() if n >= 30]
print(f"templates n>=30: {len(big)}")
rows = []
for t in big:
    sub = df[df.trip == t]
    n = len(sub)
    iH, iD, iA = 1/t[0], 1/t[1], 1/t[2]
    s = iH+iD+iA
    eH, eD, eA = iH/s, iD/s, iA/s
    obs = np.array([(sub.res=='H').sum(), (sub.res=='D').sum(), (sub.res=='A').sum()])
    exp = np.array([eH, eD, eA]) * n
    chi2 = ((obs-exp)**2/exp).sum()
    p_gof = 1 - stats.chi2.cdf(chi2, 2)
    p_stab = np.nan
    c1 = sub[sub.half==0].res.value_counts().reindex(['H','D','A'], fill_value=0)
    c2 = sub[sub.half==1].res.value_counts().reindex(['H','D','A'], fill_value=0)
    if c1.sum() >= 10 and c2.sum() >= 10:
        tab = np.array([c1.values, c2.values])
        tab = tab[:, tab.sum(axis=0) > 0]
        try:
            _, p_stab, _, _ = stats.chi2_contingency(tab)
        except Exception:
            pass
    rows.append((t, n, obs[0]/n, obs[1]/n, obs[2]/n, eH, eD, eA, chi2, p_gof, p_stab))

tdf = pd.DataFrame(rows, columns=['trip','n','pH','pD','pA','eH','eD','eA','chi2','p_gof','p_stab'])
tdf = tdf.sort_values('n', ascending=False)
print(tdf.head(15).to_string(index=False,
      formatters={'pH':'{:.3f}'.format,'pD':'{:.3f}'.format,'pA':'{:.3f}'.format,
                  'eH':'{:.3f}'.format,'eD':'{:.3f}'.format,'eA':'{:.3f}'.format,
                  'chi2':'{:.2f}'.format,'p_gof':'{:.3f}'.format,'p_stab':'{:.3f}'.format}))
print(f"\ntemplates p_gof<0.05: {(tdf.p_gof<0.05).sum()} / {len(tdf)} "
      f"(attendu sous H0 ~{0.05*len(tdf):.1f})")
print(f"templates p_stab<0.05: {(tdf.p_stab.dropna()<0.05).sum()} / {tdf.p_stab.notna().sum()}")
chi2_total = tdf.chi2.sum(); dof = 2*len(tdf)
print(f"chi2 global agrege: {chi2_total:.1f} sur dof={dof}, p={1-stats.chi2.cdf(chi2_total, dof):.4f}")
fisher = -2*np.log(np.maximum(tdf.p_gof.values, 1e-300)).sum()
print(f"Fisher combine: stat={fisher:.1f}, dof={2*len(tdf)}, p={1-stats.chi2.cdf(fisher, 2*len(tdf)):.4f}")

print("\nScores exacts pour les 3 templates les + frequents (split moitie1 | moitie2):")
for t, n in cnt.most_common(3):
    sub = df[df.trip == t]
    for h_ in (0, 1):
        s2 = sub[sub.half==h_]
        sc = Counter(zip(s2.score_a.astype(int), s2.score_b.astype(int)))
        top = sc.most_common(6)
        print(f"  T={t} n={len(s2)} (moitie {h_+1}): " + ", ".join(f"{a}-{b}:{k}" for (a,b),k in top))

# ---------- Q5: meme matchup -> meme template? ----------
print("\n" + "="*70)
print("Q5 — MATCHUP (paire ordonnee) vs TEMPLATE")
g = df.groupby(['team_a','team_b'])
mt = g['trip'].nunique()
nmatch = g.size()
print(f"paires ordonnees: {len(mt)}; matchs/paire: mean={nmatch.mean():.1f}")
print(f"paires avec 1 seul template: {(mt==1).sum()}; >1: {(mt>1).sum()}")
both = pd.DataFrame({'ntpl': mt, 'nmatch': nmatch})
both['ratio'] = both.ntpl / both.nmatch
print(f"ratio templates/matchs par paire: mean={both[both.nmatch>=2].ratio.mean():.3f} (1.0 = jamais le meme template)")
pair = nmatch.idxmax()
sub = df[(df.team_a==pair[0]) & (df.team_b==pair[1])].sort_values('expected_start')
print(f"\nExemple paire {pair} ({len(sub)} matchs) — cotes au fil du temps:")
print("  odds_home:", list(sub.odds_home.values[:15]))
print("  odds_draw:", list(sub.odds_draw.values[:15]))
print("  odds_away:", list(sub.odds_away.values[:15]))
within_std = g['odds_home'].std().mean()
print(f"\nstd(odds_home) intra-paire (moyenne): {within_std:.3f} vs std globale: {df.odds_home.std():.3f}")

# ---- determinant de la variation : saison + forme courante ----
df['round_i'] = df.round_info.astype(int)
# detection saisons: le round repart en arriere => nouvelle saison
season, cur, prev = [], 0, None
for r_ in df['round_i']:
    if prev is not None and r_ < prev:
        cur += 1
    season.append(cur)
    prev = r_
df['season'] = season
ssz = df.groupby('season').size()
print(f"saisons detectees: {df.season.nunique()} (taille mediane {ssz.median():.0f} matchs)")

# points/diff de buts cumules AVANT le match
df['pts_a_match'] = np.where(df.res=='H', 3, np.where(df.res=='D', 1, 0))
df['pts_b_match'] = np.where(df.res=='A', 3, np.where(df.res=='D', 1, 0))
pre_pts_diff = np.zeros(len(df)); pre_gd_diff = np.zeros(len(df))
for s_, idx in df.groupby('season').groups.items():
    pts, gd = {}, {}
    for i in sorted(idx, key=lambda j: (df.at[j,'round_i'], df.at[j,'id'])):
        ta, tb = df.at[i,'team_a'], df.at[i,'team_b']
        pre_pts_diff[i] = pts.get(ta,0) - pts.get(tb,0)
        pre_gd_diff[i] = gd.get(ta,0) - gd.get(tb,0)
        pts[ta] = pts.get(ta,0) + df.at[i,'pts_a_match']
        pts[tb] = pts.get(tb,0) + df.at[i,'pts_b_match']
        gda = int(df.at[i,'score_a']) - int(df.at[i,'score_b'])
        gd[ta] = gd.get(ta,0) + gda
        gd[tb] = gd.get(tb,0) - gda
df['pre_pts_diff'] = pre_pts_diff
df['pre_gd_diff'] = pre_gd_diff
sH = 1/df.odds_home + 1/df.odds_draw + 1/df.odds_away
df['ph_devig'] = (1/df.odds_home) / sH

# intra-paire: residu de p_home vs residu de forme (rounds >= 5)
df['pair_key'] = list(zip(df.team_a, df.team_b))
pn = df.groupby('pair_key')['ph_devig'].transform('size')
m5 = (pn >= 3) & (df.round_i >= 5)
rp = df['ph_devig'] - df.groupby('pair_key')['ph_devig'].transform('mean')
rd = df['pre_pts_diff'] - df.groupby('pair_key')['pre_pts_diff'].transform('mean')
rg = df['pre_gd_diff'] - df.groupby('pair_key')['pre_gd_diff'].transform('mean')
rr = df['round_i'] - df.groupby('pair_key')['round_i'].transform('mean')
sl, ic, r1, p1, se = stats.linregress(rd[m5], rp[m5])
print(f"\nintra-paire (n={m5.sum()}): resid p_home ~ resid pts_diff : pente={sl:.5f}, r={r1:.4f}, p={p1:.2e}")
sl2, _, r2, p2, _ = stats.linregress(rg[m5], rp[m5])
print(f"intra-paire: resid p_home ~ resid gd_diff  : pente={sl2:.5f}, r={r2:.4f}, p={p2:.2e}")
sl3, _, r3, p3, _ = stats.linregress(rr[m5], rp[m5])
print(f"intra-paire: resid p_home ~ round centre   : pente={sl3:.5f}, r={r3:.4f}, p={p3:.2e}")
# R2 multiple (pts_diff + gd_diff)
Xf = np.column_stack([rd[m5], rg[m5], np.ones(m5.sum())])
cf2, *_ = np.linalg.lstsq(Xf, rp[m5], rcond=None)
prf = Xf @ cf2
r2m = 1 - ((rp[m5]-prf)**2).sum() / ((rp[m5]-rp[m5].mean())**2).sum()
print(f"R2 multiple intra-paire (pts_diff+gd_diff -> p_home): {r2m:.4f}")
# part de variance de p_home expliquee par la paire seule
grand = df.ph_devig.mean()
ss_between = ((df.groupby('pair_key')['ph_devig'].transform('mean') - grand)**2).sum()
ss_tot = ((df.ph_devig - grand)**2).sum()
print(f"part de variance de p_home expliquee par l'identite de la paire: {ss_between/ss_tot*100:.1f}%")

# ---------- Q6: WALK-FORWARD ----------
print("\n" + "="*70)
print("Q6 — WALK-FORWARD train 70% / OOS 30% (edge par template)")
cut = int(len(df)*0.70)
train, oos = df.iloc[:cut], df.iloc[cut:]
print(f"train n={len(train)}, OOS n={len(oos)}")

def run_wf(min_n, ev_thr, p_thr):
    sigs = []
    vc_tr = train['trip'].value_counts()
    for t, c in vc_tr.items():
        if c < min_n:
            break
        sub = train[train.trip == t]
        for out, odd in [('H', t[0]), ('D', t[1]), ('A', t[2])]:
            k = int((sub.res == out).sum())
            p_emp = k / c
            ev = p_emp * odd - 1
            if ev <= ev_thr:
                continue
            pv = stats.binomtest(k, c, 1/odd, alternative='greater').pvalue
            if pv < p_thr:
                sigs.append((t, out, odd, c, ev, pv))
    bets, pnl, wins, odd_sum = 0, 0.0, 0, 0.0
    for t, out, odd, c, ev, pv in sigs:
        sub = oos[oos.trip == t]
        if not len(sub):
            continue
        w = int((sub.res == out).sum())
        bets += len(sub); wins += w
        pnl += w*odd - len(sub); odd_sum += odd*len(sub)
    return sigs, bets, wins, pnl, odd_sum

for (min_n, ev_thr, p_thr) in [(30, 0.05, 0.05), (50, 0.10, 0.05), (30, 0.0, 0.10), (20, 0.15, 0.02)]:
    sigs, bets, wins, pnl, odd_sum = run_wf(min_n, ev_thr, p_thr)
    tag = f"min_n={min_n}, EV>{ev_thr}, p<{p_thr}"
    if not sigs:
        print(f"[{tag}] aucun signal sur train")
        continue
    roi = pnl/bets*100 if bets else float('nan')
    print(f"[{tag}] signaux train: {len(sigs)} | OOS: {bets} paris, {wins} gagnes, "
          f"PnL={pnl:+.1f}u, ROI={roi:+.1f}%, cote moy={odd_sum/bets if bets else 0:.2f}")
    for s_ in sigs[:8]:
        print(f"    T={s_[0]} bet={s_[1]} odd={s_[2]} n_train={s_[3]} EV_train={s_[4]:.3f} p={s_[5]:.4f}")

# baselines OOS
for out, arr in [('H', oos.odds_home.values), ('D', oos.odds_draw.values), ('A', oos.odds_away.values)]:
    w = (oos.res == out).values
    roi = ((w*arr).sum() / len(oos) - 1) * 100
    print(f"baseline OOS: parier {out} partout -> ROI={roi:+.2f}% (n={len(oos)})")

print("\nDONE")
