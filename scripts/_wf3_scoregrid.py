# -*- coding: utf-8 -*-
"""
WF3 — FACETTE 'GRILLE DES SCORES'
Le marché 'Score exact' dévigorisé EST-IL le générateur ?

Sections:
  0. Chargement + dédup + parsing du marché Score exact (cotes d'ouverture = MIN(id))
  1. Fréquence réelle vs proba devig par cellule (Poisson-binomial z-test + FDR BH)
  2. Même test conditionné par profil de cotes (favori extrême / fort / léger / équilibré)
  3. Cohérence inter-marchés : grille -> 1X2 / Total de buts / +/- / Pair-Impair
     + lequel matche la réalité (log-loss / Brier pairé)
  4. Résidu Dixon-Coles sur la diagonale basse (0-0, 1-0, 0-1, 1-1) vs grille marché
  5. Le cap à 6 buts : pricing des cellules hautes + 'Total de buts = 6'
  6. WALK-FORWARD (train 70% temporel / OOS 30%) sur tout signal exploitable
"""
import sys, json, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import create_engine, text
from scraper.config import load_settings

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 50)

eng = create_engine(load_settings().db_url)

# ----------------------------------------------------------------------------
# 0. LOAD
# ----------------------------------------------------------------------------
q = """
SELECT e.id AS event_id, e.round_info, e.team_a, e.team_b, e.expected_start,
       r.score_a, r.score_b,
       o.odds_home, o.odds_draw, o.odds_away, o.extra_markets
FROM events e
JOIN results r ON r.event_id = e.id
JOIN odds_snapshots o ON o.id = (SELECT MIN(id) FROM odds_snapshots os WHERE os.event_id = e.id)
WHERE e.round_info != '0' AND r.score_a IS NOT NULL AND r.score_b IS NOT NULL
"""
df = pd.read_sql(text(q), eng)
df = df.sort_values('event_id').drop_duplicates(
    subset=['team_a', 'team_b', 'expected_start'], keep='first').reset_index(drop=True)
df['expected_start'] = pd.to_datetime(df['expected_start'])
df = df.sort_values('expected_start').reset_index(drop=True)
print(f"[0] matchs finis dédupliqués: {len(df)}")

CELLS = [f"{a}-{b}" for a in range(7) for b in range(7) if a + b <= 6]  # 28 cellules
N_CELLS = len(CELLS)

def parse_em(s):
    if s is None: return {}
    return json.loads(s) if isinstance(s, str) else s

ems = df['extra_markets'].map(parse_em)

# grille de cotes Score exact -> matrices
odds_grid = np.full((len(df), N_CELLS), np.nan)
keysets = set()
for i, em in enumerate(ems):
    cs = em.get('Score exact')
    if not cs: continue
    keysets.add(frozenset(cs.keys()))
    for j, c in enumerate(CELLS):
        v = cs.get(c)
        if v and v > 1.0:
            odds_grid[i, j] = v

print(f"[0] keysets distincts du marché Score exact: {len(keysets)}; "
      f"matchs avec grille complète: {(~np.isnan(odds_grid).any(axis=1)).sum()}")

mask = ~np.isnan(odds_grid).any(axis=1)
df = df[mask].reset_index(drop=True)
odds_grid = odds_grid[mask]
ems = ems[mask].reset_index(drop=True)
n = len(df)
print(f"[0] N final = {n}")

imp = 1.0 / odds_grid                      # probas implicites brutes
overround = imp.sum(axis=1)
devig = imp / overround[:, None]           # devig proportionnel
capped = (odds_grid >= 99.99)
print(f"[0] overround Score exact: mean={overround.mean():.4f} sd={overround.std():.4f} "
      f"min={overround.min():.4f} max={overround.max():.4f}")
print(f"[0] cellules cappées à 100.0: {capped.mean()*100:.1f}% des cellules "
      f"(moy {capped.sum(axis=1).mean():.1f}/match)")

# résultat -> index cellule
res_key = df['score_a'].astype(int).astype(str) + '-' + df['score_b'].astype(int).astype(str)
cell_idx = {c: j for j, c in enumerate(CELLS)}
y = res_key.map(cell_idx).values
assert not np.isnan(y.astype(float)).any(), "score hors grille !"
y = y.astype(int)
Y = np.zeros((n, N_CELLS)); Y[np.arange(n), y] = 1.0

# devig 1X2
imp3 = np.column_stack([1/df['odds_home'], 1/df['odds_draw'], 1/df['odds_away']])
ovr3 = imp3.sum(axis=1)
p1x2 = imp3 / ovr3[:, None]
print(f"[0] overround 1X2: mean={ovr3.mean():.4f}")

def bh_fdr(pvals):
    p = np.asarray(pvals); m = len(p)
    order = np.argsort(p); ranked = p[order] * m / (np.arange(m) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m); out[order] = np.clip(ranked, 0, 1)
    return out

def cell_table(sub_devig, sub_capped, sub_Y, sub_odds, label):
    """test Poisson-binomial (approx normale) par cellule + FDR BH"""
    rows = []
    nn = len(sub_Y)
    for j, c in enumerate(CELLS):
        p = sub_devig[:, j]
        k = sub_Y[:, j].sum()
        E = p.sum(); V = (p * (1 - p)).sum()
        z = (k - E) / np.sqrt(V) if V > 0 else 0.0
        pv = 2 * stats.norm.sf(abs(z))
        roi = (sub_Y[:, j] * sub_odds[:, j]).sum() / nn - 1.0  # flat bet chaque match
        rows.append(dict(cell=c, obs=int(k), exp=E, freq=k/nn, devig=p.mean(),
                         raw=(1/sub_odds[:, j]).mean(), z=z, p=pv, roi_flat=roi,
                         pct_cap=sub_capped[:, j].mean()))
    t = pd.DataFrame(rows)
    t['q_fdr'] = bh_fdr(t['p'].values)
    t['sig'] = np.where(t['q_fdr'] < 0.05, '***', '')
    # chi2 global
    chi2 = ((t['obs'] - t['exp'])**2 / np.maximum(t['exp'], 1e-9)).sum()
    pglob = stats.chi2.sf(chi2, N_CELLS - 1)
    print(f"\n=== [{label}] n={nn} | chi2 global grille={chi2:.1f} (df={N_CELLS-1}) p={pglob:.2e} ===")
    print(t.sort_values('q_fdr')[['cell','obs','exp','freq','devig','raw','z','p','q_fdr','sig','roi_flat','pct_cap']]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return t

# ----------------------------------------------------------------------------
# 1. CELLULE PAR CELLULE — GLOBAL
# ----------------------------------------------------------------------------
print("\n" + "="*100)
print("[1] FREQUENCE REELLE vs DEVIG 'Score exact' — PAR CELLULE (FDR BH)")
t_all = cell_table(devig, capped, Y, odds_grid, "GLOBAL")

# log-loss du devig comme prédicteur du score exact
ll_devig = -np.log(np.clip(devig[np.arange(n), y], 1e-12, 1)).mean()
print(f"\n[1] log-loss devig grille sur score exact: {ll_devig:.4f} "
      f"(uniforme=log(28)={np.log(28):.4f})")

# ----------------------------------------------------------------------------
# 2. CONDITIONNE PAR PROFIL DE COTES
# ----------------------------------------------------------------------------
print("\n" + "="*100)
print("[2] PAR PROFIL DE COTES (pmax = max(pH,pA) devig 1X2)")
pmax = np.maximum(p1x2[:, 0], p1x2[:, 2])
print(pd.Series(pmax).describe().to_string())
buckets = pd.Series(pd.cut(pmax, [0, 0.40, 0.50, 0.62, 1.0],
                 labels=['equilibre<40', 'leger40-50', 'fort50-62', 'extreme>62']))
prof_tables = {}
for b in buckets.cat.categories:
    m = (buckets == b).values
    if m.sum() < 300: continue
    prof_tables[b] = cell_table(devig[m], capped[m], Y[m], odds_grid[m], f"profil={b}")

# ----------------------------------------------------------------------------
# 3. COHERENCE INTER-MARCHES
# ----------------------------------------------------------------------------
print("\n" + "="*100)
print("[3] COHERENCE INTER-MARCHES")

# --- 3a. grille -> 1X2
h_cells = [j for j, c in enumerate(CELLS) if int(c[0]) > int(c[2])]
d_cells = [j for j, c in enumerate(CELLS) if int(c[0]) == int(c[2])]
a_cells = [j for j, c in enumerate(CELLS) if int(c[0]) < int(c[2])]
g1x2 = np.column_stack([devig[:, h_cells].sum(1), devig[:, d_cells].sum(1), devig[:, a_cells].sum(1)])
diff = g1x2 - p1x2
print("\n[3a] grille->1X2 vs 1X2 devig (diff = grille - 1X2):")
for k, lab in enumerate(['Home', 'Draw', 'Away']):
    print(f"  {lab}: mean diff={diff[:,k].mean():+.4f}  MAE={np.abs(diff[:,k]).mean():.4f} "
          f" max|d|={np.abs(diff[:,k]).max():.4f}  corr={np.corrcoef(g1x2[:,k], p1x2[:,k])[0,1]:.5f}")
out3 = np.where(df['score_a'] > df['score_b'], 0, np.where(df['score_a'] == df['score_b'], 1, 2))
ll_g = -np.log(np.clip(g1x2[np.arange(n), out3], 1e-12, 1))
ll_m = -np.log(np.clip(p1x2[np.arange(n), out3], 1e-12, 1))
w = stats.wilcoxon(ll_g, ll_m)
print(f"  log-loss 1X2: grille={ll_g.mean():.5f} vs marché 1X2={ll_m.mean():.5f} "
      f"(wilcoxon p={w.pvalue:.3g}) -> {'GRILLE' if ll_g.mean()<ll_m.mean() else '1X2'} meilleur")

# --- 3b. grille -> Total de buts (0..6)
tot_market = np.full((n, 7), np.nan)
for i, em in enumerate(ems):
    tm = em.get('Total de buts')
    if not tm: continue
    for t_ in range(7):
        v = tm.get(str(t_))
        if v and v > 1.0:
            tot_market[i, t_] = 1.0 / v
m_tot = ~np.isnan(tot_market).any(axis=1)
tm_devig = tot_market[m_tot] / tot_market[m_tot].sum(axis=1)[:, None]
tot_cells = [[j for j, c in enumerate(CELLS) if int(c[0]) + int(c[2]) == t_] for t_ in range(7)]
g_tot = np.column_stack([devig[:, js].sum(1) for js in tot_cells])[m_tot]
y_tot = (df['score_a'] + df['score_b']).values[m_tot]
print(f"\n[3b] grille->Total vs marché 'Total de buts' (n={m_tot.sum()}):")
ovr_tot = tot_market[m_tot].sum(axis=1)
print(f"  overround 'Total de buts': mean={ovr_tot.mean():.4f}")
for t_ in range(7):
    fr = (y_tot == t_).mean()
    print(f"  T={t_}: réel={fr:.4f}  grille={g_tot[:,t_].mean():.4f}  marchéTot={tm_devig[:,t_].mean():.4f} "
          f" diff(g-m)={g_tot[:,t_].mean()-tm_devig[:,t_].mean():+.4f}")
nt = m_tot.sum()
ll_gt = -np.log(np.clip(g_tot[np.arange(nt), y_tot], 1e-12, 1))
ll_mt = -np.log(np.clip(tm_devig[np.arange(nt), y_tot], 1e-12, 1))
w = stats.wilcoxon(ll_gt, ll_mt)
print(f"  log-loss Total: grille={ll_gt.mean():.5f} vs marchéTot={ll_mt.mean():.5f} "
      f"(wilcoxon p={w.pvalue:.3g}) -> {'GRILLE' if ll_gt.mean()<ll_mt.mean() else 'MARCHE TOTAL'} meilleur")
print(f"  corr cellule à cellule grille vs marchéTot: "
      + " ".join(f"T{t_}={np.corrcoef(g_tot[:,t_], tm_devig[:,t_])[0,1]:.4f}" for t_ in range(7)))

# --- 3c. grille -> +/- (ligne dynamique)
rows = []
for i, em in enumerate(ems):
    ou = em.get('+/-')
    if not ou: continue
    over_k = [k for k in ou if k.startswith('>')]
    under_k = [k for k in ou if k.startswith('<')]
    if len(over_k) != 1 or len(under_k) != 1: continue
    line = float(over_k[0].replace('>', '').strip())
    io, iu = 1/ou[over_k[0]], 1/ou[under_k[0]]
    p_over_mkt = io / (io + iu)
    p_over_grid = sum(devig[i, j] for j, c in enumerate(CELLS) if int(c[0]) + int(c[2]) > line)
    tot_real = df['score_a'].iat[i] + df['score_b'].iat[i]
    rows.append(dict(i=i, line=line, p_mkt=p_over_mkt, p_grid=p_over_grid,
                     over=int(tot_real > line), odds_over=ou[over_k[0]], odds_under=ou[under_k[0]],
                     ovr=io+iu))
ou_df = pd.DataFrame(rows)
print(f"\n[3c] '+/-' : n={len(ou_df)}, lignes={sorted(ou_df['line'].unique())}, "
      f"overround mean={ou_df['ovr'].mean():.4f}")
print(f"  P(over): réel={ou_df['over'].mean():.4f}  grille={ou_df['p_grid'].mean():.4f}  marché={ou_df['p_mkt'].mean():.4f}")
b_g = ((ou_df['p_grid'] - ou_df['over'])**2).mean()
b_m = ((ou_df['p_mkt'] - ou_df['over'])**2).mean()
w = stats.wilcoxon((ou_df['p_grid'] - ou_df['over'])**2, (ou_df['p_mkt'] - ou_df['over'])**2)
print(f"  Brier: grille={b_g:.5f} vs marché={b_m:.5f} (wilcoxon p={w.pvalue:.3g})")
print(f"  corr p_grid vs p_mkt = {np.corrcoef(ou_df['p_grid'], ou_df['p_mkt'])[0,1]:.5f}; "
      f"mean(p_grid - p_mkt) = {(ou_df['p_grid']-ou_df['p_mkt']).mean():+.5f}")

# --- 3d. Pair/Impair
rows = []
for i, em in enumerate(ems):
    pi = em.get('Pair/Impair')
    if not pi or 'Pair' not in pi or 'Impair' not in pi: continue
    ip_, ii_ = 1/pi['Pair'], 1/pi['Impair']
    p_pair_mkt = ip_ / (ip_ + ii_)
    p_pair_grid = sum(devig[i, j] for j, c in enumerate(CELLS) if (int(c[0]) + int(c[2])) % 2 == 0)
    rows.append(dict(p_mkt=p_pair_mkt, p_grid=p_pair_grid,
                     pair=int((df['score_a'].iat[i] + df['score_b'].iat[i]) % 2 == 0)))
pi_df = pd.DataFrame(rows)
print(f"\n[3d] Pair/Impair: n={len(pi_df)} | réel P(pair)={pi_df['pair'].mean():.4f} "
      f"grille={pi_df['p_grid'].mean():.4f} marché={pi_df['p_mkt'].mean():.4f}")
b_g = ((pi_df['p_grid'] - pi_df['pair'])**2).mean(); b_m = ((pi_df['p_mkt'] - pi_df['pair'])**2).mean()
print(f"  Brier: grille={b_g:.5f} vs marché={b_m:.5f} ; "
      f"corr={np.corrcoef(pi_df['p_grid'], pi_df['p_mkt'])[0,1]:.5f}")

# ----------------------------------------------------------------------------
# 4. RESIDU DIXON-COLES (0-0, 1-0, 0-1, 1-1) — la réalité vs la grille marché
# ----------------------------------------------------------------------------
print("\n" + "="*100)
print("[4] RESIDU DIXON-COLES sur diagonale basse (base = grille devig)")
dc_cells = ['0-0', '1-0', '0-1', '1-1']
chi2_dc = 0.0
for c in dc_cells:
    j = cell_idx[c]
    p = devig[:, j]; k = Y[:, j].sum(); E = p.sum(); V = (p*(1-p)).sum()
    z = (k - E)/np.sqrt(V)
    chi2_dc += z*z
    print(f"  {c}: obs={int(k)} exp={E:.1f} ratio={k/E:.4f} z={z:+.3f} p={2*stats.norm.sf(abs(z)):.4f}")
print(f"  chi2 joint (df=4) = {chi2_dc:.2f}, p = {stats.chi2.sf(chi2_dc, 4):.4f}")
# tau effectif: ratio moyen pondéré obs/exp diag vs off-diag basse
lo = ['0-0','1-1']; off = ['1-0','0-1']
rl = sum(Y[:, cell_idx[c]].sum() for c in lo) / sum(devig[:, cell_idx[c]].sum() for c in lo)
ro = sum(Y[:, cell_idx[c]].sum() for c in off) / sum(devig[:, cell_idx[c]].sum() for c in off)
print(f"  ratio diag(0-0,1-1)={rl:.4f} vs off-diag(1-0,0-1)={ro:.4f} -> tau résiduel ~ {rl/ro:.4f}")

# ----------------------------------------------------------------------------
# 5. LE CAP A 6 BUTS
# ----------------------------------------------------------------------------
print("\n" + "="*100)
print("[5] CAP A 6 BUTS")
tot_all = (df['score_a'] + df['score_b']).values
print(f"  max(total)={tot_all.max()} ; P(total=6) réel={np.mean(tot_all==6):.4f}")
hi_cells = [c for c in CELLS if int(c[0]) + int(c[2]) >= 5]
print(f"  cellules hautes (somme>=5):")
for c in hi_cells:
    j = cell_idx[c]
    k = Y[:, j].sum(); E = devig[:, j].sum(); Eraw = (1/odds_grid[:, j]).sum()
    cap_pct = capped[:, j].mean()
    mo = odds_grid[:, j][~capped[:, j]]
    print(f"   {c}: obs={int(k)} expDevig={E:.1f} expRaw={Eraw:.1f} (ratio devig={k/max(E,1e-9):.3f}) "
          f"%cap={cap_pct*100:.0f}% cote_mediane_noncap={np.median(mo) if len(mo) else float('nan'):.1f}")
# cellules cappées: implied brut 1% chacune. réalité ?
k_cap = (Y * capped).sum(); n_capcells = capped.sum()
print(f"  TOUTES cellules cappées: obs={int(k_cap)} sur {int(n_capcells)} cellule-matchs "
      f"-> freq={k_cap/n_capcells:.5f} vs implied brut 0.01 (devig moy={devig[capped].mean():.5f})")
pv = stats.binomtest(int(k_cap), int(n_capcells), 0.01).pvalue
print(f"  binomial vs p=0.01: p-value={pv:.3g}")
pv2 = stats.binomtest(int(k_cap), int(n_capcells), float(devig[capped].mean())).pvalue
print(f"  binomial vs devig moyen: p-value={pv2:.3g}")

# ----------------------------------------------------------------------------
# 6. WALK-FORWARD 70/30 TEMPOREL
# ----------------------------------------------------------------------------
print("\n" + "="*100)
print("[6] WALK-FORWARD (train 70% / OOS 30% par temps)")
cut = int(n * 0.7)
tr = np.arange(n) < cut; te = ~tr
print(f"  train n={tr.sum()} ({df['expected_start'].iloc[0]} -> {df['expected_start'].iloc[cut-1]})")
print(f"  test  n={te.sum()} ({df['expected_start'].iloc[cut]} -> {df['expected_start'].iloc[n-1]})")

# S1: cellules avec ROI flat > +3% ET z>1.5 sur train -> ROI OOS
print("\n[6-S1] bet flat par cellule sélectionnée sur train:")
sel = []
for j, c in enumerate(CELLS):
    p = devig[tr, j]; k = Y[tr, j].sum(); E = p.sum(); V = (p*(1-p)).sum()
    z = (k - E)/np.sqrt(V) if V > 0 else 0
    roi_tr = (Y[tr, j] * odds_grid[tr, j]).sum()/tr.sum() - 1
    if roi_tr > 0.03 and z > 1.5:
        sel.append(c)
        roi_te = (Y[te, j] * odds_grid[te, j]).sum()/te.sum() - 1
        print(f"   {c}: train ROI={roi_tr:+.3f} z={z:+.2f} -> OOS ROI={roi_te:+.3f} (n={te.sum()})")
if not sel:
    print("   aucune cellule sélectionnée (aucun edge train)")

# S2: discordance grille vs marché Total -> bet le total favorisé par le marché gagnant (train)
print("\n[6-S2] arbitrage grille vs 'Total de buts':")
tot_odds = np.full((n, 7), np.nan)
for i, em in enumerate(ems):
    tm = em.get('Total de buts')
    if not tm: continue
    for t_ in range(7):
        v = tm.get(str(t_))
        if v and v > 1.0: tot_odds[i, t_] = v
g_tot_all = np.column_stack([devig[:, js].sum(1) for js in tot_cells])
y_tot_all = (df['score_a'] + df['score_b']).values
ok = ~np.isnan(tot_odds).any(axis=1)
# quel marché est meilleur sur train ?
trm = tr & ok
tm_dv_all = np.where(ok[:, None], (1/tot_odds) / np.nansum(1/tot_odds, axis=1, keepdims=True), np.nan)
ll_g_tr = -np.log(np.clip(g_tot_all[trm, :][np.arange(trm.sum()), y_tot_all[trm]], 1e-12, 1)).mean()
ll_m_tr = -np.log(np.clip(tm_dv_all[trm, :][np.arange(trm.sum()), y_tot_all[trm]], 1e-12, 1)).mean()
best_is_grid = ll_g_tr < ll_m_tr
print(f"   train: log-loss grille={ll_g_tr:.5f} vs marchéTot={ll_m_tr:.5f} -> best={'grille' if best_is_grid else 'marchéTot'}")
best_p = g_tot_all if best_is_grid else tm_dv_all
for thr in [0.02, 0.05, 0.10]:
    ev = best_p * tot_odds - 1.0     # EV de parier le total t au marché Total
    pick = (ev > thr) & ok[:, None] & te[:, None]
    npick = pick.sum()
    if npick == 0:
        print(f"   thr={thr}: 0 pari OOS"); continue
    ret = 0.0
    ii, jj = np.where(pick)
    for i_, t_ in zip(ii, jj):
        ret += tot_odds[i_, t_] if y_tot_all[i_] == t_ else 0.0
    print(f"   thr={thr}: n_OOS={npick} ROI={(ret-npick)/npick:+.4f} cote_moy={tot_odds[pick].mean():.2f}")

# S3: pareil avec le 1X2 vs grille
print("\n[6-S3] arbitrage grille->1X2 vs marché 1X2 (bet au 1X2):")
odds3 = df[['odds_home', 'odds_draw', 'odds_away']].values
ll_g_tr = -np.log(np.clip(g1x2[tr, :][np.arange(tr.sum()), out3[tr]], 1e-12, 1)).mean()
ll_m_tr = -np.log(np.clip(p1x2[tr, :][np.arange(tr.sum()), out3[tr]], 1e-12, 1)).mean()
best_is_grid = ll_g_tr < ll_m_tr
print(f"   train: log-loss grille={ll_g_tr:.5f} vs 1X2={ll_m_tr:.5f} -> best={'grille' if best_is_grid else '1X2'}")
best_p3 = g1x2 if best_is_grid else p1x2
for thr in [0.01, 0.03, 0.05]:
    ev = best_p3 * odds3 - 1.0
    pick = (ev > thr) & te[:, None]
    npick = pick.sum()
    if npick == 0:
        print(f"   thr={thr}: 0 pari OOS"); continue
    ii, jj = np.where(pick)
    ret = sum(odds3[i_, k_] for i_, k_ in zip(ii, jj) if out3[i_] == k_)
    print(f"   thr={thr}: n_OOS={npick} ROI={(ret-npick)/npick:+.4f} cote_moy={odds3[pick].mean():.2f}")

print("\nDONE")
