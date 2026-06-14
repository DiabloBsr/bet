# -*- coding: utf-8 -*-
"""
WF3 — CHASSE AUX ANOMALIES : les donnees corrompues sont-elles structurees ?
1. Recensement exhaustif : HT>FT, goals_json vs FT, goals_json vs HT, placeholders,
   cotes manquantes, doublons, scores aberrants, resultats dupliques.
2. Structure : par round, par heure, par date (KS/CUSUM), par scrape_run_id -> bug feed ?
3. Impact : calibration favori devig, distribution des scores, taux BTTS, AVEC vs SANS.
4. Export exports/corrupted_events.json
"""
import sys, json, math, collections
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from scipy import stats
from scraper.config import load_settings
from sqlalchemy import create_engine

pd.set_option('display.width', 200)
eng = create_engine(load_settings().db_url)

# ---------------------------------------------------------------- load
ev = pd.read_sql("SELECT id AS event_id, round_info, team_a, team_b, expected_start, competition FROM events", eng)
res = pd.read_sql("SELECT id AS res_id, event_id, score_a, score_b, ht_score_a, ht_score_b, goals_json, scrape_run_id, finished_at FROM results", eng)
snaps = pd.read_sql("SELECT id AS snap_id, event_id, odds_home, odds_draw, odds_away, extra_markets, captured_at FROM odds_snapshots", eng)

print(f"events={len(ev)} results={len(res)} snapshots={len(snaps)}")

# ---------------------------------------------------------------- A0: doublons
# doublons de results par event_id
res_dup = res.groupby('event_id').size()
res_dup_ids = set(res_dup[res_dup > 1].index)
print(f"\n[A0a] results dupliques (>=2 lignes results pour le meme event_id): {len(res_dup_ids)} events")
if res_dup_ids:
    sub = res[res.event_id.isin(list(res_dup_ids)[:5])].sort_values('event_id')
    print(sub[['event_id','score_a','score_b','ht_score_a','ht_score_b','scrape_run_id']].head(10))
    # les scores divergent-ils entre doublons ?
    div = 0
    for eid, g in res[res.event_id.isin(res_dup_ids)].groupby('event_id'):
        if g[['score_a','score_b']].drop_duplicates().shape[0] > 1:
            div += 1
    print(f"      dont scores DIVERGENTS entre doublons: {div}")

# garder la derniere ligne result par event
res1 = res.sort_values('res_id').drop_duplicates('event_id', keep='last')

# doublons d'events (team_a, team_b, expected_start)
ev_fin = ev.merge(res1, on='event_id', how='inner')
dup_groups = ev_fin.groupby(['team_a','team_b','expected_start'])['event_id'].apply(list)
dup_groups = dup_groups[dup_groups.str.len() > 1]
dup_extra_ids = set()
dup_divergent = 0
for ids in dup_groups:
    ids = sorted(ids)
    sc = ev_fin[ev_fin.event_id.isin(ids)][['score_a','score_b']].drop_duplicates()
    if len(sc) > 1:
        dup_divergent += 1
        dup_extra_ids.update(ids)         # divergents: tous corrompus (on ne sait pas lequel croire)
    else:
        dup_extra_ids.update(ids[1:])     # copies identiques: on garde la 1ere
print(f"[A0b] doublons (team_a,team_b,expected_start) finis: {len(dup_groups)} groupes, {len(dup_extra_ids)} events a exclure, {dup_divergent} groupes a scores DIVERGENTS")

# dataset de travail dedup (garde 1er event_id par cle)
ev_fin = ev_fin.sort_values('event_id').drop_duplicates(['team_a','team_b','expected_start'], keep='first')
ev_fin = ev_fin[ev_fin.round_info != '0'].reset_index(drop=True)
print(f"dataset travail (fini, dedup, round!=0): {len(ev_fin)}")

# ---------------------------------------------------------------- parse goals_json
def parse_goals(g):
    if g is None: return None
    try:
        arr = json.loads(g) if isinstance(g, str) else g
    except Exception:
        return 'PARSE_ERROR'
    if arr is None: return None
    return arr

ev_fin['goals'] = ev_fin['goals_json'].apply(parse_goals)

# ---------------------------------------------------------------- A1: HT > FT
m_htft = (ev_fin.ht_score_a > ev_fin.score_a) | (ev_fin.ht_score_b > ev_fin.score_b)
m_htft = m_htft.fillna(False)
print(f"\n[A1] HT > FT (buts qui disparaissent): {int(m_htft.sum())}")
if m_htft.sum():
    print(ev_fin[m_htft][['event_id','team_a','team_b','score_a','score_b','ht_score_a','ht_score_b','expected_start','round_info']].head(15).to_string())

# ---------------------------------------------------------------- A2: goals_json vs FT
def check_ft(row):
    g = row.goals
    if not isinstance(g, list): return None
    nh = sum(1 for it in g if it.get('team') == 'Home')
    na = sum(1 for it in g if it.get('team') == 'Away')
    bad_count = (nh != row.score_a) or (na != row.score_b)
    bad_last = False
    bad_mono = False
    if g:
        last = g[-1]
        bad_last = (last.get('homeScore') != row.score_a) or (last.get('awayScore') != row.score_b)
        hs, as_ = 0, 0
        for it in g:
            h2, a2 = it.get('homeScore'), it.get('awayScore')
            if h2 is None or a2 is None or h2 < hs or a2 < as_ or (h2 + a2) != (hs + as_ + 1):
                bad_mono = True; break
            hs, as_ = h2, a2
    elif row.score_a + row.score_b > 0:
        bad_count = True
    return bad_count or bad_last or bad_mono

ev_fin['bad_ft'] = ev_fin.apply(check_ft, axis=1)
ev_fin['goals'] = ev_fin.goals.apply(lambda g: g if isinstance(g, (list, str)) else None)
n_gj_missing = int(ev_fin.goals.apply(lambda g: g is None).sum())
n_gj_parse = int((ev_fin.goals == 'PARSE_ERROR').sum())
m_ft = ev_fin.bad_ft.fillna(False).astype(bool)
print(f"\n[A2] goals_json absent/null: {n_gj_missing} ; parse error: {n_gj_parse}")
print(f"[A2] goals_json INCOHERENT avec score final (compte/dernier cumul/monotonie): {int(m_ft.sum())}")
# parmi les missing, combien ont des buts au score ?
miss_with_goals = ev_fin[ev_fin.goals.isna() & ((ev_fin.score_a + ev_fin.score_b) > 0)]
print(f"     goals_json manquant ALORS QUE score>0 (perte de timeline): {len(miss_with_goals)}")

# ---------------------------------------------------------------- A3: goals_json vs HT — calibrer le cutoff
has_both = ev_fin[ev_fin.goals.apply(lambda g: isinstance(g, list)) & ev_fin.ht_score_a.notna()]
for cut in (44, 45, 46):
    ok = 0
    for row in has_both.itertuples():
        h = sum(1 for it in row.goals if it['team'] == 'Home' and it['minute'] <= cut)
        a = sum(1 for it in row.goals if it['team'] == 'Away' and it['minute'] <= cut)
        ok += (h == row.ht_score_a and a == row.ht_score_b)
    print(f"[A3] cutoff minute<={cut}: coherence HT {ok}/{len(has_both)} = {ok/len(has_both)*100:.2f}%")

CUT = 45
def check_ht(row):
    g = row.goals
    if not isinstance(g, list) or pd.isna(row.ht_score_a): return None
    h = sum(1 for it in g if it['team'] == 'Home' and it['minute'] <= CUT)
    a = sum(1 for it in g if it['team'] == 'Away' and it['minute'] <= CUT)
    return (h != row.ht_score_a) or (a != row.ht_score_b)

ev_fin['bad_ht'] = ev_fin.apply(check_ht, axis=1)
m_ht = ev_fin.bad_ht.fillna(False).astype(bool)
n_ht_missing = int(ev_fin.ht_score_a.isna().sum())
print(f"[A3] ht_score manquant: {n_ht_missing} ; goals_json INCOHERENT avec ht_score (cut={CUT}): {int(m_ht.sum())}")

# ---------------------------------------------------------------- A3bis: MECANISME du ht_score faux
def ht_from_goals(g):
    if not isinstance(g, list): return None
    return (sum(1 for it in g if it['team'] == 'Home' and it['minute'] <= CUT),
            sum(1 for it in g if it['team'] == 'Away' and it['minute'] <= CUT))

ev_fin['ht_true'] = ev_fin.goals.apply(ht_from_goals)
bad_rows = ev_fin[(m_ht | m_htft) & ev_fin.ht_score_a.notna()].copy()
n_swap = n_borrow = n_other = 0
borrow_detail = []
# index: batch (expected_start arrondi a la minute) -> liste des HT vrais
ev_fin['batch'] = pd.to_datetime(ev_fin.expected_start).dt.floor('min')
batch_map = ev_fin.groupby('batch').apply(lambda g: list(zip(g.event_id, g.ht_true)), include_groups=False)
ht_global = ev_fin.ht_true.dropna().value_counts(normalize=True)
exp_borrow = 0.0
for row in bad_rows.itertuples():
    rec = (row.ht_score_a, row.ht_score_b)
    true = row.ht_true
    if true is not None and rec == (true[1], true[0]) and true[0] != true[1]:
        n_swap += 1; continue
    mates = [(eid, h) for eid, h in batch_map.get(pd.to_datetime(row.expected_start).floor('min'), [])
             if eid != row.event_id and h is not None]
    hit = [eid for eid, h in mates if h == rec]
    if mates:
        p_cell = float(ht_global.get(rec, 0.0))
        exp_borrow += 1 - (1 - p_cell) ** len(mates)
    if hit and (true is None or rec != true):
        n_borrow += 1; borrow_detail.append((row.event_id, hit[:2]))
    else:
        n_other += 1
print(f"[A3bis] mecanisme ht_score faux (n={len(bad_rows)}): SWAP home/away={n_swap}, "
      f"== HT d'un match simultane={n_borrow} (attendu par HASARD: {exp_borrow:.1f}), autre={n_other}")
if borrow_detail[:5]:
    print(f"        exemples empruntes (event -> donneurs candidats): {borrow_detail[:5]}")
# le HT vrai (goals_json) est-il fiable quand FT colle ? -> ht reconstructible
n_fixable = int(((m_ht | m_htft) & ~m_ft & ev_fin.ht_true.notna()).sum())
print(f"[A3bis] HT faux mais goals_json coherent avec FT (HT RECONSTRUCTIBLE): {n_fixable}/{len(bad_rows)}")

# ---------------------------------------------------------------- A4: cotes — placeholders / manquantes / overround
op = snaps.sort_values('snap_id').drop_duplicates('event_id', keep='first')  # ouverture
ev_fin = ev_fin.merge(op[['event_id','odds_home','odds_draw','odds_away','extra_markets','captured_at']], on='event_id', how='left')
m_noodds = ev_fin.odds_home.isna()
print(f"\n[A4] events finis SANS aucun snapshot de cotes: {int(m_noodds.sum())}")

PLACEHOLDERS = (1.01, 100.0, 1000.0)
m_ph_1x2 = ev_fin[['odds_home','odds_draw','odds_away']].isin(PLACEHOLDERS).any(axis=1)
print(f"[A4] cotes 1X2 placeholder (1.01/100/1000): {int(m_ph_1x2.fillna(False).sum())}")

inv = 1/ev_fin.odds_home + 1/ev_fin.odds_draw + 1/ev_fin.odds_away
m_overround = ((inv < 1.0) | (inv > 1.30)) & inv.notna()
print(f"[A4] overround 1X2 aberrant (<1.00 ou >1.30): {int(m_overround.sum())}  [range observe: {inv.min():.4f} - {inv.max():.4f}, mediane {inv.median():.4f}]")

# extra markets : caps de cotes (100.0 plafond / 1.01 plancher) — discovery, PAS corruption
all_vals = collections.Counter()
parse_fail = 0
for xm in ev_fin.extra_markets.dropna().sample(min(3000, int(ev_fin.extra_markets.notna().sum())), random_state=7):
    try:
        d = json.loads(xm) if isinstance(xm, str) else xm
    except Exception:
        parse_fail += 1; continue
    def walk(o):
        if isinstance(o, dict):
            for v in o.values(): walk(v)
        elif isinstance(o, list):
            for v in o: walk(v)
        elif isinstance(o, (int, float)) and not isinstance(o, bool):
            all_vals[round(float(o), 2)] += 1
    walk(d)
vals = np.array(sorted(all_vals))
n_xm_missing = int(ev_fin.extra_markets.isna().sum() - m_noodds.sum())
print(f"[A4] extra_markets manquant (snapshot present): {n_xm_missing} ; parse fail: {parse_fail}")
print(f"[A4] CAPS (echantillon 3000 events): min cote={vals.min()}, max cote={vals.max()}")
hi = [v for v in vals if v > 50]
print(f"     valeurs >50: {hi[:20]}  | n(=100.0)={all_vals.get(100.0,0)}  n(95<v<100)={sum(all_vals[v] for v in all_vals if 95<v<100)}")
lo = [v for v in vals if v < 1.10]
print(f"     valeurs <1.10: {lo[:20]} | n(=1.01)={all_vals.get(1.01,0)}  n(1.01<v<1.05)={sum(all_vals[v] for v in all_vals if 1.01<v<1.05)}")
m_xm_ph = pd.Series(False, index=ev_fin.index)  # caps = prix legitimes clippes, pas corruption

# ---------------------------------------------------------------- A5: scores aberrants
tot = ev_fin.score_a + ev_fin.score_b
sc_max = pd.read_sql("SELECT MAX(score_a) ma, MAX(score_b) mb FROM results", eng)
print(f"\n[A5] distribution totale buts: max indiv {ev_fin.score_a.max()}/{ev_fin.score_b.max()}, total max {tot.max()}")
print(ev_fin.groupby(tot).size().to_string())
m_aberr = (ev_fin.score_a > 6) | (ev_fin.score_b > 6) | (tot > 8) | (ev_fin.score_a < 0) | (ev_fin.score_b < 0)
print(f"[A5] scores aberrants (indiv>6 ou total>8 ou negatif): {int(m_aberr.sum())}")
if m_aberr.sum():
    print(ev_fin[m_aberr][['event_id','team_a','team_b','score_a','score_b','round_info','expected_start']].to_string())

# ---------------------------------------------------------------- bilan census
flags = pd.DataFrame({
    'HT_GT_FT': m_htft.values, 'GOALS_VS_FT': m_ft.values, 'GOALS_VS_HT': m_ht.values,
    'ODDS_PLACEHOLDER': m_ph_1x2.fillna(False).values, 'OVERROUND_ABERRANT': m_overround.values,
    'SCORE_ABERRANT': m_aberr.values,
    'NO_ODDS': m_noodds.values,
}, index=ev_fin.index)
ev_fin['n_flags'] = flags[['HT_GT_FT','GOALS_VS_FT','GOALS_VS_HT','ODDS_PLACEHOLDER','OVERROUND_ABERRANT','SCORE_ABERRANT']].sum(axis=1)
ev_fin['corrupt'] = ev_fin.n_flags > 0
print("\n===== BILAN CENSUS =====")
for c in flags.columns:
    print(f"  {c:20s}: {int(flags[c].sum()):5d}  ({flags[c].mean()*100:.2f}%)")
print(f"  CORROMPU HARD (hors NO_ODDS/soft): {int(ev_fin.corrupt.sum())} / {len(ev_fin)} = {ev_fin.corrupt.mean()*100:.2f}%")
print("  co-occurrence des flags (n_flags):", ev_fin[ev_fin.corrupt].n_flags.value_counts().to_dict())

# chevauchement HT_GT_FT vs GOALS_VS_HT/FT
both = int((m_htft & (m_ht | m_ft)).sum())
print(f"  HT>FT qui sont AUSSI incoherents goals_json: {both}/{int(m_htft.sum())}")

# ---------------------------------------------------------------- MECANISME GOALS_VS_FT
print("\n===== MECANISME GOALS_VS_FT =====")
bad_ft_rows = ev_fin[m_ft]
direc = collections.Counter()
internally_ok = 0
for row in bad_ft_rows.itertuples():
    g = row.goals
    if not isinstance(g, list): continue
    nh = sum(1 for it in g if it.get('team') == 'Home'); na = len(g) - nh
    direc['json>FT' if nh+na > row.score_a+row.score_b else ('json<FT' if nh+na < row.score_a+row.score_b else 'json==FT(mais repartition/ordre faux)')] += 1
    # timeline interne coherente (cumuls monotones) ?
    hs = as_ = 0; ok = True
    for it in g:
        h2, a2 = it.get('homeScore'), it.get('awayScore')
        if h2 is None or h2 < hs or a2 < as_ or (h2+a2) != (hs+as_+1): ok = False; break
        hs, as_ = h2, a2
    internally_ok += ok
print(f"direction du desaccord: {dict(direc)}")
print(f"timeline INTERNE coherente (le json est un vrai match, juste pas le bon): {internally_ok}/{len(bad_ft_rows)}")

# le timeline appartient-il a un match SIMULTANE ? (FT du donneur == dernier cumul du json)
ft_map = ev_fin.groupby('batch').apply(lambda g: list(zip(g.event_id, g.score_a, g.score_b)), include_groups=False) \
    if 'batch' in ev_fin.columns else None
n_donor = 0; n_test = 0
ft_global = ev_fin.groupby(['score_a','score_b']).size() / len(ev_fin)
exp_donor = 0.0
for row in bad_ft_rows.itertuples():
    g = row.goals
    if not isinstance(g, list) or not g: continue
    last = (g[-1].get('homeScore'), g[-1].get('awayScore'))
    if last[0] is None: continue
    mates = [m for m in ft_map.get(row.batch, []) if m[0] != row.event_id]
    if not mates: continue
    n_test += 1
    if any((sa, sb) == last for _, sa, sb in mates): n_donor += 1
    p_cell = float(ft_global.get(last, 0.0))
    exp_donor += 1 - (1 - p_cell) ** len(mates)
print(f"dernier cumul json == FT d'un match simultane: {n_donor}/{n_test} (attendu par hasard: {exp_donor:.1f})")
if n_test:
    bt = stats.binomtest(n_donor, n_test, exp_donor/n_test)
    print(f"   binomial test vs hasard: p={bt.pvalue:.4f}")

# bug au niveau du scrape_run : all-or-nothing ?
runs = ev_fin.groupby('scrape_run_id').agg(n=('event_id','size'), bad=('bad_ft', lambda s: s.fillna(False).sum()))
runs = runs[runs.n >= 5]
p_glob = m_ft.sum()/len(ev_fin)
full_bad = runs[runs.bad/runs.n >= 0.8]
exp_full = sum(stats.binom.sf(math.ceil(0.8*n)-1, n, p_glob) for n in runs.n)
print(f"runs (>=5 resultats) avec >=80% de GOALS_VS_FT: {len(full_bad)} (attendu si i.i.d.: {exp_full:.2f})")
print(f"   -> runs touches: {[(int(i), f'{int(b)}/{int(n)}') for i, (n, b) in full_bad.iterrows()][:12]}")

# delai finished_at - expected_start : resultats moissonnes trop tard ?
ev_fin['lag_h'] = (pd.to_datetime(ev_fin.finished_at) - pd.to_datetime(ev_fin.expected_start)).dt.total_seconds()/3600
lag_bad = ev_fin[m_ft].lag_h.dropna(); lag_good = ev_fin[~m_ft & ev_fin.goals.apply(lambda g: isinstance(g, list))].lag_h.dropna()
if len(lag_bad) > 10:
    u = stats.mannwhitneyu(lag_bad, lag_good)
    print(f"delai scrape (h): bad mediane={lag_bad.median():.2f} vs good mediane={lag_good.median():.2f}  Mann-Whitney p={u.pvalue:.2e}")

# ---------------------------------------------------------------- PART 2: structure
print("\n===== STRUCTURE DES ANOMALIES =====")
ev_fin['ts'] = pd.to_datetime(ev_fin.expected_start)
ev_fin['hour'] = ev_fin.ts.dt.hour
ev_fin['date'] = ev_fin.ts.dt.date
ev_fin['round_i'] = pd.to_numeric(ev_fin.round_info, errors='coerce')

def structure_test(mask, name):
    sub = ev_fin[mask]
    if len(sub) < 8:
        print(f"\n-- {name}: n={len(sub)} trop petit pour tests, dates: {sorted(set(sub.date))[:6]}")
        return
    print(f"\n-- {name} (n={len(sub)})")
    # par round (38 bins)
    obs = sub.groupby('round_i').size().reindex(range(1, 39), fill_value=0)
    exp = ev_fin.groupby('round_i').size().reindex(range(1, 39), fill_value=0) * (len(sub)/len(ev_fin))
    keep = exp > 0
    chi2, p = stats.chisquare(obs[keep], exp[keep])
    print(f"   chi2 par round: chi2={chi2:.1f} p={p:.4f}")
    # par heure
    obs_h = sub.groupby('hour').size().reindex(range(24), fill_value=0)
    exp_h = ev_fin.groupby('hour').size().reindex(range(24), fill_value=0) * (len(sub)/len(ev_fin))
    keep = exp_h > 0
    chi2h, ph = stats.chisquare(obs_h[keep], exp_h[keep])
    print(f"   chi2 par heure: chi2={chi2h:.1f} p={ph:.4f}")
    # clustering temporel: KS des timestamps anomalies vs tous
    ks, pks = stats.ks_2samp(sub.ts.astype('int64'), ev_fin.ts.astype('int64'))
    print(f"   KS temporel vs population: D={ks:.3f} p={pks:.2e}")
    # CUSUM simple sur l'indicateur ordonne par temps
    s = ev_fin.sort_values('ts')
    ind = mask.reindex(s.index).astype(float).values
    dev = np.cumsum(ind - ind.mean())
    imax = int(np.argmax(np.abs(dev)))
    print(f"   CUSUM max |dev|={np.abs(dev).max():.1f} a la date {s.iloc[imax].ts} (position {imax}/{len(s)})")
    # dates les plus touchees
    top_d = sub.groupby('date').size().sort_values(ascending=False).head(5)
    print(f"   top dates: {dict(top_d)}")
    # scrape_run_id
    if sub.scrape_run_id.notna().any():
        top_r = sub.groupby('scrape_run_id').size().sort_values(ascending=False).head(5)
        cover = ev_fin.groupby('scrape_run_id').size()
        print("   top scrape_run_id: " + ", ".join(f"run {int(k)}: {v}/{cover.get(k,0)}" for k, v in top_r.items()))

for msk, nm in [(m_htft, 'HT_GT_FT'), (m_ft, 'GOALS_VS_FT'), (m_ht, 'GOALS_VS_HT'),
                (ev_fin.ht_score_a.isna(), 'HT_MISSING'), (ev_fin.goals.isna(), 'GOALS_JSON_MISSING'),
                (m_noodds, 'NO_ODDS'), (ev_fin.corrupt, 'TOUT_CORROMPU')]:
    structure_test(msk.fillna(False).astype(bool), nm)

# ---------------------------------------------------------------- PART 3: impact
print("\n===== IMPACT SUR 3 MESURES CLES =====")
val = ev_fin[~m_noodds].copy()
val['p_fav'] = np.nan
invq = 1/val.odds_home + 1/val.odds_draw + 1/val.odds_away
val['ph_'] = (1/val.odds_home)/invq; val['pd_'] = (1/val.odds_draw)/invq; val['pa_'] = (1/val.odds_away)/invq

def calib_report(df, label):
    p_fav = df[['ph_','pd_','pa_']].max(axis=1)
    fav_side = df[['ph_','pd_','pa_']].values.argmax(axis=1)
    out = np.where(df.score_a > df.score_b, 0, np.where(df.score_a == df.score_b, 1, 2))
    win = (fav_side == out).astype(int)
    bins = pd.qcut(p_fav, 8, duplicates='drop')
    g = pd.DataFrame({'p': p_fav, 'w': win, 'b': bins}).groupby('b', observed=True).agg(n=('w','size'), p_mean=('p','mean'), w_rate=('w','mean'))
    mace = float((g.w_rate - g.p_mean).abs().mul(g.n).sum() / g.n.sum())
    # test global: binomial somme (z de Spiegelhalter simplifie)
    z = (win.sum() - p_fav.sum()) / math.sqrt((p_fav*(1-p_fav)).sum())
    print(f"\n  [{label}] n={len(df)}  favori: attendu {p_fav.mean()*100:.2f}% vs observe {win.mean()*100:.2f}%  | MACE={mace*100:.2f}pts | z={z:.2f} p={2*(1-stats.norm.cdf(abs(z))):.4f}")
    for b, r in g.iterrows():
        print(f"     bin p~{r.p_mean:.3f}: obs {r.w_rate:.3f} (n={int(r.n)})")
    return p_fav, win

print("\n--- 3a. Calibration favori (devig proportionnel) ---")
calib_report(val, 'AVEC corrompus')
calib_report(val[~val.corrupt], 'SANS corrompus')
calib_report(val[val.corrupt], 'corrompus seuls') if val.corrupt.sum() >= 30 else None

print("\n--- 3b. Distribution des scores ---")
def score_dist(df):
    s = df.apply(lambda r: (min(int(r.score_a), 5), min(int(r.score_b), 5)), axis=1)
    return s.value_counts()
d_all = score_dist(ev_fin); d_cln = score_dist(ev_fin[~ev_fin.corrupt])
cells = sorted(set(d_all.index) | set(d_cln.index))
o1 = np.array([d_all.get(c, 0) for c in cells]); o2 = np.array([d_cln.get(c, 0) for c in cells])
chi2, p, dof, _ = stats.chi2_contingency(np.vstack([o1, o2]))
print(f"  chi2 homogeneite AVEC vs SANS: chi2={chi2:.2f} dof={dof} p={p:.4f}")
top = d_all.head(8)
for c in top.index:
    pa = d_all[c]/d_all.sum(); pc = d_cln.get(c,0)/d_cln.sum()
    print(f"   {c}: AVEC {pa*100:.2f}%  SANS {pc*100:.2f}%  (delta {abs(pa-pc)*100:.3f}pts)")
print(f"  moyenne buts: AVEC {float((ev_fin.score_a+ev_fin.score_b).mean()):.4f}  SANS {float((ev_fin[~ev_fin.corrupt].score_a+ev_fin[~ev_fin.corrupt].score_b).mean()):.4f}")

print("\n--- 3c. Taux BTTS ---")
btts_all = ((ev_fin.score_a > 0) & (ev_fin.score_b > 0))
btts_cln = btts_all[~ev_fin.corrupt]
n1, n2 = len(btts_all), len(btts_cln)
p1, p2 = btts_all.mean(), btts_cln.mean()
pp = (btts_all.sum() + btts_cln.sum())/(n1+n2)
z = (p1-p2)/math.sqrt(pp*(1-pp)*(1/n1+1/n2)) if pp>0 else 0
print(f"  BTTS AVEC: {p1*100:.2f}% (n={n1})  SANS: {p2*100:.2f}% (n={n2})  z={z:.3f} p={2*(1-stats.norm.cdf(abs(z))):.4f}")
# BTTS des corrompus seuls
if ev_fin.corrupt.sum() >= 20:
    pc = btts_all[ev_fin.corrupt].mean()
    bt = stats.binomtest(int(btts_all[ev_fin.corrupt].sum()), int(ev_fin.corrupt.sum()), p2)
    print(f"  BTTS corrompus seuls: {pc*100:.2f}% (n={int(ev_fin.corrupt.sum())}) vs clean {p2*100:.2f}% -> binomial p={bt.pvalue:.4f}")

# ---------------------------------------------------------------- PART 4: export
out = {}
def add(ids, cat):
    for i in ids: out.setdefault(int(i), []).append(cat)
add(res_dup_ids, 'RESULT_ROW_DUPLICATE')
add(dup_extra_ids, 'EVENT_DUPLICATE')
add(ev_fin[m_htft].event_id, 'HT_GT_FT')
add(ev_fin[m_ft].event_id, 'GOALS_JSON_VS_FT')
add(ev_fin[m_ht].event_id, 'GOALS_JSON_VS_HT')
add(ev_fin[m_ph_1x2.fillna(False)].event_id, 'ODDS_PLACEHOLDER_1X2')
add(ev_fin[m_overround].event_id, 'OVERROUND_ABERRANT')
add(ev_fin[m_aberr].event_id, 'SCORE_ABERRANT')
payload = {
    'generated_at': pd.Timestamp.now().isoformat(),
    'definition': 'event_ids a exclure des analyses; categories par event',
    'n_corrupted': len(out),
    'categories_count': dict(collections.Counter(c for v in out.values() for c in v)),
    'events': {str(k): v for k, v in sorted(out.items())},
    'soft_incomplete': {
        'ht_missing': sorted(int(x) for x in ev_fin[ev_fin.ht_score_a.isna()].event_id),
        'goals_json_missing': sorted(int(x) for x in ev_fin[ev_fin.goals.isna()].event_id),
        'no_odds_snapshot': sorted(int(x) for x in ev_fin[m_noodds].event_id),
    },
}
with open('exports/corrupted_events.json', 'w', encoding='utf-8') as f:
    json.dump(payload, f, indent=1)
print(f"\nexports/corrupted_events.json ecrit: {len(out)} events corrompus (hard), "
      f"+ soft: ht_missing={len(payload['soft_incomplete']['ht_missing'])}, "
      f"gj_missing={len(payload['soft_incomplete']['goals_json_missing'])}, no_odds={len(payload['soft_incomplete']['no_odds_snapshot'])}")
