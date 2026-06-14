# -*- coding: utf-8 -*-
"""WF2 - Force attaque/defense basee sur les BUTS : ajoute-t-elle au-dela des cotes ?

Tests (walk-forward strict, train=70% temporel, OOS=30%):
  1. GF/GA rolling intra-saison (3/5/10) + all-time home/away splits -> logistic cote vs cote+goals
  2. Niveau all-time par equipe (WR home, GF home) : stabilite inter-periodes + value vs cote
  3. Minute des buts (finisseurs 75+ / starters <15) : stabilite + predictif Over2.5 / BTTS au-dela du marche
  4. Clean sheets home vs marche 'G/NG equipe exterieur' (Non)

Aucun leakage : toutes les features avant-match n'utilisent que les matchs STRICTEMENT anterieurs
(groupes par expected_start identique -> features calculees pour tout le groupe avant update).
"""
import sys, json, math
from collections import Counter, defaultdict
from datetime import datetime
sys.path.insert(0, '.')
import numpy as np
from scipy.optimize import minimize
from scraper.config import load_settings
from sqlalchemy import create_engine, text

SEP = "=" * 78
def parse_dt(s):
    return datetime.fromisoformat(str(s).replace('Z', ''))

# ============================================================ 0. LOAD DATASET
eng = create_engine(load_settings().db_url)
with eng.connect() as c:
    rows = c.execute(text("""
        select e.id, cast(e.round_info as int) rd, e.team_a, e.team_b, e.expected_start,
               r.score_a, r.score_b, r.goals_json,
               os.odds_home, os.odds_draw, os.odds_away, os.extra_markets
        from events e
        join results r on r.event_id = e.id
        join (select event_id, min(id) mid from odds_snapshots group by event_id) f on f.event_id = e.id
        join odds_snapshots os on os.id = f.mid
        where cast(e.round_info as int) >= 1
        order by e.expected_start, e.id
    """)).fetchall()

# dedup (team_a, team_b, expected_start) en gardant min id (tous ont un result ici)
seen = {}
for r in rows:
    k = (r[2], r[3], str(r[4]))
    if k not in seen:
        seen[k] = r
rows = sorted(seen.values(), key=lambda r: (str(r[4]), r[0]))

matches = []
for r in rows:
    oh, od, oa = r[8], r[9], r[10]
    if not oh or not od or not oa or oh <= 1.0 or od <= 1.0 or oa <= 1.0:
        continue
    em = r[11]
    if isinstance(em, str):
        try: em = json.loads(em)
        except Exception: em = None
    gj = r[7]
    if isinstance(gj, str):
        try: gj = json.loads(gj)
        except Exception: gj = None
    inv = 1/oh + 1/od + 1/oa
    m = {
        'id': r[0], 'rd': r[1], 'home': r[2], 'away': r[3], 't': parse_dt(r[4]),
        'sa': int(r[5]), 'sb': int(r[6]), 'goals': gj if isinstance(gj, list) else [],
        'oh': oh, 'od': od, 'oa': oa,
        'ph': (1/oh)/inv, 'pd': (1/od)/inv, 'pa': (1/oa)/inv,
        'em': em if isinstance(em, dict) else {},
    }
    m['y'] = 0 if m['sa'] > m['sb'] else (2 if m['sa'] < m['sb'] else 1)
    matches.append(m)

print(SEP); print(f"DATASET: {len(matches)} matchs avec result + cotes ouverture (dedup)")
print(f"span: {matches[0]['t']} -> {matches[-1]['t']}")

# saisons robustes : split quand round redescend de >=5 ou gap > 45 min
sid = 0; last_rd = None; last_t = None
for m in matches:
    if last_rd is not None and (m['rd'] < last_rd - 4 or (m['t'] - last_t).total_seconds() > 45*60):
        sid += 1
        last_rd = m['rd']
    else:
        last_rd = m['rd'] if last_rd is None else max(last_rd, m['rd'])
    last_t = m['t']
    m['sid'] = sid
print(f"saisons reconstruites: {sid+1}")

# ============================================================ FEATURE ENGINE (no leakage)
# etat par equipe
season_hist = defaultdict(list)        # (sid, team) -> [(gf, ga)] chrono intra-saison
home_hist = defaultdict(list)          # team -> [(gf, ga)] all-time a domicile
away_hist = defaultdict(list)          # team -> [(gf, ga)] all-time a l'exterieur
goal_minutes = defaultdict(lambda: [0, 0, 0])  # team -> [early(<=15), late(>=75), total] buts marques all-time

def roll_mean(lst, w, idx):
    sub = [x[idx] for x in lst[-w:]]
    return sum(sub)/len(sub) if sub else None

def compute_features(m):
    f = {}
    h, a, s = m['home'], m['away'], m['sid']
    sh, sa_ = season_hist[(s, h)], season_hist[(s, a)]
    f['n_season_h'], f['n_season_a'] = len(sh), len(sa_)
    for w in (3, 5, 10):
        f[f'gf{w}_h'] = roll_mean(sh, w, 0); f[f'ga{w}_h'] = roll_mean(sh, w, 1)
        f[f'gf{w}_a'] = roll_mean(sa_, w, 0); f[f'ga{w}_a'] = roll_mean(sa_, w, 1)
    hh, aa = home_hist[h], away_hist[a]
    f['n_home_h'], f['n_away_a'] = len(hh), len(aa)
    if hh:
        f['gf_home_h'] = sum(x[0] for x in hh)/len(hh)
        f['ga_home_h'] = sum(x[1] for x in hh)/len(hh)
        f['wr_home_h'] = sum(1 for x in hh if x[0] > x[1])/len(hh)
        f['cs_home_h'] = sum(1 for x in hh if x[1] == 0)/len(hh)
    if aa:
        f['gf_away_a'] = sum(x[0] for x in aa)/len(aa)
        f['ga_away_a'] = sum(x[1] for x in aa)/len(aa)
        f['wr_away_a'] = sum(1 for x in aa if x[0] > x[1])/len(aa)
    gm_h, gm_a = goal_minutes[h], goal_minutes[a]
    f['ngoals_h'], f['ngoals_a'] = gm_h[2], gm_a[2]
    f['early_h'] = gm_h[0]/gm_h[2] if gm_h[2] else None
    f['late_h'] = gm_h[1]/gm_h[2] if gm_h[2] else None
    f['early_a'] = gm_a[0]/gm_a[2] if gm_a[2] else None
    f['late_a'] = gm_a[1]/gm_a[2] if gm_a[2] else None
    return f

def update_state(m):
    h, a, s = m['home'], m['away'], m['sid']
    season_hist[(s, h)].append((m['sa'], m['sb']))
    season_hist[(s, a)].append((m['sb'], m['sa']))
    home_hist[h].append((m['sa'], m['sb']))
    away_hist[a].append((m['sb'], m['sa']))
    for g in m['goals']:
        team = h if g.get('team') == 'Home' else a
        mn = g.get('minute')
        if mn is None: continue
        gm = goal_minutes[team]
        gm[2] += 1
        if mn <= 15: gm[0] += 1
        if mn >= 75: gm[1] += 1

# passe chronologique, groupes par expected_start identique
i = 0
while i < len(matches):
    j = i
    while j < len(matches) and matches[j]['t'] == matches[i]['t']:
        j += 1
    grp = matches[i:j]
    for m in grp:
        m['f'] = compute_features(m)
    for m in grp:
        update_state(m)
    i = j

# ============================================================ SPLIT 70/30 temporel
n = len(matches)
cut = int(n * 0.70)
t_cut = matches[cut]['t']
TRAIN = matches[:cut]; OOS = matches[cut:]
print(f"split: train={len(TRAIN)} (jusqu'a {t_cut}), OOS={len(OOS)}")

# ============================================================ HELPERS modeles
def softmax_fit(X, y, K, l2=1e-3):
    n_, d = X.shape
    Y = np.eye(K)[y]
    def obj(w):
        W = w.reshape(K, d)
        Z = X @ W.T
        Z = Z - Z.max(axis=1, keepdims=True)
        E = np.exp(Z); S = E.sum(axis=1)
        P = E / S[:, None]
        ll = np.log(P[np.arange(n_), y] + 1e-300)
        loss = -ll.mean() + l2 * (W**2).sum()
        G = (P - Y).T @ X / n_ + 2*l2*W
        return loss, G.ravel()
    res = minimize(obj, np.zeros(K*d), jac=True, method='L-BFGS-B', options={'maxiter': 1000})
    return res.x.reshape(K, d)

def softmax_pred(W, X):
    Z = X @ W.T
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)

def standardize(Xtr, Xte):
    mu = Xtr.mean(axis=0); sd = Xtr.std(axis=0); sd[sd < 1e-9] = 1.0
    return (Xtr - mu)/sd, (Xte - mu)/sd

def add_icept(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])

def eval_1x2(P, ms):
    y = np.array([m['y'] for m in ms])
    pick = P.argmax(axis=1)
    acc = (pick == y).mean()
    ll = -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1)).mean()
    odds = np.array([[m['oh'], m['od'], m['oa']] for m in ms])
    ret = np.where(pick == y, odds[np.arange(len(y)), pick], 0.0)
    roi = (ret.sum() - len(y)) / len(y)
    avg_cote = odds[np.arange(len(y)), pick].mean()
    return acc, ll, roi, avg_cote

def run_1x2_compare(name, feat_fn_extra, subset_filter, min_train=500):
    """compare logistic cote-only vs cote+extra sur le MEME sous-ensemble."""
    base_feats = lambda m: [math.log(m['ph']), math.log(m['pd']), math.log(m['pa'])]
    tr = [m for m in TRAIN if subset_filter(m)]
    te = [m for m in OOS if subset_filter(m)]
    if len(tr) < min_train or len(te) < 40:
        print(f"[{name}] subset trop petit: train={len(tr)} oos={len(te)}"); return None
    ytr = np.array([m['y'] for m in tr])
    Xtr_b = np.array([base_feats(m) for m in tr]); Xte_b = np.array([base_feats(m) for m in te])
    Xtr_v = np.array([base_feats(m) + feat_fn_extra(m) for m in tr])
    Xte_v = np.array([base_feats(m) + feat_fn_extra(m) for m in te])
    A, B = standardize(Xtr_b, Xte_b); A, B = add_icept(A), add_icept(B)
    Wb = softmax_fit(A, ytr, 3); Pb = softmax_pred(Wb, B)
    A2, B2 = standardize(Xtr_v, Xte_v); A2, B2 = add_icept(A2), add_icept(B2)
    Wv = softmax_fit(A2, ytr, 3); Pv = softmax_pred(Wv, B2)
    acc_b, ll_b, roi_b, _ = eval_1x2(Pb, te)
    acc_v, ll_v, roi_v, cote_v = eval_1x2(Pv, te)
    # favori devig brut sur le meme subset
    Pd = np.array([[m['ph'], m['pd'], m['pa']] for m in te])
    acc_f, ll_f, roi_f, _ = eval_1x2(Pd, te)
    print(f"\n[{name}] n_train={len(tr)} n_oos={len(te)}")
    print(f"  favori devig    : acc={acc_f:.4f} logloss={ll_f:.4f} roi={roi_f:+.4f}")
    print(f"  logit cote-only : acc={acc_b:.4f} logloss={ll_b:.4f} roi={roi_b:+.4f}")
    print(f"  logit cote+{name}: acc={acc_v:.4f} logloss={ll_v:.4f} roi={roi_v:+.4f}")
    print(f"  DELTA vs cote-only: acc={acc_v-acc_b:+.4f} logloss={ll_v-ll_b:+.4f} roi={roi_v-roi_b:+.4f}")
    return dict(n_oos=len(te), acc_b=acc_b, ll_b=ll_b, roi_b=roi_b,
                acc_v=acc_v, ll_v=ll_v, roi_v=roi_v, acc_f=acc_f, avg_cote=cote_v)

# ============================================================ 1. BASELINE GLOBALE
print(); print(SEP); print("1. BASELINE COTE-ONLY (split global)"); print(SEP)
Pd_oos = np.array([[m['ph'], m['pd'], m['pa']] for m in OOS])
acc_fav, ll_fav, roi_fav, cote_fav = eval_1x2(Pd_oos, OOS)
print(f"favori devig (OOS n={len(OOS)}): acc={acc_fav:.4f} logloss={ll_fav:.4f} roi={roi_fav:+.4f} avg_cote={cote_fav:.3f}")
base_feats = lambda m: [math.log(m['ph']), math.log(m['pd']), math.log(m['pa'])]
ytr_all = np.array([m['y'] for m in TRAIN])
Xtr = np.array([base_feats(m) for m in TRAIN]); Xte = np.array([base_feats(m) for m in OOS])
A, B = standardize(Xtr, Xte); A, B = add_icept(A), add_icept(B)
W0 = softmax_fit(A, ytr_all, 3)
P0 = softmax_pred(W0, B)
acc0, ll0, roi0, _ = eval_1x2(P0, OOS)
print(f"logistic cote-only (OOS n={len(OOS)}): acc={acc0:.4f} logloss={ll0:.4f} roi={roi0:+.4f}")

# ============================================================ 2. TEST 1 : GF/GA rolling + all-time
print(); print(SEP); print("2. TEST 1 : GF/GA ROLLING (3/5/10) ET ALL-TIME HOME/AWAY"); print(SEP)

def has_roll(m):
    return m['f']['n_season_h'] >= 3 and m['f']['n_season_a'] >= 3
def roll_feats(m):
    f = m['f']; out = []
    for w in (3, 5, 10):
        out += [f[f'gf{w}_h'], f[f'ga{w}_h'], f[f'gf{w}_a'], f[f'ga{w}_a']]
    return out
res_roll = run_1x2_compare('rollingGFGA', roll_feats, has_roll)

def has_at(m):
    f = m['f']; return f['n_home_h'] >= 10 and f['n_away_a'] >= 10
def at_feats(m):
    f = m['f']
    egd = (f['gf_home_h'] + f['ga_away_a'])/2 - (f['gf_away_a'] + f['ga_home_h'])/2
    return [f['gf_home_h'], f['ga_home_h'], f['gf_away_a'], f['ga_away_a'], egd,
            f['wr_home_h'], f['wr_away_a']]
res_at = run_1x2_compare('alltimeHA', at_feats, has_at)

def has_both(m): return has_roll(m) and has_at(m)
def both_feats(m): return roll_feats(m) + at_feats(m)
res_both = run_1x2_compare('roll+alltime', both_feats, has_both)

# ============================================================ 3. TEST 2 : NIVEAU ALL-TIME, STABILITE + VALUE
print(); print(SEP); print("3. TEST 2 : NIVEAU ALL-TIME PAR EQUIPE - STABILITE + VALUE"); print(SEP)
# stabilite inter-periodes : moitie 1 vs moitie 2 (toutes donnees)
half_t = matches[n//2]['t']
stats_h1 = defaultdict(lambda: defaultdict(list))
stats_h2 = defaultdict(lambda: defaultdict(list))
for m in matches:
    tgt = stats_h1 if m['t'] < half_t else stats_h2
    tgt[m['home']]['home_win'].append(1 if m['y'] == 0 else 0)
    tgt[m['home']]['gf_home'].append(m['sa'])
    tgt[m['home']]['cs_home'].append(1 if m['sb'] == 0 else 0)
    tgt[m['away']]['away_win'].append(1 if m['y'] == 2 else 0)
    for g in m['goals']:
        team = m['home'] if g.get('team') == 'Home' else m['away']
        mn = g.get('minute')
        if mn is None: continue
        tgt[team]['late'].append(1 if mn >= 75 else 0)
        tgt[team]['early'].append(1 if mn <= 15 else 0)

teams = sorted(set(list(stats_h1.keys()) + list(stats_h2.keys())))
def corr_metric(key, min_n=30):
    xs, ys = [], []
    for t in teams:
        a, b = stats_h1[t].get(key, []), stats_h2[t].get(key, [])
        if len(a) >= min_n and len(b) >= min_n:
            xs.append(sum(a)/len(a)); ys.append(sum(b)/len(b))
    if len(xs) < 5: return None, len(xs)
    r = np.corrcoef(xs, ys)[0, 1]
    return r, len(xs)
print("correlation inter-moities (par equipe, n_team utilisees):")
for key, label in [('home_win', 'WR home'), ('gf_home', 'GF home moy'), ('cs_home', 'CS home'),
                   ('away_win', 'WR away'), ('late', 'part buts 75+'), ('early', 'part buts <=15')]:
    r, k = corr_metric(key)
    print(f"  {label:<16}: r={r if r is None else round(float(r),3)} (n_teams={k})")
# dispersion des WR home (h2)
wrs = [(t, sum(stats_h2[t]['home_win'])/len(stats_h2[t]['home_win']), len(stats_h2[t]['home_win']))
       for t in teams if len(stats_h2[t].get('home_win', [])) >= 30]
wrs.sort(key=lambda x: -x[1])
print("\nWR home par equipe (moitie 2): top3", [(t, round(w,3), k) for t, w, k in wrs[:3]],
      "bottom3", [(t, round(w,3), k) for t, w, k in wrs[-3:]])

# value : edge = wr_home_alltime - p_h devig ; bet home si edge >= thr (OOS, expanding pre-match)
print("\n--- value 'cote anormalement haute vs niveau all-time home' (OOS) ---")
for thr in (0.00, 0.05, 0.10, 0.15):
    sel = [m for m in OOS if m['f']['n_home_h'] >= 20 and (m['f']['wr_home_h'] - m['ph']) >= thr]
    if len(sel) < 40:
        print(f"thr={thr:.2f}: n={len(sel)} (<40, skip)"); continue
    wins = sum(1 for m in sel if m['y'] == 0)
    ret = sum(m['oh'] for m in sel if m['y'] == 0)
    roi = (ret - len(sel))/len(sel)
    avg_c = sum(m['oh'] for m in sel)/len(sel)
    # baseline cote-only sur le MEME subset : suivre le favori devig
    Pd = np.array([[m['ph'], m['pd'], m['pa']] for m in sel])
    accf, llf, roif, _ = eval_1x2(Pd, sel)
    print(f"thr={thr:.2f}: n={len(sel)} bet-home WR={wins/len(sel):.4f} ROI={roi:+.4f} avg_cote={avg_c:.3f} "
          f"| baseline favori-devig meme subset: acc={accf:.4f} ROI={roif:+.4f}")

# ============================================================ 4. TEST 3 : MINUTE DES BUTS -> O2.5 / BTTS
print(); print(SEP); print("4. TEST 3 : MINUTE DES BUTS (finisseurs/starters) -> OVER2.5 / BTTS"); print(SEP)
# distribution minutes (info)
mins = [g.get('minute') for m in matches for g in m['goals'] if g.get('minute') is not None]
mins = np.array(mins)
print(f"buts avec minute: {len(mins)}, min={mins.min()}, max={mins.max()}, "
      f"part <=15: {(mins<=15).mean():.3f}, part >=75: {(mins>=75).mean():.3f}")

def devig_binary(o_yes, o_no):
    iy, in_ = 1/o_yes, 1/o_no
    return iy/(iy+in_)

def market_over25(m):
    tdb = m['em'].get('Total de buts')
    if not tdb: return None
    try:
        inv = {k: 1/float(v) for k, v in tdb.items() if float(v) > 1.0}
    except Exception:
        return None
    s = sum(inv.values())
    if s <= 0: return None
    p_over = sum(v for k, v in inv.items() if k.isdigit() and int(k) >= 3)/s
    return p_over

def market_btts(m):
    g = m['em'].get('G/NG')
    if not g or 'Oui' not in g or 'Non' not in g: return None
    try: return devig_binary(float(g['Oui']), float(g['Non']))
    except Exception: return None

def logit(p): return math.log(max(min(p, 1-1e-9), 1e-9)/(1-max(min(p, 1-1e-9), 1e-9)))

def binlog_fit(X, y, l2=1e-3):
    Y = np.array(y)
    def obj(w):
        z = X @ w
        p = 1/(1+np.exp(-z))
        ll = -(Y*np.log(p+1e-300) + (1-Y)*np.log(1-p+1e-300)).mean() + l2*(w@w)
        g = X.T @ (p - Y)/len(Y) + 2*l2*w
        return ll, g
    res = minimize(obj, np.zeros(X.shape[1]), jac=True, method='L-BFGS-B', options={'maxiter': 1000})
    return res.x

def run_binary_compare(name, target_fn, mkt_fn, extra_fn, subset_filter, odds_fn=None):
    tr = [m for m in TRAIN if subset_filter(m) and mkt_fn(m) is not None]
    te = [m for m in OOS if subset_filter(m) and mkt_fn(m) is not None]
    if len(tr) < 500 or len(te) < 40:
        print(f"[{name}] subset trop petit train={len(tr)} oos={len(te)}"); return None
    ytr = np.array([target_fn(m) for m in tr]); yte = np.array([target_fn(m) for m in te])
    Xtr_b = np.array([[logit(mkt_fn(m))] for m in tr]); Xte_b = np.array([[logit(mkt_fn(m))] for m in te])
    Xtr_v = np.array([[logit(mkt_fn(m))] + extra_fn(m) for m in tr])
    Xte_v = np.array([[logit(mkt_fn(m))] + extra_fn(m) for m in te])
    A, B = standardize(Xtr_b, Xte_b); A, B = add_icept(A), add_icept(B)
    wb = binlog_fit(A, ytr); pb = 1/(1+np.exp(-(B @ wb)))
    A2, B2 = standardize(Xtr_v, Xte_v); A2, B2 = add_icept(A2), add_icept(B2)
    wv = binlog_fit(A2, ytr); pv = 1/(1+np.exp(-(B2 @ wv)))
    def ev(p):
        acc = ((p > 0.5).astype(int) == yte).mean()
        ll = -(yte*np.log(p+1e-12) + (1-yte)*np.log(1-p+1e-12)).mean()
        return acc, ll
    acc_b, ll_b = ev(pb); acc_v, ll_v = ev(pv)
    # baseline marche brut (proba devig sans fit)
    pm = np.array([mkt_fn(m) for m in te])
    acc_m, ll_m = ev(pm)
    out = dict(n_oos=len(te), acc_b=acc_b, ll_b=ll_b, acc_v=acc_v, ll_v=ll_v, acc_m=acc_m, ll_m=ll_m)
    print(f"\n[{name}] n_train={len(tr)} n_oos={len(te)} taux_cible_oos={yte.mean():.3f}")
    print(f"  marche devig brut : acc={acc_m:.4f} logloss={ll_m:.4f}")
    print(f"  logit marche-only : acc={acc_b:.4f} logloss={ll_b:.4f}")
    print(f"  logit marche+team : acc={acc_v:.4f} logloss={ll_v:.4f}")
    print(f"  DELTA vs marche-only: acc={acc_v-acc_b:+.4f} logloss={ll_v-ll_b:+.4f}")
    if odds_fn is not None:
        # ROI : parier le cote choisi par le modele a sa cote
        roi_rows = [(m, odds_fn(m)) for m in te]
        def roi_of(p):
            tot, stake = 0.0, 0
            cotes = []
            for (m, oo), pi, yi in zip(roi_rows, p, yte):
                if oo is None: continue
                o_yes, o_no = oo
                pick_yes = pi > 0.5
                stake += 1
                cotes.append(o_yes if pick_yes else o_no)
                if pick_yes and yi == 1: tot += o_yes
                if (not pick_yes) and yi == 0: tot += o_no
            return (tot - stake)/stake if stake else None, (sum(cotes)/len(cotes) if cotes else None)
        roi_b, _ = roi_of(pb); roi_v, cote_v = roi_of(pv)
        print(f"  ROI marche-only={roi_b:+.4f}  ROI marche+team={roi_v:+.4f}  delta={roi_v-roi_b:+.4f}")
        out.update(roi_b=roi_b, roi_v=roi_v, avg_cote=cote_v)
    return out

def has_minutes(m):
    f = m['f']; return f['ngoals_h'] >= 30 and f['ngoals_a'] >= 30
def minute_feats(m):
    f = m['f']
    return [f['early_h'], f['late_h'], f['early_a'], f['late_a']]

res_o25 = run_binary_compare('Over2.5+minutes', lambda m: 1 if m['sa']+m['sb'] >= 3 else 0,
                             market_over25, minute_feats, has_minutes)
def btts_odds(m):
    g = m['em'].get('G/NG')
    if not g: return None
    try:
        oy, on = float(g['Oui']), float(g['Non'])
        if oy <= 1.0 or on <= 1.0: return None
        return (oy, on)
    except Exception: return None
res_btts = run_binary_compare('BTTS+minutes', lambda m: 1 if (m['sa'] > 0 and m['sb'] > 0) else 0,
                              market_btts, minute_feats, has_minutes, odds_fn=btts_odds)

# ============================================================ 5. TEST 4 : CLEAN SHEETS vs G/NG EXTERIEUR
print(); print(SEP); print("5. TEST 4 : CLEAN SHEET HOME vs MARCHE 'G/NG equipe exterieur' (Non)"); print(SEP)
def gng_ext(m):
    g = m['em'].get('G/NG equipe extérieur')
    if not g: return None
    try:
        oy, on = float(g.get('Oui', 0)), float(g.get('Non', 0))
        if oy <= 1.0 or on <= 1.0: return None
        return (oy, on)
    except Exception: return None

def p_non_mkt(m):
    oo = gng_ext(m)
    if oo is None: return None
    return devig_binary(oo[1], oo[0])  # p(Non) = p(away ne marque pas)

def has_cs(m):
    return m['f']['n_home_h'] >= 20 and p_non_mkt(m) is not None

# logistic : marche-only vs marche + cs_rate_home + ga_away (l'attaque adverse)
def cs_feats(m):
    f = m['f']
    return [f['cs_home_h'], f['ga_home_h'], (f['gf_away_a'] if f['n_away_a'] >= 10 else 1.2)]
res_cs = run_binary_compare('CSnon+team', lambda m: 1 if m['sb'] == 0 else 0,
                            p_non_mkt, cs_feats, has_cs,
                            odds_fn=lambda m: (gng_ext(m)[1], gng_ext(m)[0]))  # (o_yes=Non, o_no=Oui)

# strategie edge : bet 'Non' quand cs_rate_emp - p_non_mkt >= thr (OOS)
print("\n--- strategie edge clean-sheet (OOS) ---")
all_non = [m for m in OOS if p_non_mkt(m) is not None]
ret_all = sum(gng_ext(m)[1] for m in all_non if m['sb'] == 0)
roi_all_non = (ret_all - len(all_non))/len(all_non)
print(f"baseline: bet 'Non' sur TOUS les matchs OOS: n={len(all_non)} ROI={roi_all_non:+.4f}")
for thr in (0.00, 0.03, 0.05, 0.10):
    sel = [m for m in OOS if has_cs(m) and (m['f']['cs_home_h'] - p_non_mkt(m)) >= thr]
    if len(sel) < 40:
        print(f"thr={thr:.2f}: n={len(sel)} (<40, skip)"); continue
    wins = sum(1 for m in sel if m['sb'] == 0)
    ret = sum(gng_ext(m)[1] for m in sel if m['sb'] == 0)
    roi = (ret - len(sel))/len(sel)
    avg_c = sum(gng_ext(m)[1] for m in sel)/len(sel)
    print(f"thr={thr:.2f}: n={len(sel)} WR={wins/len(sel):.4f} ROI={roi:+.4f} avg_cote_Non={avg_c:.3f}")

# value systematique par equipe : equipes dont ROI 'Non' > 0 en TRAIN -> ROI OOS
print("\n--- value 'Non' par equipe : selection en train, mesure en OOS ---")
team_train = defaultdict(lambda: [0.0, 0])
for m in TRAIN:
    oo = gng_ext(m)
    if oo is None: continue
    team_train[m['home']][1] += 1
    if m['sb'] == 0: team_train[m['home']][0] += oo[1]
sel_teams = [t for t, (ret, k) in team_train.items() if k >= 30 and (ret-k)/k > 0]
print(f"equipes ROI train>0: {len(sel_teams)} -> {sorted(sel_teams)}")
sel = [m for m in OOS if m['home'] in sel_teams and gng_ext(m) is not None]
if len(sel) >= 40:
    wins = sum(1 for m in sel if m['sb'] == 0)
    ret = sum(gng_ext(m)[1] for m in sel if m['sb'] == 0)
    roi = (ret - len(sel))/len(sel)
    avg_c = sum(gng_ext(m)[1] for m in sel)/len(sel)
    print(f"OOS sur ces equipes: n={len(sel)} WR={wins/len(sel):.4f} ROI={roi:+.4f} avg_cote={avg_c:.3f} "
          f"(baseline all-Non ROI={roi_all_non:+.4f})")
else:
    print(f"n OOS={len(sel)} insuffisant")

print("\nFIN.")
