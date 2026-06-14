# -*- coding: utf-8 -*-
"""WF2 - H2H : l'historique de la paire (team_a vs team_b) ajoute-t-il du pouvoir
predictif AU-DELA des cotes ?

Tests:
 1. Logistic cote-only vs cote+h2h (walk-forward 70/30 temporel, expanding window, zero leakage)
 2. Desaccords H2H (n>=8, WR>=65%) vs favori devig -> WR/ROI de suivre le H2H contre le marche
 3. Stabilite du H2H par segment de saison (DS J1-3 vs FS J34-38)
 4. Score exact : top1/top3 H2H vs top1/top3 marche (Score exact) vs global train

Conventions:
 - cotes ouverture = snapshot MIN(id) par event
 - dedup events par (team_a, team_b, expected_start), on garde celui avec result
 - H2H avant-match = matchs STRICTEMENT anterieurs (expected_start), groupes par timestamp
   identique traites en bloc (features calculees avant mise a jour de l'etat)
 - split: 70% premiers matchs (temporel) = train, 30% derniers = OOS
"""
import sys, json
from collections import defaultdict, Counter
from datetime import datetime

sys.path.insert(0, '.')
import numpy as np
from scipy.optimize import minimize
from scraper.config import load_settings
from sqlalchemy import create_engine, text

SEP = "=" * 78


def parse_dt(s):
    return datetime.fromisoformat(str(s).replace('Z', ''))


# ------------------------------------------------------------------ load data
eng = create_engine(load_settings().db_url)
with eng.connect() as c:
    rows = c.execute(text(
        "select e.id, e.round_info, e.team_a, e.team_b, e.expected_start, "
        "       r.score_a, r.score_b, "
        "       o.odds_home, o.odds_draw, o.odds_away, o.extra_markets "
        "from events e "
        "left join results r on r.event_id = e.id "
        "left join (select event_id, odds_home, odds_draw, odds_away, extra_markets, "
        "           min(id) mid from odds_snapshots group by event_id) o "
        "       on o.event_id = e.id "
        "order by e.expected_start, e.id")).fetchall()

# dedup (team_a, team_b, expected_start): garder celui avec result, sinon min id
seen = {}
for r in rows:
    k = (r[2], r[3], str(r[4]))
    if k not in seen or (seen[k][5] is None and r[5] is not None):
        seen[k] = r
rows = sorted(seen.values(), key=lambda r: (str(r[4]), r[0]))

matches = []  # tous les matchs AVEC result (alimentent le H2H)
for r in rows:
    if r[5] is None:
        continue
    try:
        rd = int(r[1])
    except (TypeError, ValueError):
        rd = None
    sa, sb = int(r[5]), int(r[6])
    y = 0 if sa > sb else (1 if sa == sb else 2)
    se = None
    if r[10]:
        try:
            se = json.loads(r[10]).get("Score exact")
        except Exception:
            se = None
    matches.append(dict(
        id=r[0], rd=rd, home=r[2], away=r[3], t=parse_dt(r[4]),
        sa=sa, sb=sb, y=y,
        oh=r[7], od=r[8], oa=r[9], score_market=se))

print(f"matchs avec result (dedup): {len(matches)}")
n_odds = sum(1 for m in matches if m['oh'] and m['od'] and m['oa'])
print(f"  dont avec cotes ouverture 1X2: {n_odds}")

# ------------------------------------------------------------------ H2H expanding window
# etat par paire ORIENTEE (home, away) et NON-ORIENTEE frozenset
ori = defaultdict(lambda: dict(n=0, hw=0, dr=0, aw=0, scores=Counter(), seg=defaultdict(lambda: [0, 0, 0])))
uno = defaultdict(lambda: dict(n=0, dr=0, wins=Counter()))


def seg_of(rd):
    if rd is None:
        return None
    if rd <= 3:
        return 'DS'
    if rd >= 34:
        return 'FS'
    return 'MS'


# traiter par groupe de timestamp identique (pas de leakage intra-groupe)
i = 0
while i < len(matches):
    j = i
    while j < len(matches) and matches[j]['t'] == matches[i]['t']:
        j += 1
    grp = matches[i:j]
    for m in grp:  # 1) features avant-match
        ko = (m['home'], m['away'])
        ku = frozenset(ko)
        so, su = ori[ko], uno[ku]
        m['h2h_n'] = so['n']
        m['h2h_hw'] = so['hw']; m['h2h_dr'] = so['dr']; m['h2h_aw'] = so['aw']
        m['h2h_scores'] = dict(so['scores'])
        m['h2h_seg'] = {k: list(v) for k, v in so['seg'].items()}
        m['u_n'] = su['n']
        m['u_home_w'] = su['wins'][m['home']]
        m['u_away_w'] = su['wins'][m['away']]
        m['u_dr'] = su['dr']
    for m in grp:  # 2) mise a jour de l'etat
        ko = (m['home'], m['away'])
        ku = frozenset(ko)
        so, su = ori[ko], uno[ku]
        so['n'] += 1
        so['scores'][f"{m['sa']}-{m['sb']}"] += 1
        s = seg_of(m['rd'])
        if m['y'] == 0:
            so['hw'] += 1; su['wins'][m['home']] += 1
        elif m['y'] == 1:
            so['dr'] += 1; su['dr'] += 1
        else:
            so['aw'] += 1; su['wins'][m['away']] += 1
        if s:
            so['seg'][s][m['y']] += 1
        su['n'] += 1
    i = j

# ------------------------------------------------------------------ dataset predictif + split
data = [m for m in matches if m['oh'] and m['od'] and m['oa']]
n = len(data)
cut = int(n * 0.70)
train, oos = data[:cut], data[cut:]
print(f"dataset predictif: n={n}  train={len(train)} (jusqu'a {train[-1]['t']})  "
      f"oos={len(oos)} (depuis {oos[0]['t']})")

for m in data:
    inv = np.array([1.0 / m['oh'], 1.0 / m['od'], 1.0 / m['oa']])
    m['p'] = inv / inv.sum()

base = np.array([np.mean([m['y'] == k for m in train]) for k in range(3)])
print(f"base rates train (H/D/A): {base.round(4)}")

K_SHRINK = 8.0
for m in data:
    nn = m['h2h_n']
    m['f_hw'] = (m['h2h_hw'] + K_SHRINK * base[0]) / (nn + K_SHRINK) - base[0]
    m['f_dr'] = (m['h2h_dr'] + K_SHRINK * base[1]) / (nn + K_SHRINK) - base[1]
    m['f_aw'] = (m['h2h_aw'] + K_SHRINK * base[2]) / (nn + K_SHRINK) - base[2]
    m['f_n'] = np.log1p(nn)
    un = m['u_n']
    wh = (m['u_home_w'] + K_SHRINK * 0.5) / (un + K_SHRINK)
    wa = (m['u_away_w'] + K_SHRINK * 0.5) / (un + K_SHRINK)
    m['f_udiff'] = wh - wa
    m['f_un'] = np.log1p(un)

# ------------------------------------------------------------------ logistic multinomiale (scipy)

def fit_mnlogit(X, y, l2=1e-3):
    n, d = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])
    idx = np.arange(n)

    def nll(w):
        W = w.reshape(2, d + 1)
        Z = np.hstack([np.zeros((n, 1)), Xb @ W.T])
        mx = Z.max(axis=1, keepdims=True)
        lse = (mx[:, 0] + np.log(np.exp(Z - mx).sum(axis=1)))
        return -(Z[idx, y] - lse).mean() + l2 * np.sum(w ** 2)

    res = minimize(nll, np.zeros(2 * (d + 1)), method='L-BFGS-B',
                   options=dict(maxiter=2000))
    return res.x


def predict_mnlogit(w, X):
    n, d = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])
    W = w.reshape(2, d + 1)
    Z = np.hstack([np.zeros((n, 1)), Xb @ W.T])
    Z -= Z.max(axis=1, keepdims=True)
    P = np.exp(Z)
    return P / P.sum(axis=1, keepdims=True)


def metrics(P, rows_, label, subset=None):
    """acc, logloss, roi (mise 1 sur l'argmax au prix marche)."""
    idxs = range(len(rows_)) if subset is None else subset
    y = np.array([rows_[i]['y'] for i in idxs])
    Ps = P[list(idxs)] if subset is not None else P
    pick = Ps.argmax(axis=1)
    acc = (pick == y).mean()
    ll = -np.log(np.clip(Ps[np.arange(len(y)), y], 1e-12, 1)).mean()
    odds_mat = np.array([[rows_[i]['oh'], rows_[i]['od'], rows_[i]['oa']] for i in idxs])
    won = (pick == y)
    roi = (np.where(won, odds_mat[np.arange(len(y)), pick], 0.0) - 1.0).mean()
    print(f"  {label:<38} n={len(y):>5}  acc={acc:.4f}  logloss={ll:.4f}  roi={roi:+.4f}")
    return acc, ll, roi


def feats(rows_, cols):
    return np.array([[m[c] for c in cols] for m in rows_])


COLS_BASE = []  # rempli ci-dessous
for m in data:
    p = m['p']
    m['x_lha'] = np.log(p[0] / p[2])
    m['x_lda'] = np.log(p[1] / p[2])

ytr = np.array([m['y'] for m in train])
yoo = np.array([m['y'] for m in oos])

print(); print(SEP)
print("TEST 1 - LOGISTIC COTE-ONLY vs COTE+H2H (OOS = derniers 30%)")
print(SEP)

# Baseline 0 : favori devig brut (probas devig non recalibrees)
P_devig_oos = np.array([m['p'] for m in oos])
print("[B0] favori devig (probas devig brutes)")
b0 = metrics(P_devig_oos, oos, "B0 devig OOS global")

variants = {
    'B1 cote-only': ['x_lha', 'x_lda'],
    'V1 cote+H2H oriente': ['x_lha', 'x_lda', 'f_hw', 'f_dr', 'f_aw', 'f_n'],
    'V2 cote+H2H non-oriente': ['x_lha', 'x_lda', 'f_udiff', 'f_un'],
    'V3 cote+H2H les deux': ['x_lha', 'x_lda', 'f_hw', 'f_dr', 'f_aw', 'f_n', 'f_udiff', 'f_un'],
    'V4 H2H seul (sans cote)': ['f_hw', 'f_dr', 'f_aw', 'f_n', 'f_udiff'],
}
results = {}
P_oos = {}
for name, cols in variants.items():
    Xtr, Xoo = feats(train, cols), feats(oos, cols)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
    w = fit_mnlogit((Xtr - mu) / sd, ytr)
    P = predict_mnlogit(w, (Xoo - mu) / sd)
    P_oos[name] = P
    print(f"[{name}] cols={cols}")
    results[name] = metrics(P, oos, f"{name} OOS global")

b1 = results['B1 cote-only']
print("\n--- deltas vs B1 cote-only (OOS global) ---")
for name in variants:
    if name == 'B1 cote-only':
        continue
    a, l, r = results[name]
    print(f"  {name:<28} d_acc={a-b1[0]:+.4f}  d_logloss={l-b1[1]:+.4f}  d_roi={r-b1[2]:+.4f}")

# sous-ensemble OOS h2h_n >= 8 (la ou le H2H a de la matiere)
sub8 = [i for i, m in enumerate(oos) if m['h2h_n'] >= 8]
print(f"\n--- sous-ensemble OOS h2h_n>=8 (n={len(sub8)}) : memes modeles, evalues sur subset ---")
for name in ['B1 cote-only', 'V1 cote+H2H oriente', 'V3 cote+H2H les deux']:
    results[name + '_sub8'] = metrics(P_oos[name], oos, f"{name} OOS h2h_n>=8", subset=sub8)
metrics(P_devig_oos, oos, "B0 devig OOS h2h_n>=8", subset=sub8)

# ------------------------------------------------------------------ TEST 2 desaccords
print(); print(SEP)
print("TEST 2 - DESACCORDS : H2H (n>=8, WR>=65%) CONTREDIT le favori devig")
print(SEP)


def disagree_eval(rows_, n_min, wr_min, oriented=True, label=""):
    picks = []
    for m in rows_:
        if oriented:
            nn = m['h2h_n']
            if nn < n_min:
                continue
            wr_h, wr_a = m['h2h_hw'] / nn, m['h2h_aw'] / nn
            h2h_side = 0 if (wr_h >= wr_min and wr_h > wr_a) else (2 if (wr_a >= wr_min and wr_a > wr_h) else None)
        else:
            nn = m['u_n']
            if nn < n_min:
                continue
            wr_h, wr_a = m['u_home_w'] / nn, m['u_away_w'] / nn
            h2h_side = 0 if (wr_h >= wr_min and wr_h > wr_a) else (2 if (wr_a >= wr_min and wr_a > wr_h) else None)
        if h2h_side is None:
            continue
        fav = int(np.argmax(m['p']))
        if fav == h2h_side:
            continue
        picks.append((m, h2h_side, fav))
    if not picks:
        print(f"  {label}: aucun cas"); return None
    odds3 = lambda m: [m['oh'], m['od'], m['oa']]
    wins_h2h = sum(1 for m, s, f in picks if m['y'] == s)
    roi_h2h = np.mean([(odds3(m)[s] if m['y'] == s else 0) - 1 for m, s, f in picks])
    wins_fav = sum(1 for m, s, f in picks if m['y'] == f)
    roi_fav = np.mean([(odds3(m)[f] if m['y'] == f else 0) - 1 for m, s, f in picks])
    n_ = len(picks)
    avg_cote_h2h = np.mean([odds3(m)[s] for m, s, f in picks])
    print(f"  {label}: n={n_}")
    print(f"    suivre H2H contre marche : WR={wins_h2h/n_:.3f} ({wins_h2h}/{n_})  ROI={roi_h2h:+.4f}  cote_moy={avg_cote_h2h:.2f}")
    print(f"    suivre favori (memes matchs): WR={wins_fav/n_:.3f} ({wins_fav}/{n_})  ROI={roi_fav:+.4f}")
    dist = Counter(m['y'] for m, s, f in picks)
    print(f"    issue reelle H/D/A sur ces matchs: {dict(dist)}")
    return dict(n=n_, wr=wins_h2h / n_, roi=roi_h2h, wr_fav=wins_fav / n_, roi_fav=roi_fav,
                avg_cote=avg_cote_h2h)


print("[OOS]")
d_or = disagree_eval(oos, 8, 0.65, True, "oriente n>=8 WR>=65%")
d_un = disagree_eval(oos, 8, 0.65, False, "non-oriente n>=8 WR>=65%")
print("\n[OOS] variantes de seuil (sensibilite)")
disagree_eval(oos, 12, 0.70, True, "oriente n>=12 WR>=70%")
disagree_eval(oos, 8, 0.55, True, "oriente n>=8 WR>=55%")
print("\n[TRAIN pour reference (in-sample, ne pas citer comme preuve)]")
disagree_eval(train, 8, 0.65, True, "oriente n>=8 WR>=65%")

# ------------------------------------------------------------------ TEST 3 segments
print(); print(SEP)
print("TEST 3 - H2H PAR SEGMENT (DS J1-3 vs FS J34-38)")
print(SEP)

# (a) stabilite train : pour les paires orientees, correlation WR_home entre segments
pair_seg = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))  # pair -> seg -> [hw, dr, aw]
for m in train:
    s = seg_of(m['rd'])
    if s:
        pair_seg[(m['home'], m['away'])][s][m['y']] += 1

for s1, s2, nmin in [('DS', 'FS', 3), ('DS', 'MS', 5), ('FS', 'MS', 5)]:
    xs, ys = [], []
    for pair, segs in pair_seg.items():
        a, b = segs.get(s1), segs.get(s2)
        if a and b and sum(a) >= nmin and sum(b) >= nmin:
            xs.append(a[0] / sum(a)); ys.append(b[0] / sum(b))
    if len(xs) >= 10:
        r = np.corrcoef(xs, ys)[0, 1]
        print(f"  corr WR_home {s1} vs {s2} (paires avec n>={nmin} dans chaque, n_paires={len(xs)}): r={r:+.3f}")
    else:
        print(f"  {s1} vs {s2}: paires insuffisantes ({len(xs)})")

# (b) OOS : desaccords par segment du match courant
print("\n[OOS] desaccords (oriente n>=8 WR>=65%) par segment du match courant")
for s in ['DS', 'MS', 'FS']:
    rows_s = [m for m in oos if seg_of(m['rd']) == s]
    print(f" segment {s} (n_matchs={len(rows_s)}):")
    disagree_eval(rows_s, 8, 0.65, True, f"  {s}")

# (c) le H2H restreint au meme segment predit-il mieux que le H2H global ?
# pour matchs OOS en DS ou FS avec >=4 rencontres anterieures dans le MEME segment
print("\n[OOS] accuracy du 'cote H2H majoritaire' : H2H meme-segment vs H2H global (memes matchs)")
for s in ['DS', 'FS']:
    rows_s = []
    for m in oos:
        if seg_of(m['rd']) != s:
            continue
        segc = m['h2h_seg'].get(s)
        if not segc or sum(segc) < 4 or m['h2h_n'] < 8:
            continue
        rows_s.append(m)
    if len(rows_s) < 40:
        print(f"  segment {s}: n={len(rows_s)} < 40 -> insuffisant, non conclu")
        continue
    acc_seg = acc_glob = acc_fav = 0
    for m in rows_s:
        segc = m['h2h_seg'][s]
        pick_seg = int(np.argmax(segc))
        pick_glob = int(np.argmax([m['h2h_hw'], m['h2h_dr'], m['h2h_aw']]))
        acc_seg += (pick_seg == m['y'])
        acc_glob += (pick_glob == m['y'])
        acc_fav += (int(np.argmax(m['p'])) == m['y'])
    nl = len(rows_s)
    print(f"  segment {s}: n={nl}  acc(H2H seg)={acc_seg/nl:.3f}  acc(H2H global)={acc_glob/nl:.3f}  acc(favori devig)={acc_fav/nl:.3f}")

# ------------------------------------------------------------------ TEST 4 score exact
print(); print(SEP)
print("TEST 4 - SCORE EXACT : H2H top1/top3 vs marche 'Score exact' vs global train")
print(SEP)

glob_scores = Counter(f"{m['sa']}-{m['sb']}" for m in train)
glob_rank = [s for s, _ in glob_scores.most_common()]

rows4 = [m for m in oos if m['h2h_n'] >= 8 and m['score_market']]
print(f"matchs OOS avec h2h_n>=8 ET marche Score exact: {len(rows4)}")

hit = dict(h2h1=0, h2h3=0, mkt1=0, mkt3=0, glo1=0, glo3=0)
for m in rows4:
    actual = f"{m['sa']}-{m['sb']}"
    # H2H : scores les plus frequents (tie-break: frequence globale train)
    sc = m['h2h_scores']
    h2h_rank = sorted(sc, key=lambda s: (-sc[s], -glob_scores.get(s, 0)))
    # marche : cotes les plus basses
    mk = {k: v for k, v in m['score_market'].items() if isinstance(v, (int, float))}
    mkt_rank = sorted(mk, key=lambda s: mk[s])
    hit['h2h1'] += (len(h2h_rank) > 0 and actual == h2h_rank[0])
    hit['h2h3'] += (actual in h2h_rank[:3])
    hit['mkt1'] += (len(mkt_rank) > 0 and actual == mkt_rank[0])
    hit['mkt3'] += (actual in mkt_rank[:3])
    hit['glo1'] += (actual == glob_rank[0])
    hit['glo3'] += (actual in glob_rank[:3])
n4 = len(rows4)
if n4:
    print(f"{'methode':<26} {'top1':>8} {'top3':>8}")
    for lab, k1, k3 in [("H2H paire", 'h2h1', 'h2h3'), ("marche Score exact", 'mkt1', 'mkt3'),
                        ("global train", 'glo1', 'glo3')]:
        print(f"{lab:<26} {hit[k1]/n4:>8.4f} {hit[k3]/n4:>8.4f}   ({hit[k1]}/{n4}, {hit[k3]}/{n4})")

# hybride : H2H top1 quand h2h_n>=12 et score modal >=25% des rencontres, sinon marche
hyb1 = 0
n_used_h2h = 0
for m in rows4:
    actual = f"{m['sa']}-{m['sb']}"
    sc = m['h2h_scores']
    h2h_rank = sorted(sc, key=lambda s: (-sc[s], -glob_scores.get(s, 0)))
    mk = {k: v for k, v in m['score_market'].items() if isinstance(v, (int, float))}
    mkt_rank = sorted(mk, key=lambda s: mk[s])
    if m['h2h_n'] >= 12 and h2h_rank and sc[h2h_rank[0]] / m['h2h_n'] >= 0.25:
        pick = h2h_rank[0]; n_used_h2h += 1
    else:
        pick = mkt_rank[0] if mkt_rank else (glob_rank[0])
    hyb1 += (actual == pick)
if n4:
    print(f"hybride (H2H modal>=25% & n>=12 sinon marche): top1={hyb1/n4:.4f} ({hyb1}/{n4}), H2H utilise {n_used_h2h}x")

# ------------------------------------------------------------------ distribution h2h_n OOS
print(); print(SEP)
print("DISTRIBUTION h2h_n (oriente) dans l'OOS")
print(SEP)
dist = Counter()
for m in oos:
    b = min(m['h2h_n'] // 4 * 4, 40)
    dist[b] += 1
print({f"{k}-{k+3}" if k < 40 else "40+": v for k, v in sorted(dist.items())})

print("\nFIN.")
