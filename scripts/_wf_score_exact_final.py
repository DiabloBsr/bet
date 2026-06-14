# -*- coding: utf-8 -*-
"""Chiffres finaux exacts pour le rapport (wr_train, aggregats manquants)."""
import sys, collections
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
from _wf_score_exact import load_data, roi_line

def orient(m, s):
    a, b = s.split('-')
    return f'{a}-{b}' if m['oh'] <= m['oa'] else f'{b}-{a}'

data = load_data()
cut = int(len(data) * 0.7)
train, oos = data[:cut], data[cut:]

def report(label, train_bets, oos_bets):
    n_t, wr_t, c_t, r_t = roi_line(train_bets)
    n_o, wr_o, c_o, r_o = roi_line(oos_bets)
    print(f'{label}: TRAIN n={n_t} wr={wr_t*100:.3f}% cote~{c_t:.1f} roi={r_t*100:+.1f}% | OOS n={n_o} hits={sum(w for w,_ in oos_bets)} wr={wr_o*100:.3f}% cote~{c_o:.1f} roi={r_o*100:+.2f}%')

def bets(ds, fn):
    out = []
    for m in ds:
        for s, o in m['se'].items():
            if o >= 100.0: continue
            if fn(m, s): out.append((1 if m['ft'] == s else 0, o))
    return out

# 1. fav-oriente 4-2
f1 = lambda m, s: min(m['oh'], m['oa']) <= 2.60 and orient(m, s) == '4-2'
report('FAV-4-2 (fav<=2.60)', bets(train, f1), bets(oos, f1))
# 2. 0-4 quand A favori
f2 = lambda m, s: m['oa'] < m['oh'] and s == '0-4'
report('0-4 A-fav          ', bets(train, f2), bets(oos, f2))
# 3. portfolio global {4-2, 0-4}
f3 = lambda m, s: s in ('4-2', '0-4')
report('Global {4-2,0-4}   ', bets(train, f3), bets(oos, f3))
# 4. 6-0 (sur-cote)
f4 = lambda m, s: s == '6-0'
report('6-0 (contre-signal)', bets(train, f4), bets(oos, f4))

# 5. 2MT-CS : parier toute la grille (mesure du vig realise)
def bets2h(ds):
    out = []
    for m in ds:
        for s, o in m['h2'].items():
            if o >= 100.0: continue
            out.append((1 if m['h2s'] == s else 0, o))
    return out
report('2MT-CS grille totale', bets2h(train), bets2h(oos))

# 6. ROI de parier le top1 marche / top1 blend bayesien prior=8 / top1 marche x ratio
prof_dist = collections.defaultdict(collections.Counter)
pair_dist = collections.defaultdict(collections.Counter)
glob = collections.Counter()
for m in train:
    prof_dist[m['profile']][m['ft']] += 1
    pair_dist[(m['ta'], m['tb'])][m['ft']] += 1
    glob[m['ft']] += 1

def blend(m, prior=8):
    base = prof_dist[m['profile']] or glob
    nb = sum(base.values())
    pp = {s: c / nb for s, c in base.items()}
    pc = pair_dist.get((m['ta'], m['tb']))
    if not pc: return pp
    npair = sum(pc.values())
    pdd = {s: c / npair for s, c in pc.items()}
    w = npair / (npair + prior)
    return {s: w * pdd.get(s, 0) + (1 - w) * pp.get(s, 0) for s in set(pp) | set(pdd)}

imp_sum = collections.defaultdict(float)
act_n = collections.defaultdict(int)
for m in train:
    fs = 'H' if m['oh'] <= m['oa'] else 'A'
    inv = {s: 1.0 / o for s, o in m['se'].items() if o > 1.0}
    tot = sum(inv.values())
    for s in m['se']:
        imp_sum[(fs, s)] += inv[s] / tot
    act_n[(fs, m['ft'])] += 1
ratio = {}
for k in imp_sum:
    nh = act_n.get(k, 0)
    raw = nh / imp_sum[k] if imp_sum[k] > 0 else 1.0
    ratio[k] = (nh * raw + 30) / (nh + 30)

def p_market(m):
    inv = {s: 1.0 / o for s, o in m['se'].items() if o > 1.0}
    tot = sum(inv.values())
    return {s: v / tot for s, v in inv.items()}

def top_bets(ds, top_fn, k=1):
    out = []
    for m in ds:
        for s in top_fn(m)[:k]:
            out.append((1 if m['ft'] == s else 0, m['se'].get(s, 100.0)))
    return out

t_mkt = lambda m: [s for s, _ in sorted(m['se'].items(), key=lambda kv: kv[1])[:3]]
t_blend = lambda m: [s for s, _ in sorted(blend(m).items(), key=lambda kv: -kv[1])[:3]]
def t_ratio(m):
    fs = 'H' if m['oh'] <= m['oa'] else 'A'
    pm = p_market(m)
    pc = {s: p * ratio.get((fs, s), 1.0) for s, p in pm.items()}
    return [s for s, _ in sorted(pc.items(), key=lambda kv: -kv[1])[:3]]

report('BET top1 marche     ', top_bets(train, t_mkt), top_bets(oos, t_mkt))
report('BET top1 blend p=8  ', top_bets(train, t_blend), top_bets(oos, t_blend))
report('BET top1 mkt x ratio', top_bets(train, t_ratio), top_bets(oos, t_ratio))
report('BET top3 mkt x ratio', top_bets(train, t_ratio, 3), top_bets(oos, t_ratio, 3))
