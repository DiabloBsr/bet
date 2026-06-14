# -*- coding: utf-8 -*-
"""WF3 - FACETTE HT/FT, partie 4 : VALIDATION HAUTE PUISSANCE DES POCHES EV>0

Probleme partie 3 : ROI realise sur cotes 30-100 = bruit pur (CI +/-100%).
Tests a forte puissance :
  1. LL apparie : p_modele(G1xG2) vs p_marche(devig) sur l'issue observee,
     marche par marche -> qui connait la verite ?
  2. calibration in-pocket : nb de wins observe vs attendu sous modele vs sous marche
     (test du rapport de vraisemblance binomial, pooled Poisson-binomial)
  3. mecanique de la poche 'Mi-tps 1X2' : la cote 1X2-MT est-elle l'intrus
     (vs G1 et vs 'Mi-tps DC') ?
  4. walk-forward fige : regle unique pre-declaree = 'Mi-tps 1X2 EV_mod>0'
     (choisie sur train partie 3) -> OOS detail
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
CELLS = [(i, j) for i in range(4) for j in range(4) if i + j <= 3]


def parse_cell(s):
    a, b = s.split('-')
    return int(a), int(b)


def devig(d, keys=None):
    imp = {}
    for k, v in d.items():
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None, None
        if v < 1.01:
            return None, None
        imp[k] = 1.0 / v
    if keys and set(imp) != set(keys):
        return None, None
    s = sum(imp.values())
    return {k: p / s for k, p in imp.items()}, s


def res_of(h, a):
    return '1' if h > a else ('2' if h < a else 'X')


def convolve(g1, g2):
    out = defaultdict(float)
    for (i, j), p1 in g1.items():
        for (k, l), p2 in g2.items():
            out[(i + k, j + l)] += p1 * p2
    return out


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
            if hh > h or ha > a:
                continue
            gj = r[13]
            goals = []
            if gj:
                try:
                    goals = json.loads(gj) if isinstance(gj, str) else gj
                except Exception:
                    goals = []
            if goals and len(goals) == h + a:
                hh = sum(1 for g in goals if g['minute'] <= 45 and g['team'] == 'Home')
                ha = sum(1 for g in goals if g['minute'] <= 45 and g['team'] == 'Away')
            if hh + ha > 3 or (h - hh) + (a - ha) > 3:
                continue
            g1d = em.get('Mi-tps CS'); g2d = em.get('2ème mi-tps - CS') or em.get('2eme mi-tps - CS')
            if not (isinstance(g1d, dict) and isinstance(g2d, dict)):
                continue
            try:
                k1 = {parse_cell(k) for k in g1d}; k2 = {parse_cell(k) for k in g2d}
            except Exception:
                continue
            if k1 != set(CELLS) or k2 != set(CELLS):
                continue
            g1n, _ = devig({f"{c[0]}-{c[1]}": g1d[f"{c[0]}-{c[1]}"] for c in CELLS})
            g2n, _ = devig({f"{c[0]}-{c[1]}": g2d[f"{c[0]}-{c[1]}"] for c in CELLS})
            if g1n is None or g2n is None:
                continue
            g1 = {parse_cell(k): v for k, v in g1n.items()}
            g2 = {parse_cell(k): v for k, v in g2n.items()}
            out.append(dict(id=r[0], start=r[2], oh=r[5], od=r[6], oa=r[7], em=em,
                            h=h, a=a, hh=hh, ha=ha, h2h=h - hh, h2a=a - ha,
                            g1=g1, g2=g2, g1raw=g1d, g2raw=g2d))
    return out

M = load()
print(f"matchs : {len(M)}")
for m in M:
    m['cv'] = convolve(m['g1'], m['g2'])
    m['p1x2'] = {'1': sum(p for c, p in m['cv'].items() if c[0] > c[1]),
                 'X': sum(p for c, p in m['cv'].items() if c[0] == c[1]),
                 '2': sum(p for c, p in m['cv'].items() if c[0] < c[1])}
    m['pht'] = {'1': sum(p for c, p in m['g1'].items() if c[0] > c[1]),
                'X': sum(p for c, p in m['g1'].items() if c[0] == c[1]),
                '2': sum(p for c, p in m['g1'].items() if c[0] < c[1])}
    m['phtft'] = defaultdict(float)
    for c1_, p1 in m['g1'].items():
        r1_ = res_of(*c1_)
        for c2_, p2 in m['g2'].items():
            m['phtft'][f"{r1_}/{res_of(c1_[0]+c2_[0], c1_[1]+c2_[1])}"] += p1 * p2

# ---------------------------------------------------------------- 1. LL apparie
print(SEP); print("1. LL APPARIE modele vs marche (sur l'issue observee)"); print(SEP)

def ll_test(label, items):
    """items: liste (p_model, p_market) pour l'issue observee."""
    d = [math.log(max(pm, 1e-9)) - math.log(max(pk, 1e-9)) for pm, pk in items]
    t, pv = stats.ttest_1samp(d, 0)
    side = 'MODELE gagne' if np.mean(d) > 0 else 'MARCHE gagne'
    print(f"  {label:42s} n={len(d):5d}  dLL/match={np.mean(d):+.5f}  t={t:+6.2f}  p={pv:.2e}  {side if pv<0.05 else 'egalite'}")

items = []
for m in M:
    if not all((m['oh'], m['od'], m['oa'])):
        continue
    dv, _ = devig({'1': m['oh'], 'X': m['od'], '2': m['oa']})
    if dv:
        r = res_of(m['h'], m['a'])
        items.append((m['p1x2'][r], dv[r]))
ll_test("1X2 final : conv vs devig(1X2 principal)", items)

items = []
for m in M:
    d = m['em'].get('Mi-tps 1X2')
    if isinstance(d, dict):
        dv, _ = devig(d, keys={'1', 'X', '2'})
        if dv:
            r = res_of(m['hh'], m['ha'])
            items.append((m['pht'][r], dv[r]))
ll_test("1X2 mi-temps : G1 vs devig(Mi-tps 1X2)", items)

items = []
COMBOS = ['1/1', '1/X', '1/2', 'X/1', 'X/X', 'X/2', '2/1', '2/X', '2/2']
for m in M:
    d = m['em'].get('HT/FT')
    if isinstance(d, dict):
        dv, _ = devig(d, keys=set(COMBOS))
        if dv:
            r = f"{res_of(m['hh'], m['ha'])}/{res_of(m['h'], m['a'])}"
            items.append((m['phtft'][r], dv[r]))
ll_test("HT/FT : G1xG2 vs devig(HT/FT)", items)

items = []
for m in M:
    d = m['em'].get('Score exact')
    if isinstance(d, dict):
        dv, _ = devig(d)
        if dv:
            r = f"{m['h']}-{m['a']}"
            if r in dv:
                items.append((m['cv'].get((m['h'], m['a']), 1e-9), dv[r]))
ll_test("Score exact : conv vs devig(Score exact)", items)

# version sans cellules cappees a 100 (artefact de cap ?)
items = []
for m in M:
    d = m['em'].get('Score exact')
    if isinstance(d, dict) and all(float(v) < 99.5 for v in d.values()):
        dv, _ = devig(d)
        if dv:
            r = f"{m['h']}-{m['a']}"
            if r in dv:
                items.append((m['cv'].get((m['h'], m['a']), 1e-9), dv[r]))
ll_test("Score exact (grilles SANS cote cappee 100)", items)

# ---------------------------------------------------------------- 2. calibration in-pocket
print(SEP); print("2. CALIBRATION DANS LES POCHES EV_modele>0"); print(SEP)
print("(attendu_modele = somme p_mod ; attendu_marche = somme p_devig ; lequel colle ?)")

def pocket_test(label, rows):
    """rows: (p_mod, p_mkt, odds, won)"""
    if not rows:
        return
    n = len(rows)
    e_mod = sum(r[0] for r in rows); e_mkt = sum(r[1] for r in rows)
    w = sum(1 for r in rows if r[3])
    v_mod = sum(r[0] * (1 - r[0]) for r in rows); v_mkt = sum(r[1] * (1 - r[1]) for r in rows)
    z_mod = (w - e_mod) / math.sqrt(max(v_mod, 1e-9))
    z_mkt = (w - e_mkt) / math.sqrt(max(v_mkt, 1e-9))
    roi = sum(r[2] for r in rows if r[3]) / n - 1
    print(f"  {label:24s} n={n:5d} wins={w:4d}  E_mod={e_mod:7.1f}(z={z_mod:+5.2f})  "
          f"E_mkt={e_mkt:7.1f}(z={z_mkt:+5.2f})  ROI={roi:+7.2%}")

def collect(mk_label, get_pm, get_mkt_dict, keys, outcome_of):
    rows_all, by_sel = [], defaultdict(list)
    for m in M:
        d = m['em'].get(get_mkt_dict)
        if not isinstance(d, dict):
            continue
        dv, s = devig(d, keys=keys)
        if not dv:
            continue
        obs = outcome_of(m)
        for sel in dv:
            try:
                o = float(d[sel])
            except (TypeError, ValueError):
                continue
            pm = get_pm(m, sel)
            if pm * o - 1 > 0:
                rows_all.append((pm, dv[sel], o, sel == obs))
                by_sel[sel].append((pm, dv[sel], o, sel == obs))
    pocket_test(mk_label + ' [ALL]', rows_all)
    for sel, rows in sorted(by_sel.items(), key=lambda kv: -len(kv[1])):
        if len(rows) >= 40:
            pocket_test(f"{mk_label} [{sel}]", rows)
    return rows_all

rows_ht1x2 = collect('Mi-tps 1X2', lambda m, s: m['pht'][s], 'Mi-tps 1X2', {'1', 'X', '2'},
                     lambda m: res_of(m['hh'], m['ha']))
rows_htft = collect('HT/FT', lambda m, s: m['phtft'][s], 'HT/FT', set(COMBOS),
                    lambda m: f"{res_of(m['hh'], m['ha'])}/{res_of(m['h'], m['a'])}")
rows_se = collect('Score exact', lambda m, s: m['cv'].get(parse_cell(s), 0.0), 'Score exact', None,
                  lambda m: f"{m['h']}-{m['a']}")

# 1X2 principal
rows_all = []
for m in M:
    if not all((m['oh'], m['od'], m['oa'])):
        continue
    d = {'1': m['oh'], 'X': m['od'], '2': m['oa']}
    dv, _ = devig(d)
    obs = res_of(m['h'], m['a'])
    for sel in '1X2':
        o = float(d[sel]); pm = m['p1x2'][sel]
        if pm * o - 1 > 0:
            rows_all.append((pm, dv[sel], o, sel == obs))
pocket_test('1X2 principal [ALL]', rows_all)

# pooled
pooled = rows_ht1x2 + rows_htft + rows_se + rows_all
pocket_test('>>> POOLED toutes poches', pooled)

# ---------------------------------------------------------------- 3. mecanique poche Mi-tps 1X2
print(SEP); print("3. POCHE 'Mi-tps 1X2' : QUI EST L'INTRUS ?"); print(SEP)
ex = []
agree_g1, agree_1x2 = [], []
for m in M:
    d = m['em'].get('Mi-tps 1X2'); dc = m['em'].get('Mi-tps DC')
    if not (isinstance(d, dict) and isinstance(dc, dict)):
        continue
    dv, _ = devig(d, keys={'1', 'X', '2'})
    dvc, _ = devig(dc, keys={'1X', 'X2', '12'})
    if not (dv and dvc):
        continue
    # probas issues de DC (systeme : p1+pX, pX+p2, p1+p2 -> p = (somme paires)/2)
    p1 = dvc['1X'] + dvc['12'] - dvc['X2']
    px = dvc['1X'] + dvc['X2'] - dvc['12']
    p2 = dvc['X2'] + dvc['12'] - dvc['1X']
    pdc = {'1': p1 / 1.0, 'X': px, '2': p2}
    has_pocket = any(m['pht'][s] * float(d[s]) - 1 > 0 for s in '1X2')
    dg1 = sum(abs(pdc[s] - m['pht'][s]) for s in '1X2')
    d1x2 = sum(abs(pdc[s] - dv[s]) for s in '1X2')
    (ex if has_pocket else None) is not None and has_pocket and ex.append((m, dv, pdc))
    agree_g1.append((dg1, has_pocket)); agree_1x2.append((d1x2, has_pocket))

a_g1 = np.array([x[0] for x in agree_g1]); a_12 = np.array([x[0] for x in agree_1x2])
pk = np.array([x[1] for x in agree_g1], dtype=bool)
print(f"distance |p_DC - p_G1| (somme abs)   : tous={np.median(a_g1):.4f}  poche={np.median(a_g1[pk]):.4f}")
print(f"distance |p_DC - p_(1X2MT devig)|    : tous={np.median(a_12):.4f}  poche={np.median(a_12[pk]):.4f}")
print("=> si en poche DC colle a G1 mais pas au 1X2-MT, la cote 1X2-MT est l'intrus (stale/altere)")
print(f"\nexemples poche (5 premiers) :")
for m, dv, pdc in ex[:5]:
    sels = [s for s in '1X2' if m['pht'][s] * float(m['em']['Mi-tps 1X2'][s]) - 1 > 0]
    print(f"  id={m['id']} cotesMT={m['em']['Mi-tps 1X2']} pG1={ {s: round(m['pht'][s],3) for s in '1X2'} } "
          f"pDC={ {s: round(pdc[s],3) for s in '1X2'} } poche={sels} HT reel {m['hh']}-{m['ha']}")

# ---------------------------------------------------------------- 4. walk-forward fige
print(SEP); print("4. WALK-FORWARD REGLE FIGEE : 'Mi-tps 1X2, EV_mod>0'"); print(SEP)
bets = []
for m in M:
    d = m['em'].get('Mi-tps 1X2')
    if not isinstance(d, dict):
        continue
    obs = res_of(m['hh'], m['ha'])
    for sel in '1X2':
        try:
            o = float(d[sel])
        except (TypeError, ValueError, KeyError):
            continue
        pm = m['pht'][sel]
        if pm * o - 1 > 0:
            bets.append((m['start'], pm, o, sel == obs))
bets.sort(key=lambda b: b[0])
cut = bets[int(len(bets) * 0.7)][0] if bets else None
tr = [b for b in bets if b[0] <= cut]; oo = [b for b in bets if b[0] > cut]
for lbl, lst in [('train(70%)', tr), ('OOS(30%)', oo), ('TOTAL', bets)]:
    n = len(lst); w = sum(1 for b in lst if b[3])
    roi = sum(b[2] for b in lst if b[3]) / n - 1 if n else 0
    e_mod = sum(b[1] for b in lst)
    ao = np.mean([b[2] for b in lst]) if n else 0
    print(f"  {lbl:10s} n={n:4d}  wins={w:3d} (E_mod={e_mod:.1f})  ROI={roi:+7.2%}  cote_moy={ao:.2f}")
