# -*- coding: utf-8 -*-
"""WF3 - FACETTE HT/FT, partie 3 :
A. parametrisation du moteur : grilles par MT = bivarie Poisson tronque ? lien MT1/MT2 ?
B. coherence 1X2 principal vs convolution
C. scan EV de TOUTES les selections HT/FT-famille avec p_modele = G1(x)G2 (valide partie 2)
D. walk-forward 70/30 sur les groupes EV>0
"""
import sys, json, math
from collections import Counter, defaultdict
sys.path.insert(0, '.')
from scraper.config import load_settings
from sqlalchemy import create_engine, text
import numpy as np
from scipy import stats, optimize

eng = create_engine(load_settings().db_url)
SEP = "=" * 78
CELLS = [(i, j) for i in range(4) for j in range(4) if i + j <= 3]


def parse_cell(s):
    a, b = s.split('-')
    return int(a), int(b)


def devig_grid(d):
    imp = {}
    for k, v in d.items():
        try:
            c = parse_cell(k)
        except Exception:
            return None, None
        if v is None or float(v) < 1.01:
            return None, None
        imp[c] = 1.0 / float(v)
    s = sum(imp.values())
    return {c: p / s for c, p in imp.items()}, s


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
                jh = sum(1 for g in goals if g['minute'] <= 45 and g['team'] == 'Home')
                ja = sum(1 for g in goals if g['minute'] <= 45 and g['team'] == 'Away')
                hh, ha = jh, ja
            if hh + ha > 3 or (h - hh) + (a - ha) > 3:
                continue
            g1d = em.get('Mi-tps CS'); g2d = em.get('2ème mi-tps - CS') or em.get('2eme mi-tps - CS')
            if not (isinstance(g1d, dict) and isinstance(g2d, dict)):
                continue
            try:
                if set(map(parse_cell, g1d.keys())) != set(CELLS) or set(map(parse_cell, g2d.keys())) != set(CELLS):
                    continue
            except Exception:
                continue
            g1, mg1 = devig_grid(g1d)
            g2, mg2 = devig_grid(g2d)
            if g1 is None or g2 is None:
                continue
            out.append(dict(id=r[0], start=r[2], oh=r[5], od=r[6], oa=r[7], em=em,
                            h=h, a=a, hh=hh, ha=ha, h2h=h - hh, h2a=a - ha,
                            g1=g1, g2=g2, g1raw=g1d, g2raw=g2d, mg1=mg1, mg2=mg2))
    return out

M = load()
print(f"matchs exploitables : {len(M)}")

# ---------------------------------------------------------------- A. parametrisation
print(SEP); print("A. PARAMETRISATION DES GRILLES PAR MI-TEMPS"); print(SEP)

def lam(g):
    return (sum(p * c[0] for c, p in g.items()), sum(p * c[1] for c, p in g.items()))

r_h, r_a, l1t, l2t = [], [], [], []
for m in M:
    lh1, la1 = lam(m['g1']); lh2, la2 = lam(m['g2'])
    if lh1 > 0.05 and la1 > 0.05:
        r_h.append(lh2 / lh1); r_a.append(la2 / la1)
    l1t.append(lh1 + la1); l2t.append(lh2 + la2)
print(f"E[buts] MT1 (grille, tronquee) : median={np.median(l1t):.3f}  MT2 : {np.median(l2t):.3f}")
print(f"ratio intensites MT2/MT1 par equipe : home median={np.median(r_h):.3f} IQR=({np.percentile(r_h,25):.3f},{np.percentile(r_h,75):.3f})")
print(f"                                      away median={np.median(r_a):.3f} IQR=({np.percentile(r_a,25):.3f},{np.percentile(r_a,75):.3f})")

# fit bivarie Poisson tronque (avec / sans terme diagonal) sur un echantillon
def fit_grid(gobs, with_cov):
    """min somme (log p_model - log p_obs)^2 sur les 10 cellules."""
    def model(params):
        if with_cov:
            lh, la, lc = np.exp(params)
        else:
            lh, la = np.exp(params); lc = 0.0
        pr = {}
        for (i, j) in CELLS:
            s = 0.0
            for k in range(min(i, j) + 1):
                s += (lh ** (i - k)) / math.factorial(i - k) * (la ** (j - k)) / math.factorial(j - k) * \
                     ((lc ** k) / math.factorial(k) if lc > 0 or k == 0 else 0.0)
            pr[(i, j)] = s
        tot = sum(pr.values())
        return {c: v / tot for c, v in pr.items()}
    def loss(params):
        pm = model(params)
        return sum((math.log(max(pm[c], 1e-12)) - math.log(max(gobs[c], 1e-12))) ** 2 for c in CELLS)
    x0 = [math.log(0.7), math.log(0.5)] + ([math.log(0.1)] if with_cov else [])
    res = optimize.minimize(loss, x0, method='Nelder-Mead',
                            options=dict(maxiter=2000, xatol=1e-8, fatol=1e-12))
    return res.fun, model(res.x), (np.exp(res.x))

rng = np.random.default_rng(42)
idx = rng.choice(len(M), size=150, replace=False)
loss_ind, loss_cov, lc_vals, resid11 = [], [], [], []
for i in idx:
    g = M[i]['g1']
    li, pmi, _ = fit_grid(g, False)
    lcv, pmc, prm = fit_grid(g, True)
    loss_ind.append(li); loss_cov.append(lcv)
    lc_vals.append(prm[2])
    resid11.append(math.log(g[(1, 1)]) - math.log(pmi[(1, 1)]))
print(f"\nfit grille MT1 (150 matchs echantillon, somme des (dlog)^2 sur 10 cellules):")
print(f"  Poisson bivarie tronque SANS covariance : median loss={np.median(loss_ind):.5f}")
print(f"  AVEC covariance (lambda_c)              : median loss={np.median(loss_cov):.5f}")
print(f"  lambda_c median={np.median(lc_vals):.4f} IQR=({np.percentile(lc_vals,25):.4f},{np.percentile(lc_vals,75):.4f})")
print(f"  residu log cellule 1-1 du fit independant : median={np.median(resid11):+.4f} "
      f"({'1-1 sur-pondere vs indep' if np.median(resid11) > 0 else '1-1 sous-pondere'})")

# ---------------------------------------------------------------- B. 1X2 principal vs conv
print(SEP); print("B. 1X2 PRINCIPAL vs CONVOLUTION"); print(SEP)
rows = []
for m in M:
    if not all((m['oh'], m['od'], m['oa'])):
        continue
    cv = convolve(m['g1'], m['g2'])
    p1 = sum(p for c, p in cv.items() if c[0] > c[1])
    px = sum(p for c, p in cv.items() if c[0] == c[1])
    p2 = 1 - p1 - px
    imp = [1 / float(m['oh']), 1 / float(m['od']), 1 / float(m['oa'])]
    s = sum(imp)
    rows.append((p1, px, p2, imp[0] / s, imp[1] / s, imp[2] / s, s))
rows = np.array(rows)
print(f"marge mediane 1X2 principal : {np.median(rows[:,6]):.4f}")
for i, lbl in [(0, 'P1'), (1, 'PX'), (2, 'P2')]:
    d = rows[:, i] - rows[:, i + 3]
    print(f"  {lbl}: conv - devig1X2  mean={d.mean():+.5f}  median={np.median(d):+.5f}  |d| moy={np.abs(d).mean():.5f}")

# ---------------------------------------------------------------- C. scan EV
print(SEP); print("C. SCAN EV (p_modele = G1xG2) sur les marches famille HT/FT"); print(SEP)

def model_probs(m):
    """probas modele pour chaque (marche, selection) -> p."""
    g1, g2 = m['g1'], m['g2']
    cv = convolve(g1, g2)
    P = {}
    # marches par MT (cotes brutes des grilles)
    for c in CELLS:
        P[('Mi-tps CS', f"{c[0]}-{c[1]}")] = g1[c]
        P[('2eme MT CS', f"{c[0]}-{c[1]}")] = g2[c]
    # Mi-tps 1X2 / DC
    p1h = sum(p for c, p in g1.items() if c[0] > c[1])
    pxh = sum(p for c, p in g1.items() if c[0] == c[1])
    p2h = 1 - p1h - pxh
    P[('Mi-tps 1X2', '1')], P[('Mi-tps 1X2', 'X')], P[('Mi-tps 1X2', '2')] = p1h, pxh, p2h
    P[('Mi-tps DC', '1X')], P[('Mi-tps DC', 'X2')], P[('Mi-tps DC', '12')] = p1h + pxh, pxh + p2h, p1h + p2h
    # HT/FT
    for c1_, p1 in g1.items():
        r1_ = res_of(*c1_)
        for c2_, p2 in g2.items():
            rf = res_of(c1_[0] + c2_[0], c1_[1] + c2_[1])
            P[('HT/FT', f"{r1_}/{rf}")] = P.get(('HT/FT', f"{r1_}/{rf}"), 0.0) + p1 * p2
    # Score exact + 1X2 final
    for c, p in cv.items():
        P[('Score exact', f"{c[0]}-{c[1]}")] = p
    P[('1X2', '1')] = sum(p for c, p in cv.items() if c[0] > c[1])
    P[('1X2', 'X')] = sum(p for c, p in cv.items() if c[0] == c[1])
    P[('1X2', '2')] = sum(p for c, p in cv.items() if c[0] < c[1])
    return P


def market_odds(m):
    """(marche, selection) -> cote brute."""
    em = m['em']
    O = {}
    for c in CELLS:
        O[('Mi-tps CS', f"{c[0]}-{c[1]}")] = float(m['g1raw'][f"{c[0]}-{c[1]}"])
        O[('2eme MT CS', f"{c[0]}-{c[1]}")] = float(m['g2raw'][f"{c[0]}-{c[1]}"])
    for mk, key in [('Mi-tps 1X2', 'Mi-tps 1X2'), ('Mi-tps DC', 'Mi-tps DC'),
                    ('HT/FT', 'HT/FT'), ('Score exact', 'Score exact')]:
        d = em.get(key)
        if isinstance(d, dict):
            for k, v in d.items():
                try:
                    O[(mk, k)] = float(v)
                except (TypeError, ValueError):
                    pass
    if m['oh']:
        O[('1X2', '1')], O[('1X2', 'X')], O[('1X2', '2')] = float(m['oh']), float(m['od']), float(m['oa'])
    return O


def won(m, mk, sel):
    h, a, hh, ha = m['h'], m['a'], m['hh'], m['ha']
    h2h, h2a = m['h2h'], m['h2a']
    if mk == 'Mi-tps CS':
        return sel == f"{hh}-{ha}"
    if mk == '2eme MT CS':
        return sel == f"{h2h}-{h2a}"
    if mk == 'Score exact':
        return sel == f"{h}-{a}"
    if mk == 'Mi-tps 1X2':
        return sel == res_of(hh, ha)
    if mk == 'Mi-tps DC':
        return res_of(hh, ha) in {'1X': '1X', 'X2': 'X2', '12': '12'}[sel]
    if mk == 'HT/FT':
        return sel == f"{res_of(hh, ha)}/{res_of(h, a)}"
    if mk == '1X2':
        return sel == res_of(h, a)
    raise ValueError(mk)

BETS = []   # (start, market, sel, p_model, odds, ev, won)
for m in M:
    P = model_probs(m)
    O = market_odds(m)
    for key, p in P.items():
        o = O.get(key)
        if o is None or o < 1.01:
            continue
        BETS.append((m['start'], key[0], key[1], p, o, p * o - 1, won(m, key[0], key[1])))

print(f"{len(BETS)} selections evaluees sur {len(M)} matchs")
by_mkt = defaultdict(list)
for b in BETS:
    by_mkt[b[1]].append(b)
print("\nmarche          n_sel    EV_modele moyen   %sel EV>0   EV>0: EV moyen")
for mk, lst in sorted(by_mkt.items()):
    evs = np.array([b[5] for b in lst])
    pos = evs > 0
    print(f"  {mk:14s} {len(lst):6d}    {evs.mean():+8.4f}      {pos.mean():7.2%}    "
          f"{evs[pos].mean():+8.4f}" if pos.any() else
          f"  {mk:14s} {len(lst):6d}    {evs.mean():+8.4f}      {pos.mean():7.2%}        -")

# detail des poches EV>0
print("\n--- selections EV_modele > 0 : detail par (marche, selection) [si n>=20] ---")
by_sel = defaultdict(list)
for b in BETS:
    if b[5] > 0:
        by_sel[(b[1], b[2])].append(b)
for (mk, sel), lst in sorted(by_sel.items(), key=lambda kv: -len(kv[1])):
    if len(lst) < 20:
        continue
    evs = np.array([b[5] for b in lst]); odds = np.array([b[4] for b in lst])
    w = np.array([b[6] for b in lst], dtype=bool)
    roi = (w * odds).sum() / len(lst) - 1
    print(f"  {mk:14s} {sel:5s} n={len(lst):5d}  EV_mod={evs.mean():+.4f}  cote_med={np.median(odds):6.2f}  "
          f"ROI_realise={roi:+7.2%}  WR={w.mean():.3f}")

# ---------------------------------------------------------------- D. walk-forward
print(SEP); print("D. WALK-FORWARD 70/30 (split temporel)"); print(SEP)
BETS.sort(key=lambda b: b[0])
starts = sorted({b[0] for b in BETS})
cut = starts[int(len(starts) * 0.7)]
train = [b for b in BETS if b[0] <= cut]
oos = [b for b in BETS if b[0] > cut]
print(f"train: {len(train)} sel (jusqu'a {cut})   oos: {len(oos)} sel")

def roi_of(lst):
    if not lst:
        return 0.0, 0, 0.0, 0.0
    odds = np.array([b[4] for b in lst]); w = np.array([b[6] for b in lst], dtype=bool)
    return (w * odds).sum() / len(lst) - 1, len(lst), w.mean(), odds.mean()

for thr in (0.0, 0.02, 0.05):
    tr = [b for b in train if b[5] > thr]
    oo = [b for b in oos if b[5] > thr]
    r_tr, n_tr, _, _ = roi_of(tr)
    r_oo, n_oo, wr_oo, ao_oo = roi_of(oo)
    print(f"\nEV_modele > {thr:.2f} : train n={n_tr} ROI={r_tr:+7.2%}   OOS n={n_oo} ROI={r_oo:+7.2%} WR={wr_oo:.3f} cote_moy={ao_oo:.2f}")
    # par marche sur train (selection des marches train-positifs)
    bym = defaultdict(list)
    for b in tr:
        bym[b[1]].append(b)
    good = []
    for mk, lst in bym.items():
        r, n, _, _ = roi_of(lst)
        ev = np.mean([b[5] for b in lst])
        print(f"    train {mk:14s} n={n:5d} EV_mod={ev:+.4f} ROI={r:+7.2%}")
        if r > 0 and n >= 50:
            good.append(mk)
    if good:
        sel_oo = [b for b in oo if b[1] in good]
        r, n, wr, ao = roi_of(sel_oo)
        # bootstrap CI
        if n > 30:
            arr = np.array([(b[6] * b[4] - 1) for b in sel_oo])
            bs = [np.mean(np.random.default_rng(s).choice(arr, size=n)) for s in range(400)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            print(f"    >>> OOS marches train-positifs {good}: n={n} ROI={r:+7.2%} (CI95 [{lo:+.2%},{hi:+.2%}]) WR={wr:.3f} cote_moy={ao:.2f}")
