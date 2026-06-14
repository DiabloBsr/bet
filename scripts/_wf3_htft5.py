# -*- coding: utf-8 -*-
"""WF3 - FACETTE HT/FT, partie 5 : VERDICT FINAL sur les deux imperfections candidates.

1. Poche 'Mi-tps 1X2 EV_mod>0' : tests EXACTS Poisson-binomial (vs p_modele et vs
   p_marche devig), stabilite temporelle, et calibration du marche Mi-tps 1X2 par
   bucket de cote (biais favori-outsider ?).
2. 'Score exact' : mecanisme du cap de cote a 100 -> combien de cellules cappees,
   ou va la masse au devig, EV reel de parier les cellules ou conv > devig.
3. Recap parametres moteur (p_MT1, ratio lambda MT2/MT1, marges).
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


def poisson_binom_sf(ps, k_obs):
    """P(W >= k_obs) exacte, W = somme de Bernoulli(ps). DP."""
    pmf = np.zeros(len(ps) + 1)
    pmf[0] = 1.0
    for p in ps:
        pmf[1:] = pmf[1:] * (1 - p) + pmf[:-1] * p
        pmf[0] *= (1 - p)
    return float(pmf[k_obs:].sum()), pmf


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
            g1n, mg1 = devig({f"{c[0]}-{c[1]}": g1d[f"{c[0]}-{c[1]}"] for c in CELLS})
            g2n, mg2 = devig({f"{c[0]}-{c[1]}": g2d[f"{c[0]}-{c[1]}"] for c in CELLS})
            if g1n is None or g2n is None:
                continue
            g1 = {parse_cell(k): v for k, v in g1n.items()}
            g2 = {parse_cell(k): v for k, v in g2n.items()}
            out.append(dict(id=r[0], start=r[2], oh=r[5], od=r[6], oa=r[7], em=em,
                            h=h, a=a, hh=hh, ha=ha, h2h=h - hh, h2a=a - ha,
                            g1=g1, g2=g2, g1raw=g1d, g2raw=g2d, mg1=mg1, mg2=mg2))
    return out

M = load()
print(f"matchs : {len(M)}")
for m in M:
    m['pht'] = {'1': sum(p for c, p in m['g1'].items() if c[0] > c[1]),
                'X': sum(p for c, p in m['g1'].items() if c[0] == c[1]),
                '2': sum(p for c, p in m['g1'].items() if c[0] < c[1])}

# ================================================================ 1. poche Mi-tps 1X2
print(SEP); print("1. POCHE 'Mi-tps 1X2 EV_mod>0' : TESTS EXACTS"); print(SEP)
bets = []
for m in M:
    d = m['em'].get('Mi-tps 1X2')
    if not isinstance(d, dict):
        continue
    dv, _ = devig(d, keys={'1', 'X', '2'})
    if not dv:
        continue
    obs = res_of(m['hh'], m['ha'])
    for sel in '1X2':
        try:
            o = float(d[sel])
        except (TypeError, ValueError, KeyError):
            continue
        pm = m['pht'][sel]
        if pm * o - 1 > 0:
            bets.append(dict(start=m['start'], sel=sel, pm=pm, pk=dv[sel], o=o,
                             won=sel == obs, id=m['id']))
bets.sort(key=lambda b: b['start'])
n = len(bets); w = sum(b['won'] for b in bets)
ps_mod = [b['pm'] for b in bets]; ps_mkt = [b['pk'] for b in bets]
ps_raw = [1.0 / b['o'] for b in bets]   # break-even prob
sf_mod, _ = poisson_binom_sf(ps_mod, w)
sf_mkt, _ = poisson_binom_sf(ps_mkt, w)
sf_raw, _ = poisson_binom_sf(ps_raw, w)
roi = sum(b['o'] for b in bets if b['won']) / n - 1
print(f"n={n} wins={w} ROI={roi:+.2%} cote_moy={np.mean([b['o'] for b in bets]):.2f}")
print(f"  E_mod={sum(ps_mod):.2f}  P(W>={w} | modele G1)  = {sf_mod:.4f}")
print(f"  E_mkt={sum(ps_mkt):.2f}  P(W>={w} | marche devig)= {sf_mkt:.4f}")
print(f"  E_raw={sum(ps_raw):.2f}  P(W>={w} | break-even)  = {sf_raw:.4f}  (<0.05 => ROI>0 significatif)")

# stabilite temporelle : 3 tiers
print("\n  par tiers temporel :")
for i in range(3):
    sub = bets[i * n // 3:(i + 1) * n // 3]
    ww = sum(b['won'] for b in sub)
    rr = sum(b['o'] for b in sub if b['won']) / len(sub) - 1
    print(f"    tiers {i+1}: n={len(sub)} wins={ww} E_mod={sum(b['pm'] for b in sub):.1f} ROI={rr:+.2%}")

# par selection
print("\n  par selection :")
for sel in '12X':
    sub = [b for b in bets if b['sel'] == sel]
    if not sub:
        continue
    ww = sum(b['won'] for b in sub)
    sf_m, _ = poisson_binom_sf([b['pm'] for b in sub], ww)
    rr = sum(b['o'] for b in sub if b['won']) / len(sub) - 1
    print(f"    {sel}: n={len(sub)} wins={ww} E_mod={sum(b['pm'] for b in sub):.1f} "
          f"P(W>=w|mod)={sf_m:.3f} ROI={rr:+.2%} cote_moy={np.mean([b['o'] for b in sub]):.2f}")

# la poche est-elle un artefact 'G1 intrus' ? -> verite terrain par replication :
# matchs APPARIES hors-poche avec meme p_mkt : freq obs vs p_mkt
print("\n  >>> CALIBRATION du marche 'Mi-tps 1X2' par bucket de proba devig (TOUTES sel.):")
rows = []
for m in M:
    d = m['em'].get('Mi-tps 1X2')
    if not isinstance(d, dict):
        continue
    dv, _ = devig(d, keys={'1', 'X', '2'})
    if not dv:
        continue
    obs = res_of(m['hh'], m['ha'])
    for sel in '1X2':
        rows.append((dv[sel], m['pht'][sel], float(d[sel]), sel == obs))
rows = np.array(rows, dtype=float)
edges = [0, .05, .08, .12, .20, .30, .45, .60, 1.0]
print("  bucket p_mkt      n      p_mkt_moy  p_G1_moy   freq_obs   [CI95 binom]      EV@cote")
for lo, hi in zip(edges[:-1], edges[1:]):
    sub = rows[(rows[:, 0] >= lo) & (rows[:, 0] < hi)]
    if len(sub) < 50:
        continue
    f = sub[:, 3].mean()
    nn = len(sub)
    ci = stats.binom.interval(0.95, nn, sub[:, 0].mean())
    ev = (sub[:, 3] * sub[:, 2]).mean() - 1
    flag = ' <<<' if not (ci[0] <= sub[:, 3].sum() <= ci[1]) else ''
    print(f"  [{lo:.2f},{hi:.2f})  {nn:6d}   {sub[:,0].mean():.4f}    {sub[:,1].mean():.4f}    {f:.4f}   "
          f"[{ci[0]/nn:.4f},{ci[1]/nn:.4f}]   {ev:+.3f}{flag}")

# ================================================================ 2. Score exact : cap 100
print(SEP); print("2. 'SCORE EXACT' : LE CAP DE COTE A 100"); print(SEP)
n_grids = 0; capped_counts = Counter(); cell_capped = Counter()
ev_capped, ev_uncapped_tail = [], []
bets_conv_pos = []
for m in M:
    d = m['em'].get('Score exact')
    if not isinstance(d, dict):
        continue
    try:
        vals = {k: float(v) for k, v in d.items()}
    except (TypeError, ValueError):
        continue
    n_grids += 1
    cv = convolve(m['g1'], m['g2'])
    ncap = sum(1 for v in vals.values() if v >= 99.5)
    capped_counts[ncap] += 1
    for k, v in vals.items():
        try:
            c = parse_cell(k)
        except Exception:
            continue
        p = cv.get(c, 0.0)
        ev = p * v - 1
        if v >= 99.5:
            cell_capped[k] += 1
            ev_capped.append(ev)
        if ev > 0:
            bets_conv_pos.append((m['start'], k, p, v, (m['h'], m['a']) == c))
print(f"grilles SE analysees : {n_grids}")
print(f"nb cellules cappees a ~100 par grille : {dict(sorted(capped_counts.items()))}")
print(f"cellules les plus souvent cappees : {cell_capped.most_common(8)}")
print(f"EV(conv) des cellules cappees : mean={np.mean(ev_capped):+.4f} "
      f"frac EV>0 = {np.mean(np.array(ev_capped) > 0):.3f}")

bets_conv_pos.sort(key=lambda b: b[0])
nb = len(bets_conv_pos)
wb = sum(b[4] for b in bets_conv_pos)
ps = [b[2] for b in bets_conv_pos]
ps_be = [1 / b[3] for b in bets_conv_pos]
sf_m, _ = poisson_binom_sf(ps, wb)
sf_be, _ = poisson_binom_sf(ps_be, wb)
roi = sum(b[3] for b in bets_conv_pos if b[4]) / nb - 1
print(f"\nbets 'SE cellule EV_conv>0' : n={nb} wins={wb} E_mod={sum(ps):.1f} E_be={sum(ps_be):.1f}")
print(f"  P(W>=w|conv)={sf_m:.3f}   P(W>=w|break-even)={sf_be:.3f}   ROI={roi:+.2%}")
cut = nb * 7 // 10
tr, oo = bets_conv_pos[:cut], bets_conv_pos[cut:]
for lbl, sub in [('train70', tr), ('OOS30  ', oo)]:
    ww = sum(b[4] for b in sub)
    rr = sum(b[3] for b in sub if b[4]) / len(sub) - 1 if sub else 0
    print(f"  {lbl}: n={len(sub)} wins={ww} E_mod={sum(b[2] for b in sub):.1f} ROI={rr:+.2%}")

# ================================================================ 3. recap parametres
print(SEP); print("3. RECAP PARAMETRES MOTEUR"); print(SEP)
lam1h = [sum(p * c[0] for c, p in m['g1'].items()) for m in M]
lam1a = [sum(p * c[1] for c, p in m['g1'].items()) for m in M]
lam2h = [sum(p * c[0] for c, p in m['g2'].items()) for m in M]
lam2a = [sum(p * c[1] for c, p in m['g2'].items()) for m in M]
tot1 = np.array(lam1h) + np.array(lam1a); tot2 = np.array(lam2h) + np.array(lam2a)
print(f"E[buts MT1] grille : median={np.median(tot1):.3f}  E[buts MT2] : {np.median(tot2):.3f} "
      f" ratio global={np.median(tot2/tot1):.4f}")
gh1 = sum(m['hh'] + m['ha'] for m in M); gtot = sum(m['h'] + m['a'] for m in M)
print(f"frac buts reels en MT1 : {gh1/gtot:.4f}  ({gh1}/{gtot})")
print(f"marges medianes : G1={np.median([m['mg1'] for m in M]):.4f} G2={np.median([m['mg2'] for m in M]):.4f}")
ht1 = sum(1 for m in M if m['hh'] + m['ha'] == 3)
h2_3 = sum(1 for m in M if m['h2h'] + m['h2a'] == 3)
print(f"frac matchs au cap (3 buts) : MT1={ht1/len(M):.4f}  MT2={h2_3/len(M):.4f}")
print(f"frac FT total=6 (double cap) : {sum(1 for m in M if m['h']+m['a']==6)/len(M):.4f}")
