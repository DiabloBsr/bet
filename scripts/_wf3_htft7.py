# -*- coding: utf-8 -*-
"""WF3 - FACETTE HT/FT, partie 7 : cross-market sur la poche longshot Mi-tps 1X2.
Sur les memes selections (p_mkt(1X2MT)<0.08), que disent les AUTRES marches de la
proba HT : G1 ('Mi-tps CS'), HT/FT (somme sel/*), Mi-tps DC (systeme) ?
Si tous s'accordent ~0.07 et obs=0.11, c'est du bruit ; si les autres marches
pointaient plus haut, la cote 1X2-MT est structurellement fausse -> edge reel.
"""
import sys, json
sys.path.insert(0, '.')
from scraper.config import load_settings
from sqlalchemy import create_engine, text
import numpy as np

eng = create_engine(load_settings().db_url)
CELLS = [(i, j) for i in range(4) for j in range(4) if i + j <= 3]
COMBOS = ['1/1', '1/X', '1/2', 'X/1', 'X/X', 'X/2', '2/1', '2/X', '2/2']


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


def poisson_binom_sf(ps, k_obs):
    pmf = np.zeros(len(ps) + 1)
    pmf[0] = 1.0
    for p in ps:
        pmf[1:] = pmf[1:] * (1 - p) + pmf[:-1] * p
        pmf[0] *= (1 - p)
    return float(pmf[k_obs:].sum())


q = text('''
    SELECT e.id, e.expected_start, e.team_a, e.team_b, o.extra_markets,
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
seen, M = set(), []
with create_engine(load_settings().db_url).connect() as c:
    for r in c.execute(q):
        key = (r[2], r[3], r[1])
        if key in seen:
            continue
        seen.add(key)
        em = r[4]
        if isinstance(em, str):
            try:
                em = json.loads(em)
            except Exception:
                em = {}
        em = em or {}
        h, a, hh, ha = int(r[5]), int(r[6]), int(r[7]), int(r[8])
        if hh > h or ha > a:
            continue
        gj = r[9]
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
        M.append(dict(em=em, hh=hh, ha=ha))

rows = []
for m in M:
    em = m['em']
    d = em.get('Mi-tps 1X2')
    if not isinstance(d, dict):
        continue
    dv, _ = devig(d, keys={'1', 'X', '2'})
    if not dv:
        continue
    g1d = em.get('Mi-tps CS')
    htft = em.get('HT/FT')
    dc = em.get('Mi-tps DC')
    obs = res_of(m['hh'], m['ha'])
    for sel in '12':
        if dv[sel] >= 0.08 or float(d[sel]) >= 99.5:
            continue
        p_g1 = p_htft = p_dc = None
        if isinstance(g1d, dict):
            gv, _ = devig(g1d)
            if gv:
                p_g1 = sum(p for k, p in gv.items() if res_of(*parse_cell(k)) == sel)
        if isinstance(htft, dict):
            hv, _ = devig(htft, keys=set(COMBOS))
            if hv:
                p_htft = sum(p for k, p in hv.items() if k.startswith(sel + '/'))
        # DC : floor a ~0.97-1.00 sur la paire favorite dans ces matchs -> on
        # devigue quand meme (info = la paire qui CONTIENT sel, pas trop floored)
        if isinstance(dc, dict):
            try:
                imp = {k: 1.0 / max(float(v), 0.90) for k, v in dc.items()}
                s = sum(imp.values())
                cv = {k: v / s for k, v in imp.items()}
                if sel == '1':
                    p_dc = cv['1X'] + cv['12'] - cv['X2']
                else:
                    p_dc = cv['X2'] + cv['12'] - cv['1X']
            except Exception:
                p_dc = None
        rows.append((dv[sel], p_g1, p_htft, p_dc, float(d[sel]), sel == obs))

rows = [r for r in rows if None not in r[:3]]
n = len(rows); w = sum(r[5] for r in rows)
print(f"selections longshot 1X2-MT (p<0.08, G1+HT/FT dispos) : n={n} wins={w} obs_freq={w/n:.4f}")
labels = ['p_mkt(1X2MT)', 'p_G1(CS)', 'p_HT/FT', 'p_DC(syst,floor)']
for i, lbl in enumerate(labels):
    ps = [r[i] for r in rows if r[i] is not None]
    ww = sum(r[5] for r in rows if r[i] is not None)
    sf = poisson_binom_sf(ps, ww)
    print(f"  {lbl:16s} n={len(ps):3d} moy={np.mean(ps):.4f}  E[w]={sum(ps):5.1f}  P(W>={ww}|cette source)={sf:.4f}")
roi = sum(r[4] for r in rows if r[5]) / n - 1
print(f"ROI cote brute = {roi:+.2%}")
print("\n=> si TOUTES les sources donnent P(W>=w) < ~0.05, l'anomalie est partagee (bruit du tirage improbable)")
print("   si seule 1X2-MT est en dessous, sa cote est structurellement trop haute (edge reel)")
