# -*- coding: utf-8 -*-
"""WF3 - FACETTE HT/FT, partie 6 : regle pre-declaree 'longshot Mi-tps 1X2'
(devig p_mkt < 0.08, sans filtre EV) -> test exact + walk-forward 70/30.
+ replication sur HT/FT (combos longshot p_mkt<0.08) pour voir si le biais est
   propre au marche Mi-tps 1X2 ou general aux longshots de la famille.
"""
import sys, json, math
from collections import defaultdict
sys.path.insert(0, '.')
from scraper.config import load_settings
from sqlalchemy import create_engine, text
import numpy as np

eng = create_engine(load_settings().db_url)
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


def poisson_binom_sf(ps, k_obs):
    pmf = np.zeros(len(ps) + 1)
    pmf[0] = 1.0
    for p in ps:
        pmf[1:] = pmf[1:] * (1 - p) + pmf[:-1] * p
        pmf[0] *= (1 - p)
    return float(pmf[k_obs:].sum())


def load():
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
    seen, out = set(), []
    with eng.connect() as c:
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
            out.append(dict(start=r[1], em=em, h=h, a=a, hh=hh, ha=ha))
    return out

M = load()
print(f"matchs : {len(M)}")
COMBOS = ['1/1', '1/X', '1/2', 'X/1', 'X/X', 'X/2', '2/1', '2/X', '2/2']


def run_rule(label, market_key, keys, outcome_of, pmax):
    bets = []
    for m in M:
        d = m['em'].get(market_key)
        if not isinstance(d, dict):
            continue
        dv, _ = devig(d, keys=keys)
        if not dv:
            continue
        obs = outcome_of(m)
        for sel, p in dv.items():
            if p < pmax:
                o = float(d[sel])
                if o >= 99.5:      # exclut cotes cappees
                    continue
                bets.append((m['start'], sel, p, o, sel == obs))
    bets.sort(key=lambda b: b[0])
    n = len(bets)
    if n == 0:
        return
    w = sum(b[4] for b in bets)
    roi = sum(b[3] for b in bets if b[4]) / n - 1
    sf_mkt = poisson_binom_sf([b[2] for b in bets], w)
    sf_be = poisson_binom_sf([1 / b[3] for b in bets], w)
    print(f"\n{label} (p_mkt<{pmax}) : n={n} wins={w} ROI={roi:+.2%} cote_moy={np.mean([b[3] for b in bets]):.2f}")
    print(f"   E_mkt={sum(b[2] for b in bets):.1f}  P(W>=w|mkt)={sf_mkt:.4f}   "
          f"E_be={sum(1/b[3] for b in bets):.1f}  P(W>=w|break-even)={sf_be:.4f}")
    cut = n * 7 // 10
    for lbl, sub in [('train70', bets[:cut]), ('OOS30  ', bets[cut:])]:
        ww = sum(b[4] for b in sub)
        rr = sum(b[3] for b in sub if b[4]) / len(sub) - 1
        sf = poisson_binom_sf([1 / b[3] for b in sub], ww)
        print(f"   {lbl}: n={len(sub)} wins={ww} ROI={rr:+.2%} P(W>=w|be)={sf:.3f}")


run_rule("Mi-tps 1X2 longshot", 'Mi-tps 1X2', {'1', 'X', '2'},
         lambda m: res_of(m['hh'], m['ha']), 0.08)
run_rule("Mi-tps 1X2 longshot strict", 'Mi-tps 1X2', {'1', 'X', '2'},
         lambda m: res_of(m['hh'], m['ha']), 0.07)
run_rule("HT/FT longshot", 'HT/FT', set(COMBOS),
         lambda m: f"{res_of(m['hh'], m['ha'])}/{res_of(m['h'], m['a'])}", 0.08)
