"""Walk-forward parlays v3 — stress-test de la cellule FOCUS (away 2.2-3.2).

Questions :
  1. away[2.2,3.2) par segment sur TRAIN — la cellule MS_early est-elle un pic
     isole ou un gradient debut de saison ?
  2. variante elargie DS+MS_early (decision prise sur train uniquement)
  3. FOCUS x2 MEME round (plusieurs aways 2.2-3.2 par round MS_early ?)
  4. coherence temporelle : OOS coupe en 2 moities
  5. courbe d equity OOS du FOCUS single (drawdown max)
"""
from __future__ import annotations

import itertools
import sys
from collections import defaultdict

sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from scraper.config import load_settings

RNG = np.random.default_rng(7)


def segment_of(round_info):
    try:
        j = int(round_info)
    except (TypeError, ValueError):
        return '?'
    if j <= 3:
        return 'DS'
    if j <= 12:
        return 'MS_early'
    if j <= 25:
        return 'MS_mid'
    if j <= 33:
        return 'MS_late'
    return 'FS'


def roi_of(sub):
    return (sub.won * (sub.cote - 1) - (1 - sub.won)).mean()


def boot_ci(sub, n_boot=4000):
    pnl = (sub.won * (sub.cote - 1) - (1 - sub.won)).to_numpy()
    idx = RNG.integers(0, len(pnl), size=(n_boot, len(pnl)))
    means = pnl[idx].mean(axis=1)
    return (np.percentile(means, 2.5), np.percentile(means, 97.5))


def stat(name, sub, ci=False):
    n = len(sub)
    if n == 0:
        print(f'  {name:<46} n=0')
        return None
    wr, cote, roi = sub.won.mean(), sub.cote.mean(), roi_of(sub)
    extra = ''
    if ci:
        lo, hi = boot_ci(sub)
        extra = f'  CI95=[{lo*100:+.1f}%,{hi*100:+.1f}%]'
    flag = '' if n >= 30 else '  [INSTABLE n<30]'
    print(f'  {name:<46} n={n:<5} wr={wr*100:5.1f}%  cote={cote:5.2f}  ROI={roi*100:+6.1f}%{extra}{flag}')
    return dict(n=n, wr=wr, cote=cote, roi=roi)


def seq_parlay(rows, k):
    rows = sorted(rows, key=lambda x: (x['round_id'], x['ev_id']))
    out, buf, used = [], [], set()
    for leg in rows:
        if leg['round_id'] in used:
            continue
        buf.append(leg); used.add(leg['round_id'])
        if len(buf) == k:
            out.append(dict(won=int(all(l['won'] for l in buf)),
                            cote=float(np.prod([l['cote'] for l in buf]))))
            buf, used = [], set()
    return pd.DataFrame(out)


def same_round_parlay(rows, k, max_combos=6):
    g = defaultdict(list)
    for r in rows:
        g[r['round_id']].append(r)
    out = []
    for rid, ls in g.items():
        if len(ls) < k:
            continue
        combos = list(itertools.combinations(ls, k))
        if len(combos) > max_combos:
            idx = RNG.choice(len(combos), size=max_combos, replace=False)
            combos = [combos[i] for i in idx]
        for combo in combos:
            if len({c['ev_id'] for c in combo}) < k:
                continue
            out.append(dict(won=int(all(c['won'] for c in combo)),
                            cote=float(np.prod([c['cote'] for c in combo]))))
    return pd.DataFrame(out)


def main():
    engine = create_engine(load_settings().db_url)
    df = pd.read_sql(
        """
        SELECT e.id ev_id, e.team_a, e.team_b, e.expected_start, e.round_info,
               o.odds_home, o.odds_draw, o.odds_away,
               r.score_a, r.score_b
        FROM events e
        JOIN odds_snapshots o
             ON o.id = (SELECT MIN(id) FROM odds_snapshots WHERE event_id = e.id)
        JOIN results r ON r.event_id = e.id
        WHERE r.score_a IS NOT NULL AND r.score_b IS NOT NULL
              AND o.odds_home IS NOT NULL AND o.odds_draw IS NOT NULL
              AND o.odds_away IS NOT NULL
        ORDER BY e.expected_start, e.id
        """,
        engine,
    )
    df = df.drop_duplicates(subset=['team_a', 'team_b', 'expected_start'], keep='first').reset_index(drop=True)
    df['round_id'] = df['expected_start']
    df['segment'] = df['round_info'].map(segment_of)

    rounds = sorted(df.round_id.unique())
    counts = df.round_id.value_counts()
    cum, cut = 0, rounds[-1]
    for rid in rounds:
        cum += counts[rid]
        if cum >= int(len(df) * 0.70):
            cut = rid
            break

    aw = []
    for r in df.itertuples():
        if not (2.2 <= r.odds_away < 3.2):
            continue
        aw.append(dict(ev_id=r.ev_id, round_id=r.round_id, segment=r.segment,
                       cote=float(r.odds_away), won=int(int(r.score_b) > int(r.score_a))))
    aw = pd.DataFrame(aw)
    aw_tr, aw_oo = aw[aw.round_id < cut], aw[aw.round_id >= cut]

    print('=== 1. away[2.2,3.2) PAR SEGMENT (train | oos) ===')
    for seg in ['DS', 'MS_early', 'MS_mid', 'MS_late', 'FS']:
        stat(f'away223 {seg} [train]', aw_tr[aw_tr.segment == seg])
        stat(f'away223 {seg} [oos]  ', aw_oo[aw_oo.segment == seg])

    print('\n=== 2. VARIANTE ELARGIE DS+MS_early (J1-J12) ===')
    early = ('DS', 'MS_early')
    e_tr = aw_tr[aw_tr.segment.isin(early)]
    e_oo = aw_oo[aw_oo.segment.isin(early)]
    stat('away223 J1-12 [train]', e_tr)
    stat('away223 J1-12 [oos]  ', e_oo, ci=True)
    stat('away223 J1-12 x2 seq [train]', seq_parlay(e_tr.to_dict('records'), 2))
    stat('away223 J1-12 x2 seq [oos]  ', seq_parlay(e_oo.to_dict('records'), 2), ci=True)
    stat('away223 J1-12 x2 SAME round [train]', same_round_parlay(e_tr.to_dict('records'), 2))
    stat('away223 J1-12 x2 SAME round [oos]  ', same_round_parlay(e_oo.to_dict('records'), 2), ci=True)

    print('\n=== 3. FOCUS MS_early seul : x2 meme round ===')
    f_tr = aw_tr[aw_tr.segment == 'MS_early']
    f_oo = aw_oo[aw_oo.segment == 'MS_early']
    stat('FOCUS x2 SAME round [train]', same_round_parlay(f_tr.to_dict('records'), 2))
    stat('FOCUS x2 SAME round [oos]  ', same_round_parlay(f_oo.to_dict('records'), 2), ci=True)

    print('\n=== 4. COHERENCE TEMPORELLE FOCUS single (4 tranches) ===')
    tr_rounds = sorted(f_tr.round_id.unique())
    h_cut = tr_rounds[len(tr_rounds) // 2]
    oo_rounds = sorted(f_oo.round_id.unique())
    o_cut = oo_rounds[len(oo_rounds) // 2]
    stat('FOCUS train H1', f_tr[f_tr.round_id < h_cut])
    stat('FOCUS train H2', f_tr[f_tr.round_id >= h_cut])
    stat('FOCUS oos H1  ', f_oo[f_oo.round_id < o_cut])
    stat('FOCUS oos H2  ', f_oo[f_oo.round_id >= o_cut])
    # idem variante elargie
    e_oo_s = e_oo.sort_values('round_id')
    eo_rounds = sorted(e_oo.round_id.unique())
    eo_cut = eo_rounds[len(eo_rounds) // 2]
    stat('J1-12 oos H1  ', e_oo_s[e_oo_s.round_id < eo_cut])
    stat('J1-12 oos H2  ', e_oo_s[e_oo_s.round_id >= eo_cut])

    print('\n=== 5. EQUITY OOS FOCUS single (1u/pari) ===')
    pnl = (f_oo.sort_values('round_id').won * (f_oo.sort_values('round_id').cote - 1)
           - (1 - f_oo.sort_values('round_id').won)).to_numpy()
    eq = np.cumsum(pnl)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak).min()
    print(f'  n={len(pnl)}  pnl_total={eq[-1]:+.1f}u  max_drawdown={dd:+.1f}u  '
          f'plus_longue_serie_pertes={max(len(list(g)) for k, g in itertools.groupby(pnl < 0) if k)}')

    print('\n=== 6. CONTROLE NEGATIF : home[2.2,3.2) MS_early (meme bucket, autre cote) ===')
    hm = []
    for r in df.itertuples():
        if r.segment != 'MS_early' or not (2.2 <= r.odds_home < 3.2):
            continue
        hm.append(dict(round_id=r.round_id, cote=float(r.odds_home),
                       won=int(int(r.score_a) > int(r.score_b))))
    hm = pd.DataFrame(hm)
    stat('home223 MS_early [train]', hm[hm.round_id < cut])
    stat('home223 MS_early [oos]  ', hm[hm.round_id >= cut])


if __name__ == '__main__':
    main()
