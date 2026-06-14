"""Walk-forward parlays v2 — tests cibles apres _wf_parlays.py.

Memes regles anti-leakage : dedupe, split temporel 70/30 a frontiere de round,
selection sur train uniquement, metriques OOS reportees avec bootstrap CI.

Focus :
  A. T1 par segment (selection train -> eval OOS) + parlays T1 segmentes
  B. Cellule robuste 'away cote [2.2,3.2) MS_early' (seule cellule durcie
     survivante OOS du v1) : single, x2 cross-round, + T1 meme round
  C. T1 haute proba (p>=0.70 / 0.75) en x2 / x3 -> objectif (a) et cote >= 2
  D. Verification independance au niveau parlay : WR observe vs produit des WR
  E. Bootstrap 95% CI sur les ROI OOS cles
"""
from __future__ import annotations

import itertools
import json
import sys
from collections import defaultdict

sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from scraper.config import load_settings

RNG = np.random.default_rng(7)


def load_data():
    engine = create_engine(load_settings().db_url)
    df = pd.read_sql(
        """
        SELECT e.id ev_id, e.team_a, e.team_b, e.expected_start, e.round_info,
               o.odds_home, o.odds_draw, o.odds_away, o.extra_markets,
               r.score_a, r.score_b, r.ht_score_a, r.ht_score_b
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
    df = df.drop_duplicates(subset=['team_a', 'team_b', 'expected_start'], keep='first')
    df = df.reset_index(drop=True)
    df['round_id'] = df['expected_start']
    return df


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
    if len(sub) == 0:
        return (float('nan'), float('nan'))
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


def group_by_round(rows):
    g = defaultdict(list)
    for r in rows:
        g[r['round_id']].append(r)
    return g


def seq_parlay(rows, k):
    """Parlays non chevauchants, jambes de rounds distincts, ordre chrono."""
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
    g = group_by_round(rows)
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


def cross_with(rows_a, rows_b):
    """1 jambe A + 1 jambe B du MEME round, matchs differents, 1 paire max
    par jambe A (la premiere B dispo)."""
    gb = group_by_round(rows_b)
    out = []
    for a in rows_a:
        for b in gb.get(a['round_id'], []):
            if b['ev_id'] == a['ev_id']:
                continue
            out.append(dict(won=int(a['won'] and b['won']), cote=a['cote'] * b['cote']))
            break
    return pd.DataFrame(out)


def main():
    df = load_data()
    rounds = sorted(df.round_id.unique())
    counts = df.round_id.value_counts()
    cum, cut = 0, rounds[-1]
    target = int(len(df) * 0.70)
    for rid in rounds:
        cum += counts[rid]
        if cum >= target:
            cut = rid
            break
    df['segment'] = df['round_info'].map(segment_of)
    tr_df, oo_df = df[df.round_id < cut], df[df.round_id >= cut]
    print(f'matchs={len(df)} train={len(tr_df)} oos={len(oo_df)} cut={cut}')

    # ---- jambes T1 (devig >= 0.65, cote <= 1.55) ----
    t1 = []
    for r in df.itertuples():
        sa, sb = int(r.score_a), int(r.score_b)
        out = '1' if sa > sb else ('X' if sa == sb else '2')
        imp = np.array([1 / r.odds_home, 1 / r.odds_draw, 1 / r.odds_away])
        p = imp / imp.sum()
        if p[0] >= 0.65 and r.odds_home <= 1.55:
            t1.append(dict(ev_id=r.ev_id, round_id=r.round_id, segment=r.segment,
                           cote=float(r.odds_home), won=int(out == '1'), p=float(p[0])))
        elif p[2] >= 0.65 and r.odds_away <= 1.55:
            t1.append(dict(ev_id=r.ev_id, round_id=r.round_id, segment=r.segment,
                           cote=float(r.odds_away), won=int(out == '2'), p=float(p[2])))
    t1 = pd.DataFrame(t1)
    t1_tr, t1_oo = t1[t1.round_id < cut], t1[t1.round_id >= cut]

    # ---- jambe FOCUS : away cote [2.2,3.2) en MS_early (cellule durcie v1) --
    foc = []
    for r in df.itertuples():
        if r.segment != 'MS_early' or not (2.2 <= r.odds_away < 3.2):
            continue
        sa, sb = int(r.score_a), int(r.score_b)
        foc.append(dict(ev_id=r.ev_id, round_id=r.round_id, segment=r.segment,
                        cote=float(r.odds_away), won=int(sb > sa)))
    foc = pd.DataFrame(foc)
    foc_tr, foc_oo = foc[foc.round_id < cut], foc[foc.round_id >= cut]

    # ============================ A. T1 par segment ==========================
    print('\n=== A. T1 PAR SEGMENT (train -> OOS) ===')
    seg_pos = []
    for seg in ['DS', 'MS_early', 'MS_mid', 'MS_late', 'FS']:
        s_tr = stat(f'T1 {seg} [train]', t1_tr[t1_tr.segment == seg])
        s_oo = stat(f'T1 {seg} [oos]  ', t1_oo[t1_oo.segment == seg])
        if s_tr and s_tr['roi'] > -0.02:
            seg_pos.append(seg)
    print(f'  segments T1 retenus sur train (roi>-2%) : {seg_pos}')
    if seg_pos:
        rows_oo = t1_oo[t1_oo.segment.isin(seg_pos)].to_dict('records')
        rows_tr = t1_tr[t1_tr.segment.isin(seg_pos)].to_dict('records')
        stat('T1 segsel single [train]', pd.DataFrame(rows_tr))
        stat('T1 segsel single [oos]  ', pd.DataFrame(rows_oo), ci=True)
        for k in (2, 3):
            stat(f'T1 segsel x{k} seq [train]', seq_parlay(rows_tr, k))
            stat(f'T1 segsel x{k} seq [oos]  ', seq_parlay(rows_oo, k), ci=True)

    # ============================ B. cellule FOCUS ===========================
    print('\n=== B. FOCUS away[2.2,3.2) MS_early (selection durcie sur train, v1) ===')
    stat('FOCUS single [train]', foc_tr)
    s = stat('FOCUS single [oos]  ', foc_oo, ci=True)
    rows_tr = foc_tr.to_dict('records')
    rows_oo = foc_oo.to_dict('records')
    stat('FOCUS x2 seq [train]', seq_parlay(rows_tr, 2))
    stat('FOCUS x2 seq [oos]  ', seq_parlay(rows_oo, 2), ci=True)
    # FOCUS + T1 meme round
    stat('FOCUS + T1 meme round [train]', cross_with(rows_tr, t1_tr.to_dict('records')))
    stat('FOCUS + T1 meme round [oos]  ', cross_with(rows_oo, t1_oo.to_dict('records')), ci=True)
    # FOCUS + T1 p>=0.75 meme round (jambe la plus sure)
    t1h_tr = t1_tr[t1_tr.p >= 0.75].to_dict('records')
    t1h_oo = t1_oo[t1_oo.p >= 0.75].to_dict('records')
    stat('FOCUS + T1p75 meme round [train]', cross_with(rows_tr, t1h_tr))
    stat('FOCUS + T1p75 meme round [oos]  ', cross_with(rows_oo, t1h_oo), ci=True)

    # variante : away [2.2,3.2) TOUTES saisons (pooled, vu OOS +3.9% en v1 —
    # mais selection sur train : roi train ?)
    print('\n--- variante pooled away[2.2,3.2) (verif train avant lecture OOS) ---')
    pool = []
    for r in df.itertuples():
        if not (2.2 <= r.odds_away < 3.2):
            continue
        sa, sb = int(r.score_a), int(r.score_b)
        pool.append(dict(ev_id=r.ev_id, round_id=r.round_id,
                         cote=float(r.odds_away), won=int(sb > sa)))
    pool = pd.DataFrame(pool)
    stat('away[2.2,3.2) pooled [train]', pool[pool.round_id < cut])
    stat('away[2.2,3.2) pooled [oos]  ', pool[pool.round_id >= cut], ci=True)

    # ============================ C. T1 haute proba parlays ==================
    print('\n=== C. T1 HAUTE PROBA EN PARLAY (objectif accuracy / cote>=2) ===')
    for pmin in (0.70, 0.75):
        rt = t1_tr[t1_tr.p >= pmin].to_dict('records')
        ro = t1_oo[t1_oo.p >= pmin].to_dict('records')
        for k in (2, 3):
            stat(f'T1 p>={pmin:.2f} x{k} seq [train]', seq_parlay(rt, k))
            stat(f'T1 p>={pmin:.2f} x{k} seq [oos]  ', seq_parlay(ro, k), ci=True)

    # ============================ D. independance niveau parlay ==============
    print('\n=== D. WR PARLAY OBSERVE vs PRODUIT DES WR INDIVIDUELS (OOS) ===')
    wr1 = t1_oo.won.mean()
    for k in (2, 3, 4):
        obs = same_round_parlay(t1_oo.to_dict('records'), k)
        if len(obs):
            print(f'  T1 x{k} meme round : observe={obs.won.mean()*100:5.1f}%  '
                  f'theorique={wr1**k*100:5.1f}%  n={len(obs)}')
    # ROI attendu d un parlay = produit des (1+roi) des jambes - 1
    roi1 = roi_of(t1_oo)
    print(f'  ROI T1 single OOS = {roi1*100:+.1f}%  ->  attendu x2={((1+roi1)**2-1)*100:+.1f}%  '
          f'x3={((1+roi1)**3-1)*100:+.1f}%  x4={((1+roi1)**4-1)*100:+.1f}%')

    # ============================ E. CI sur findings v1 ======================
    print('\n=== E. BOOTSTRAP CI95 SUR LES PARLAYS T1 v1 (OOS) ===')
    stat('T1 x2 meme round [oos]', same_round_parlay(t1_oo.to_dict('records'), 2), ci=True)
    stat('T1 x3 meme round [oos]', same_round_parlay(t1_oo.to_dict('records'), 3), ci=True)
    stat('T1 x4 meme round [oos]', same_round_parlay(t1_oo.to_dict('records'), 4), ci=True)


if __name__ == '__main__':
    main()
