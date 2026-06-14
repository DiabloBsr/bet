# -*- coding: utf-8 -*-
"""
Walk-forward HT/FT + marches mi-temps (jamais exploites par le systeme actuel).

Methodo anti-leakage :
- tri par expected_start, train = premiers 70%, OOS = derniers 30%
- toute selection de cellule (combo x bracket x segment) se fait sur le TRAIN uniquement
- on rapporte UNIQUEMENT les metriques OOS (n_oos, wr_oos, avg_cote, roi_oos)
- roi = mean(won*(cote-1) - (1-won))   [mise 1 unite par pick]
"""
import sys, json
sys.path.insert(0, '.')
from collections import defaultdict
from scraper.config import load_settings
from sqlalchemy import create_engine, text

SEGMENTS = [(1, 3, 'DS'), (4, 12, 'MS_early'), (13, 25, 'MS_mid'),
            (26, 33, 'MS_late'), (34, 38, 'FS')]

def seg_of(rnd):
    for lo, hi, name in SEGMENTS:
        if lo <= rnd <= hi:
            return name
    return None

HOME_BRACKETS = [(1.0, 1.25, 'H<=1.25'), (1.25, 1.45, 'H1.25-1.45'),
                 (1.45, 1.70, 'H1.45-1.70'), (1.70, 2.00, 'H1.70-2.00'),
                 (2.00, 2.50, 'H2.00-2.50'), (2.50, 3.50, 'H2.50-3.50'),
                 (3.50, 99.0, 'H>3.50')]

def bracket_of(odds, brackets):
    for lo, hi, name in brackets:
        if lo <= odds < hi:
            return name
    return None

def load_matches():
    eng = create_engine(load_settings().db_url)
    q = text('''
        SELECT e.id, e.round_info, e.expected_start, e.team_a, e.team_b,
               o.odds_home, o.odds_draw, o.odds_away, o.extra_markets,
               r.score_a, r.score_b, r.ht_score_a, r.ht_score_b
        FROM events e
        JOIN results r ON r.event_id = e.id
        JOIN odds_snapshots o ON o.event_id = e.id
         AND o.id = (SELECT MIN(id) FROM odds_snapshots WHERE event_id = e.id)
        WHERE e.round_info != '0'
          AND r.ht_score_a IS NOT NULL AND r.ht_score_b IS NOT NULL
          AND r.score_a IS NOT NULL AND r.score_b IS NOT NULL
          AND o.extra_markets IS NOT NULL
          AND o.odds_home IS NOT NULL AND o.odds_draw IS NOT NULL AND o.odds_away IS NOT NULL
        ORDER BY e.expected_start ASC, e.id ASC
    ''')
    rows = []
    with eng.connect() as c:
        for r in c.execute(q):
            try:
                rnd = int(r[1])
            except (TypeError, ValueError):
                continue
            if not (1 <= rnd <= 38):
                continue
            em = r[8]
            if isinstance(em, str):
                try:
                    em = json.loads(em)
                except Exception:
                    continue
            if not isinstance(em, dict):
                continue
            sa, sb, ha, hb = int(r[9]), int(r[10]), int(r[11]), int(r[12])
            ht = '1' if ha > hb else ('2' if hb > ha else 'X')
            ft = '1' if sa > sb else ('2' if sb > sa else 'X')
            rows.append({
                'id': r[0], 'round': rnd, 'seg': seg_of(rnd), 'start': r[2],
                'team_a': r[3], 'team_b': r[4],
                'oh': float(r[5]), 'od': float(r[6]), 'oa': float(r[7]),
                'em': em,
                'sa': sa, 'sb': sb, 'ha': ha, 'hb': hb,
                'ht': ht, 'ft': ft, 'htft': ht + '/' + ft,
                'ht_cs': f'{ha}-{hb}', 'tot': sa + sb,
            })
    return rows

def roi_stats(picks):
    """picks = list of (won_bool, odds). Returns n, wr, avg_odds, roi."""
    n = len(picks)
    if n == 0:
        return 0, 0.0, 0.0, 0.0
    wins = sum(1 for w, _ in picks if w)
    avg_o = sum(o for _, o in picks) / n
    roi = sum((o - 1.0) if w else -1.0 for w, o in picks) / n
    return n, wins / n, avg_o, roi

def fmt(name, n, wr, ao, roi, extra=''):
    return f'{name:<58s} n={n:<5d} wr={wr*100:5.1f}% cote={ao:5.2f} roi={roi*100:+6.1f}% {extra}'

# ---------------------------------------------------------------- pick builders
def pick_htft(m, combo):
    mk = m['em'].get('HT/FT')
    if not isinstance(mk, dict):
        return None
    o = mk.get(combo)
    if not o:
        return None
    return (m['htft'] == combo, float(o))

def pick_mt1x2(m, sel):
    mk = m['em'].get('Mi-tps 1X2')
    if not isinstance(mk, dict):
        return None
    o = mk.get(sel)
    if not o:
        return None
    return (m['ht'] == sel, float(o))

def pick_mtcs(m, cs):
    mk = m['em'].get('Mi-tps CS')
    if not isinstance(mk, dict):
        return None
    o = mk.get(cs)
    if not o:
        return None
    return (m['ht_cs'] == cs, float(o))

def pick_1x2total(m, sel):  # sel like '1 / < 3.5'
    mk = m['em'].get('1X2 & Total')
    if not isinstance(mk, dict):
        return None
    o = mk.get(sel)
    if not o:
        return None
    side, line = [s.strip() for s in sel.split('/')]
    over = '>' in line
    won_side = m['ft'] == side
    won_line = (m['tot'] > 3.5) if over else (m['tot'] < 3.5)
    return (won_side and won_line, float(o))

def pick_mtdc(m, sel):  # 'Mi-tps DC' : '1X','12','X2'
    mk = m['em'].get('Mi-tps DC')
    if not isinstance(mk, dict):
        return None
    o = mk.get(sel)
    if not o:
        return None
    return (m['ht'] in sel, float(o))

# ---------------------------------------------------------------- main
def main():
    rows = load_matches()
    n = len(rows)
    cut = int(n * 0.70)
    train, oos = rows[:cut], rows[cut:]
    print(f'matchs={n}  train={len(train)} (-> {train[-1]["start"]})  oos={len(oos)} (from {oos[0]["start"]})')
    print()

    # ============ 1) GRILLE HT/FT : combo x bracket(home odds) x segment ============
    combos = ['1/1', '1/X', '1/2', 'X/1', 'X/X', 'X/2', '2/1', '2/X', '2/2']

    def cells_for(dataset):
        cells = defaultdict(list)  # key -> picks
        for m in dataset:
            bh = bracket_of(m['oh'], HOME_BRACKETS)
            for combo in combos:
                p = pick_htft(m, combo)
                if p is None:
                    continue
                keys = [
                    ('HTFT', combo, bh, m['seg']),
                    ('HTFT', combo, bh, 'ALL'),
                    ('HTFT', combo, 'ALLBR', m['seg']),
                ]
                for k in keys:
                    cells[k].append(p)
        return cells

    tr_cells = cells_for(train)
    oo_cells = cells_for(oos)

    print('=== HT/FT : cellules selectionnees sur TRAIN (n_train>=60, roi_train>=+10%) -> eval OOS ===')
    selected = []
    for k, picks in tr_cells.items():
        ntr, wtr, aotr, rtr = roi_stats(picks)
        if ntr >= 60 and rtr >= 0.10:
            selected.append((k, ntr, wtr, aotr, rtr))
    selected.sort(key=lambda x: -x[4])
    for k, ntr, wtr, aotr, rtr in selected:
        no, wo, aoo, ro = roi_stats(oo_cells.get(k, []))
        tag = '' if no >= 30 else '[instable n<30]'
        print(fmt(f'{k[1]} | {k[2]} | {k[3]}', no, wo, aoo, ro,
                  f'| train: n={ntr} wr={wtr*100:.1f}% roi={rtr*100:+.1f}% {tag}'))
    print()

    # ============ 2) HT/FT frequences brutes OOS par combo (reference) ============
    print('=== HT/FT reference : frequence OOS + cote moyenne par combo (tous matchs) ===')
    for combo in combos:
        picks = [pick_htft(m, combo) for m in oos]
        picks = [p for p in picks if p]
        no, wo, aoo, ro = roi_stats(picks)
        print(fmt(f'{combo} (blanket OOS)', no, wo, aoo, ro))
    print()

    # ============ 3) Mi-tps 1X2 ============
    print('=== Mi-tps 1X2 : blanket + par bracket/segment (selection train) ===')
    for sel in ['1', 'X', '2']:
        picks = [pick_mt1x2(m, sel) for m in oos]
        picks = [p for p in picks if p]
        print(fmt(f'MT-1X2 {sel} (blanket OOS)', *roi_stats(picks)))
    # grille
    def mt_cells(dataset):
        cells = defaultdict(list)
        for m in dataset:
            bh = bracket_of(m['oh'], HOME_BRACKETS)
            for sel in ['1', 'X', '2']:
                p = pick_mt1x2(m, sel)
                if p is None:
                    continue
                for k in [('MT1X2', sel, bh, m['seg']), ('MT1X2', sel, bh, 'ALL'),
                          ('MT1X2', sel, 'ALLBR', m['seg'])]:
                    cells[k].append(p)
        return cells
    trc, ooc = mt_cells(train), mt_cells(oos)
    sel2 = [(k, *roi_stats(v)) for k, v in trc.items()]
    sel2 = [s for s in sel2 if s[1] >= 60 and s[4] >= 0.08]
    sel2.sort(key=lambda x: -x[4])
    for k, ntr, wtr, aotr, rtr in sel2:
        no, wo, aoo, ro = roi_stats(ooc.get(k, []))
        tag = '' if no >= 30 else '[instable n<30]'
        print(fmt(f'MT-1X2 {k[1]} | {k[2]} | {k[3]}', no, wo, aoo, ro,
                  f'| train: n={ntr} roi={rtr*100:+.1f}% {tag}'))
    print()

    # ============ 4) Mi-tps CS ============
    print('=== Mi-tps CS : blanket OOS (0-0, 1-0, 0-1, 1-1) ===')
    for cs in ['0-0', '1-0', '0-1', '1-1']:
        picks = [pick_mtcs(m, cs) for m in oos]
        picks = [p for p in picks if p]
        print(fmt(f'MT-CS {cs} (blanket OOS)', *roi_stats(picks)))
    # grille train-select
    def cs_cells(dataset):
        cells = defaultdict(list)
        for m in dataset:
            bh = bracket_of(m['oh'], HOME_BRACKETS)
            for cs in ['0-0', '1-0', '0-1', '1-1', '2-0', '0-2']:
                p = pick_mtcs(m, cs)
                if p is None:
                    continue
                for k in [('MTCS', cs, bh, m['seg']), ('MTCS', cs, bh, 'ALL'),
                          ('MTCS', cs, 'ALLBR', m['seg'])]:
                    cells[k].append(p)
        return cells
    trc, ooc = cs_cells(train), cs_cells(oos)
    sel3 = [(k, *roi_stats(v)) for k, v in trc.items()]
    sel3 = [s for s in sel3 if s[1] >= 60 and s[4] >= 0.10]
    sel3.sort(key=lambda x: -x[4])
    for k, ntr, wtr, aotr, rtr in sel3:
        no, wo, aoo, ro = roi_stats(ooc.get(k, []))
        tag = '' if no >= 30 else '[instable n<30]'
        print(fmt(f'MT-CS {k[1]} | {k[2]} | {k[3]}', no, wo, aoo, ro,
                  f'| train: n={ntr} roi={rtr*100:+.1f}% {tag}'))
    print()

    # ============ 5) 1X2 & Total ============
    print('=== 1X2 & Total : blanket OOS ===')
    for sel in ['1 / < 3.5', '1 / > 3.5', '2 / < 3.5', '2 / > 3.5', 'X / < 3.5']:
        picks = [pick_1x2total(m, sel) for m in oos]
        picks = [p for p in picks if p]
        print(fmt(f'1X2&T {sel} (blanket OOS)', *roi_stats(picks)))
    def t_cells(dataset):
        cells = defaultdict(list)
        sels = ['1 / < 3.5', '1 / > 3.5', '2 / < 3.5', '2 / > 3.5', 'X / < 3.5', 'X / > 3.5']
        for m in dataset:
            bh = bracket_of(m['oh'], HOME_BRACKETS)
            for sel in sels:
                p = pick_1x2total(m, sel)
                if p is None:
                    continue
                for k in [('T', sel, bh, m['seg']), ('T', sel, bh, 'ALL'),
                          ('T', sel, 'ALLBR', m['seg'])]:
                    cells[k].append(p)
        return cells
    trc, ooc = t_cells(train), t_cells(oos)
    sel4 = [(k, *roi_stats(v)) for k, v in trc.items()]
    sel4 = [s for s in sel4 if s[1] >= 60 and s[4] >= 0.10]
    sel4.sort(key=lambda x: -x[4])
    for k, ntr, wtr, aotr, rtr in sel4:
        no, wo, aoo, ro = roi_stats(ooc.get(k, []))
        tag = '' if no >= 30 else '[instable n<30]'
        print(fmt(f'1X2&T {k[1]} | {k[2]} | {k[3]}', no, wo, aoo, ro,
                  f'| train: n={ntr} roi={rtr*100:+.1f}% {tag}'))
    print()

    # ============ 6) Mi-tps DC ============
    print('=== Mi-tps DC : blanket OOS ===')
    for sel in ['1X', '12', 'X2']:
        picks = [pick_mtdc(m, sel) for m in oos]
        picks = [p for p in picks if p]
        print(fmt(f'MT-DC {sel} (blanket OOS)', *roi_stats(picks)))
    print()

    # ============ 7) Dutching HT/FT : couvrir 1/1 + X/1 pour favoris home ============
    print('=== Dutching 1/1 + X/1 (favori home, mise 1+1) : ROI par bracket, eval OOS apres select train ===')
    def dutch(dataset, lo, hi):
        out = []
        for m in dataset:
            if not (lo <= m['oh'] < hi):
                continue
            mk = m['em'].get('HT/FT')
            if not isinstance(mk, dict) or '1/1' not in mk or 'X/1' not in mk:
                continue
            o11, ox1 = float(mk['1/1']), float(mk['X/1'])
            if m['htft'] == '1/1':
                ret = o11
            elif m['htft'] == 'X/1':
                ret = ox1
            else:
                ret = 0.0
            out.append(ret - 2.0)  # mise 2
        return out
    for lo, hi, name in HOME_BRACKETS:
        tr = dutch(train, lo, hi)
        oo = dutch(oos, lo, hi)
        if len(tr) < 60:
            continue
        rtr = sum(tr) / (2 * len(tr)) if tr else 0
        roo = sum(oo) / (2 * len(oo)) if oo else 0
        wo = sum(1 for x in oo if x > -2.0) / len(oo) if oo else 0
        print(f'dutch 1/1+X/1 | {name:<12s} n_oos={len(oo):<4d} hit_oos={wo*100:5.1f}% roi_oos={roo*100:+6.1f}% | train n={len(tr)} roi={rtr*100:+.1f}%')
    print()

if __name__ == '__main__':
    main()
