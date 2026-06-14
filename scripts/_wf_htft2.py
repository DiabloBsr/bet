# -*- coding: utf-8 -*-
"""
Iteration 2 : strategies HT/FT + mi-temps candidates, avec controles de stabilite.
- selection sur TRAIN (70%) uniquement, eval OOS (30%)
- stabilite : split-half du train (les 2 moities doivent etre >0) + OOS coupe en 3 chunks
- on plafonne les cotes moyennes pour eviter les loteries (cellules cote>25 = ininterpretables)
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
                'oh': float(r[5]), 'od': float(r[6]), 'oa': float(r[7]),
                'em': em, 'sa': sa, 'sb': sb, 'ha': ha, 'hb': hb,
                'ht': ht, 'ft': ft, 'htft': ht + '/' + ft,
                'ht_cs': f'{ha}-{hb}', 'tot': sa + sb,
            })
    return rows

def roi_stats(picks):
    n = len(picks)
    if n == 0:
        return 0, 0.0, 0.0, 0.0
    wins = sum(1 for w, _ in picks if w)
    avg_o = sum(o for _, o in picks) / n
    roi = sum((o - 1.0) if w else -1.0 for w, o in picks) / n
    return n, wins / n, avg_o, roi

# ---- pick helpers --------------------------------------------------------
def get_odds(m, market, key):
    mk = m['em'].get(market)
    if not isinstance(mk, dict):
        return None
    o = mk.get(key)
    return float(o) if o else None

def p_htft(m, combo):
    o = get_odds(m, 'HT/FT', combo)
    return None if o is None else (m['htft'] == combo, o)

def p_mt1x2(m, sel):
    o = get_odds(m, 'Mi-tps 1X2', sel)
    return None if o is None else (m['ht'] == sel, o)

def p_mtcs(m, cs):
    o = get_odds(m, 'Mi-tps CS', cs)
    return None if o is None else (m['ht_cs'] == cs, o)

def p_1x2t(m, sel):
    o = get_odds(m, '1X2 & Total', sel)
    if o is None:
        return None
    side, line = [s.strip() for s in sel.split('/')]
    over = '>' in line
    won = (m['ft'] == side) and ((m['tot'] > 3.5) if over else (m['tot'] < 3.5))
    return (won, o)

# ---- candidate strategies (definies a priori sur la base de l'iteration 1,
#      la SELECTION finale = train roi>0 sur les 2 moities du train) ----------
STRATS = [
    # name, filter(m)->bool, pick(m)->(won,odds)|None
    ('MT1X2-1 home longshot oh>=3.5 (ALL)',
     lambda m: m['oh'] >= 3.5, lambda m: p_mt1x2(m, '1')),
    ('MT1X2-1 home longshot oh>=3.0 (ALL)',
     lambda m: m['oh'] >= 3.0, lambda m: p_mt1x2(m, '1')),
    ('MT1X2-1 home longshot oh>=4.0 (ALL)',
     lambda m: m['oh'] >= 4.0, lambda m: p_mt1x2(m, '1')),
    ('MT1X2-1 home longshot oh>=3.5 MS_early',
     lambda m: m['oh'] >= 3.5 and m['seg'] == 'MS_early', lambda m: p_mt1x2(m, '1')),
    ('MT1X2-1 home longshot oh>=3.5 hors MS_early',
     lambda m: m['oh'] >= 3.5 and m['seg'] != 'MS_early', lambda m: p_mt1x2(m, '1')),
    ('HTFT 1/1 home longshot oh>=3.5 (ALL)',
     lambda m: m['oh'] >= 3.5, lambda m: p_htft(m, '1/1')),
    ('HTFT 1/1 home longshot oh>=3.5 MS_early',
     lambda m: m['oh'] >= 3.5 and m['seg'] == 'MS_early', lambda m: p_htft(m, '1/1')),
    ('HTFT X/2 home fav 1.25-1.70 MS_mid',
     lambda m: 1.25 <= m['oh'] < 1.70 and m['seg'] == 'MS_mid', lambda m: p_htft(m, 'X/2')),
    ('HTFT X/2 home fav 1.25-1.70 (ALL segs)',
     lambda m: 1.25 <= m['oh'] < 1.70, lambda m: p_htft(m, 'X/2')),
    ('HTFT X/2 home fav 1.25-1.45 (ALL segs)',
     lambda m: 1.25 <= m['oh'] < 1.45, lambda m: p_htft(m, 'X/2')),
    ('HTFT 2/1 MS_mid (tous matchs)',
     lambda m: m['seg'] == 'MS_mid', lambda m: p_htft(m, '2/1')),
    ('HTFT 2/1 home fav oh<2.0 MS_mid',
     lambda m: m['oh'] < 2.0 and m['seg'] == 'MS_mid', lambda m: p_htft(m, '2/1')),
    ('HTFT 1/X home longshot oh>=3.5 MS_early',
     lambda m: m['oh'] >= 3.5 and m['seg'] == 'MS_early', lambda m: p_htft(m, '1/X')),
    ('1X2&T 2/>3.5 FS (tous)',
     lambda m: m['seg'] == 'FS', lambda m: p_1x2t(m, '2 / > 3.5')),
    ('1X2&T 2/>3.5 FS away fav (oa<oh)',
     lambda m: m['seg'] == 'FS' and m['oa'] < m['oh'], lambda m: p_1x2t(m, '2 / > 3.5')),
    ('1X2&T 2/>3.5 FS away non-fav (oa>=oh)',
     lambda m: m['seg'] == 'FS' and m['oa'] >= m['oh'], lambda m: p_1x2t(m, '2 / > 3.5')),
    ('HTFT 2/2 FS away fav oa<=2.0',
     lambda m: m['seg'] == 'FS' and m['oa'] <= 2.0, lambda m: p_htft(m, '2/2')),
    ('HTFT X/2 FS away fav oa<=2.0',
     lambda m: m['seg'] == 'FS' and m['oa'] <= 2.0, lambda m: p_htft(m, 'X/2')),
    ('HTFT X/1 FS home fav oh<=2.0',
     lambda m: m['seg'] == 'FS' and m['oh'] <= 2.0, lambda m: p_htft(m, 'X/1')),
    ('HTFT X/1 home fav oh<=1.45 (ALL segs)',
     lambda m: m['oh'] <= 1.45, lambda m: p_htft(m, 'X/1')),
    ('HTFT X/X MS_early fav home 1.25-1.45',
     lambda m: 1.25 <= m['oh'] < 1.45 and m['seg'] == 'MS_early', lambda m: p_htft(m, 'X/X')),
    ('MTCS 0-1 home fav oh<=1.25 (ALL segs)',
     lambda m: m['oh'] <= 1.25, lambda m: p_mtcs(m, '0-1')),
    ('1X2&T 2/>3.5 away longshot oa>=3.5 FS',
     lambda m: m['seg'] == 'FS' and m['oa'] >= 3.5, lambda m: p_1x2t(m, '2 / > 3.5')),
    # symetrie du signal longshot : away longshot mene a la mi-temps ?
    ('MT1X2-2 away longshot oa>=3.5 (ALL)',
     lambda m: m['oa'] >= 3.5, lambda m: p_mt1x2(m, '2')),
    ('HTFT 2/2 away longshot oa>=3.5 (ALL)',
     lambda m: m['oa'] >= 3.5, lambda m: p_htft(m, '2/2')),
]

def chunked_roi(picks, k=3):
    out = []
    n = len(picks)
    if n == 0:
        return out
    sz = max(1, n // k)
    for i in range(0, n, sz):
        ch = picks[i:i + sz]
        if len(ch) < max(5, sz // 2):
            # merge tiny tail into previous
            if out:
                continue
        _, _, _, r = roi_stats(ch)
        out.append(r)
    return out[:k + 1]

def main():
    rows = load_matches()
    n = len(rows)
    cut = int(n * 0.70)
    train, oos = rows[:cut], rows[cut:]
    h1, h2 = train[:cut // 2], train[cut // 2:]
    print(f'matchs={n} train={len(train)} oos={len(oos)}')
    print()
    hdr = f'{"strategie":<46s} {"tr_n":>5s} {"tr_roi":>7s} {"h1":>6s} {"h2":>6s} | {"n_oos":>5s} {"wr":>6s} {"cote":>6s} {"roi_oos":>8s} | chunks_oos'
    print(hdr)
    print('-' * len(hdr))
    for name, filt, pick in STRATS:
        tr_p = [pick(m) for m in train if filt(m)]
        tr_p = [p for p in tr_p if p]
        h1_p = [pick(m) for m in h1 if filt(m)]
        h1_p = [p for p in h1_p if p]
        h2_p = [pick(m) for m in h2 if filt(m)]
        h2_p = [p for p in h2_p if p]
        oo_p = [pick(m) for m in oos if filt(m)]
        oo_p = [p for p in oo_p if p]
        ntr, wtr, aotr, rtr = roi_stats(tr_p)
        _, _, _, r1 = roi_stats(h1_p)
        _, _, _, r2 = roi_stats(h2_p)
        no, wo, aoo, ro = roi_stats(oo_p)
        chunks = chunked_roi(oo_p, 3)
        ch_s = ' '.join(f'{c*100:+.0f}%' for c in chunks)
        stable = 'OK ' if (r1 > 0 and r2 > 0 and rtr > 0.05) else '-- '
        print(f'{name:<46s} {ntr:>5d} {rtr*100:>+6.1f}% {r1*100:>+5.0f}% {r2*100:>+5.0f}% | {no:>5d} {wo*100:>5.1f}% {aoo:>6.2f} {ro*100:>+7.1f}% | {ch_s}  {stable}')
    print()
    print('NB: "OK" = train roi>+5% ET les 2 moities du train positives (critere de selection pre-OOS).')

if __name__ == '__main__':
    main()
