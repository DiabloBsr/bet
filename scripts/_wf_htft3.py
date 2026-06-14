# -*- coding: utf-8 -*-
"""
Iteration 3 : robustesse des strategies finalistes HT/FT + mi-temps.
- 3 splits temporels (60/40, 70/30, 80/20) -> ROI OOS pour chacun
- p-value binomiale : P(wins >= obs | p_breakeven) sur l'OOS 70/30
- portfolio combine
"""
import sys, json, math
sys.path.insert(0, '.')
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
        SELECT e.id, e.round_info, e.expected_start,
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
            em = r[6]
            if isinstance(em, str):
                try:
                    em = json.loads(em)
                except Exception:
                    continue
            if not isinstance(em, dict):
                continue
            sa, sb, ha, hb = int(r[7]), int(r[8]), int(r[9]), int(r[10])
            ht = '1' if ha > hb else ('2' if hb > ha else 'X')
            ft = '1' if sa > sb else ('2' if sb > sa else 'X')
            rows.append({
                'round': rnd, 'seg': seg_of(rnd), 'start': r[2],
                'oh': float(r[3]), 'od': float(r[4]), 'oa': float(r[5]),
                'em': em, 'ht': ht, 'ft': ft, 'htft': ht + '/' + ft,
                'ht_cs': f'{ha}-{hb}', 'tot': sa + sb,
            })
    return rows

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

def roi_stats(picks):
    n = len(picks)
    if n == 0:
        return 0, 0.0, 0.0, 0.0
    wins = sum(1 for w, _ in picks if w)
    avg_o = sum(o for _, o in picks) / n
    roi = sum((o - 1.0) if w else -1.0 for w, o in picks) / n
    return n, wins / n, avg_o, roi

def binom_pval(n, k, p):
    """P(X >= k) pour X~Bin(n,p)."""
    if k <= 0:
        return 1.0
    s = 0.0
    for i in range(k, n + 1):
        s += math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
    return s

FINALISTS = [
    ('A. MT1X2-1 | oh>=3.5 | ALL segs',
     lambda m: m['oh'] >= 3.5, lambda m: p_mt1x2(m, '1')),
    ('B. MT1X2-1 | oh>=4.0 | ALL segs',
     lambda m: m['oh'] >= 4.0, lambda m: p_mt1x2(m, '1')),
    ('C. MT1X2-1 | oh>=3.5 | MS_early',
     lambda m: m['oh'] >= 3.5 and m['seg'] == 'MS_early', lambda m: p_mt1x2(m, '1')),
    ('D. HTFT 1/X | oh>=3.5 | MS_early',
     lambda m: m['oh'] >= 3.5 and m['seg'] == 'MS_early', lambda m: p_htft(m, '1/X')),
    ('E. HTFT 2/1 | oh<2.0 | MS_mid',
     lambda m: m['oh'] < 2.0 and m['seg'] == 'MS_mid', lambda m: p_htft(m, '2/1')),
    ('F. HTFT X/2 | 1.25<=oh<1.70 | MS_mid',
     lambda m: 1.25 <= m['oh'] < 1.70 and m['seg'] == 'MS_mid', lambda m: p_htft(m, 'X/2')),
    ('G. 1X2&T 2/>3.5 | oa>=oh | FS',
     lambda m: m['seg'] == 'FS' and m['oa'] >= m['oh'], lambda m: p_1x2t(m, '2 / > 3.5')),
    ('H. MTCS 0-1 | oh<=1.25 | ALL segs',
     lambda m: m['oh'] <= 1.25, lambda m: p_mtcs(m, '0-1')),
    ('I. HTFT 1/1 | oh>=3.5 | MS_early',
     lambda m: m['oh'] >= 3.5 and m['seg'] == 'MS_early', lambda m: p_htft(m, '1/1')),
]

def main():
    rows = load_matches()
    n = len(rows)
    print(f'matchs={n}')
    print()
    for name, filt, pick in FINALISTS:
        line = f'{name:<42s}'
        # multi-splits
        for frac in (0.60, 0.70, 0.80):
            cut = int(n * frac)
            oos = rows[cut:]
            pk = [pick(m) for m in oos if filt(m)]
            pk = [p for p in pk if p]
            no, wo, aoo, ro = roi_stats(pk)
            line += f' | {int(frac*100)}/{int(100-frac*100)}: n={no:<4d} roi={ro*100:+6.1f}%'
        print(line)
        # detail OOS 70/30 + p-value
        cut = int(n * 0.70)
        oos = rows[cut:]
        pk = [pick(m) for m in oos if filt(m)]
        pk = [p for p in pk if p]
        no, wo, aoo, ro = roi_stats(pk)
        wins = sum(1 for w, _ in pk if w)
        p_be = 1.0 / aoo if aoo > 0 else 1.0
        pv = binom_pval(no, wins, p_be)
        print(f'    OOS70/30: n={no} wins={wins} wr={wo*100:.1f}% cote_moy={aoo:.2f} '
              f'roi={ro*100:+.1f}% | p_breakeven={p_be*100:.1f}% p-value(binom)={pv:.4f}')
        print()

    # ---------------- PORTFOLIO ----------------
    print('=== PORTFOLIO (1 unite par pick, OOS 70/30) ===')
    combos = {
        'P1: A+E+F (gros n, cotes 6-30)': ['A.', 'E.', 'F.'],
        'P2: C+D+I (MS_early home longshot pack)': ['C.', 'D.', 'I.'],
        'P3: tout (A,C-only-extra,D,E,F,G,H)': ['A.', 'D.', 'E.', 'F.', 'G.', 'H.'],
    }
    cut = int(n * 0.70)
    oos = rows[cut:]
    strat_by_prefix = {nm.split()[0]: (filt, pick) for nm, filt, pick in FINALISTS}
    for pname, prefixes in combos.items():
        all_picks = []
        for pref in prefixes:
            filt, pick = strat_by_prefix[pref]
            pk = [pick(m) for m in oos if filt(m)]
            all_picks += [p for p in pk if p]
        no, wo, aoo, ro = roi_stats(all_picks)
        wins = sum(1 for w, _ in all_picks if w)
        print(f'{pname:<44s} n={no} wins={wins} wr={wo*100:.1f}% cote={aoo:.2f} roi={ro*100:+.1f}%')
    print()

    # ---------------- MS_mid favorite-trap : dutching X/2 + 2/1 ----------------
    print('=== MS_mid favori home oh<1.70 : dutching X/2 + 2/1 (2 unites par match, OOS) ===')
    tot_staked, tot_ret, nm_, hits = 0.0, 0.0, 0, 0
    for m in oos:
        if not (m['seg'] == 'MS_mid' and 1.0 <= m['oh'] < 1.70):
            continue
        o_x2 = get_odds(m, 'HT/FT', 'X/2')
        o_21 = get_odds(m, 'HT/FT', '2/1')
        if not o_x2 or not o_21:
            continue
        nm_ += 1
        tot_staked += 2.0
        if m['htft'] == 'X/2':
            tot_ret += o_x2; hits += 1
        elif m['htft'] == '2/1':
            tot_ret += o_21; hits += 1
    if tot_staked:
        print(f'n_matchs={nm_} hits={hits} ({hits/nm_*100:.1f}%) roi={100*(tot_ret-tot_staked)/tot_staked:+.1f}%')

if __name__ == '__main__':
    main()
