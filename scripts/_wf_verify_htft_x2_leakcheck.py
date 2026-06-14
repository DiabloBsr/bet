# -*- coding: utf-8 -*-
"""
Check leakage complementaire pour HTFT X/2 MS_mid favori home :
1. Distribution du lag (captured_at - expected_start) sur les picks
2. ROI des picks selon capture avant/apres expected_start (region test 50-100%)
3. Walk-forward agrege restreint aux captures STRICTEMENT avant expected_start
"""
import sys, json
sys.path.insert(0, '.')
from datetime import datetime
from scraper.config import load_settings
from sqlalchemy import create_engine, text

def parse_dt(s):
    if s is None:
        return None
    s = str(s).strip().replace('T', ' ')
    for fmt in ('%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M'):
        try:
            return datetime.strptime(s[:26], fmt)
        except ValueError:
            continue
    return None

def load():
    eng = create_engine(load_settings().db_url)
    q = text('''
        SELECT e.id, e.round_info, e.expected_start,
               o.odds_home, o.extra_markets, o.captured_at,
               r.score_a, r.score_b, r.ht_score_a, r.ht_score_b
        FROM events e
        JOIN results r ON r.event_id = e.id
        JOIN odds_snapshots o ON o.event_id = e.id
         AND o.id = (SELECT MIN(id) FROM odds_snapshots WHERE event_id = e.id)
        WHERE e.round_info != '0'
          AND r.score_a IS NOT NULL AND r.score_b IS NOT NULL
          AND r.ht_score_a IS NOT NULL AND r.ht_score_b IS NOT NULL
          AND o.odds_home IS NOT NULL AND o.extra_markets IS NOT NULL
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
            em = r[4]
            if isinstance(em, str):
                try:
                    em = json.loads(em)
                except Exception:
                    continue
            if not isinstance(em, dict):
                continue
            x2 = None
            mk = em.get('HT/FT')
            if isinstance(mk, dict):
                try:
                    v = mk.get('X/2')
                    x2 = float(v) if v not in (None, '', 0) else None
                except (TypeError, ValueError):
                    x2 = None
            sa, sb, ha, hb = int(r[6]), int(r[7]), int(r[8]), int(r[9])
            es, ca = parse_dt(r[2]), parse_dt(r[5])
            lag = (ca - es).total_seconds() if (es and ca) else None
            rows.append({
                'rnd': rnd, 'oh': float(r[3]), 'x2': x2, 'lag': lag,
                'won': (ha == hb and sb > sa),
            })
    return rows

def stats(picks):
    n = len(picks)
    if n == 0:
        return 'n=0'
    wins = sum(1 for w, _ in picks if w)
    avg_o = sum(o for _, o in picks) / n
    roi = sum((o - 1.0) if w else -1.0 for w, o in picks) / n
    return f"n={n:<4d} wins={wins:<3d} wr={wins/n*100:5.1f}% cote={avg_o:5.2f} roi={roi*100:+7.1f}%"

def main():
    rows = load()
    n = len(rows)
    cut50, cut66, cut83 = int(n*0.50), int(n*0.66), int(n*0.83)

    def is_pick(m):
        return 13 <= m['rnd'] <= 25 and 1.25 <= m['oh'] < 1.70 and m['x2'] is not None

    picks_all = [m for m in rows if is_pick(m)]
    lags = sorted(m['lag'] for m in picks_all if m['lag'] is not None)
    if lags:
        import statistics
        print(f"lag capture->KO sur les {len(lags)} picks (sec) : "
              f"min={lags[0]:.0f} p25={lags[len(lags)//4]:.0f} med={lags[len(lags)//2]:.0f} "
              f"p75={lags[3*len(lags)//4]:.0f} max={lags[-1]:.0f}")
        late = [m for m in picks_all if m['lag'] is not None and m['lag'] >= 0]
        print(f"picks captures a/apres expected_start : {len(late)}")
        if late:
            ll = sorted(m['lag'] for m in late)
            print(f"  lags tardifs (sec): min={ll[0]:.0f} med={ll[len(ll)//2]:.0f} max={ll[-1]:.0f}")
    print()

    test = rows[cut50:]
    te_picks = [m for m in test if is_pick(m)]
    before = [(m['won'], m['x2']) for m in te_picks if m['lag'] is not None and m['lag'] < 0]
    after  = [(m['won'], m['x2']) for m in te_picks if m['lag'] is None or m['lag'] >= 0]
    print('=== Region test 50-100%, picks selon moment de capture ===')
    print(f"  capture AVANT KO : {stats(before)}")
    print(f"  capture A/APRES  : {stats(after)}")
    print()

    print('=== Walk-forward agrege, restreint capture AVANT KO ===')
    for nm, lo_i, hi_i in [('W1 50-66% ', cut50, cut66), ('W2 66-83% ', cut66, cut83),
                            ('W3 83-100%', cut83, n)]:
        pk = [(m['won'], m['x2']) for m in rows[lo_i:hi_i]
              if is_pick(m) and m['lag'] is not None and m['lag'] < 0]
        print(f"  {nm} {stats(pk)}")
    agg = [(m['won'], m['x2']) for m in rows[cut50:]
           if is_pick(m) and m['lag'] is not None and m['lag'] < 0]
    print(f"  AGREGE     {stats(agg)}")

if __name__ == '__main__':
    main()
