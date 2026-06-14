# -*- coding: utf-8 -*-
"""
Follow-up adversarial : le signal MT1X2-'1' oh>=3.5 MS_early survit-il
  (a) si on exclut les snapshots captures >= expected_start (anti-leakage timing) ?
  (b) bootstrap de l'aggregat 3 fenetres (IC 90% du ROI) ?
  (c) lateness des captures : de combien de minutes ?
"""
import sys, json, random
sys.path.insert(0, '.')
from datetime import datetime
from scraper.config import load_settings
from sqlalchemy import create_engine, text

MIN_OH, RND_LO, RND_HI = 3.5, 4, 12


def parse_dt(s):
    if s is None:
        return None
    s = str(s)
    for f in ('%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S.%f',
              '%Y-%m-%dT%H:%M:%S'):
        try:
            return datetime.strptime(s, f)
        except ValueError:
            pass
    return None


def load_rows():
    eng = create_engine(load_settings().db_url)
    q = text('''
        SELECT e.id, e.round_info, e.expected_start,
               o.odds_home, o.extra_markets, o.captured_at,
               r.ht_score_a, r.ht_score_b
        FROM events e
        JOIN results r        ON r.event_id = e.id
        JOIN odds_snapshots o ON o.event_id = e.id
         AND o.id = (SELECT MIN(id) FROM odds_snapshots WHERE event_id = e.id)
        WHERE e.round_info != '0'
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
            mk = em.get('Mi-tps 1X2')
            o1 = None
            if isinstance(mk, dict):
                try:
                    o1 = float(mk.get('1'))
                except (TypeError, ValueError):
                    o1 = None
            rows.append({'rnd': rnd, 'start': parse_dt(r[2]), 'oh': float(r[3]),
                         'o1': o1, 'cap': parse_dt(r[5]),
                         'won': int(r[6]) > int(r[7])})
    return rows


def stats(picks):
    n = len(picks)
    if n == 0:
        return dict(n=0, wr=0.0, avg_o=0.0, roi=0.0)
    wins = sum(1 for w, _ in picks if w)
    return dict(n=n, wr=wins / n, avg_o=sum(o for _, o in picks) / n,
                roi=sum((o - 1.0) if w else -1.0 for w, o in picks) / n)


def main():
    rows = load_rows()
    n = len(rows)

    # (c) lateness distribution
    lates = [(m['cap'] - m['start']).total_seconds() / 60.0
             for m in rows if m['cap'] and m['start']]
    lates.sort()
    def pct(p):
        return lates[int(len(lates) * p)]
    print(f'Lateness capture-kickoff (min) : p10={pct(0.10):+.1f} p50={pct(0.50):+.1f} '
          f'p90={pct(0.90):+.1f} max={lates[-1]:+.1f}')
    very_late = sum(1 for x in lates if x > 45)
    print(f'Captures > 45 min apres kickoff (potentiellement post-HT) : {very_late}/{len(lates)}')

    cuts = [(0.50, 0.66), (0.66, 0.83), (0.83, 1.00)]

    def agg_picks(filter_fn):
        pk = []
        for a, b in cuts:
            for m in rows[int(n * a):int(n * b)]:
                if (m['o1'] and m['oh'] >= MIN_OH and RND_LO <= m['rnd'] <= RND_HI
                        and filter_fn(m)):
                    pk.append((m['won'], m['o1']))
        return pk

    all_p = agg_picks(lambda m: True)
    pre = agg_picks(lambda m: m['cap'] and m['start'] and m['cap'] < m['start'])
    post = agg_picks(lambda m: not (m['cap'] and m['start'] and m['cap'] < m['start']))
    pre_safe = agg_picks(lambda m: m['cap'] and m['start']
                         and (m['cap'] - m['start']).total_seconds() < -60)

    for nm, pk in [('TOUS', all_p), ('capture AVANT kickoff', pre),
                   ('capture >=1min AVANT kickoff', pre_safe),
                   ('capture APRES/au kickoff', post)]:
        s = stats(pk)
        print(f"{nm:<32s}: n={s['n']:>3} wr={s['wr']*100:5.1f}% "
              f"cote={s['avg_o']:.2f} roi={s['roi']*100:+6.1f}%")

    # (b) bootstrap IC du ROI agrege
    random.seed(42)
    rois = []
    for _ in range(10000):
        sample = [random.choice(all_p) for _ in range(len(all_p))]
        rois.append(stats(sample)['roi'])
    rois.sort()
    print(f"\nBootstrap ROI agrege (n={len(all_p)}, 10k resamples) : "
          f"p5={rois[500]*100:+.1f}% p25={rois[2500]*100:+.1f}% med={rois[5000]*100:+.1f}% "
          f"p95={rois[9500]*100:+.1f}% | P(ROI<=0)={sum(1 for x in rois if x <= 0)/len(rois)*100:.1f}%")

    # idem fenetre par fenetre, P(ROI<=0)
    for i, (a, b) in enumerate(cuts, 1):
        pk = [(m['won'], m['o1']) for m in rows[int(n*a):int(n*b)]
              if m['o1'] and m['oh'] >= MIN_OH and RND_LO <= m['rnd'] <= RND_HI]
        rois_w = []
        for _ in range(5000):
            sample = [random.choice(pk) for _ in range(len(pk))]
            rois_w.append(stats(sample)['roi'])
        neg = sum(1 for x in rois_w if x <= 0) / len(rois_w)
        print(f'W{i} : P(ROI<=0) bootstrap = {neg*100:.1f}%')


if __name__ == '__main__':
    main()
