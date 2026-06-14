# -*- coding: utf-8 -*-
"""
VERIFICATION ADVERSARIALE (from scratch, independante du script du mineur)
Signal pretendu : HT/FT 'X/2' | 1.25 <= odds_home < 1.70 | journee 13-25 (MS_mid)
Claims OOS (70/30) : n=197, WR=9.6%, cote moy=14.93, ROI=+39.9%

Protocole impose : walk-forward 3 fenetres
  W1: train [0%,50%)  -> test [50%,66%)
  W2: train [0%,66%)  -> test [66%,83%)
  W3: train [0%,83%)  -> test [83%,100%)
Le signal est re-evalue sur chaque train (gate: ROI_train > 0) puis mesure sur le test.

Checks adversariaux en plus :
  - doublons (match_key, expected_start)
  - cotes capturees APRES le coup d'envoi (leakage in-play)
  - placebo : meme bracket de cotes dans les AUTRES segments (meme region test)
  - brackets voisins dans MS_mid (sensibilite au cherry-picking du bracket)
  - baseline : tous les X/2 (sans filtre) sur la region test
  - concentration du profit (top wins), p-value binomiale agregee
"""
import sys, json, math
sys.path.insert(0, '.')
from scraper.config import load_settings
from sqlalchemy import create_engine, text

SEG = [(1, 3, 'DS'), (4, 12, 'MS_early'), (13, 25, 'MS_mid'),
       (26, 33, 'MS_late'), (34, 38, 'FS')]

def seg_of(rnd):
    for lo, hi, nm in SEG:
        if lo <= rnd <= hi:
            return nm
    return None

def parse_dt(s):
    # expected_start / captured_at sont des strings ISO en SQLite
    from datetime import datetime
    if s is None:
        return None
    if not isinstance(s, str):
        return s
    s = s.strip().replace('T', ' ')
    for fmt in ('%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M'):
        try:
            return datetime.strptime(s[:26], fmt)
        except ValueError:
            continue
    return None

def load():
    eng = create_engine(load_settings().db_url)
    q = text('''
        SELECT e.id, e.match_key, e.round_info, e.expected_start,
               o.odds_home, o.extra_markets, o.captured_at,
               r.score_a, r.score_b, r.ht_score_a, r.ht_score_b
        FROM events e
        JOIN results r ON r.event_id = e.id
        JOIN odds_snapshots o ON o.event_id = e.id
         AND o.id = (SELECT MIN(id) FROM odds_snapshots WHERE event_id = e.id)
        WHERE e.round_info != '0'
          AND r.score_a IS NOT NULL AND r.score_b IS NOT NULL
          AND r.ht_score_a IS NOT NULL AND r.ht_score_b IS NOT NULL
          AND o.odds_home IS NOT NULL
          AND o.extra_markets IS NOT NULL
        ORDER BY e.expected_start ASC, e.id ASC
    ''')
    rows, seen, dups, late_cap = [], set(), 0, 0
    htft_key_samples = set()
    with eng.connect() as c:
        for r in c.execute(q):
            try:
                rnd = int(r[2])
            except (TypeError, ValueError):
                continue
            if not (1 <= rnd <= 38):
                continue
            em = r[5]
            if isinstance(em, str):
                try:
                    em = json.loads(em)
                except Exception:
                    continue
            if not isinstance(em, dict):
                continue
            key = (r[1], str(r[3]))
            if key in seen:
                dups += 1
                continue  # on EXCLUT les doublons (le mineur ne le faisait pas)
            seen.add(key)
            es, ca = parse_dt(str(r[3])), parse_dt(str(r[6]))
            if es and ca and ca >= es:
                late_cap += 1
            sa, sb, ha, hb = int(r[7]), int(r[8]), int(r[9]), int(r[10])
            ht = '1' if ha > hb else ('2' if hb > ha else 'X')
            ft = '1' if sa > sb else ('2' if sb > sa else 'X')
            # cote X/2 directement depuis le marche HT/FT
            x2 = None
            mk = em.get('HT/FT')
            if isinstance(mk, dict):
                if len(htft_key_samples) < 12:
                    htft_key_samples.update(mk.keys())
                v = mk.get('X/2')
                try:
                    x2 = float(v) if v not in (None, '', 0) else None
                except (TypeError, ValueError):
                    x2 = None
            rows.append({
                'rnd': rnd, 'seg': seg_of(rnd), 'start': str(r[3]),
                'oh': float(r[4]), 'x2': x2,
                'won': (ht == 'X' and ft == '2'),
            })
    return rows, dups, late_cap, sorted(htft_key_samples)

def picks_of(rows, lo=1.25, hi=1.70, seg='MS_mid'):
    out = []
    for m in rows:
        if m['seg'] != seg:
            continue
        if not (lo <= m['oh'] < hi):
            continue
        if m['x2'] is None:
            continue
        out.append((m['won'], m['x2']))
    return out

def stats(picks):
    n = len(picks)
    if n == 0:
        return dict(n=0, wins=0, wr=0.0, avg_o=0.0, roi=0.0, profit=0.0)
    wins = sum(1 for w, _ in picks if w)
    avg_o = sum(o for _, o in picks) / n
    profit = sum((o - 1.0) if w else -1.0 for w, o in picks)
    return dict(n=n, wins=wins, wr=wins / n, avg_o=avg_o, roi=profit / n, profit=profit)

def binom_pval_ge(n, k, p):
    if k <= 0:
        return 1.0
    return sum(math.comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k, n + 1))

def fmt(s):
    return (f"n={s['n']:<4d} wins={s['wins']:<3d} wr={s['wr']*100:5.1f}% "
            f"cote={s['avg_o']:5.2f} roi={s['roi']*100:+7.1f}%")

def main():
    rows, dups, late_cap, keys = load()
    n = len(rows)
    print(f"matchs uniques={n} | doublons exclus={dups} | cotes capturees apres KO={late_cap}")
    print(f"cles marche HT/FT (echantillon): {keys}")
    print()

    cuts = [int(n * f) for f in (0.50, 0.66, 0.83)]
    windows = [
        ('W1 train 0-50%  test 50-66% ', rows[:cuts[0]], rows[cuts[0]:cuts[1]]),
        ('W2 train 0-66%  test 66-83% ', rows[:cuts[1]], rows[cuts[1]:cuts[2]]),
        ('W3 train 0-83%  test 83-100%', rows[:cuts[2]], rows[cuts[2]:]),
    ]

    print('=== WALK-FORWARD 3 FENETRES : HTFT X/2 | 1.25<=oh<1.70 | MS_mid ===')
    agg, agg_gated, pos_windows = [], [], 0
    for name, tr, te in windows:
        st_tr = stats(picks_of(tr))
        pk_te = picks_of(te)
        st_te = stats(pk_te)
        gate = st_tr['roi'] > 0
        if st_te['roi'] > 0:
            pos_windows += 1
        agg += pk_te
        if gate:
            agg_gated += pk_te
        print(f"{name} | TRAIN {fmt(st_tr)}  gate={'PASS' if gate else 'FAIL'}")
        print(f"{' '*len(name)} | TEST  {fmt(st_te)}")
    print()
    sa = stats(agg)
    p_be = (1.0 / sa['avg_o']) if sa['avg_o'] else 1.0
    pv = binom_pval_ge(sa['n'], sa['wins'], p_be)
    print(f"AGREGE 3 fenetres (toutes) : {fmt(sa)}")
    print(f"  fenetres ROI>0 : {pos_windows}/3 | p_breakeven={p_be*100:.2f}% "
          f"p-value binom (>=wins)={pv:.4f}")
    sg = stats(agg_gated)
    print(f"AGREGE gate train ROI>0   : {fmt(sg)}")
    print()

    # concentration du profit : que devient le ROI sans la meilleure win ?
    wins_odds = sorted([o for w, o in agg if w], reverse=True)
    print(f"cotes des wins (agrege, desc): {[round(o,2) for o in wins_odds]}")
    if wins_odds and sa['n'] > 1:
        prof_wo_best = sa['profit'] - (wins_odds[0] - 1.0) - 1.0  # best win devient une perte
        print(f"ROI agrege si la meilleure win etait perdue : "
              f"{prof_wo_best/sa['n']*100:+.1f}%")
        if len(wins_odds) >= 2:
            prof_wo_2 = prof_wo_best - (wins_odds[1] - 1.0) - 1.0
            print(f"ROI agrege si les 2 meilleures wins etaient perdues : "
                  f"{prof_wo_2/sa['n']*100:+.1f}%")
    print()

    # ---- PLACEBOS (region test 50-100% uniquement) ----
    test_all = rows[cuts[0]:]
    print('=== PLACEBO segments (meme bracket 1.25-1.70, X/2, region 50-100%) ===')
    for s in ('DS', 'MS_early', 'MS_mid', 'MS_late', 'FS'):
        print(f"  {s:<9s} {fmt(stats(picks_of(test_all, seg=s)))}")
    print()
    print('=== Sensibilite bracket dans MS_mid (X/2, region 50-100%) ===')
    for lo, hi in [(1.00, 1.25), (1.25, 1.70), (1.70, 2.20), (2.20, 3.00), (3.00, 99.0)]:
        print(f"  oh [{lo:.2f},{hi:.2f}) {fmt(stats(picks_of(test_all, lo=lo, hi=hi)))}")
    print()
    print('=== Baseline X/2 sans aucun filtre (region 50-100%) ===')
    base = [(m['won'], m['x2']) for m in test_all if m['x2'] is not None]
    print(f"  ALL       {fmt(stats(base))}")

if __name__ == '__main__':
    main()
