# -*- coding: utf-8 -*-
"""
VERIFICATION ADVERSARIALE (from scratch, independante du script du mineur)
Signal: HT/FT '1/X' | odds_home >= 3.5 | journee 4-12 (MS_early)
Pari: marche 'HT/FT' combo '1/X' aux cotes du snapshot d'OUVERTURE (MIN(id)).
Win: home mene a la HT (ht_a > ht_b) ET nul au FT (ft_a == ft_b).

Protocole:
1. Audits d'integrite: doublons d'events, look-ahead (captured_at vs expected_start),
   coherence HT<=FT.
2. Walk-forward 3 fenetres: train 0-50% -> test 50-66% ; train 0-66% -> test 66-83% ;
   train 0-83% -> test 83-100%. La regle est fixe; le "recalcul" sur train = gate
   (on verifie que le ROI train est >0 avant de parier la fenetre test).
3. Metriques agregees sur les 3 fenetres test.
4. Reproduction du split 70/30 du mineur AVEC et SANS doublons pour quantifier le biais.
"""
import sys, json, math
from datetime import datetime
sys.path.insert(0, '.')
from scraper.config import load_settings
from sqlalchemy import create_engine, text


def _parse_dt(s):
    s = str(s)
    for f in ('%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S'):
        try:
            return datetime.strptime(s, f)
        except ValueError:
            pass
    return None

OH_MIN = 3.5
RND_LO, RND_HI = 4, 12
MARKET, COMBO = 'HT/FT', '1/X'


def load(dedupe=True):
    eng = create_engine(load_settings().db_url)
    q = text('''
        SELECT e.id, e.team_a, e.team_b, e.expected_start, e.round_info,
               o.odds_home, o.extra_markets, o.captured_at,
               r.score_a, r.score_b, r.ht_score_a, r.ht_score_b
        FROM events e
        JOIN results r ON r.event_id = e.id
        JOIN odds_snapshots o ON o.event_id = e.id
         AND o.id = (SELECT MIN(id) FROM odds_snapshots WHERE event_id = e.id)
        WHERE e.round_info IS NOT NULL AND e.round_info != '0'
          AND r.score_a IS NOT NULL AND r.score_b IS NOT NULL
          AND r.ht_score_a IS NOT NULL AND r.ht_score_b IS NOT NULL
          AND o.odds_home IS NOT NULL AND o.extra_markets IS NOT NULL
        ORDER BY e.expected_start ASC, e.id ASC
    ''')
    rows, seen = [], set()
    n_dup = n_badht = n_lookahead = 0
    with eng.connect() as c:
        for r in c.execute(q):
            try:
                rnd = int(r[4])
            except (TypeError, ValueError):
                continue
            if not (1 <= rnd <= 38):
                continue
            key = (r[1], r[2], str(r[3]))  # team_a, team_b, expected_start
            if key in seen:
                n_dup += 1
                if dedupe:
                    continue
            seen.add(key)
            sa, sb, ha, hb = int(r[8]), int(r[9]), int(r[10]), int(r[11])
            if ha > sa or hb > sb:          # score HT > score FT = donnee corrompue
                n_badht += 1
                continue
            es, ca = _parse_dt(r[3]), _parse_dt(r[7]) if r[7] else None
            late = (ca is not None and es is not None and ca > es)
            if late:
                n_lookahead += 1
            em = r[6]
            if isinstance(em, str):
                try:
                    em = json.loads(em)
                except Exception:
                    continue
            if not isinstance(em, dict):
                continue
            rows.append({
                'id': r[0], 'rnd': rnd, 'start': str(r[3]),
                'oh': float(r[5]), 'em': em, 'late': late,
                'win': (ha > hb) and (sa == sb),
            })
    return rows, n_dup, n_badht, n_lookahead


def pick_odds(m):
    mk = m['em'].get(MARKET)
    if not isinstance(mk, dict):
        return None
    o = mk.get(COMBO)
    try:
        o = float(o)
    except (TypeError, ValueError):
        return None
    return o if o > 1.0 else None


def signal_picks(rows, strict_prekickoff=False):
    out = []
    for m in rows:
        if strict_prekickoff and m['late']:
            continue
        if m['oh'] >= OH_MIN and RND_LO <= m['rnd'] <= RND_HI:
            o = pick_odds(m)
            if o is not None:
                out.append((m['win'], o, m['start']))
    return out


def stats(picks):
    n = len(picks)
    if n == 0:
        return dict(n=0, wins=0, wr=0.0, avg_o=0.0, roi=0.0)
    wins = sum(1 for w, _, _ in picks if w)
    avg_o = sum(o for _, o, _ in picks) / n
    roi = sum((o - 1.0) if w else -1.0 for w, o, _ in picks) / n
    return dict(n=n, wins=wins, wr=wins / n, avg_o=avg_o, roi=roi)


def binom_pval_ge(n, k, p):
    if k <= 0:
        return 1.0
    return sum(math.comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k, n + 1))


def main():
    rows, n_dup, n_badht, n_la = load(dedupe=True)
    n = len(rows)
    print(f'matchs exploitables (dedupliques) = {n}')
    print(f'AUDIT: doublons retires={n_dup} | HT>FT corrompus={n_badht} | '
          f'cotes capturees apres kickoff={n_la}')
    print()

    # ---- walk-forward 3 fenetres (standard + strict pre-kickoff) ----
    bounds = [(0.50, 0.66), (0.66, 0.83), (0.83, 1.00)]
    results = {}
    for strict in (False, True):
        label = 'STRICT pre-kickoff (cotes capturees AVANT le coup d envoi uniquement)' \
                if strict else 'STANDARD (toutes cotes d ouverture)'
        print(f'=== WALK-FORWARD 3 FENETRES — {label} ===')
        agg, pos_windows = [], 0
        for lo, hi in bounds:
            a, b = int(n * lo), int(n * hi)
            train, test = rows[:a], rows[a:b]
            st_tr = stats(signal_picks(train, strict))
            st_te = stats(signal_picks(test, strict))
            gate = 'OUVERT' if st_tr['roi'] > 0 else 'FERME (ROI train <= 0)'
            if st_te['roi'] > 0:
                pos_windows += 1
            agg += signal_picks(test, strict)
            print(f"  train 0-{int(lo*100)}%: n={st_tr['n']:<4d} wr={st_tr['wr']*100:5.1f}% "
                  f"roi={st_tr['roi']*100:+7.1f}%  -> gate {gate}")
            print(f"  test {int(lo*100)}-{int(hi*100)}%: n={st_te['n']:<4d} wins={st_te['wins']} "
                  f"wr={st_te['wr']*100:5.1f}% cote={st_te['avg_o']:6.2f} roi={st_te['roi']*100:+7.1f}%")
        st = stats(agg)
        p_be = 1.0 / st['avg_o'] if st['avg_o'] > 0 else 1.0
        pv = binom_pval_ge(st['n'], st['wins'], p_be)
        print(f"  AGREGE: n={st['n']} wins={st['wins']} wr={st['wr']*100:.1f}% "
              f"cote_moy={st['avg_o']:.2f} roi={st['roi']*100:+.1f}% | fenetres ROI>0: {pos_windows}/3 "
              f"| p-value vs breakeven={pv:.4f}")
        print()
        results[strict] = (st, pos_windows, agg)
    st, pos_windows, agg = results[False]

    # ---- reproduction du split 70/30 du mineur, avec et sans doublons ----
    print('=== REPRODUCTION SPLIT 70/30 DU MINEUR ===')
    for dd, label in [(False, 'AVEC doublons (comme le mineur)'),
                      (True, 'SANS doublons (corrige)')]:
        rws, *_ = load(dedupe=dd)
        cut = int(len(rws) * 0.70)
        st_o = stats(signal_picks(rws[cut:]))
        print(f"  {label:<34s}: n={st_o['n']} wins={st_o['wins']} wr={st_o['wr']*100:.1f}% "
              f"cote={st_o['avg_o']:.2f} roi={st_o['roi']*100:+.1f}%")
    print()

    # ---- detail des wins agreges (inspection manuelle anti-bug) ----
    print('=== DETAIL DES WINS (fenetres test agregees) ===')
    for w, o, start in agg:
        if w:
            print(f'  WIN cote={o:6.2f} start={start}')

    # ---- verdict ----
    print()
    if st['roi'] * 100 >= 8.0 and pos_windows >= 2:
        v = 'CONFIRMED'
    elif st['roi'] > 0:
        v = 'WEAKENED'
    else:
        v = 'REFUTED'
    print(f"VERDICT={v} (roi_agrege={st['roi']*100:+.1f}%, fenetres positives={pos_windows}/3)")


if __name__ == '__main__':
    main()
