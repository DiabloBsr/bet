# -*- coding: utf-8 -*-
"""
VERIFICATION ADVERSARIALE du signal :
  "HT/FT '2/1' favori home MS_mid (comeback du favori)"
  Definition : odds_home < 2.0 ET journee 13-25 (MS_mid) -> parier HT/FT combo '2/1'.
  Claims OOS (70/30) : n=327, WR=5.2%, cote moyenne=30.4, ROI=+60.7%.

Methodo INDEPENDANTE (pas le code du mineur) :
  - Chargement from scratch + controles d'integrite (doublons, coherence HT/FT, cles marche)
  - Reproduction du split 70/30 du mineur pour valider le pipeline
  - Walk-forward 3 fenetres :
      W1: train 0-50%  -> test 50-66%
      W2: train 0-66%  -> test 66-83%
      W3: train 0-83%  -> test 83-100%
    Le signal (regle fixe) est re-evalue sur chaque train (gating) puis sur chaque test.
  - Metriques agregees sur l'union des 3 fenetres test + p-value binomiale + bootstrap.
"""
import sys, json, math, random
sys.path.insert(0, '.')
from scraper.config import load_settings
from sqlalchemy import create_engine, text


def load_matches():
    eng = create_engine(load_settings().db_url)
    q = text('''
        SELECT e.id, e.match_key, e.round_info, e.expected_start, e.team_a, e.team_b,
               o.odds_home, o.extra_markets,
               r.score_a, r.score_b, r.ht_score_a, r.ht_score_b
        FROM events e
        JOIN results r ON r.event_id = e.id
        JOIN odds_snapshots o ON o.event_id = e.id
         AND o.id = (SELECT MIN(id) FROM odds_snapshots WHERE event_id = e.id)
        WHERE e.round_info != '0'
          AND r.ht_score_a IS NOT NULL AND r.ht_score_b IS NOT NULL
          AND r.score_a IS NOT NULL AND r.score_b IS NOT NULL
          AND o.extra_markets IS NOT NULL
          AND o.odds_home IS NOT NULL
        ORDER BY e.expected_start ASC, e.id ASC
    ''')
    rows = []
    with eng.connect() as c:
        for r in c.execute(q):
            try:
                rnd = int(r[2])
            except (TypeError, ValueError):
                continue
            if not (1 <= rnd <= 38):
                continue
            em = r[7]
            if isinstance(em, str):
                try:
                    em = json.loads(em)
                except Exception:
                    continue
            if not isinstance(em, dict):
                continue
            sa, sb, ha, hb = int(r[8]), int(r[9]), int(r[10]), int(r[11])
            # coherence : score HT ne peut pas depasser le score FT
            if ha > sa or hb > sb:
                continue  # donnee corrompue
            ht = '1' if ha > hb else ('2' if hb > ha else 'X')
            ft = '1' if sa > sb else ('2' if sb > sa else 'X')
            rows.append({
                'id': r[0], 'mk': r[1], 'round': rnd, 'start': r[3],
                'teams': (r[4], r[5]), 'oh': float(r[6]), 'em': em,
                'htft': ht + '/' + ft,
            })
    return rows


def get_21_odds(m):
    mk = m['em'].get('HT/FT')
    if not isinstance(mk, dict):
        return None
    o = mk.get('2/1')
    if o is None:
        return None
    try:
        o = float(o)
    except (TypeError, ValueError):
        return None
    return o if o > 1.0 else None


def is_candidate(m):
    return m['oh'] < 2.0 and 13 <= m['round'] <= 25


def stats(picks):
    n = len(picks)
    if n == 0:
        return dict(n=0, wins=0, wr=0.0, avg_o=0.0, roi=0.0)
    wins = sum(1 for w, _ in picks if w)
    avg_o = sum(o for _, o in picks) / n
    roi = sum((o - 1.0) if w else -1.0 for w, o in picks) / n
    return dict(n=n, wins=wins, wr=wins / n, avg_o=avg_o, roi=roi)


def binom_pval_ge(n, k, p):
    if k <= 0:
        return 1.0
    return sum(math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(k, n + 1))


def main():
    rows = load_matches()
    n = len(rows)
    print(f'matchs charges (tries par expected_start) : {n}')

    # ---------- CONTROLE 1 : doublons ----------
    seen, dups = {}, 0
    for m in rows:
        key = (m['mk'], str(m['start']))
        seen[key] = seen.get(key, 0) + 1
    dups = sum(c - 1 for c in seen.values() if c > 1)
    print(f'doublons (match_key, expected_start) : {dups}')

    # ---------- CONTROLE 2 : sanity du marche HT/FT 2/1 ----------
    cands = [m for m in rows if is_candidate(m)]
    with_odds = [(m, get_21_odds(m)) for m in cands]
    with_odds = [(m, o) for m, o in with_odds if o is not None]
    missing = len(cands) - len(with_odds)
    odds_list = sorted(o for _, o in with_odds)
    if odds_list:
        med = odds_list[len(odds_list) // 2]
        print(f'candidats (oh<2.0, J13-25) : {len(cands)} | avec cote 2/1 : {len(with_odds)} '
              f'(manquants {missing}) | cote 2/1 min/med/max = '
              f'{odds_list[0]:.2f}/{med:.2f}/{odds_list[-1]:.2f}')

    # frequence reelle du combo 2/1 chez les candidats vs proba implicite
    hits_all = sum(1 for m, _ in with_odds if m['htft'] == '2/1')
    avg_o_all = sum(o for _, o in with_odds) / len(with_odds)
    print(f'frequence reelle 2/1 (TOUT historique) : {hits_all}/{len(with_odds)} '
          f'= {hits_all/len(with_odds)*100:.2f}% | implicite 1/cote_moy = {100/avg_o_all:.2f}%')

    # ---------- CONTROLE 3 : reproduction du claim 70/30 du mineur ----------
    cut70 = int(n * 0.70)
    oos_m = [(m['htft'] == '2/1', o) for m, o in with_odds
             if rows.index(m) >= 0]  # placeholder, on refait proprement ci-dessous
    idx_of = {id(m): i for i, m in enumerate(rows)}
    oos_m = [(m['htft'] == '2/1', o) for m, o in with_odds if idx_of[id(m)] >= cut70]
    s = stats(oos_m)
    print(f"\n[reproduction claim 70/30] n={s['n']} wins={s['wins']} wr={s['wr']*100:.1f}% "
          f"cote_moy={s['avg_o']:.1f} roi={s['roi']*100:+.1f}%  "
          f"(claim: n=327 wr=5.2% cote=30.4 roi=+60.7%)")
    tr_m = [(m['htft'] == '2/1', o) for m, o in with_odds if idx_of[id(m)] < cut70]
    st = stats(tr_m)
    print(f"[train 70% du mineur]      n={st['n']} wins={st['wins']} wr={st['wr']*100:.1f}% "
          f"cote_moy={st['avg_o']:.1f} roi={st['roi']*100:+.1f}%")

    # ---------- WALK-FORWARD 3 FENETRES ----------
    print('\n=== WALK-FORWARD 3 FENETRES (decoupage different du mineur) ===')
    windows = [(0.00, 0.50, 0.66), (0.00, 0.66, 0.83), (0.00, 0.83, 1.00)]
    agg = []
    for wi, (a, b, c) in enumerate(windows, 1):
        i0, i1, i2 = int(n * a), int(n * b), int(n * c)
        train_rows = rows[i0:i1]
        test_rows = rows[i1:i2]
        # signal re-evalue sur le train (regle fixe -> gating : ROI train > 0 ?)
        tr = [( m['htft'] == '2/1', get_21_odds(m)) for m in train_rows
              if is_candidate(m) and get_21_odds(m) is not None]
        te = [( m['htft'] == '2/1', get_21_odds(m)) for m in test_rows
              if is_candidate(m) and get_21_odds(m) is not None]
        st, se = stats(tr), stats(te)
        gate = 'OUI' if st['roi'] > 0 else 'NON'
        print(f"W{wi} train[0-{int(b*100)}%]  n={st['n']:4d} wins={st['wins']:3d} "
              f"wr={st['wr']*100:4.1f}% roi={st['roi']*100:+6.1f}%  (signal positif en train ? {gate})")
        print(f"W{wi} test [{int(b*100)}-{int(c*100)}%] n={se['n']:4d} wins={se['wins']:3d} "
              f"wr={se['wr']*100:4.1f}% cote_moy={se['avg_o']:5.1f} roi={se['roi']*100:+6.1f}%")
        agg.extend(te)

    sa_ = stats(agg)
    print(f"\n=== AGREGE 3 fenetres test (50-100%) ===")
    print(f"n={sa_['n']} wins={sa_['wins']} wr={sa_['wr']*100:.2f}% "
          f"cote_moy={sa_['avg_o']:.1f} roi={sa_['roi']*100:+.1f}%")
    if sa_['n']:
        p_be = 1.0 / sa_['avg_o']
        pv = binom_pval_ge(sa_['n'], sa_['wins'], p_be)
        print(f"p-value binomiale (H0 : wr = 1/cote_moy = {p_be*100:.2f}%) : {pv:.4f}")

        # bootstrap du ROI agrege (10000 resamples)
        random.seed(42)
        rets = [(o - 1.0) if w else -1.0 for w, o in agg]
        boots = []
        for _ in range(10000):
            sample = [rets[random.randrange(len(rets))] for _ in range(len(rets))]
            boots.append(sum(sample) / len(sample))
        boots.sort()
        lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]
        neg_share = sum(1 for b in boots if b <= 0) / len(boots)
        print(f"bootstrap ROI IC95% = [{lo*100:+.1f}%, {hi*100:+.1f}%] | P(ROI<=0) = {neg_share*100:.1f}%")

    # ---------- SENSIBILITE : wins individuels & dependance aux outliers ----------
    print('\n=== SENSIBILITE (poids des wins individuels, fenetres test) ===')
    win_odds = sorted((o for w, o in agg if w), reverse=True)
    print(f"cotes des wins (desc) : {[f'{o:.0f}' for o in win_odds]}")
    if sa_['n'] and win_odds:
        # ROI si on retire les k plus gros wins
        rets = [(o - 1.0) if w else -1.0 for w, o in agg]
        total = sum(rets)
        for k in (1, 2, 3):
            if k <= len(win_odds):
                reduced = total - sum(o - 1.0 for o in win_odds[:k]) - k  # win devient perte (-1)
                print(f"ROI agrege si les {k} plus gros wins etaient perdus : "
                      f"{reduced/len(rets)*100:+.1f}%")

    # ---------- DECOUPAGE ALTERNATIF : 5 chunks egaux sur TOUT l'historique ----------
    print('\n=== 5 CHUNKS EGAUX (tout historique, regle fixe) ===')
    for k in range(5):
        i1, i2 = int(n * k / 5), int(n * (k + 1) / 5)
        ch = [( m['htft'] == '2/1', get_21_odds(m)) for m in rows[i1:i2]
              if is_candidate(m) and get_21_odds(m) is not None]
        sc = stats(ch)
        print(f"chunk {k+1}/5 [{k*20}-{(k+1)*20}%] n={sc['n']:4d} wins={sc['wins']:3d} "
              f"wr={sc['wr']*100:4.1f}% roi={sc['roi']*100:+6.1f}%")


if __name__ == '__main__':
    main()
