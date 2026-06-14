# -*- coding: utf-8 -*-
"""
VERIFICATEUR ADVERSARIAL — signal "1X2 & Total '2 / > 3.5' en FS, away non-favori"
Re-implementation from scratch (independante du script du mineur _wf_htft3.py).

Definition testee :
  - round_info 34..38 (FS)
  - odds_away >= odds_home (cotes d'ouverture = MIN(id) snapshot)
  - pari : extra_markets['1X2 & Total']['2 / > 3.5']
  - gagne ssi score_b > score_a (away win) ET score_a+score_b >= 4 (total > 3.5)

Protocole : walk-forward 3 fenetres
  W1: train [0,50%)   -> test [50%,66%)
  W2: train [0,66%)   -> test [66%,83%)
  W3: train [0,83%)   -> test [83%,100%)
La regle est fixe (pas de parametre ajuste) ; sur chaque train on verifie que la
regle aurait ete "selectionnable" (ROI train > 0), puis on l'evalue sur le test.

Checks adversariaux :
  - doublons d'events (match_key + expected_start)
  - inventaire des cles du marche '1X2 & Total' (la ligne est-elle toujours 3.5 ?)
  - plausibilite des cotes (distribution)
  - controle : meme pari hors FS (le signal est-il specifique a FS ?)
  - p-value binomiale vs breakeven sur l'agrege
"""
import sys, json, math
from collections import Counter
sys.path.insert(0, '.')
from scraper.config import load_settings
from sqlalchemy import create_engine, text

MARKET = '1X2 & Total'
KEY = '2 / > 3.5'


def load_matches():
    eng = create_engine(load_settings().db_url)
    q = text('''
        SELECT e.id, e.match_key, e.round_info, e.expected_start,
               o.odds_home, o.odds_draw, o.odds_away, o.extra_markets,
               r.score_a, r.score_b
        FROM events e
        JOIN results r ON r.event_id = e.id
        JOIN odds_snapshots o ON o.event_id = e.id
         AND o.id = (SELECT MIN(id) FROM odds_snapshots WHERE event_id = e.id)
        WHERE e.round_info != '0'
          AND r.score_a IS NOT NULL AND r.score_b IS NOT NULL
          AND o.odds_home IS NOT NULL AND o.odds_away IS NOT NULL
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
                    em = None
            if not isinstance(em, dict):
                em = {}
            rows.append({
                'eid': r[0], 'mkey': r[1], 'round': rnd, 'start': r[3],
                'oh': float(r[4]), 'oa': float(r[6]), 'em': em,
                'sa': int(r[8]), 'sb': int(r[9]),
            })
    return rows


def combo_odds(m):
    mk = m['em'].get(MARKET)
    if not isinstance(mk, dict):
        return None
    o = mk.get(KEY)
    try:
        o = float(o)
    except (TypeError, ValueError):
        return None
    return o if o > 1.0 else None


def is_fs_awaynonfav(m):
    return 34 <= m['round'] <= 38 and m['oa'] >= m['oh']


def outcome(m):
    return (m['sb'] > m['sa']) and ((m['sa'] + m['sb']) > 3.5)


def evaluate(rows):
    """Retourne la liste des picks (won, odds) parmi rows."""
    picks = []
    for m in rows:
        if not is_fs_awaynonfav(m):
            continue
        o = combo_odds(m)
        if o is None:
            continue
        picks.append((outcome(m), o))
    return picks


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
    return sum(math.comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k, n + 1))


def main():
    rows = load_matches()
    n = len(rows)
    print(f'matchs charges (cotes ouverture + score FT + journee 1-38) : {n}')

    # ---------- CHECK 1 : doublons ----------
    dup_eid = [k for k, v in Counter(m['eid'] for m in rows).items() if v > 1]
    dup_match = [k for k, v in Counter((m['mkey'], str(m['start'])) for m in rows).items() if v > 1]
    print(f'CHECK doublons : event_id dupliques={len(dup_eid)} ; (match_key,start) dupliques={len(dup_match)}')

    # ---------- CHECK 2 : inventaire des cles du marche ----------
    keys = Counter()
    for m in rows:
        mk = m['em'].get(MARKET)
        if isinstance(mk, dict):
            for k in mk:
                keys[k] += 1
    print(f"CHECK marche '{MARKET}' : {len(keys)} cles distinctes")
    for k, v in sorted(keys.items()):
        print(f'   {k!r:<22s} n={v}')
    other_lines = [k for k in keys if '3.5' not in k]
    print(f"CHECK ligne unique 3.5 : cles sans '3.5' = {other_lines if other_lines else 'AUCUNE (ligne toujours 3.5, OK)'}")

    # ---------- CHECK 3 : plausibilite des cotes du combo sur le filtre ----------
    all_picks = evaluate(rows)
    odds_list = sorted(o for _, o in all_picks)
    if odds_list:
        def pct(p):
            return odds_list[min(len(odds_list) - 1, int(p * len(odds_list)))]
        print(f'CHECK cotes combo (filtre FS away-non-fav, TOUT historique) : n={len(odds_list)} '
              f'min={odds_list[0]:.2f} p25={pct(.25):.2f} med={pct(.5):.2f} p75={pct(.75):.2f} max={odds_list[-1]:.2f}')
    full = stats(all_picks)
    print(f"REF tout-historique : n={full['n']} wins={full['wins']} wr={full['wr']*100:.1f}% "
          f"cote_moy={full['avg_o']:.2f} roi={full['roi']*100:+.1f}%")

    # ---------- CHECK 4 : base rate vs implied ----------
    fs_pool = [m for m in rows if is_fs_awaynonfav(m) and combo_odds(m) is not None]
    if fs_pool:
        base = sum(1 for m in fs_pool if outcome(m)) / len(fs_pool)
        imp = sum(1.0 / combo_odds(m) for m in fs_pool) / len(fs_pool)
        print(f'CHECK base rate FS : P(away win & tot>3.5)={base*100:.2f}% vs implied moyen={imp*100:.2f}%')

    # ---------- WALK-FORWARD 3 FENETRES ----------
    print()
    print('=== WALK-FORWARD 3 FENETRES (regle fixe, validee sur train, evaluee sur test) ===')
    windows = [
        ('W1', 0.00, 0.50, 0.50, 0.66),
        ('W2', 0.00, 0.66, 0.66, 0.83),
        ('W3', 0.00, 0.83, 0.83, 1.00),
    ]
    agg = []
    pos_windows = 0
    for name, tr0, tr1, te0, te1 in windows:
        train = rows[int(n * tr0):int(n * tr1)]
        test = rows[int(n * te0):int(n * te1)]
        st_tr = stats(evaluate(train))
        pk_te = evaluate(test)
        st_te = stats(pk_te)
        selectable = st_tr['roi'] > 0
        if selectable:
            agg += pk_te          # on ne mise en test que si le train confirme la regle
        if st_te['roi'] > 0:
            pos_windows += 1
        print(f"{name} train[{int(tr0*100)}-{int(tr1*100)}%]: n={st_tr['n']:<4d} wins={st_tr['wins']:<3d} "
              f"roi={st_tr['roi']*100:+7.1f}%  (selectable={selectable})")
        print(f"   test[{int(te0*100)}-{int(te1*100)}%]:  n={st_te['n']:<4d} wins={st_te['wins']:<3d} "
              f"wr={st_te['wr']*100:5.1f}% cote_moy={st_te['avg_o']:6.2f} roi={st_te['roi']*100:+7.1f}%")

    print()
    a = stats(agg)
    print(f"=== AGREGE 3 fenetres (test 50-100%) : n={a['n']} wins={a['wins']} wr={a['wr']*100:.2f}% "
          f"cote_moy={a['avg_o']:.2f} roi={a['roi']*100:+.2f}% | fenetres ROI>0 : {pos_windows}/3 ===")
    if a['n'] and a['avg_o'] > 1:
        p_be = 1.0 / a['avg_o']
        pv = binom_pval_ge(a['n'], a['wins'], p_be)
        print(f'p-value binomiale (>= {a["wins"]} wins | p_breakeven={p_be*100:.2f}%) = {pv:.4f}')
        # sensibilite : ROI agrege si on retire la plus grosse cote gagnante
        won_odds = sorted((o for w, o in agg if w), reverse=True)
        if won_odds:
            roi_minus1 = (sum((o - 1.0) if w else -1.0 for w, o in agg) - (won_odds[0] - 1.0) + (-1.0)) / a['n']
            print(f'SENSIBILITE : sans le plus gros win (cote {won_odds[0]:.2f}) -> roi={roi_minus1*100:+.1f}%')
            print(f'cotes des wins : {[round(o,2) for o in won_odds]}')

    # ---------- CONTROLE : meme pari hors FS (sur la meme periode test 50-100%) ----------
    print()
    print('=== CONTROLE : meme pari (2 / > 3.5, oa>=oh) hors FS, periode 50-100% ===')
    half = rows[int(n * 0.50):]
    segs = [(1, 3, 'DS'), (4, 12, 'MS_early'), (13, 25, 'MS_mid'), (26, 33, 'MS_late'), (34, 38, 'FS')]
    for lo, hi, sname in segs:
        pk = []
        for m in half:
            if not (lo <= m['round'] <= hi and m['oa'] >= m['oh']):
                continue
            o = combo_odds(m)
            if o is None:
                continue
            pk.append((outcome(m), o))
        s = stats(pk)
        print(f"{sname:<9s} n={s['n']:<4d} wins={s['wins']:<3d} wr={s['wr']*100:5.1f}% "
              f"cote_moy={s['avg_o']:6.2f} roi={s['roi']*100:+7.1f}%")


if __name__ == '__main__':
    main()
