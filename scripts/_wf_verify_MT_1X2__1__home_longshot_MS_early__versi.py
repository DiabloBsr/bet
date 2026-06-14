# -*- coding: utf-8 -*-
"""
VERIFICATEUR ADVERSARIAL — signal du mineur :
  "MT-1X2 '1' home longshot MS_early"
  Definition : odds_home >= 3.5 ET round 4-12 (MS_early)
               -> pari 'Mi-tps 1X2' selection '1' (home mene a la mi-temps)
  Claims OOS (70/30) : n=83, WR=25.3%, cote moy=6.65, ROI=+60.4%

Re-implementation from scratch (requete SQL + parsing independants du script
du mineur) avec walk-forward 3 fenetres :
  W1 : train [0, 50%)   -> test [50%, 66%)
  W2 : train [0, 66%)   -> test [66%, 83%)
  W3 : train [0, 83%)   -> test [83%, 100%)
La regle est statique (aucun parametre appris) ; le "recalcul sur train"
= verification que l'edge existe deja dans chaque train avant de tester.

Checks anti-leakage / anti-bug en plus :
  - doublons d'events (meme match_key / memes equipes + meme expected_start)
  - snapshot d'ouverture capture APRES le coup d'envoi (cotes post-info)
  - coherence HT vs goals_json (le score HT utilise pour regler le pari)
  - sanity des cotes 'Mi-tps 1X2' '1' (bornes, correlation avec odds_home)
"""
import sys, json, math
sys.path.insert(0, '.')
from scraper.config import load_settings
from sqlalchemy import create_engine, text

MARKET = 'Mi-tps 1X2'
SEL = '1'
MIN_OH = 3.5
RND_LO, RND_HI = 4, 12  # MS_early


def load_rows():
    eng = create_engine(load_settings().db_url)
    q = text('''
        SELECT e.id, e.match_key, e.team_a, e.team_b, e.round_info,
               e.expected_start,
               o.odds_home, o.extra_markets, o.captured_at,
               r.ht_score_a, r.ht_score_b, r.goals_json
        FROM events e
        JOIN results r        ON r.event_id = e.id
        JOIN odds_snapshots o ON o.event_id = e.id
         AND o.id = (SELECT MIN(id) FROM odds_snapshots WHERE event_id = e.id)
        WHERE e.round_info != '0'
          AND r.ht_score_a IS NOT NULL AND r.ht_score_b IS NOT NULL
          AND o.odds_home IS NOT NULL
          AND o.extra_markets IS NOT NULL
        ORDER BY e.expected_start ASC, e.id ASC
    ''')
    rows = []
    with eng.connect() as c:
        for r in c.execute(q):
            try:
                rnd = int(r[4])
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
            rows.append({
                'id': r[0], 'mk': r[1], 'ta': r[2], 'tb': r[3], 'rnd': rnd,
                'start': r[5], 'oh': float(r[6]), 'em': em, 'cap': r[8],
                'ha': int(r[9]), 'hb': int(r[10]), 'gj': r[11],
            })
    return rows


def mt1_odds(m):
    mk = m['em'].get(MARKET)
    if not isinstance(mk, dict):
        return None
    o = mk.get(SEL)
    try:
        o = float(o)
    except (TypeError, ValueError):
        return None
    return o if o > 1.0 else None


def is_pick(m):
    return m['oh'] >= MIN_OH and RND_LO <= m['rnd'] <= RND_HI


def settle(m):
    """(won, odds) ou None si marche absent."""
    o = mt1_odds(m)
    if o is None:
        return None
    return (m['ha'] > m['hb'], o)


def stats(picks):
    n = len(picks)
    if n == 0:
        return dict(n=0, wins=0, wr=0.0, avg_o=0.0, roi=0.0)
    wins = sum(1 for w, _ in picks if w)
    avg_o = sum(o for _, o in picks) / n
    roi = sum((o - 1.0) if w else -1.0 for w, o in picks) / n
    return dict(n=n, wins=wins, wr=wins / n, avg_o=avg_o, roi=roi)


def binom_p_ge(n, k, p):
    if k <= 0:
        return 1.0
    return sum(math.comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k, n + 1))


def ht_from_goals_json(gj, ha, hb):
    """Recalcule le score HT depuis goals_json (buts minute <= 45).
    Retourne True si coherent avec (ha, hb), None si non verifiable."""
    if not gj:
        return None
    try:
        goals = json.loads(gj) if isinstance(gj, str) else gj
    except Exception:
        return None
    if not isinstance(goals, list):
        return None
    h = a = 0
    last = (0, 0)
    ok_parse = True
    for g in goals:
        try:
            mn = int(str(g.get('minute', '')).split('+')[0].replace("'", ''))
        except Exception:
            ok_parse = False
            break
        if mn <= 45:
            last = (int(g.get('homeScore', 0)), int(g.get('awayScore', 0)))
    if not ok_parse:
        return None
    return last == (ha, hb)


def main():
    rows = load_rows()
    n = len(rows)
    print(f'Total matchs charges (tries par expected_start) : {n}')

    # ---------- CHECK 1 : doublons ----------
    seen, dup_mk, dup_match = {}, 0, 0
    seen2 = set()
    for m in rows:
        k = (m['mk'], str(m['start']))
        if k in seen:
            dup_mk += 1
        seen[k] = True
        k2 = (m['ta'], m['tb'], str(m['start']))
        if k2 in seen2:
            dup_match += 1
        seen2.add(k2)
    print(f'CHECK doublons : (match_key,start) dupliques={dup_mk} | '
          f'(team_a,team_b,start) dupliques={dup_match}')

    # ---------- CHECK 2 : snapshot capture apres kickoff ----------
    late = sum(1 for m in rows if m['cap'] and m['start'] and str(m['cap']) >= str(m['start']))
    print(f'CHECK timing : snapshots ouverture captures >= expected_start : '
          f'{late}/{n} ({100*late/max(n,1):.1f}%)')

    # ---------- CHECK 3 : coherence HT vs goals_json (sur les picks) ----------
    picks_all = [m for m in rows if is_pick(m) and settle(m) is not None]
    bad_ht, checked = 0, 0
    for m in picks_all:
        ok = ht_from_goals_json(m['gj'], m['ha'], m['hb'])
        if ok is not None:
            checked += 1
            if not ok:
                bad_ht += 1
    print(f'CHECK score HT vs goals_json (picks) : verifies={checked} incoherents={bad_ht}')

    # ---------- CHECK 4 : sanity des cotes MT-1X2 "1" ----------
    odds_list = sorted(o for _, o in (settle(m) for m in picks_all))
    if odds_list:
        med = odds_list[len(odds_list)//2]
        print(f"CHECK cotes MT1X2-'1' (picks) : n={len(odds_list)} "
              f"min={odds_list[0]:.2f} med={med:.2f} max={odds_list[-1]:.2f}")
    no_market = sum(1 for m in rows if is_pick(m) and settle(m) is None)
    print(f'Picks eligibles (oh>=3.5, J4-12) : {len(picks_all)} | marche absent : {no_market}')

    # ---------- Baseline : WR HT-home-win global vs implied ----------
    base = [settle(m) for m in rows if settle(m) is not None]
    bs = stats(base)
    print(f"Baseline tous matchs MT1X2-'1' : n={bs['n']} wr={bs['wr']*100:.1f}% "
          f"cote_moy={bs['avg_o']:.2f} roi={bs['roi']*100:+.1f}%")
    print()

    # ---------- WALK-FORWARD 3 FENETRES ----------
    cuts = [(0.50, 0.66), (0.66, 0.83), (0.83, 1.00)]
    agg = []
    wins_windows = 0
    print('=== WALK-FORWARD 3 FENETRES ===')
    for i, (a, b) in enumerate(cuts, 1):
        i0, i1 = int(n * a), int(n * b)
        train, test_ = rows[:i0], rows[i0:i1]
        tr_picks = [settle(m) for m in train if is_pick(m)]
        tr_picks = [p for p in tr_picks if p is not None]
        te_picks = [settle(m) for m in test_ if is_pick(m)]
        te_picks = [p for p in te_picks if p is not None]
        ts, es = stats(tr_picks), stats(te_picks)
        agg += te_picks
        if es['roi'] > 0:
            wins_windows += 1
        print(f"W{i} train[0:{int(a*100)}%] n={ts['n']:>3} wr={ts['wr']*100:5.1f}% "
              f"roi={ts['roi']*100:+6.1f}%  ||  test[{int(a*100)}%:{int(b*100)}%] "
              f"n={es['n']:>3} wins={es['wins']:>2} wr={es['wr']*100:5.1f}% "
              f"cote={es['avg_o']:.2f} roi={es['roi']*100:+6.1f}%")

    print()
    ag = stats(agg)
    p_be = 1.0 / ag['avg_o'] if ag['avg_o'] > 0 else 1.0
    pv = binom_p_ge(ag['n'], ag['wins'], p_be)
    print(f"=== AGREGE (3 fenetres test = 50%->100% de l'historique) ===")
    print(f"n={ag['n']} wins={ag['wins']} wr={ag['wr']*100:.1f}% cote_moy={ag['avg_o']:.2f} "
          f"roi={ag['roi']*100:+.1f}% | p_breakeven={p_be*100:.1f}% p-value={pv:.4f}")
    print(f"Fenetres ROI>0 : {wins_windows}/3")

    # ---------- Reproduction du claim du mineur (70/30) pour comparaison ----------
    cut = int(n * 0.70)
    oos = [settle(m) for m in rows[cut:] if is_pick(m)]
    oos = [p for p in oos if p is not None]
    os_ = stats(oos)
    print(f"\nReproduction split 70/30 du mineur : n={os_['n']} wr={os_['wr']*100:.1f}% "
          f"cote={os_['avg_o']:.2f} roi={os_['roi']*100:+.1f}% "
          f"(claim: n=83 wr=25.3% cote=6.65 roi=+60.4%)")

    # ---------- Sensibilite : seuils voisins (robustesse, pas cherry-pick) ----------
    print('\n=== SENSIBILITE (agrege 3 fenetres test) ===')
    for lo in (3.0, 3.25, 3.5, 4.0, 4.5):
        pk = []
        for a, b in cuts:
            i0, i1 = int(n * a), int(n * b)
            for m in rows[i0:i1]:
                if m['oh'] >= lo and RND_LO <= m['rnd'] <= RND_HI:
                    s = settle(m)
                    if s:
                        pk.append(s)
        s_ = stats(pk)
        print(f"oh>={lo:<4} : n={s_['n']:>3} wr={s_['wr']*100:5.1f}% "
              f"cote={s_['avg_o']:.2f} roi={s_['roi']*100:+6.1f}%")
    # segments voisins au meme seuil 3.5
    for lo_r, hi_r, nm in [(1, 3, 'DS'), (4, 12, 'MS_early'), (13, 25, 'MS_mid'),
                           (26, 33, 'MS_late'), (34, 38, 'FS')]:
        pk = []
        for a, b in cuts:
            i0, i1 = int(n * a), int(n * b)
            for m in rows[i0:i1]:
                if m['oh'] >= MIN_OH and lo_r <= m['rnd'] <= hi_r:
                    s = settle(m)
                    if s:
                        pk.append(s)
        s_ = stats(pk)
        print(f"seg {nm:<9}: n={s_['n']:>3} wr={s_['wr']*100:5.1f}% "
              f"cote={s_['avg_o']:.2f} roi={s_['roi']*100:+6.1f}%")


if __name__ == '__main__':
    main()
