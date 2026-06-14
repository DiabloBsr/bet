# -*- coding: utf-8 -*-
"""
VERIFICATION ADVERSARIALE — PORTFOLIO P1 (A + E + F)
Re-implementation from scratch (independante du script du mineur _wf_htft3.py).

Definition verifiee :
  A : 'Mi-tps 1X2' selection '1'  si odds_home >= 3.5            (tous segments)
  E : 'HT/FT'      selection '2/1' si odds_home < 2.0  et MS_mid (J13-25)
  F : 'HT/FT'      selection 'X/2' si 1.25 <= odds_home < 1.70 et MS_mid
  1 unite par pick, union des 3 jambes.

Protocole : walk-forward 3 fenetres (tri par expected_start puis id) :
  W1 : train [0,50%)  -> test [50%,66%)
  W2 : train [0,66%)  -> test [66%,83%)
  W3 : train [0,83%)  -> test [83%,100%)
Les regles sont STATIQUES (aucun parametre ajuste) ; la "recalculation sur le
train" se traduit par un controle de decouvrabilite : la regle est-elle deja
profitable sur chaque train ? (sinon = suspicion de chance pure sur l'OOS).

Checks adversariaux supplementaires :
  - dedupe (team_a, team_b, expected_start) — garde l'event id le plus bas
  - sanity de la condition de gain (echantillons affiches avec scores bruts)
  - concentration du profit (part du PnL venant des 5 plus grosses cotes gagnees)
  - ROI avec cotes tronquees a 40 (robustesse aux outliers)
"""
import sys, json
sys.path.insert(0, '.')
from scraper.config import load_settings
from sqlalchemy import create_engine, text


# ---------------------------------------------------------------- data
def segment(rnd):
    if 1 <= rnd <= 3:    return 'DS'
    if 4 <= rnd <= 12:   return 'MS_early'
    if 13 <= rnd <= 25:  return 'MS_mid'
    if 26 <= rnd <= 33:  return 'MS_late'
    if 34 <= rnd <= 38:  return 'FS'
    return None


def load():
    eng = create_engine(load_settings().db_url)
    q = text("""
        SELECT e.id, e.team_a, e.team_b, e.expected_start, e.round_info,
               o.odds_home, o.extra_markets,
               r.score_a, r.score_b, r.ht_score_a, r.ht_score_b
        FROM events e
        JOIN results r        ON r.event_id = e.id
        JOIN odds_snapshots o ON o.event_id = e.id
                             AND o.id = (SELECT MIN(id) FROM odds_snapshots
                                         WHERE event_id = e.id)
        WHERE e.round_info != '0'
          AND r.score_a    IS NOT NULL AND r.score_b    IS NOT NULL
          AND r.ht_score_a IS NOT NULL AND r.ht_score_b IS NOT NULL
          AND o.odds_home  IS NOT NULL
          AND o.extra_markets IS NOT NULL
        ORDER BY e.expected_start ASC, e.id ASC
    """)
    out, seen = [], set()
    with eng.connect() as c:
        for r in c.execute(q):
            try:
                rnd = int(r[4])
            except (TypeError, ValueError):
                continue
            if not (1 <= rnd <= 38):
                continue
            key = (r[1], r[2], str(r[3]))
            if key in seen:                       # dedupe defensive
                continue
            seen.add(key)
            em = r[6]
            if isinstance(em, str):
                try:
                    em = json.loads(em)
                except Exception:
                    continue
            if not isinstance(em, dict):
                continue
            sa, sb = int(r[7]), int(r[8])
            ha, hb = int(r[9]), int(r[10])
            ht = '1' if ha > hb else ('2' if hb > ha else 'X')
            ft = '1' if sa > sb else ('2' if sb > sa else 'X')
            out.append({
                'id': r[0], 'teams': f'{r[1]} vs {r[2]}', 'start': str(r[3]),
                'rnd': rnd, 'seg': segment(rnd), 'oh': float(r[5]), 'em': em,
                'ht_res': ht, 'ft_res': ft, 'htft': ht + '/' + ft,
                'score': f'{sa}-{sb}', 'ht_score': f'{ha}-{hb}',
            })
    return out


def market_odds(m, market, key):
    mk = m['em'].get(market)
    if not isinstance(mk, dict):
        return None
    v = mk.get(key)
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if v > 1.0 else None


# ---------------------------------------------------------------- legs
def leg_A(m):
    """MT-1X2 '1' si oh >= 3.5 (tous segments)."""
    if m['oh'] < 3.5:
        return None
    o = market_odds(m, 'Mi-tps 1X2', '1')
    if o is None:
        return None
    return ('A', m['ht_res'] == '1', o, m)


def leg_E(m):
    """HT/FT '2/1' si oh < 2.0 et MS_mid."""
    if not (m['oh'] < 2.0 and m['seg'] == 'MS_mid'):
        return None
    o = market_odds(m, 'HT/FT', '2/1')
    if o is None:
        return None
    return ('E', m['htft'] == '2/1', o, m)


def leg_F(m):
    """HT/FT 'X/2' si 1.25 <= oh < 1.70 et MS_mid."""
    if not (1.25 <= m['oh'] < 1.70 and m['seg'] == 'MS_mid'):
        return None
    o = market_odds(m, 'HT/FT', 'X/2')
    if o is None:
        return None
    return ('F', m['htft'] == 'X/2', o, m)


def portfolio(rows):
    picks = []
    for m in rows:
        for fn in (leg_A, leg_E, leg_F):
            p = fn(m)
            if p:
                picks.append(p)
    return picks


def stats(picks):
    n = len(picks)
    if n == 0:
        return dict(n=0, wins=0, wr=0.0, avg_o=0.0, roi=0.0, pnl=0.0)
    wins = sum(1 for _, w, _, _ in picks if w)
    avg_o = sum(o for _, _, o, _ in picks) / n
    pnl = sum((o - 1.0) if w else -1.0 for _, w, o, _ in picks)
    return dict(n=n, wins=wins, wr=wins / n, avg_o=avg_o, roi=pnl / n, pnl=pnl)


def fmt(s):
    return (f"n={s['n']:<4d} wins={s['wins']:<3d} wr={s['wr']*100:5.1f}% "
            f"cote_moy={s['avg_o']:5.2f} roi={s['roi']*100:+7.1f}%")


# ---------------------------------------------------------------- main
def main():
    rows = load()
    n = len(rows)
    print(f'univers analyse (dedupe) : {n} matchs '
          f'({rows[0]["start"]} -> {rows[-1]["start"]})')
    print()

    # ---- sanity : 3 picks gagnants affiches avec scores bruts -------------
    sample = [p for p in portfolio(rows) if p[1]][:3]
    print('--- SANITY conditions de gain (3 exemples gagnants) ---')
    for tag, w, o, m in sample:
        print(f'  [{tag}] {m["teams"]} J{m["rnd"]} | HT {m["ht_score"]} '
              f'FT {m["score"]} -> htft={m["htft"]} | cote={o} | oh={m["oh"]}')
    print()

    # ---- walk-forward 3 fenetres ------------------------------------------
    cuts = [(0.50, 0.66), (0.66, 0.83), (0.83, 1.00)]
    agg, per_window = [], []
    print('--- WALK-FORWARD 3 FENETRES ---')
    for i, (a, b) in enumerate(cuts, 1):
        lo, hi = int(n * a), int(n * b)
        train, test = rows[:lo], rows[lo:hi]
        st_tr = stats(portfolio(train))
        st_te = stats(portfolio(test))
        per_window.append(st_te)
        agg.extend(portfolio(test))
        print(f'W{i} train[0:{lo}]      : {fmt(st_tr)}   (decouvrabilite)')
        print(f'W{i} TEST [{lo}:{hi}] : {fmt(st_te)}')
        # per-leg dans la fenetre test
        for tag in ('A', 'E', 'F'):
            st_leg = stats([p for p in portfolio(test) if p[0] == tag])
            print(f'     leg {tag}          : {fmt(st_leg)}')
        print()

    st = stats(agg)
    print('--- AGREGE (3 fenetres test = 50%->100% des donnees) ---')
    print(fmt(st))
    pos = sum(1 for s in per_window if s['roi'] > 0)
    print(f'fenetres ROI>0 : {pos}/3')
    print()

    # ---- concentration du profit ------------------------------------------
    win_odds = sorted([o for _, w, o, _ in agg if w], reverse=True)
    if win_odds:
        top5 = sum(o - 1.0 for o in win_odds[:5])
        print(f'--- CONCENTRATION ---')
        print(f'gains totaux bruts = {sum(o-1 for o in win_odds):.1f}u, '
              f'top5 cotes gagnees = {win_odds[:5]} (= {top5:.1f}u)')
        print(f'PnL total = {st["pnl"]:+.1f}u ; PnL sans les 5 plus grosses '
              f'cotes gagnees = {st["pnl"]-top5:+.1f}u '
              f'(roi {100*(st["pnl"]-top5)/st["n"]:+.1f}%)')
        capped = sum((min(o, 40.0) - 1.0) if w else -1.0 for _, w, o, _ in agg)
        print(f'ROI avec cotes tronquees a 40 : {100*capped/st["n"]:+.1f}%')
    print()

    # ---- replication du claim mineur (split 70/30) a titre de controle ----
    cut = int(n * 0.70)
    st70 = stats(portfolio(rows[cut:]))
    print('--- CONTROLE : replication du claim OOS 70/30 du mineur ---')
    print(f'claim mineur : n=821 wr=11.1% cote=18.08 roi=+39.7%')
    print(f'ma replication: {fmt(st70)}')


if __name__ == '__main__':
    main()
