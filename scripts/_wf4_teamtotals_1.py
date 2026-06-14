# -*- coding: utf-8 -*-
"""
WF4 — team totals & micro-markets miner — step 1: dataset build.

Builds a flat per-bet table over ALL 9 leagues:
  one row = (event, market, selection, opening odd, settled won 0/1)
for every exotic market we can settle from results (FT score, HT score,
goals_json). Opening snapshot = MIN(odds_snapshots.id) per event.

Guards:
  - exclude exports/corrupted_events.json ids (covers 8035)
  - generic guard for ALL leagues: drop event if ht_score > ft_score (corruption)
  - goals_json-based markets only settled when goals_json is parseable,
    len == score_a+score_b and last cumulative score == FT score
  - odds capped at 100.0 are dropped (known dead cells)

Outputs:
  exports/_wf4_teamtotals_bets.pkl   (DataFrame of bets)
  exports/wf4_teamtotals_margins.json (margin per market per league)
"""
import sys, json, collections
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from scraper.config import load_settings

CORRUPTED = set()
with open('exports/corrupted_events.json', 'r', encoding='utf-8') as f:
    d = json.load(f)
    CORRUPTED = set(int(k) for k in d['events'].keys())
print(f"corrupted ids loaded: {len(CORRUPTED)}")

eng = create_engine(load_settings().db_url)
q = """
SELECT e.id AS event_id, e.competition, e.team_a, e.team_b, e.expected_start,
       os.id AS snap_id, os.odds_home, os.odds_draw, os.odds_away, os.extra_markets,
       r.score_a, r.score_b, r.ht_score_a, r.ht_score_b, r.goals_json
FROM events e
JOIN (SELECT event_id, MIN(id) AS mid FROM odds_snapshots GROUP BY event_id) m
     ON m.event_id = e.id
JOIN odds_snapshots os ON os.id = m.mid
JOIN results r ON r.event_id = e.id
WHERE r.score_a IS NOT NULL AND r.score_b IS NOT NULL
ORDER BY e.expected_start, e.id
"""
with eng.connect() as c:
    rows = c.execute(text(q)).fetchall()
print(f"raw joined rows: {len(rows)}")


def parse_goals(gj, sa, sb):
    """Return list of goal dicts if consistent with FT score, else None."""
    if not gj:
        return None
    try:
        g = json.loads(gj) if isinstance(gj, str) else gj
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(g, list) or len(g) != sa + sb:
        return None
    try:
        last = max(g, key=lambda x: (x['minute'],))
        # cumulative check on final entry
        hs = max(int(x['homeScore']) for x in g)
        as_ = max(int(x['awayScore']) for x in g)
        if hs != sa or as_ != sb:
            return None
    except (KeyError, ValueError, TypeError):
        return None
    return g


def settle(market, sel, sa, sb, hta, htb, goals):
    """Return 1/0 if settleable, None if not settleable for this event."""
    tot = sa + sb
    ht_ok = hta is not None and htb is not None
    if market in ('Total equipe domicile',):
        return int(sa >= 4) if sel == '> 3.5' else (int(sa <= 3) if sel == '< 3.5' else None)
    if market in ('Total equipe extérieur',):
        return int(sb >= 4) if sel == '> 3.5' else (int(sb <= 3) if sel == '< 3.5' else None)
    if market == '+/-':
        return int(tot >= 4) if sel == '> 3.5' else (int(tot <= 3) if sel == '< 3.5' else None)
    if market == 'Total de buts':
        try:
            k = int(sel)
        except ValueError:
            return None
        return int(tot >= 6) if k == 6 else int(tot == k)
    if market == 'Pair/Impair':
        return int(tot % 2 == 0) if sel == 'Pair' else int(tot % 2 == 1)
    if market == 'G/NG':
        w = sa > 0 and sb > 0
        return int(w) if sel == 'Oui' else int(not w)
    if market == 'G/NG equipe domicile':
        return int(sa > 0) if sel == 'Oui' else int(sa == 0)
    if market == 'G/NG equipe extérieur':
        return int(sb > 0) if sel == 'Oui' else int(sb == 0)
    if market == 'Les deux équipes marquent / 1ère mi temps':
        if not ht_ok:
            return None
        w = hta > 0 and htb > 0
        return int(w) if sel == 'Oui' else int(not w)
    if market == '1X2 & G/NG':
        if 'aucun but' in sel:
            return int(sa == sb == 0)
        if sel.startswith('X et les deux'):
            return int(sa == sb and sa > 0)
        if sel.startswith('1 gagne et les deux'):
            return int(sa > sb and sb > 0)
        if sel.startswith('2 gagne et les deux'):
            return int(sb > sa and sa > 0)
        if sel.startswith('1 gagne et seulement'):
            return int(sa > sb and sb == 0)
        if sel.startswith('2 gagne et seulement'):
            return int(sb > sa and sa == 0)
        return None
    if market == '1X2 & Total':
        parts = sel.split(' / ')
        if len(parts) != 2:
            return None
        res, line = parts
        r_ok = (sa > sb) if res == '1' else ((sa == sb) if res == 'X' else (sb > sa))
        t_ok = (tot <= 3) if line.startswith('<') else (tot >= 4)
        return int(r_ok and t_ok)
    if market == 'Double Chance':
        if sel == '1X': return int(sa >= sb)
        if sel == 'X2': return int(sb >= sa)
        if sel == '12': return int(sa != sb)
        return None
    if market == 'Mi-tps DC':
        if not ht_ok: return None
        if sel == '1X': return int(hta >= htb)
        if sel == 'X2': return int(htb >= hta)
        if sel == '12': return int(hta != htb)
        return None
    if market == 'Mi-tps 1X2':
        if not ht_ok: return None
        if sel == '1': return int(hta > htb)
        if sel == 'X': return int(hta == htb)
        if sel == '2': return int(htb > hta)
        return None
    if market == 'Mi-tps CS':
        if not ht_ok: return None
        try:
            h, a = sel.split('-'); h, a = int(h), int(a)
        except ValueError:
            return None
        return int(hta == h and htb == a)
    if market == '2ème mi-tps - CS':
        if not ht_ok: return None
        h2a, h2b = sa - hta, sb - htb
        try:
            h, a = sel.split('-'); h, a = int(h), int(a)
        except ValueError:
            return None
        return int(h2a == h and h2b == a)
    if market == 'HT/FT':
        if not ht_ok: return None
        try:
            p1, p2 = sel.split('/')
        except ValueError:
            return None
        htr = '1' if hta > htb else ('2' if htb > hta else 'X')
        ftr = '1' if sa > sb else ('2' if sb > sa else 'X')
        return int(htr == p1 and ftr == p2)
    if market == 'FTTS':
        if sel == 'Pas de but':
            return int(tot == 0)
        if goals is None:
            return None
        if tot == 0:
            return 0
        first = min(goals, key=lambda x: (x['minute'], int(x['homeScore']) + int(x['awayScore'])))
        # first goal = entry with cumulative sum == 1
        firsts = [x for x in goals if int(x['homeScore']) + int(x['awayScore']) == 1]
        if not firsts:
            return None
        ft = firsts[0]['team']
        return int(ft == 'Home') if sel == '1' else int(ft == 'Away')
    if market == 'Minute du premier but':
        if sel == 'Pas de but':
            return int(tot == 0)
        if tot == 0:
            return 0
        if goals is None:
            return None
        firsts = [x for x in goals if int(x['homeScore']) + int(x['awayScore']) == 1]
        if not firsts:
            return None
        m = firsts[0]['minute']
        try:
            lo, hi = sel.split('-'); lo, hi = int(lo), int(hi)
        except ValueError:
            return None
        return int(lo <= m <= hi)
    if market == 'Score exact':
        try:
            h, a = sel.split('-'); h, a = int(h), int(a)
        except ValueError:
            return None
        return int(sa == h and sb == a)
    return None  # Multi-Buts and anything unknown -> skip


SKIP_MARKETS = {'Multi-Buts'}

bet_rows = []
margins = collections.defaultdict(lambda: collections.defaultdict(list))
n_kept = 0
n_drop_corrupt = n_drop_htft = 0
for r in rows:
    eid = r.event_id
    if eid in CORRUPTED:
        n_drop_corrupt += 1
        continue
    sa, sb = int(r.score_a), int(r.score_b)
    hta = int(r.ht_score_a) if r.ht_score_a is not None else None
    htb = int(r.ht_score_b) if r.ht_score_b is not None else None
    if hta is not None and (hta > sa or htb > sb):
        n_drop_htft += 1
        continue
    if r.extra_markets is None:
        continue
    try:
        em = json.loads(r.extra_markets) if isinstance(r.extra_markets, str) else r.extra_markets
    except (json.JSONDecodeError, TypeError):
        continue
    if not isinstance(em, dict):
        continue
    goals = parse_goals(r.goals_json, sa, sb)
    n_kept += 1
    league = r.competition.replace('InstantLeague-', '')
    for mk, sels in em.items():
        if mk in SKIP_MARKETS or not isinstance(sels, dict):
            continue
        # margin (full book incl. capped cells)
        try:
            ov = sum(1.0 / float(o) for o in sels.values() if o and float(o) > 1.0)
            margins[mk][league].append(ov)
        except (TypeError, ValueError):
            pass
        for sel, odd in sels.items():
            try:
                odd = float(odd)
            except (TypeError, ValueError):
                continue
            if odd <= 1.0 or odd >= 100.0:
                continue
            won = settle(mk, sel, sa, sb, hta, htb, goals)
            if won is None:
                continue
            bet_rows.append((eid, league, r.expected_start, mk, sel, odd, won,
                             r.odds_home, r.odds_draw, r.odds_away))

print(f"events kept: {n_kept}  dropped corrupted: {n_drop_corrupt}  dropped ht>ft: {n_drop_htft}")
df = pd.DataFrame(bet_rows, columns=['event_id', 'league', 'expected_start', 'market', 'sel',
                                     'odd', 'won', 'oh', 'od', 'oa'])
print(f"bet rows: {len(df)}")
df.to_pickle('exports/_wf4_teamtotals_bets.pkl')

marg_out = {}
for mk, per_lg in margins.items():
    marg_out[mk] = {lg: round(float(np.mean(v)) - 1.0, 4) for lg, v in per_lg.items()}
with open('exports/wf4_teamtotals_margins.json', 'w', encoding='utf-8') as f:
    json.dump(marg_out, f, ensure_ascii=False, indent=1)

print("\n--- mean margin per market (pooled across leagues) ---")
for mk in sorted(marg_out):
    allv = [v for lg in marg_out[mk].values() for v in [lg]]
    print(f"{mk!r}: pooled mean margin = {np.mean(allv):+.3f}  per-league: {marg_out[mk]}")
