"""Walk-forward parlays — TIER1 simplifie + jambes marches annexes.

Methodo stricte anti-leakage :
  - dedupe (team_a, team_b, expected_start) -> MIN(event id)
  - tri temporel par expected_start, split 70/30 AU NIVEAU ROUND (un round
    n'est jamais a cheval train/OOS)
  - toutes les regles de selection (seuils, types de jambes) sont fixes
    a priori ou choisies sur le train ; seules les metriques OOS sont reportees
  - roi = mean(won*(cote_combinee-1) - (1-won)), 1 unite par parlay

TIER1 simplifie (spec mission) : proba 1X2 devigorisee >= 0.65 ET cote <= 1.55
(home ou away).

NOTE DONNEES : le marche '+/-' ne cote QUE la ligne 3.5 (jamais 1.5) et il
n'existe aucun marche 'Over 0.5 1ere mi-temps'. Les jambes 'sures' testees sont
donc celles reellement bettables : '< 3.5', 'G/NG Non',
'Les deux equipes marquent / 1ere mi temps' = Non, 'Double Chance' favori,
'Multi-Buts 1-2-3', et '> 3.5' comme jambe haute cote.
"""
from __future__ import annotations

import itertools
import json
import sys
from collections import defaultdict

sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from scraper.config import load_settings

EPS = 1e-9
RNG = np.random.default_rng(42)

BTTS1H = 'Les deux équipes marquent / 1ère mi temps'
MB123 = 'Le total de buts est de 1, 2 ou 3'
MB012 = 'Le total de buts est de 0, 1 ou 2'
MB234 = 'Le total de buts est de 2, 3 ou 4'
MB4P = 'Le total de buts est supérieur à 4'


# ---------------------------------------------------------------- data ------

def load_data():
    engine = create_engine(load_settings().db_url)
    df = pd.read_sql(
        """
        SELECT e.id ev_id, e.team_a, e.team_b, e.expected_start, e.round_info,
               o.odds_home, o.odds_draw, o.odds_away, o.extra_markets,
               r.score_a, r.score_b, r.ht_score_a, r.ht_score_b
        FROM events e
        JOIN odds_snapshots o
             ON o.id = (SELECT MIN(id) FROM odds_snapshots WHERE event_id = e.id)
        JOIN results r ON r.event_id = e.id
        WHERE r.score_a IS NOT NULL AND r.score_b IS NOT NULL
              AND o.odds_home IS NOT NULL AND o.odds_draw IS NOT NULL
              AND o.odds_away IS NOT NULL
        ORDER BY e.expected_start, e.id
        """,
        engine,
    )
    # dedupe doublons de scraping (2 formats de match_key pour le meme match)
    df = df.drop_duplicates(subset=['team_a', 'team_b', 'expected_start'], keep='first')
    df = df.reset_index(drop=True)
    df['round_id'] = df['expected_start']  # un round = matchs simultanes
    return df


def parse_em(em):
    if em is None:
        return {}
    if isinstance(em, str):
        try:
            em = json.loads(em)
        except (json.JSONDecodeError, TypeError):
            return {}
    return em if isinstance(em, dict) else {}


def segment_of(round_info):
    try:
        j = int(round_info)
    except (TypeError, ValueError):
        return '?'
    if j <= 3:
        return 'DS'
    if j <= 12:
        return 'MS_early'
    if j <= 25:
        return 'MS_mid'
    if j <= 33:
        return 'MS_late'
    return 'FS'


# ---------------------------------------------------------------- legs ------

def build_legs(df):
    """Une ligne par jambe candidate. Champs : ev_id, round_id, leg_type,
    cote, won, p_devig (si dispo), segment."""
    legs = []
    for r in df.itertuples():
        sa, sb = int(r.score_a), int(r.score_b)
        total = sa + sb
        out = '1' if sa > sb else ('X' if sa == sb else '2')
        seg = segment_of(r.round_info)
        em = parse_em(r.extra_markets)

        imp = np.array([1.0 / r.odds_home, 1.0 / r.odds_draw, 1.0 / r.odds_away])
        p = imp / imp.sum()
        base = dict(ev_id=r.ev_id, round_id=r.round_id, segment=seg,
                    teams=f'{r.team_a}-{r.team_b}')

        # --- TIER1 simplifie 1X2
        if p[0] >= 0.65 and r.odds_home <= 1.55:
            legs.append(dict(base, leg_type='T1', cote=float(r.odds_home),
                             won=int(out == '1'), p_devig=float(p[0])))
        elif p[2] >= 0.65 and r.odds_away <= 1.55:
            legs.append(dict(base, leg_type='T1', cote=float(r.odds_away),
                             won=int(out == '2'), p_devig=float(p[2])))

        # --- Double chance cote du favori (jambe tres sure)
        dc = em.get('Double Chance') or {}
        fav_is_home = r.odds_home <= r.odds_away
        dc_key = '1X' if fav_is_home else 'X2'
        if dc_key in dc:
            dc_won = int(out in ('1', 'X')) if fav_is_home else int(out in ('X', '2'))
            legs.append(dict(base, leg_type='DCfav', cote=float(dc[dc_key]),
                             won=dc_won, p_devig=float(p[0] + p[1]) if fav_is_home else float(p[1] + p[2])))

        # --- Under / Over 3.5 (seule ligne +/- cotee)
        ou = em.get('+/-') or {}
        if '< 3.5' in ou:
            legs.append(dict(base, leg_type='U35', cote=float(ou['< 3.5']),
                             won=int(total <= 3), p_devig=None))
        if '> 3.5' in ou:
            legs.append(dict(base, leg_type='O35', cote=float(ou['> 3.5']),
                             won=int(total >= 4), p_devig=None))

        # --- G/NG
        gng = em.get('G/NG') or {}
        if 'Non' in gng:
            legs.append(dict(base, leg_type='GNGnon', cote=float(gng['Non']),
                             won=int(sa == 0 or sb == 0), p_devig=None))
        if 'Oui' in gng:
            legs.append(dict(base, leg_type='GNGoui', cote=float(gng['Oui']),
                             won=int(sa > 0 and sb > 0), p_devig=None))

        # --- BTTS 1ere mi-temps Non (besoin du score HT)
        b1h = em.get(BTTS1H) or {}
        if 'Non' in b1h and pd.notna(r.ht_score_a) and pd.notna(r.ht_score_b):
            ha, hb = int(r.ht_score_a), int(r.ht_score_b)
            legs.append(dict(base, leg_type='B1HNon', cote=float(b1h['Non']),
                             won=int(ha == 0 or hb == 0), p_devig=None))

        # --- Multi-buts
        mb = em.get('Multi-Buts') or {}
        if MB123 in mb:
            legs.append(dict(base, leg_type='MB123', cote=float(mb[MB123]),
                             won=int(1 <= total <= 3), p_devig=None))
        if MB4P in mb:
            legs.append(dict(base, leg_type='MB4plus', cote=float(mb[MB4P]),
                             won=int(total >= 5), p_devig=None))

        # --- BOOSTERS meme match sur pick T1 (cote>=2 sur un match "sur") ---
        t1_side = None
        if p[0] >= 0.65 and r.odds_home <= 1.55:
            t1_side = '1'
        elif p[2] >= 0.65 and r.odds_away <= 1.55:
            t1_side = '2'
        if t1_side is not None:
            fav_won = int(out == t1_side)
            ht_ok = pd.notna(r.ht_score_a) and pd.notna(r.ht_score_b)
            if ht_ok:
                ha, hb = int(r.ht_score_a), int(r.ht_score_b)
                ht_out = '1' if ha > hb else ('X' if ha == hb else '2')
            # HT/FT fav/fav
            htft = em.get('HT/FT') or {}
            key = f'{t1_side}/{t1_side}'
            if key in htft and ht_ok:
                legs.append(dict(base, leg_type='T1B_HTFT',
                                 cote=float(htft[key]),
                                 won=int(fav_won and ht_out == t1_side), p_devig=None))
            # Mi-tps 1X2 fav (le fav mene a la MT)
            ht12 = em.get('Mi-tps 1X2') or {}
            if t1_side in ht12 and ht_ok:
                legs.append(dict(base, leg_type='T1B_HT1X2',
                                 cote=float(ht12[t1_side]),
                                 won=int(ht_out == t1_side), p_devig=None))
            # 1X2 & Total : fav & < 3.5  /  fav & > 3.5
            x2t = em.get('1X2 & Total') or {}
            ku, ko = f'{t1_side} / < 3.5', f'{t1_side} / > 3.5'
            if ku in x2t:
                legs.append(dict(base, leg_type='T1B_U35',
                                 cote=float(x2t[ku]),
                                 won=int(fav_won and total <= 3), p_devig=None))
            if ko in x2t:
                legs.append(dict(base, leg_type='T1B_O35',
                                 cote=float(x2t[ko]),
                                 won=int(fav_won and total >= 4), p_devig=None))
            # 1X2 & G/NG : fav gagne et les deux marquent / seul le fav marque
            xg = em.get('1X2 & G/NG') or {}
            k_btts = f'{t1_side} gagne et les deux équipes marquent'
            if k_btts in xg:
                legs.append(dict(base, leg_type='T1B_BTTS',
                                 cote=float(xg[k_btts]),
                                 won=int(fav_won and sa > 0 and sb > 0), p_devig=None))
            k_only = ('1 gagne et seulement  1  marque' if t1_side == '1'
                      else '2 gagne et seulement 2 marque')
            if k_only in xg:
                opp_zero = (sb == 0) if t1_side == '1' else (sa == 0)
                legs.append(dict(base, leg_type='T1B_ONLY',
                                 cote=float(xg[k_only]),
                                 won=int(fav_won and opp_zero), p_devig=None))
    return pd.DataFrame(legs)


# --------- jambes VALUE 1X2 / marches par segment (grille fixee sur train) --

SIDE_BUCKETS = [(1.55, 2.2), (2.2, 3.2), (3.2, 5.0), (5.0, 15.0)]
MKT_FOR_SEG = ['U35', 'O35', 'GNGoui', 'GNGnon', 'B1HNon', 'MB123', 'MB4plus']


def build_1x2_value_legs(df):
    rows = []
    for r in df.itertuples():
        sa, sb = int(r.score_a), int(r.score_b)
        out = '1' if sa > sb else ('X' if sa == sb else '2')
        seg = segment_of(r.round_info)
        for side, cote in [('1', r.odds_home), ('X', r.odds_draw), ('2', r.odds_away)]:
            for lo, hi in SIDE_BUCKETS:
                if lo <= cote < hi:
                    rows.append(dict(ev_id=r.ev_id, round_id=r.round_id,
                                     segment=seg, side=side, bucket=f'[{lo},{hi})',
                                     cote=float(cote), won=int(out == side)))
    return pd.DataFrame(rows)


def grid_select(vleg_tr, legs_tr, min_n=80, min_roi=0.08):
    """Retourne les cellules (definition, df_train) avec roi_train>=min_roi."""
    selected = []
    # 1X2 value : segment x side x bucket
    for (seg, side, bucket), sub in vleg_tr.groupby(['segment', 'side', 'bucket']):
        roi = (sub.won * (sub.cote - 1) - (1 - sub.won)).mean()
        if len(sub) >= min_n and roi >= min_roi:
            selected.append((f'1X2 {side} {seg} cote{bucket}',
                             dict(kind='1x2', seg=seg, side=side, bucket=bucket),
                             len(sub), sub.won.mean(), roi))
    # marches x segment
    for lt in MKT_FOR_SEG:
        for seg, sub in legs_tr[legs_tr.leg_type == lt].groupby('segment'):
            roi = (sub.won * (sub.cote - 1) - (1 - sub.won)).mean()
            if len(sub) >= min_n and roi >= min_roi:
                selected.append((f'{lt} {seg}',
                                 dict(kind='mkt', seg=seg, leg_type=lt),
                                 len(sub), sub.won.mean(), roi))
    return selected


def pair_sequential(rows, k=2):
    """Parlays NON chevauchants : jambes triees chronologiquement, paquets de k
    provenant de rounds distincts (matchs distincts garantis)."""
    rows = sorted(rows, key=lambda x: (x['round_id'], x['ev_id']))
    out, buf, used_rounds = [], [], set()
    for leg in rows:
        if leg['round_id'] in used_rounds:
            continue
        buf.append(leg); used_rounds.add(leg['round_id'])
        if len(buf) == k:
            out.append(dict(won=int(all(l['won'] for l in buf)),
                            cote=float(np.prod([l['cote'] for l in buf]))))
            buf, used_rounds = [], set()
    return pd.DataFrame(out)


# ---------------------------------------------------------------- utils -----

def stat_line(name, sub, extra=''):
    n = len(sub)
    if n == 0:
        print(f'  {name:<42} n=0')
        return None
    wr = sub.won.mean()
    cote = sub.cote.mean()
    roi = (sub.won * (sub.cote - 1) - (1 - sub.won)).mean()
    flag = '' if n >= 30 else '  [INSTABLE n<30]'
    print(f'  {name:<42} n={n:<5} wr={wr*100:5.1f}%  cote={cote:5.2f}  ROI={roi*100:+6.1f}%{flag}{extra}')
    return dict(n=n, wr=wr, cote=cote, roi=roi)


def parlay_rows(groups, k, max_combos=6):
    """groups: dict round_id -> list of leg rows (dict). Retourne lignes parlay
    (won, cote) en formant toutes les combinaisons C(n,k) plafonnees."""
    rows = []
    for rid, ls in groups.items():
        if len(ls) < k:
            continue
        combos = list(itertools.combinations(ls, k))
        if len(combos) > max_combos:
            idx = RNG.choice(len(combos), size=max_combos, replace=False)
            combos = [combos[i] for i in idx]
        for combo in combos:
            evs = {c['ev_id'] for c in combo}
            if len(evs) < k:      # jamais 2 jambes sur le meme match
                continue
            won = int(all(c['won'] for c in combo))
            cote = float(np.prod([c['cote'] for c in combo]))
            rows.append(dict(won=won, cote=cote, round_id=rid))
    return pd.DataFrame(rows)


def cross_round_pairs(legs_a, legs_b, max_pairs=4000):
    """Paires (jambe round t, jambe round t' != t). legs_* tries par round."""
    rows = []
    rounds_a = sorted(legs_a.keys())
    for i, ra in enumerate(rounds_a):
        # apparier avec le round suivant ayant des jambes dans legs_b
        for rb in rounds_a[i + 1:i + 4]:
            if rb == ra or rb not in legs_b:
                continue
            for la in legs_a[ra][:3]:
                for lb in legs_b[rb][:3]:
                    rows.append(dict(
                        won=int(la['won'] and lb['won']),
                        cote=la['cote'] * lb['cote'],
                        w1=la['won'], w2=lb['won']))
            break
        if len(rows) >= max_pairs:
            break
    return pd.DataFrame(rows)


def group_by_round(sub):
    g = defaultdict(list)
    for row in sub.to_dict('records'):
        g[row['round_id']].append(row)
    return g


# ---------------------------------------------------------------- main ------

def main():
    df = load_data()
    print(f'matchs uniques avec cotes+score : {len(df)}')

    # split 70/30 EN MATCHS, a une frontiere de round
    rounds = sorted(df.round_id.unique())
    counts = df.round_id.value_counts()
    cum, cut_round = 0, rounds[-1]
    target = int(len(df) * 0.70)
    for rid in rounds:
        cum += counts[rid]
        if cum >= target:
            cut_round = rid
            break
    train_df = df[df.round_id < cut_round]
    oos_df = df[df.round_id >= cut_round]
    print(f'rounds={len(rounds)}  train_matchs={len(train_df)}  oos_matchs={len(oos_df)}')
    print(f'cut={cut_round}')

    legs = build_legs(df)
    legs_tr = legs[legs.round_id < cut_round]
    legs_oo = legs[legs.round_id >= cut_round]

    # ---------------------------------------------------------- 1. jambes ---
    print('\n=== 1. JAMBES INDIVIDUELLES (train | oos) ===')
    leg_stats = {}
    for lt in ['T1', 'DCfav', 'U35', 'O35', 'GNGnon', 'GNGoui', 'B1HNon', 'MB123', 'MB4plus']:
        tr = legs_tr[legs_tr.leg_type == lt]
        oo = legs_oo[legs_oo.leg_type == lt]
        s_tr = stat_line(f'{lt} [train]', tr)
        s_oo = stat_line(f'{lt} [oos]  ', oo)
        leg_stats[lt] = dict(train=s_tr, oos=s_oo)

    # T1 par bucket de proba devig sur train (choisir raffinement eventuel)
    print('\n--- T1 par bucket p_devig (train) ---')
    t1tr = legs_tr[legs_tr.leg_type == 'T1']
    for lo, hi in [(0.65, 0.70), (0.70, 0.75), (0.75, 1.01)]:
        stat_line(f'T1 p[{lo:.2f},{hi:.2f})', t1tr[(t1tr.p_devig >= lo) & (t1tr.p_devig < hi)])

    # ------------------------------------------------- 2. correlation -------
    print('\n=== 2. CORRELATION INTRA-ROUND DES PICKS T1 ===')
    for label, sub in [('train', legs_tr), ('oos', legs_oo)]:
        t1 = sub[sub.leg_type == 'T1']
        g = group_by_round(t1)
        same = []
        for rid, ls in g.items():
            for a, b in itertools.combinations(ls, 2):
                same.append((a['won'], b['won']))
        cross = cross_round_pairs(g, g)
        p_ind = t1.won.mean() ** 2
        if same:
            sj = np.mean([a and b for a, b in same])
            w1 = np.array([a for a, _ in same]); w2 = np.array([b for _, b in same])
            phi = np.corrcoef(w1, w2)[0, 1] if w1.std() > 0 and w2.std() > 0 else float('nan')
            se = np.sqrt(sj * (1 - sj) / len(same))
            print(f'  [{label}] paires MEME round    n={len(same):<5} joint={sj*100:5.1f}%  '
                  f'(SE {se*100:.1f}pp)  attendu indep={p_ind*100:5.1f}%  phi={phi:+.3f}')
        if len(cross):
            cj = cross.won.mean()
            se = np.sqrt(cj * (1 - cj) / len(cross))
            print(f'  [{label}] paires rounds DIFF    n={len(cross):<5} joint={cj*100:5.1f}%  '
                  f'(SE {se*100:.1f}pp)  attendu indep={p_ind*100:5.1f}%')
        # conditionnel : P(win | co-pick gagne) vs P(win | co-pick perd)
        if same:
            w_w = [b for a, b in same if a] + [a for a, b in same if b]
            w_l = [b for a, b in same if not a] + [a for a, b in same if not b]
            if w_w and w_l:
                print(f'  [{label}] P(win|co-pick W)={np.mean(w_w)*100:5.1f}% (n={len(w_w)})  '
                      f'P(win|co-pick L)={np.mean(w_l)*100:5.1f}% (n={len(w_l)})')

    # ------------------------------------------------- 3. parlays T1 --------
    print('\n=== 3. PARLAYS T1 PURS (OOS uniquement) ===')
    findings = []
    t1_oo = legs_oo[legs_oo.leg_type == 'T1']
    g_oo = group_by_round(t1_oo)
    t1_tr_g = group_by_round(legs_tr[legs_tr.leg_type == 'T1'])

    for k in (2, 3, 4):
        # meme round
        tr_p = parlay_rows(t1_tr_g, k)
        oo_p = parlay_rows(g_oo, k)
        s_tr = stat_line(f'T1 x{k} MEME round [train]', tr_p)
        s_oo = stat_line(f'T1 x{k} MEME round [oos]  ', oo_p)
        findings.append((f'T1x{k}_same_round', s_tr, s_oo,
                         f'{k} jambes TIER1 (devig>=0.65, cote<=1.55) du meme round, '
                         f'toutes combinaisons C(n,{k}) plafonnees a 6/round, jamais 2 jambes meme match'))

    # cross-round 2 jambes
    tr_cross = cross_round_pairs(t1_tr_g, t1_tr_g)
    oo_cross = cross_round_pairs(g_oo, g_oo)
    s_tr = stat_line('T1 x2 rounds DIFFERENTS [train]', tr_cross)
    s_oo = stat_line('T1 x2 rounds DIFFERENTS [oos]  ', oo_cross)
    findings.append(('T1x2_cross_round', s_tr, s_oo,
                     '2 jambes TIER1 de 2 rounds distincts (round t x premier round suivant avec pick, <=3 picks chacun)'))

    # ------------------------------------------------- 4. parlays mixtes ----
    print('\n=== 4. PARLAYS MIXTES T1 + jambe sure / haute cote (OOS) ===')

    def mixed_parlay(legs_all, lt2, k2=1, require_diff_match=True):
        """1 jambe T1 + k2 jambes lt2 du MEME round (matchs differents)."""
        sub_t1 = legs_all[legs_all.leg_type == 'T1']
        sub_2 = legs_all[legs_all.leg_type == lt2]
        g1, g2 = group_by_round(sub_t1), group_by_round(sub_2)
        rows = []
        for rid, l1s in g1.items():
            l2s = [l for l in g2.get(rid, [])]
            for l1 in l1s[:2]:
                cands = [l for l in l2s if l['ev_id'] != l1['ev_id']] if require_diff_match else l2s
                if len(cands) < k2:
                    continue
                for combo in itertools.combinations(cands[:4], k2):
                    evs = {l1['ev_id']} | {c['ev_id'] for c in combo}
                    if len(evs) < 1 + k2:
                        continue
                    won = int(l1['won'] and all(c['won'] for c in combo))
                    cote = l1['cote'] * float(np.prod([c['cote'] for c in combo]))
                    rows.append(dict(won=won, cote=cote))
        return pd.DataFrame(rows)

    mixes = [
        ('T1+U35', 'U35', 1, '1 jambe TIER1 + 1 jambe Under 3.5 (marche +/-) sur un AUTRE match du meme round'),
        ('T1+B1HNon', 'B1HNon', 1, '1 jambe TIER1 + 1 jambe BTTS-1ere-MT Non sur un autre match du meme round'),
        ('T1+GNGoui', 'GNGoui', 1, '1 jambe TIER1 + 1 jambe G/NG Oui sur un autre match du meme round'),
        ('T1+MB123', 'MB123', 1, '1 jambe TIER1 + 1 jambe Multi-Buts 1-2-3 sur un autre match du meme round'),
        ('T1+O35', 'O35', 1, '1 jambe TIER1 + 1 jambe Over 3.5 sur un autre match du meme round'),
        ('T1+2xB1HNon', 'B1HNon', 2, '1 jambe TIER1 + 2 jambes BTTS-1ere-MT Non (2 autres matchs du meme round)'),
        ('T1+DCfav', 'DCfav', 1, '1 jambe TIER1 + 1 jambe Double Chance favori sur un autre match du meme round'),
    ]
    for name, lt2, k2, desc in mixes:
        tr_p = mixed_parlay(legs_tr, lt2, k2)
        oo_p = mixed_parlay(legs_oo, lt2, k2)
        s_tr = stat_line(f'{name} [train]', tr_p)
        s_oo = stat_line(f'{name} [oos]  ', oo_p)
        findings.append((name, s_tr, s_oo, desc))

    # jambes sures pures x2 / x3 (sans T1)
    print('\n=== 5. PARLAYS JAMBES MARCHES SEULES (OOS) ===')
    for lt, k in [('B1HNon', 2), ('B1HNon', 3), ('U35', 2), ('GNGoui', 2), ('O35', 2)]:
        g_tr = group_by_round(legs_tr[legs_tr.leg_type == lt])
        g_oo2 = group_by_round(legs_oo[legs_oo.leg_type == lt])
        tr_p = parlay_rows(g_tr, k)
        oo_p = parlay_rows(g_oo2, k)
        s_tr = stat_line(f'{lt} x{k} meme round [train]', tr_p)
        s_oo = stat_line(f'{lt} x{k} meme round [oos]  ', oo_p)
        findings.append((f'{lt}x{k}_same_round', s_tr, s_oo,
                         f'{k} jambes {lt} du meme round, matchs differents, combos plafonnes a 6/round'))

    # ------------------------------------------------- 6. correlation legs --
    print('\n=== 6. CORRELATION INTRA-ROUND T1 x AUTRES JAMBES (pooled, diagnostic) ===')
    for lt2 in ['U35', 'O35', 'B1HNon', 'GNGoui']:
        t1g = group_by_round(legs[legs.leg_type == 'T1'])
        l2g = group_by_round(legs[legs.leg_type == lt2])
        joint, w1m, w2m = [], [], []
        for rid, l1s in t1g.items():
            for l1 in l1s:
                for l2 in l2g.get(rid, []):
                    if l2['ev_id'] == l1['ev_id']:
                        continue
                    joint.append(l1['won'] and l2['won'])
                    w1m.append(l1['won']); w2m.append(l2['won'])
        if joint:
            j = np.mean(joint); ind = np.mean(w1m) * np.mean(w2m)
            phi = np.corrcoef(w1m, w2m)[0, 1]
            print(f'  T1 x {lt2:<8} n={len(joint):<6} joint={j*100:5.1f}%  indep={ind*100:5.1f}%  phi={phi:+.3f}')

    # ------------------------------------- 7. boosters meme match T1 --------
    print('\n=== 7. BOOSTERS MEME MATCH SUR PICK T1 (cote >= 2 sur match sur) ===')
    print('   (marches combines bookmaker : HT/FT, 1X2&Total, 1X2&G/NG, Mi-tps 1X2)')
    booster_types = ['T1B_HT1X2', 'T1B_HTFT', 'T1B_U35', 'T1B_O35', 'T1B_BTTS', 'T1B_ONLY']
    for lt in booster_types:
        s_tr = stat_line(f'{lt} [train]', legs_tr[legs_tr.leg_type == lt])
        s_oo = stat_line(f'{lt} [oos]  ', legs_oo[legs_oo.leg_type == lt])
        findings.append((lt, s_tr, s_oo, 'booster meme match sur pick T1'))

    # T1B par bucket p_devig du pick T1 sous-jacent (train -> filtre eventuel)
    print('\n--- boosters restreints aux T1 p_devig>=0.70 (jointure via ev_id) ---')
    t1p = legs[(legs.leg_type == 'T1') & (legs.p_devig >= 0.70)].ev_id
    for lt in booster_types:
        sub_tr = legs_tr[(legs_tr.leg_type == lt) & (legs_tr.ev_id.isin(t1p))]
        sub_oo = legs_oo[(legs_oo.leg_type == lt) & (legs_oo.ev_id.isin(t1p))]
        s_tr = stat_line(f'{lt}|p>=.70 [train]', sub_tr)
        s_oo = stat_line(f'{lt}|p>=.70 [oos]  ', sub_oo)
        findings.append((f'{lt}_p70', s_tr, s_oo, 'booster, pick T1 sous-jacent p_devig>=0.70'))

    # ------------------------------------- 8. jambes value (grille train) ---
    print('\n=== 8. JAMBES VALUE — grille segment x cote fixee sur TRAIN, validee OOS ===')
    vleg = build_1x2_value_legs(df)
    vleg_tr = vleg[vleg.round_id < cut_round]
    vleg_oo = vleg[vleg.round_id >= cut_round]
    selected = grid_select(vleg_tr, legs_tr)
    print(f'  cellules retenues sur train (n>=80, roi>=+8%) : {len(selected)}')
    value_oos_legs = {}
    for name, spec, n_tr, wr_tr, roi_tr in selected:
        if spec['kind'] == '1x2':
            sub_oo = vleg_oo[(vleg_oo.segment == spec['seg']) & (vleg_oo.side == spec['side'])
                             & (vleg_oo.bucket == spec['bucket'])]
        else:
            sub_oo = legs_oo[(legs_oo.leg_type == spec['leg_type'])
                             & (legs_oo.segment == spec['seg'])]
        print(f'  {name:<34} train: n={n_tr} wr={wr_tr*100:.1f}% roi={roi_tr*100:+.1f}%')
        s_oo = stat_line(f'    -> OOS', sub_oo)
        findings.append((f'VALUE {name}', dict(n=n_tr, wr=wr_tr, roi=roi_tr, cote=float('nan')),
                         s_oo, f'jambe value selectionnee sur train : {name}'))
        if s_oo and s_oo['roi'] > 0:
            value_oos_legs[name] = (spec, sub_oo)

    # ------------------------------------- 9. parlays jambes value ----------
    print('\n=== 9. PARLAYS AVEC JAMBES VALUE (selection train -> eval OOS) ===')
    # toutes jambes value train-selectionnees, poolees, en OOS
    def collect_value_rows(vleg_part, legs_part):
        rows = []
        for name, spec, *_ in selected:
            if spec['kind'] == '1x2':
                sub = vleg_part[(vleg_part.segment == spec['seg']) & (vleg_part.side == spec['side'])
                                & (vleg_part.bucket == spec['bucket'])]
                for rr in sub.to_dict('records'):
                    rows.append(dict(rr, leg_type=name))
            else:
                sub = legs_part[(legs_part.leg_type == spec['leg_type'])
                                & (legs_part.segment == spec['seg'])]
                for rr in sub.to_dict('records'):
                    rows.append(dict(rr, leg_type=name))
        # dedupe par (ev_id) pour eviter 2 jambes sur le meme match
        seen, out = set(), []
        for rr in sorted(rows, key=lambda x: (x['round_id'], x['ev_id'])):
            if rr['ev_id'] in seen:
                continue
            seen.add(rr['ev_id']); out.append(rr)
        return out

    val_tr_rows = collect_value_rows(vleg_tr, legs_tr)
    val_oo_rows = collect_value_rows(vleg_oo, legs_oo)
    print(f'  jambes value poolees : train={len(val_tr_rows)}  oos={len(val_oo_rows)}')
    for k in (2, 3):
        tr_p = pair_sequential(val_tr_rows, k)
        oo_p = pair_sequential(val_oo_rows, k)
        s_tr = stat_line(f'VALUE x{k} sequentiel [train]', tr_p)
        s_oo = stat_line(f'VALUE x{k} sequentiel [oos]  ', oo_p)
        findings.append((f'VALUEx{k}_seq', s_tr, s_oo,
                         f'{k} jambes value (cellules train) de rounds distincts, parlays non chevauchants'))

    # value + T1 (jambe sure) : pour chaque jambe value, T1 du meme round autre match
    def value_plus_t1(val_rows, legs_part):
        t1g = group_by_round(legs_part[legs_part.leg_type == 'T1'])
        rows = []
        for vr in val_rows:
            for l1 in t1g.get(vr['round_id'], []):
                if l1['ev_id'] == vr['ev_id']:
                    continue
                rows.append(dict(won=int(vr['won'] and l1['won']),
                                 cote=vr['cote'] * l1['cote']))
                break
        return pd.DataFrame(rows)

    tr_p = value_plus_t1(val_tr_rows, legs_tr)
    oo_p = value_plus_t1(val_oo_rows, legs_oo)
    s_tr = stat_line('VALUE + T1 meme round [train]', tr_p)
    s_oo = stat_line('VALUE + T1 meme round [oos]  ', oo_p)
    findings.append(('VALUE+T1', s_tr, s_oo,
                     '1 jambe value + 1 pick T1 d un autre match du meme round'))

    # boosters x2 : 2 boosters T1B_U35 de matchs differents (rounds distincts)
    print('\n--- parlays de boosters (cote tres haute) ---')
    for lt in ['T1B_U35', 'T1B_HT1X2', 'T1B_HTFT']:
        tr_rows = legs_tr[legs_tr.leg_type == lt].to_dict('records')
        oo_rows = legs_oo[legs_oo.leg_type == lt].to_dict('records')
        tr_p = pair_sequential(tr_rows, 2)
        oo_p = pair_sequential(oo_rows, 2)
        s_tr = stat_line(f'{lt} x2 sequentiel [train]', tr_p)
        s_oo = stat_line(f'{lt} x2 sequentiel [oos]  ', oo_p)
        findings.append((f'{lt}x2', s_tr, s_oo, f'2 boosters {lt} de rounds distincts'))

    # ----------------------- 9b. robustesse : cellules poolees + 2 moities --
    print('\n=== 9b. ROBUSTESSE — cellules 1X2 POOLEES (sans segment, 12 cellules) ===')
    for side in ('1', 'X', '2'):
        for lo, hi in SIDE_BUCKETS:
            b = f'[{lo},{hi})'
            tr = vleg_tr[(vleg_tr.side == side) & (vleg_tr.bucket == b)]
            oo = vleg_oo[(vleg_oo.side == side) & (vleg_oo.bucket == b)]
            s_tr = stat_line(f'1X2 {side} cote{b} [train]', tr)
            s_oo = stat_line(f'1X2 {side} cote{b} [oos]  ', oo)

    print('\n=== 9c. SELECTION DURCIE — ROI>0 dans les 2 MOITIES du train, puis OOS ===')
    tr_rounds = sorted(vleg_tr.round_id.unique())
    half_cut = tr_rounds[len(tr_rounds) // 2]
    hardened = []
    for (seg, side, bucket), sub in vleg_tr.groupby(['segment', 'side', 'bucket']):
        h1 = sub[sub.round_id < half_cut]; h2 = sub[sub.round_id >= half_cut]
        if len(h1) < 40 or len(h2) < 40:
            continue
        r1 = (h1.won * (h1.cote - 1) - (1 - h1.won)).mean()
        r2 = (h2.won * (h2.cote - 1) - (1 - h2.won)).mean()
        if r1 > 0.03 and r2 > 0.03:
            hardened.append((seg, side, bucket, len(sub), r1, r2))
    print(f'  cellules stables sur les 2 moities du train : {len(hardened)}')
    hardened_oos_rows = []
    for seg, side, bucket, n_tr, r1, r2 in hardened:
        oo = vleg_oo[(vleg_oo.segment == seg) & (vleg_oo.side == side) & (vleg_oo.bucket == bucket)]
        print(f'  1X2 {side} {seg} cote{bucket}  train n={n_tr} roi_h1={r1*100:+.1f}% roi_h2={r2*100:+.1f}%')
        s_oo = stat_line('    -> OOS', oo)
        findings.append((f'HARD 1X2 {side} {seg} {bucket}',
                         dict(n=n_tr, wr=float('nan'), roi=(r1 + r2) / 2, cote=float('nan')),
                         s_oo, f'cellule durcie (ROI>+3% sur 2 moities du train) : side={side} seg={seg} cote{bucket}'))
        hardened_oos_rows.extend(dict(rr, leg_type=f'{side}{seg}{bucket}') for rr in oo.to_dict('records'))

    if hardened_oos_rows:
        # dedupe meme match
        seen, hrows = set(), []
        for rr in sorted(hardened_oos_rows, key=lambda x: (x['round_id'], x['ev_id'])):
            if rr['ev_id'] not in seen:
                seen.add(rr['ev_id']); hrows.append(rr)
        oo_p = pair_sequential(hrows, 2)
        s_oo = stat_line('HARDENED x2 sequentiel [oos]', oo_p)
        findings.append(('HARDENEDx2_seq', None, s_oo,
                         '2 jambes des cellules durcies, rounds distincts, non chevauchant'))
        oo_p = value_plus_t1(hrows, legs_oo)
        s_oo = stat_line('HARDENED + T1 meme round [oos]', oo_p)
        findings.append(('HARDENED+T1', None, s_oo,
                         '1 jambe cellule durcie + 1 pick T1 autre match du meme round'))

    # T1 haute proba (objectif accuracy max)
    print('\n=== 10. OBJECTIF (a) ACCURACY MAX — singles haute proba OOS ===')
    for lo in (0.70, 0.75):
        sub_tr = legs_tr[(legs_tr.leg_type == 'T1') & (legs_tr.p_devig >= lo)]
        sub_oo = legs_oo[(legs_oo.leg_type == 'T1') & (legs_oo.p_devig >= lo)]
        s_tr = stat_line(f'T1 p>={lo:.2f} [train]', sub_tr)
        s_oo = stat_line(f'T1 p>={lo:.2f} [oos]  ', sub_oo)
        findings.append((f'T1_p{int(lo*100)}', s_tr, s_oo, f'single 1X2 devig>={lo}, cote<=1.55'))

    return findings


if __name__ == '__main__':
    main()
