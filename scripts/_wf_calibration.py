# -*- coding: utf-8 -*-
"""
_wf_calibration.py — Walk-forward (70/30 temporel) :
  A. Calibration probas implicites marche (devig) vs WR reel, par bucket
  B. Filtre composite : favori home cote<=1.5 + equipe NON volatile (volatilite calculee sur train)
  C. Confirmation croisee : favori home + adversaire en mauvaise forme away sur train (delta <= -5pp)
  D. Double chance synthetique : 1X sur home cote [1.8-2.4] vs marche DC reel
  E. Value DC : cote DC reelle vs cote DC fair synthetique (devig), seuil choisi sur train
  F. Grille segment x bande de cote (selection train ROI>=+10%, n>=40) -> OOS poole (hautes cotes)
Anti-leakage : tout signal calcule sur train uniquement, metriques rapportees = OOS.
"""
import sys, json
sys.path.insert(0, '.')
from collections import defaultdict

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from scraper.config import load_settings

SEGMENTS = [
    ('DS', 1, 3), ('MS_early', 4, 12), ('MS_mid', 13, 25),
    ('MS_late', 26, 33), ('FS', 34, 38),
]


def seg_of(rnd):
    for name, lo, hi in SEGMENTS:
        if lo <= rnd <= hi:
            return name
    return None


def load_data():
    eng = create_engine(load_settings().db_url)
    q = """
    SELECT e.id, e.round_info, e.team_a, e.team_b, e.expected_start,
           o.odds_home, o.odds_draw, o.odds_away, o.extra_markets,
           r.score_a, r.score_b
    FROM events e
    JOIN results r ON r.event_id = e.id
    JOIN odds_snapshots o ON o.id = (
        SELECT MIN(o2.id) FROM odds_snapshots o2 WHERE o2.event_id = e.id)
    WHERE e.round_info != '0'
      AND r.score_a IS NOT NULL AND r.score_b IS NOT NULL
      AND o.odds_home IS NOT NULL AND o.odds_draw IS NOT NULL AND o.odds_away IS NOT NULL
    """
    with eng.connect() as c:
        df = pd.read_sql(text(q), c)
    df['round'] = pd.to_numeric(df['round_info'], errors='coerce')
    df = df.dropna(subset=['round'])
    df['round'] = df['round'].astype(int)
    df = df[(df['round'] >= 1) & (df['round'] <= 38)]
    df['seg'] = df['round'].map(seg_of)
    df['expected_start'] = pd.to_datetime(df['expected_start'])
    df = df.sort_values('expected_start').reset_index(drop=True)

    # outcome
    df['out'] = np.where(df.score_a > df.score_b, 'H',
                np.where(df.score_a < df.score_b, 'A', 'D'))

    # devig probs (proportionnel)
    inv = 1/df.odds_home + 1/df.odds_draw + 1/df.odds_away
    df['p_home'] = (1/df.odds_home) / inv
    df['p_draw'] = (1/df.odds_draw) / inv
    df['p_away'] = (1/df.odds_away) / inv
    df['overround'] = inv

    # marche DC reel
    def parse_dc(em):
        if em is None:
            return (np.nan, np.nan, np.nan)
        try:
            d = json.loads(em) if isinstance(em, str) else em
            dc = d.get('Double Chance') or {}
            return (dc.get('1X', np.nan), dc.get('X2', np.nan), dc.get('12', np.nan))
        except Exception:
            return (np.nan, np.nan, np.nan)
    dc = df.extra_markets.map(parse_dc)
    df['dc_1x'] = dc.map(lambda t: t[0]); df['dc_x2'] = dc.map(lambda t: t[1]); df['dc_12'] = dc.map(lambda t: t[2])
    return df


def roi(won, odds):
    won = np.asarray(won, float); odds = np.asarray(odds, float)
    return float(np.mean(won * (odds - 1) - (1 - won)))


def report(label, sub, side_won, odds_col):
    n = len(sub)
    if n == 0:
        return dict(name=label, n=0, wr=np.nan, avg=np.nan, roi=np.nan)
    wr = float(side_won.mean())
    avg = float(sub[odds_col].mean())
    r = roi(side_won.values, sub[odds_col].values)
    return dict(name=label, n=n, wr=wr, avg=avg, roi=r)


def main():
    df = load_data()
    n = len(df)
    cut = int(n * 0.70)
    train, oos = df.iloc[:cut].copy(), df.iloc[cut:].copy()
    print(f"TOTAL={n}  train={len(train)} ({train.expected_start.min()} -> {train.expected_start.max()})")
    print(f"             oos={len(oos)}  ({oos.expected_start.min()} -> {oos.expected_start.max()})")
    cov_base = len(oos)

    # =================================================================
    # A. CALIBRATION probas implicites (HOME) par bucket
    # =================================================================
    print("\n=== A. CALIBRATION marche (devig) — side HOME ===")
    bins = [0, .30, .40, .45, .50, .55, .60, .65, .70, .75, .80, 1.01]
    for name, d in [('TRAIN', train), ('OOS', oos)]:
        d = d.copy(); d['b'] = pd.cut(d.p_home, bins)
        g = d.groupby('b', observed=True).apply(
            lambda x: pd.Series({'n': len(x), 'implied': x.p_home.mean(),
                                 'wr': (x.out == 'H').mean(),
                                 'avg_cote': x.odds_home.mean(),
                                 'roi': roi((x.out == 'H').values, x.odds_home.values)}),
            include_groups=False)
        print(f"-- {name} --")
        print(g.to_string(float_format=lambda v: f"{v: .3f}"))

    print("\n=== A2. CALIBRATION — side AWAY ===")
    for name, d in [('TRAIN', train), ('OOS', oos)]:
        d = d.copy(); d['b'] = pd.cut(d.p_away, bins)
        g = d.groupby('b', observed=True).apply(
            lambda x: pd.Series({'n': len(x), 'implied': x.p_away.mean(),
                                 'wr': (x.out == 'A').mean(),
                                 'avg_cote': x.odds_away.mean(),
                                 'roi': roi((x.out == 'A').values, x.odds_away.values)}),
            include_groups=False)
        print(f"-- {name} --")
        print(g.to_string(float_format=lambda v: f"{v: .3f}"))

    # =================================================================
    # B. FILTRE COMPOSITE : home fav cote<=1.5, equipe NON volatile (train)
    # =================================================================
    print("\n=== B. COMPOSITE : fav home <=1.5 + equipe non volatile ===")
    # volatilite home par equipe sur train : std du WR home entre segments (segments n>=5)
    vol = {}
    for team, g in train.groupby('team_a'):
        wrs = []
        for seg, gg in g.groupby('seg'):
            if len(gg) >= 5:
                wrs.append((gg.out == 'H').mean())
        if len(wrs) >= 3:
            vol[team] = max(wrs) - min(wrs)
    vol_sorted = sorted(vol.items(), key=lambda t: -t[1])
    volatile_teams = set(t for t, v in vol_sorted if v >= np.percentile(list(vol.values()), 70))
    print(f"equipes volatiles (top30% spread WR home inter-segment, train): {sorted(volatile_teams)}")

    base = oos[oos.odds_home <= 1.5]
    r0 = report('fav<=1.5 (base)', base, (base.out == 'H'), 'odds_home')
    filt = base[~base.team_a.isin(volatile_teams)]
    r1 = report('fav<=1.5 + non-volatile', filt, (filt.out == 'H'), 'odds_home')
    # train equivalents pour reference
    bt = train[train.odds_home <= 1.5]; ft = bt[~bt.team_a.isin(volatile_teams)]
    for r, tr in [(r0, bt), (r1, ft)]:
        wr_t = (tr.out == 'H').mean()
        print(f"{r['name']:32s} n_oos={r['n']:4d} wr_oos={r['wr']:.3f} (train {wr_t:.3f}) avg_cote={r['avg']:.3f} roi_oos={r['roi']:+.3f} couverture={r['n']/cov_base:.1%}")

    # variante stricte cote<=1.30
    base13 = oos[oos.odds_home <= 1.30]
    f13 = base13[~base13.team_a.isin(volatile_teams)]
    for lbl, s in [('fav<=1.30 (base)', base13), ('fav<=1.30 + non-volatile', f13)]:
        r = report(lbl, s, (s.out == 'H'), 'odds_home')
        print(f"{lbl:32s} n_oos={r['n']:4d} wr_oos={r['wr']:.3f} avg_cote={r['avg']:.3f} roi_oos={r['roi']:+.3f} couverture={r['n']/cov_base:.1%}")

    # =================================================================
    # C. CONFIRMATION CROISEE : fav home + adversaire away en mauvaise forme (train)
    # =================================================================
    print("\n=== C. CONFIRMATION CROISEE (forme away adversaire, train) ===")
    league_away_wr = (train.out == 'A').mean()
    away_wr = train.groupby('team_b').apply(lambda g: (g.out == 'A').mean(), include_groups=False)
    away_n = train.groupby('team_b').size()
    delta_away = (away_wr - league_away_wr)
    weak_away = set(delta_away[(delta_away <= -0.05) & (away_n >= 20)].index)
    print(f"league away WR train={league_away_wr:.3f}; equipes faibles away (delta<=-5pp, n>=20): {sorted(weak_away)}")
    for thr in [1.5, 1.7, 2.0]:
        s = oos[(oos.odds_home <= thr) & (oos.team_b.isin(weak_away))]
        r = report(f'home<= {thr} & adv weak-away', s, (s.out == 'H'), 'odds_home')
        st = train[(train.odds_home <= thr) & (train.team_b.isin(weak_away))]
        wr_t = (st.out == 'H').mean() if len(st) else np.nan
        print(f"home<={thr} & adv weak-away      n_oos={r['n']:4d} wr_oos={r['wr']:.3f} (train {wr_t:.3f}) avg_cote={r['avg']:.3f} roi_oos={r['roi']:+.3f} couverture={r['n']/cov_base:.1%}")
    # variante hautes cotes : home outsider contre adversaire faible away
    s = oos[(oos.odds_home >= 2.0) & (oos.odds_home <= 3.5) & (oos.team_b.isin(weak_away))]
    r = report('home 2.0-3.5 & adv weak-away', s, (s.out == 'H'), 'odds_home')
    print(f"home 2.0-3.5 & adv weak-away    n_oos={r['n']:4d} wr_oos={r['wr']:.3f} avg_cote={r['avg']:.3f} roi_oos={r['roi']:+.3f}")

    # =================================================================
    # D. DOUBLE CHANCE SYNTHETIQUE : 1X sur home cote [1.8-2.4]
    # =================================================================
    print("\n=== D. DC SYNTHETIQUE 1X (home 1.8-2.4) vs TIER1 ===")
    s = oos[(oos.odds_home >= 1.8) & (oos.odds_home <= 2.4) & oos.dc_1x.notna()]
    won = (s.out != 'A')
    r = report('DC 1X home 1.8-2.4 (cote DC reelle)', s, won, 'dc_1x')
    st = train[(train.odds_home >= 1.8) & (train.odds_home <= 2.4) & train.dc_1x.notna()]
    wr_t = (st.out != 'A').mean()
    print(f"n_oos={r['n']} wr_oos={r['wr']:.3f} (train {wr_t:.3f}) avg_cote_dc={r['avg']:.3f} roi_oos={r['roi']:+.3f} couverture={r['n']/cov_base:.1%}")
    # toutes bandes 1X
    for lo, hi in [(1.3, 1.8), (1.8, 2.4), (2.4, 3.2)]:
        s = oos[(oos.odds_home >= lo) & (oos.odds_home < hi) & oos.dc_1x.notna()]
        won = (s.out != 'A')
        r = report('', s, won, 'dc_1x')
        print(f"  1X home[{lo}-{hi})  n_oos={r['n']:4d} wr_oos={r['wr']:.3f} avg_cote_dc={r['avg']:.3f} roi_oos={r['roi']:+.3f}")
    # X2 bandes (cote DC plus haute)
    for lo, hi in [(1.8, 2.4), (2.4, 3.2), (3.2, 5.0)]:
        s = oos[(oos.odds_away >= lo) & (oos.odds_away < hi) & oos.dc_x2.notna()]
        won = (s.out != 'H')
        r = report('', s, won, 'dc_x2')
        print(f"  X2 away[{lo}-{hi})  n_oos={r['n']:4d} wr_oos={r['wr']:.3f} avg_cote_dc={r['avg']:.3f} roi_oos={r['roi']:+.3f}")

    # =================================================================
    # E. VALUE DC : cote reelle vs fair synthetique
    # =================================================================
    print("\n=== E. VALUE DC (cote reelle vs fair devig) ===")
    for col, fair_p, win_cond, lbl in [
        ('dc_1x', oos.p_home + oos.p_draw, oos.out != 'A', '1X'),
        ('dc_x2', oos.p_away + oos.p_draw, oos.out != 'H', 'X2'),
        ('dc_12', oos.p_home + oos.p_away, oos.out != 'D', '12'),
    ]:
        m = oos[col].notna()
        edge = oos.loc[m, col] * fair_p[m] - 1  # EV a probas devig
        d = oos[m].copy(); d['edge'] = edge; d['won'] = win_cond[m]
        g = d.groupby(pd.cut(d.edge, [-1, -0.10, -0.05, 0, 0.05, 1]), observed=True).apply(
            lambda x: pd.Series({'n': len(x), 'wr': x.won.mean(), 'avg_cote': x[col].mean(),
                                 'roi': roi(x.won.values, x[col].values)}), include_groups=False)
        print(f"-- {lbl} par bucket d'edge (OOS) --")
        print(g.to_string(float_format=lambda v: f"{v: .3f}"))

    # seuil choisi sur train : parier DC quand edge_train >= seuil*
    print("-- selection seuil sur train, eval OOS --")
    for col, lbl in [('dc_1x', '1X'), ('dc_x2', 'X2'), ('dc_12', '12')]:
        fair_tr = {'dc_1x': train.p_home + train.p_draw, 'dc_x2': train.p_away + train.p_draw,
                   'dc_12': train.p_home + train.p_away}[col]
        win_tr = {'dc_1x': train.out != 'A', 'dc_x2': train.out != 'H', 'dc_12': train.out != 'D'}[col]
        m = train[col].notna()
        edge_tr = train.loc[m, col] * fair_tr[m] - 1
        best = None
        for thr in [0.0, 0.02, 0.04, 0.06, 0.08]:
            sel = edge_tr >= thr
            if sel.sum() < 40:
                continue
            r_tr = roi(win_tr[m][sel].values, train.loc[m, col][sel].values)
            if best is None or r_tr > best[1]:
                best = (thr, r_tr, int(sel.sum()))
        if best is None:
            print(f"{lbl}: aucun seuil train avec n>=40"); continue
        thr = best[0]
        fair_oo = {'dc_1x': oos.p_home + oos.p_draw, 'dc_x2': oos.p_away + oos.p_draw,
                   'dc_12': oos.p_home + oos.p_away}[col]
        win_oo = {'dc_1x': oos.out != 'A', 'dc_x2': oos.out != 'H', 'dc_12': oos.out != 'D'}[col]
        mo = oos[col].notna()
        edge_oo = oos.loc[mo, col] * fair_oo[mo] - 1
        sel = edge_oo >= thr
        s = oos[mo][sel]
        w = win_oo[mo][sel]
        print(f"{lbl}: seuil*={thr:+.2f} (train roi={best[1]:+.3f}, n={best[2]}) -> "
              f"OOS n={len(s)} wr={w.mean() if len(s) else float('nan'):.3f} "
              f"avg_cote={s[col].mean() if len(s) else float('nan'):.3f} "
              f"roi={roi(w.values, s[col].values) if len(s) else float('nan'):+.3f}")

    # =================================================================
    # F. GRILLE segment x bande de cote (hautes cotes), selection sur train
    # =================================================================
    print("\n=== F. GRILLE seg x cote (selection train: roi>=+0.10, n>=40) -> OOS poole ===")
    bands = [(2.0, 2.5), (2.5, 3.0), (3.0, 3.5), (3.5, 4.5), (4.5, 6.0), (6.0, 10.0)]
    sides = [('H', 'odds_home'), ('A', 'odds_away'), ('D', 'odds_draw')]
    selected = []
    for side, col in sides:
        for segname, _, _ in SEGMENTS:
            for lo, hi in bands:
                tr = train[(train.seg == segname) & (train[col] >= lo) & (train[col] < hi)]
                if len(tr) < 40:
                    continue
                r_tr = roi((tr.out == side).values, tr[col].values)
                if r_tr >= 0.10:
                    selected.append((side, col, segname, lo, hi, len(tr), r_tr))
    print("cellules retenues (train):")
    for s_ in selected:
        print(f"  {s_[0]} {s_[2]} cote[{s_[3]}-{s_[4]}) n_train={s_[5]} roi_train={s_[6]:+.3f}")
    # OOS poole
    mask = pd.Series(False, index=oos.index)
    odds_pick = pd.Series(np.nan, index=oos.index)
    won_pick = pd.Series(False, index=oos.index)
    for side, col, segname, lo, hi, _, _ in selected:
        m = (oos.seg == segname) & (oos[col] >= lo) & (oos[col] < hi) & ~mask
        mask |= m
        odds_pick[m] = oos.loc[m, col]
        won_pick[m] = (oos.loc[m, 'out'] == side)
    s = oos[mask]
    if len(s):
        print(f"OOS poole: n={len(s)} wr={won_pick[mask].mean():.3f} avg_cote={odds_pick[mask].mean():.3f} "
              f"roi={roi(won_pick[mask].values, odds_pick[mask].values):+.3f} couverture={len(s)/cov_base:.1%}")
    else:
        print("OOS poole: aucune cellule")

    # detail OOS par cellule
    for side, col, segname, lo, hi, ntr, rtr in selected:
        m = (oos.seg == segname) & (oos[col] >= lo) & (oos[col] < hi)
        ss = oos[m]
        if len(ss) == 0:
            continue
        w = (ss.out == side)
        print(f"  {side} {segname} [{lo}-{hi}) OOS n={len(ss)} wr={w.mean():.3f} roi={roi(w.values, ss[col].values):+.3f}")

    # =================================================================
    # G. ACCURACY MAX : meilleur pick par proba devig max, seuils de couverture
    # =================================================================
    print("\n=== G. ACCURACY MAX (pick = argmax proba devig, seuil sur p_max) ===")
    oos2 = oos.copy()
    probs = oos2[['p_home', 'p_draw', 'p_away']].values
    pick_idx = probs.argmax(axis=1)
    pick_side = np.array(['H', 'D', 'A'])[pick_idx]
    pick_odds = oos2[['odds_home', 'odds_draw', 'odds_away']].values[np.arange(len(oos2)), pick_idx]
    p_max = probs.max(axis=1)
    won = (pick_side == oos2.out.values)
    for thr in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        m = p_max >= thr
        if m.sum() == 0:
            continue
        print(f"p_max>={thr:.2f}: n={m.sum():4d} wr={won[m].mean():.3f} avg_cote={pick_odds[m].mean():.3f} "
              f"roi={roi(won[m], pick_odds[m]):+.3f} couverture={m.sum()/cov_base:.1%}")
    # combinaison avec filtres B et C
    nonvol = ~oos2.team_a.isin(volatile_teams)
    weakopp = oos2.team_b.isin(weak_away)
    for lbl, extra in [('+non-volatile', nonvol.values),
                       ('+adv weak-away', weakopp.values),
                       ('+non-vol & weak-away', (nonvol & weakopp).values)]:
        for thr in [0.55, 0.60, 0.65]:
            m = (p_max >= thr) & (pick_side == 'H') & extra
            if m.sum() < 10:
                continue
            print(f"H p>={thr:.2f} {lbl:22s}: n={m.sum():4d} wr={won[m].mean():.3f} "
                  f"avg_cote={pick_odds[m].mean():.3f} roi={roi(won[m], pick_odds[m]):+.3f} couverture={m.sum()/cov_base:.1%}")


if __name__ == '__main__':
    main()
