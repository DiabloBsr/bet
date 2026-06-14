# -*- coding: utf-8 -*-
"""
_wf_calibration2.py — iteration 2 walk-forward 70/30 :
  H. Steam (mouvement de cote entre 1er et dernier snapshot pre-match)
  I. Residu marche par equipe (train) -> bets OOS (dont hautes cotes)
  J. Accuracy push : seuils calibres sur train pour WR>=80%, eval OOS
  K. Selection par journee x side (train roi>=15%) -> OOS poole
  L. Forme causale (WR 10 derniers matchs) en complement des cotes
  M. Draw hunting : matchs equilibres, bandes de cote nul
  N. DC 1X / 12 comme jeu accuracy (WR tres haut, couverture large)
"""
import sys, json
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from scraper.config import load_settings

SEGMENTS = [('DS', 1, 3), ('MS_early', 4, 12), ('MS_mid', 13, 25), ('MS_late', 26, 33), ('FS', 34, 38)]

def seg_of(r):
    for n, lo, hi in SEGMENTS:
        if lo <= r <= hi:
            return n
    return None

def roi(won, odds):
    won = np.asarray(won, float); odds = np.asarray(odds, float)
    return float(np.mean(won * (odds - 1) - (1 - won)))

def load():
    eng = create_engine(load_settings().db_url)
    q = """
    SELECT e.id, e.round_info, e.team_a, e.team_b, e.expected_start,
           o1.odds_home oh1, o1.odds_draw od1, o1.odds_away oa1, o1.extra_markets,
           o2.odds_home oh2, o2.odds_draw od2, o2.odds_away oa2, o2.captured_at cap2,
           r.score_a, r.score_b
    FROM events e
    JOIN results r ON r.event_id = e.id
    JOIN odds_snapshots o1 ON o1.id = (SELECT MIN(x.id) FROM odds_snapshots x WHERE x.event_id = e.id)
    JOIN odds_snapshots o2 ON o2.id = (SELECT MAX(x.id) FROM odds_snapshots x WHERE x.event_id = e.id)
    WHERE e.round_info != '0' AND r.score_a IS NOT NULL AND r.score_b IS NOT NULL
      AND o1.odds_home IS NOT NULL AND o1.odds_draw IS NOT NULL AND o1.odds_away IS NOT NULL
    """
    with eng.connect() as c:
        df = pd.read_sql(text(q), c)
    df['round'] = pd.to_numeric(df['round_info'], errors='coerce')
    df = df.dropna(subset=['round']); df['round'] = df['round'].astype(int)
    df = df[(df['round'] >= 1) & (df['round'] <= 38)]
    df['seg'] = df['round'].map(seg_of)
    df['expected_start'] = pd.to_datetime(df['expected_start'])
    df['cap2'] = pd.to_datetime(df['cap2'])
    df = df.sort_values('expected_start').reset_index(drop=True)
    df['out'] = np.where(df.score_a > df.score_b, 'H', np.where(df.score_a < df.score_b, 'A', 'D'))
    for pfx, h, d, a in [('1', 'oh1', 'od1', 'oa1'), ('2', 'oh2', 'od2', 'oa2')]:
        inv = 1/df[h] + 1/df[d] + 1/df[a]
        df[f'ph{pfx}'] = (1/df[h]) / inv
        df[f'pd{pfx}'] = (1/df[d]) / inv
        df[f'pa{pfx}'] = (1/df[a]) / inv

    def parse_dc(em):
        try:
            d = json.loads(em) if isinstance(em, str) else em
            dc = d.get('Double Chance') or {}
            return (dc.get('1X', np.nan), dc.get('X2', np.nan), dc.get('12', np.nan))
        except Exception:
            return (np.nan, np.nan, np.nan)
    dc = df.extra_markets.map(parse_dc)
    df['dc_1x'] = dc.map(lambda t: t[0]); df['dc_x2'] = dc.map(lambda t: t[1]); df['dc_12'] = dc.map(lambda t: t[2])
    return df

def main():
    df = load()
    n = len(df); cut = int(n * 0.70)
    train, oos = df.iloc[:cut].copy(), df.iloc[cut:].copy()
    cov = len(oos)
    print(f"TOTAL={n} train={len(train)} oos={len(oos)}")

    # =====================================================
    # H. STEAM : drift proba home entre 1er et dernier snapshot
    # =====================================================
    print("\n=== H. STEAM (drift p_home open->last, snapshot pre-match) ===")
    mv = df[(df.oh1 != df.oh2) & (df.cap2 < df.expected_start)].copy()
    mv['drift_h'] = mv.ph2 - mv.ph1
    mv['drift_a'] = mv.pa2 - mv.pa1
    mtr, moo = mv[mv.index < cut], mv[mv.index >= cut]
    print(f"events avec mouvement pre-match: train={len(mtr)} oos={len(moo)}")
    for name, d in [('TRAIN', mtr), ('OOS', moo)]:
        for lbl, m, side, oc in [
            ('home steam>=+2pp -> H @last', d.drift_h >= 0.02, 'H', 'oh2'),
            ('home steam>=+2pp -> H @open', d.drift_h >= 0.02, 'H', 'oh1'),
            ('home drift<=-2pp -> A @last', d.drift_h <= -0.02, 'A', 'oa2'),
            ('away steam>=+2pp -> A @last', d.drift_a >= 0.02, 'A', 'oa2'),
        ]:
            s = d[m]
            if len(s) < 10:
                print(f"  {name} {lbl}: n={len(s)} (trop petit)"); continue
            w = (s.out == side)
            print(f"  {name} {lbl}: n={len(s)} wr={w.mean():.3f} avg_cote={s[oc].mean():.3f} roi={roi(w.values, s[oc].values):+.3f}")

    # =====================================================
    # I. RESIDU MARCHE PAR EQUIPE (train) -> OOS
    # =====================================================
    print("\n=== I. RESIDU equipe vs marche (train, n>=25) -> OOS ===")
    res_rows = []
    for side, team_col, pcol, ocol in [('H', 'team_a', 'ph1', 'oh1'), ('A', 'team_b', 'pa1', 'oa1')]:
        for team, g in train.groupby(team_col):
            if len(g) < 25:
                continue
            resid = (g.out == side).mean() - g[pcol].mean()
            res_rows.append((side, team, len(g), resid, roi((g.out == side).values, g[ocol].values)))
    rr = pd.DataFrame(res_rows, columns=['side', 'team', 'n_tr', 'resid', 'roi_tr'])
    pos = rr[(rr.resid >= 0.05) & (rr.roi_tr >= 0.05)]
    print("equipes residu>=+5pp & roi_train>=+5%:")
    print(pos.to_string(index=False, float_format=lambda v: f"{v: .3f}"))
    # OOS poole : parier ces equipes sur leur side
    picks = []
    for _, row in pos.iterrows():
        tc, pc, oc = ('team_a', 'ph1', 'oh1') if row.side == 'H' else ('team_b', 'pa1', 'oa1')
        s = oos[oos[tc] == row.team]
        for _, e in s.iterrows():
            picks.append((e[oc], e.out == row.side))
    if picks:
        o_, w_ = np.array([p[0] for p in picks]), np.array([p[1] for p in picks])
        print(f"OOS poole tous: n={len(o_)} wr={w_.mean():.3f} avg_cote={o_.mean():.3f} roi={roi(w_, o_):+.3f}")
        hi = o_ >= 2.0
        if hi.sum():
            print(f"OOS poole cote>=2.0: n={hi.sum()} wr={w_[hi].mean():.3f} avg_cote={o_[hi].mean():.3f} roi={roi(w_[hi], o_[hi]):+.3f}")

    # version hautes cotes : residu calcule uniquement sur matchs cote>=2.0 du train
    print("-- residu equipe calcule sur cotes>=2.0 (train, n>=15) --")
    rows2 = []
    for side, team_col, pcol, ocol in [('H', 'team_a', 'ph1', 'oh1'), ('A', 'team_b', 'pa1', 'oa1')]:
        t2 = train[train[ocol] >= 2.0]
        for team, g in t2.groupby(team_col):
            if len(g) < 15:
                continue
            r_tr = roi((g.out == side).values, g[ocol].values)
            if r_tr >= 0.15:
                rows2.append((side, team, len(g), r_tr))
    print(rows2)
    picks = []
    for side, team, _, _ in rows2:
        tc, oc = ('team_a', 'oh1') if side == 'H' else ('team_b', 'oa1')
        s = oos[(oos[tc] == team) & (oos[oc] >= 2.0)]
        for _, e in s.iterrows():
            picks.append((e[oc], e.out == side))
    if picks:
        o_, w_ = np.array([p[0] for p in picks]), np.array([p[1] for p in picks])
        print(f"OOS poole (cote>=2.0): n={len(o_)} wr={w_.mean():.3f} avg_cote={o_.mean():.3f} roi={roi(w_, o_):+.3f}")

    # =====================================================
    # J. ACCURACY PUSH : seuil choisi sur train pour WR>=80%
    # =====================================================
    print("\n=== J. ACCURACY : seuil train pour WR>=80%, eval OOS ===")
    def pickmax(d):
        P = d[['ph1', 'pd1', 'pa1']].values
        idx = P.argmax(axis=1)
        side = np.array(['H', 'D', 'A'])[idx]
        odds = d[['oh1', 'od1', 'oa1']].values[np.arange(len(d)), idx]
        return side, odds, P.max(axis=1)
    s_tr, o_tr, p_tr = pickmax(train)
    w_tr = s_tr == train.out.values
    best_thr = None
    for thr in np.arange(0.50, 0.86, 0.01):
        m = p_tr >= thr
        if m.sum() < 100:
            break
        if w_tr[m].mean() >= 0.80:
            best_thr = thr
            break
    print(f"seuil train pour WR>=80%: p_max>={best_thr}")
    s_oo, o_oo, p_oo = pickmax(oos)
    w_oo = s_oo == oos.out.values
    if best_thr is not None:
        m = p_oo >= best_thr
        print(f"OOS: n={m.sum()} wr={w_oo[m].mean():.3f} avg_cote={o_oo[m].mean():.3f} roi={roi(w_oo[m], o_oo[m]):+.3f} couverture={m.sum()/cov:.1%}")

    # =====================================================
    # K. JOURNEE x SIDE (train roi>=+15%, n>=30) -> OOS poole
    # =====================================================
    print("\n=== K. JOURNEE x SIDE selection train -> OOS ===")
    cells = []
    for side, ocol in [('H', 'oh1'), ('A', 'oa1'), ('D', 'od1')]:
        for rnd, g in train.groupby('round'):
            gg = g[g[ocol] >= 2.0]
            if len(gg) < 30:
                continue
            r_tr = roi((gg.out == side).values, gg[ocol].values)
            if r_tr >= 0.15:
                cells.append((side, ocol, rnd, len(gg), r_tr))
    print("cellules:", [(c[0], int(c[2]), c[3], round(c[4], 3)) for c in cells])
    picks = []
    for side, ocol, rnd, _, _ in cells:
        s = oos[(oos['round'] == rnd) & (oos[ocol] >= 2.0)]
        for _, e in s.iterrows():
            picks.append((e[ocol], e.out == side))
    if picks:
        o_, w_ = np.array([p[0] for p in picks]), np.array([p[1] for p in picks])
        print(f"OOS poole: n={len(o_)} wr={w_.mean():.3f} avg_cote={o_.mean():.3f} roi={roi(w_, o_):+.3f}")

    # =====================================================
    # L. FORME CAUSALE : WR des 10 derniers matchs (toute la timeline)
    # =====================================================
    print("\n=== L. FORME CAUSALE (10 derniers matchs, sans leakage) ===")
    # construit l'historique sequentiel des resultats par equipe
    hist = {}
    form_a, form_b = np.full(n, np.nan), np.full(n, np.nan)
    for i, e in enumerate(df.itertuples()):
        for team, ishome in [(e.team_a, True), (e.team_b, False)]:
            h = hist.get(team, [])
            arr = form_a if ishome else form_b
            if len(h) >= 5:
                arr[i] = np.mean(h[-10:])
        won_a = 1.0 if e.out == 'H' else (0.5 if e.out == 'D' else 0.0)
        hist.setdefault(e.team_a, []).append(won_a)
        hist.setdefault(e.team_b, []).append(1.0 - won_a)
    df['form_a'], df['form_b'] = form_a, form_b
    df['form_diff'] = df.form_a - df.form_b
    tr2, oo2 = df.iloc[:cut], df.iloc[cut:]
    # parmi favoris home (<=1.6) : la forme ajoute-t-elle de l'info ?
    for name, d in [('TRAIN', tr2), ('OOS', oo2)]:
        s = d[(d.oh1 <= 1.6) & d.form_diff.notna()]
        for lbl, m in [('form_diff>=+0.15', s.form_diff >= 0.15), ('form_diff<=-0.15', s.form_diff <= -0.15)]:
            ss = s[m]
            if len(ss) < 15:
                continue
            w = (ss.out == 'H')
            print(f"  {name} fav home<=1.6 & {lbl}: n={len(ss)} wr={w.mean():.3f} avg_cote={ss.oh1.mean():.3f} roi={roi(w.values, ss.oh1.values):+.3f}")
    # hautes cotes : home outsider en forme vs adversaire pas en forme
    for name, d in [('TRAIN', tr2), ('OOS', oo2)]:
        s = d[(d.oh1 >= 2.0) & (d.oh1 <= 4.0) & (d.form_diff >= 0.15)]
        if len(s) >= 15:
            w = (s.out == 'H')
            print(f"  {name} home 2.0-4.0 & form_diff>=+0.15: n={len(s)} wr={w.mean():.3f} avg_cote={s.oh1.mean():.3f} roi={roi(w.values, s.oh1.values):+.3f}")
        s = d[(d.oa1 >= 2.0) & (d.oa1 <= 4.0) & (d.form_diff <= -0.15)]
        if len(s) >= 15:
            w = (s.out == 'A')
            print(f"  {name} away 2.0-4.0 & form_diff<=-0.15: n={len(s)} wr={w.mean():.3f} avg_cote={s.oa1.mean():.3f} roi={roi(w.values, s.oa1.values):+.3f}")

    # =====================================================
    # M. DRAW HUNTING
    # =====================================================
    print("\n=== M. DRAWS : matchs equilibres & bandes de cote nul ===")
    for name, d in [('TRAIN', train), ('OOS', oos)]:
        for lbl, m in [
            ('|ph-pa|<=0.05', (d.ph1 - d.pa1).abs() <= 0.05),
            ('|ph-pa|<=0.03', (d.ph1 - d.pa1).abs() <= 0.03),
            ('cote nul<=3.2', d.od1 <= 3.2),
        ]:
            s = d[m]
            if len(s) < 20:
                continue
            w = (s.out == 'D')
            print(f"  {name} {lbl}: n={len(s)} wr={w.mean():.3f} avg_cote={s.od1.mean():.3f} roi={roi(w.values, s.od1.values):+.3f}")

    # =====================================================
    # N. DC COMME JEU ACCURACY (WR max, couverture large)
    # =====================================================
    print("\n=== N. DC accuracy (OOS) ===")
    for lbl, m, win, oc in [
        ('1X home<=1.5', (oos.oh1 <= 1.5) & oos.dc_1x.notna(), oos.out != 'A', 'dc_1x'),
        ('1X home<=1.8', (oos.oh1 <= 1.8) & oos.dc_1x.notna(), oos.out != 'A', 'dc_1x'),
        ('1X home<=2.0', (oos.oh1 <= 2.0) & oos.dc_1x.notna(), oos.out != 'A', 'dc_1x'),
        ('12 ph+pa>=0.80', (oos.ph1 + oos.pa1 >= 0.80) & oos.dc_12.notna(), oos.out != 'D', 'dc_12'),
        ('X2 away<=1.8', (oos.oa1 <= 1.8) & oos.dc_x2.notna(), oos.out != 'H', 'dc_x2'),
    ]:
        s = oos[m]; w = win[m]
        if len(s) < 20:
            print(f"  {lbl}: n={len(s)} trop petit"); continue
        print(f"  {lbl}: n={len(s)} wr={w.mean():.3f} avg_cote={s[oc].mean():.3f} roi={roi(w.values, s[oc].values):+.3f} couverture={len(s)/cov:.1%}")

if __name__ == '__main__':
    main()
