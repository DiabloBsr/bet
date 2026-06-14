# -*- coding: utf-8 -*-
"""
_wf_calibration3.py — iteration 3 walk-forward 70/30 :
  O. STEAM raffine (seuils, drift relatif, bandes de cote)
  P. HT/FT : calibration + ROI par combo (hautes cotes : X/1, X/X, 2/2 ...)
  Q. Mi-tps 1X2 : '1' HT sur favoris FT
  R. 1X2 & Total + 1X2 & G/NG : mispricing de correlation
  S. ACCURACY : p_max derniere cote (ph2) + anti-draw filtre, seuils tunes sur train
  T. Favori x O/U (croisement '+/-')
"""
import sys, json
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from scraper.config import load_settings

def roi(won, odds):
    won = np.asarray(won, float); odds = np.asarray(odds, float)
    if len(won) == 0:
        return float('nan')
    return float(np.mean(won * (odds - 1) - (1 - won)))

def load():
    eng = create_engine(load_settings().db_url)
    q = """
    SELECT e.id, e.round_info, e.team_a, e.team_b, e.expected_start,
           o1.odds_home oh1, o1.odds_draw od1, o1.odds_away oa1, o1.extra_markets,
           o2.odds_home oh2, o2.odds_draw od2, o2.odds_away oa2, o2.captured_at cap2,
           r.score_a, r.score_b, r.ht_score_a, r.ht_score_b
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
    df['expected_start'] = pd.to_datetime(df['expected_start'])
    df['cap2'] = pd.to_datetime(df['cap2'])
    df = df.sort_values('expected_start').reset_index(drop=True)
    df['out'] = np.where(df.score_a > df.score_b, 'H', np.where(df.score_a < df.score_b, 'A', 'D'))
    df['ht_out'] = np.where(df.ht_score_a > df.ht_score_b, '1',
                   np.where(df.ht_score_a < df.ht_score_b, '2', 'X'))
    df['tot'] = df.score_a + df.score_b
    df['btts'] = (df.score_a > 0) & (df.score_b > 0)
    for p, h, d, a in [('1', 'oh1', 'od1', 'oa1'), ('2', 'oh2', 'od2', 'oa2')]:
        inv = 1/df[h] + 1/df[d] + 1/df[a]
        df[f'ph{p}'] = (1/df[h])/inv; df[f'pd{p}'] = (1/df[d])/inv; df[f'pa{p}'] = (1/df[a])/inv
    df['em'] = df.extra_markets.map(lambda x: json.loads(x) if isinstance(x, str) else (x or {}))
    return df

def main():
    df = load()
    n = len(df); cut = int(n*0.70)
    df['is_oos'] = df.index >= cut
    train, oos = df[~df.is_oos], df[df.is_oos]
    cov = len(oos)
    print(f"TOTAL={n} train={len(train)} oos={cov}")

    # =====================================================
    # O. STEAM raffine
    # =====================================================
    print("\n=== O. STEAM raffine (mouvement pre-match uniquement) ===")
    mv = df[(df.cap2 < df.expected_start)].copy()
    mv['drift_h'] = mv.ph2 - mv.ph1
    has_mv = mv[mv.oh1 != mv.oh2]
    print(f"events pre-match avec mouvement: {len(has_mv)} (train {len(has_mv[~has_mv.is_oos])}, oos {len(has_mv[has_mv.is_oos])})")
    for thr in [0.005, 0.01, 0.015, 0.02, 0.03]:
        for name, d in [('TRAIN', mv[~mv.is_oos]), ('OOS', mv[mv.is_oos])]:
            s = d[d.drift_h >= thr]
            if len(s) < 5:
                continue
            w = (s.out == 'H')
            print(f"  {name} drift_h>={thr:+.3f} -> H@open: n={len(s)} wr={w.mean():.3f} avg_cote={s.oh1.mean():.3f} roi={roi(w.values, s.oh1.values):+.3f}")
    # par bande de cote (drift>=0.01)
    print("  -- par bande de cote home (drift>=+0.01) --")
    for lo, hi in [(1.0, 1.8), (1.8, 2.5), (2.5, 4.0), (4.0, 12.0)]:
        for name, d in [('TRAIN', mv[~mv.is_oos]), ('OOS', mv[mv.is_oos])]:
            s = d[(d.drift_h >= 0.01) & (d.oh1 >= lo) & (d.oh1 < hi)]
            if len(s) < 5:
                continue
            w = (s.out == 'H')
            print(f"  {name} home[{lo}-{hi}): n={len(s)} wr={w.mean():.3f} avg_cote={s.oh1.mean():.3f} roi={roi(w.values, s.oh1.values):+.3f}")

    # =====================================================
    # P. HT/FT calibration
    # =====================================================
    print("\n=== P. HT/FT : ROI par combo (flat bet chaque match dispo) ===")
    combos = ['1/1', '1/X', '1/2', 'X/1', 'X/X', 'X/2', '2/1', '2/X', '2/2']
    def htft_actual(row):
        return f"{row.ht_out}/{'1' if row.out=='H' else ('X' if row.out=='D' else '2')}"
    df['htft'] = df.apply(htft_actual, axis=1)
    train, oos = df[~df.is_oos], df[df.is_oos]
    rows = []
    for cmb in combos:
        for name, d in [('TRAIN', df[~df.is_oos]), ('OOS', df[df.is_oos])]:
            odds = d.em.map(lambda m: (m.get('HT/FT') or {}).get(cmb, np.nan))
            m = odds.notna()
            if m.sum() < 30:
                continue
            w = (d.htft == cmb)[m]
            rows.append((cmb, name, int(m.sum()), float(w.mean()), float(odds[m].mean()),
                         roi(w.values, odds[m].values)))
    rp = pd.DataFrame(rows, columns=['combo', 'set', 'n', 'freq', 'avg_cote', 'roi'])
    print(rp.pivot_table(index='combo', columns='set', values=['n', 'freq', 'avg_cote', 'roi']).to_string(float_format=lambda v: f"{v: .3f}"))

    # combos x bande de favori FT (sur train), selection roi>=+0.08 n>=60 -> OOS
    print("-- combo x bande oh1 : selection train roi>=+0.08 n>=60 -> OOS --")
    sel = []
    bandsF = [(1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 12.0)]
    for cmb in combos:
        for lo, hi in bandsF:
            d = train[(train.oh1 >= lo) & (train.oh1 < hi)]
            odds = d.em.map(lambda m: (m.get('HT/FT') or {}).get(cmb, np.nan))
            m = odds.notna()
            if m.sum() < 60:
                continue
            w = (d.htft == cmb)[m]
            r = roi(w.values, odds[m].values)
            if r >= 0.08:
                sel.append((cmb, lo, hi, int(m.sum()), r))
    print("cellules train:", sel)
    picks_o, picks_w = [], []
    for cmb, lo, hi, _, _ in sel:
        d = oos[(oos.oh1 >= lo) & (oos.oh1 < hi)]
        odds = d.em.map(lambda m: (m.get('HT/FT') or {}).get(cmb, np.nan))
        m = odds.notna()
        picks_o += list(odds[m].values); picks_w += list((d.htft == cmb)[m].values)
    if picks_o:
        o_, w_ = np.array(picks_o), np.array(picks_w)
        print(f"OOS poole: n={len(o_)} wr={w_.mean():.3f} avg_cote={o_.mean():.3f} roi={roi(w_, o_):+.3f}")

    # =====================================================
    # Q. Mi-tps 1X2 : '1' HT sur favoris FT
    # =====================================================
    print("\n=== Q. Mi-tps 1X2 ===")
    for side_key, win_col in [('1', '1'), ('X', 'X'), ('2', '2')]:
        for name, d in [('TRAIN', train), ('OOS', oos)]:
            odds = d.em.map(lambda m: (m.get('Mi-tps 1X2') or {}).get(side_key, np.nan))
            m = odds.notna()
            if m.sum() < 30:
                continue
            w = (d.ht_out == win_col)[m]
            print(f"  {name} HT '{side_key}' (tous): n={m.sum()} wr={w.mean():.3f} avg_cote={odds[m].mean():.3f} roi={roi(w.values, odds[m].values):+.3f}")
    # HT X sur gros favoris (la mi-temps nulle est frequente) — cote ~2.3-2.6
    for lo, hi in [(1.0, 1.4), (1.4, 1.8)]:
        for name, d in [('TRAIN', train), ('OOS', oos)]:
            dd = d[(d.oh1 >= lo) & (d.oh1 < hi)]
            odds = dd.em.map(lambda m: (m.get('Mi-tps 1X2') or {}).get('X', np.nan))
            m = odds.notna()
            if m.sum() < 30:
                continue
            w = (dd.ht_out == 'X')[m]
            print(f"  {name} HT 'X' & oh1[{lo}-{hi}): n={m.sum()} wr={w.mean():.3f} avg_cote={odds[m].mean():.3f} roi={roi(w.values, odds[m].values):+.3f}")

    # =====================================================
    # R. 1X2 & Total / 1X2 & G/NG
    # =====================================================
    print("\n=== R. 1X2 & Total (cle dynamique) ===")
    def xtotal_outcomes(d):
        # retourne par match la liste (cle, cote, gagnant?)
        recs = []
        for idx, row in d.iterrows():
            mk = row.em.get('1X2 & Total') or {}
            for k, o in mk.items():
                try:
                    side, rest = [x.strip() for x in k.split('/', 1)]
                    direction = '>' if '>' in rest else '<'
                    line = float(rest.replace('>', '').replace('<', '').strip())
                except Exception:
                    continue
                ft = '1' if row.out == 'H' else ('X' if row.out == 'D' else '2')
                won = (ft == side) and ((row.tot > line) if direction == '>' else (row.tot < line))
                recs.append((idx, f"{side}/{direction}", line, o, won, row.oh1))
        return pd.DataFrame(recs, columns=['idx', 'key', 'line', 'odds', 'won', 'oh1'])
    xt_tr, xt_oo = xtotal_outcomes(train), xtotal_outcomes(oos)
    for key in sorted(xt_tr.key.unique()):
        a = xt_tr[xt_tr.key == key]; b = xt_oo[xt_oo.key == key]
        if len(a) < 50 or len(b) < 30:
            continue
        print(f"  {key:6s} TRAIN n={len(a)} wr={a.won.mean():.3f} cote={a.odds.mean():.2f} roi={roi(a.won.values, a.odds.values):+.3f} | "
              f"OOS n={len(b)} wr={b.won.mean():.3f} cote={b.odds.mean():.2f} roi={roi(b.won.values, b.odds.values):+.3f}")

    print("\n=== R2. 1X2 & G/NG ===")
    GN_KEYS = {
        '1&GG': '1 gagne et les deux équipes marquent',
        '1&NG': '1 gagne et seulement  1  marque',
        'X&GG': 'X et les deux équipes marquent',
        '2&GG': '2 gagne et les deux équipes marquent',
        '2&NG': '2 gagne et seulement 2 marque',
    }
    def gn_won(row, tag):
        if tag == '1&GG': return row.out == 'H' and row.btts
        if tag == '1&NG': return row.out == 'H' and not row.btts
        if tag == 'X&GG': return row.out == 'D' and row.btts
        if tag == '2&GG': return row.out == 'A' and row.btts
        if tag == '2&NG': return row.out == 'A' and not row.btts
    for tag, key in GN_KEYS.items():
        for name, d in [('TRAIN', train), ('OOS', oos)]:
            odds = d.em.map(lambda m: (m.get('1X2 & G/NG') or {}).get(key, np.nan))
            m = odds.notna()
            if m.sum() < 30:
                continue
            w = d[m].apply(lambda r: gn_won(r, tag), axis=1)
            print(f"  {name} {tag}: n={m.sum()} wr={w.mean():.3f} avg_cote={odds[m].mean():.2f} roi={roi(w.values, odds[m].values):+.3f}")

    # =====================================================
    # S. ACCURACY : derniere cote pre-match (snapshot 2)
    # =====================================================
    print("\n=== S. ACCURACY avec derniere cote pre-match ===")
    pre = df[df.cap2 < df.expected_start]
    def pickmax(d, pfx):
        P = d[[f'ph{pfx}', f'pd{pfx}', f'pa{pfx}']].values
        idx = P.argmax(axis=1)
        side = np.array(['H', 'D', 'A'])[idx]
        odds = d[[f'oh{pfx}', f'od{pfx}', f'oa{pfx}']].values[np.arange(len(d)), idx]
        return side, odds, P.max(axis=1)
    for pfx, lbl in [('1', 'open'), ('2', 'last')]:
        d_tr, d_oo = pre[~pre.is_oos], pre[pre.is_oos]
        s_tr, o_tr, p_tr = pickmax(d_tr, pfx); w_tr = s_tr == d_tr.out.values
        s_oo, o_oo, p_oo = pickmax(d_oo, pfx); w_oo = s_oo == d_oo.out.values
        for thr in [0.65, 0.68, 0.70, 0.72]:
            m_tr = p_tr >= thr; m_oo = p_oo >= thr
            if m_oo.sum() < 30:
                continue
            print(f"  [{lbl}] p>={thr:.2f}: TRAIN n={m_tr.sum()} wr={w_tr[m_tr].mean():.3f} | "
                  f"OOS n={m_oo.sum()} wr={w_oo[m_oo].mean():.3f} cote={o_oo[m_oo].mean():.3f} "
                  f"roi={roi(w_oo[m_oo], o_oo[m_oo]):+.3f} couv={m_oo.sum()/len(d_oo):.1%}")

    # anti-draw : parmi picks H p>=0.65, perte = D ou A ? filtre via cote nul
    print("-- decomposition pertes p_max>=0.70 (OOS, cote open) --")
    s_oo, o_oo, p_oo = pickmax(oos, '1')
    m = (p_oo >= 0.70)
    lost = oos[m][np.array(s_oo[m] != oos[m].out.values)]
    print(f"  pertes: {len(lost)} dont D={int((lost.out=='D').sum())} A={int((lost.out=='A').sum())}")
    # filtre G/NG 'Non' : favori avec defense solide (NoGoal adverse probable)
    gng_non = oos.em.map(lambda mm: (mm.get('G/NG') or {}).get('Non', np.nan))
    for gthr in [2.0, 2.2]:
        mm = m & (gng_non.values <= gthr)
        if mm.sum() >= 30:
            w = s_oo[mm] == oos[mm].out.values
            print(f"  p>=0.70 & GNG_Non<={gthr}: n={mm.sum()} wr={w.mean():.3f} couv={mm.sum()/cov:.1%}")

    # =====================================================
    # T. favori x O/U ('+/-' premiere ligne dispo)
    # =====================================================
    print("\n=== T. favori home <=1.6 x ligne '+/-' ===")
    def ou_feat(mm):
        d = mm.get('+/-') or {}
        for k, v in d.items():
            if k.startswith('>'):
                try:
                    return float(k.replace('>', '').strip()), v
                except Exception:
                    return (np.nan, np.nan)
        return (np.nan, np.nan)
    feats = df.em.map(ou_feat)
    df['ou_line'] = feats.map(lambda t: t[0]); df['ou_over'] = feats.map(lambda t: t[1])
    print("lignes O/U:", df.ou_line.value_counts().to_dict())
    for name, d in [('TRAIN', df[~df.is_oos]), ('OOS', df[df.is_oos])]:
        s = d[(d.oh1 <= 1.6) & d.ou_over.notna()]
        for lbl, m2 in [('over_cote<=1.7 (match ouvert)', s.ou_over <= 1.7),
                        ('over_cote>1.7 (match ferme)', s.ou_over > 1.7)]:
            ss = s[m2]
            if len(ss) < 30:
                continue
            w = (ss.out == 'H')
            print(f"  {name} fav<=1.6 & {lbl}: n={len(ss)} wr={w.mean():.3f} cote={ss.oh1.mean():.3f} roi={roi(w.values, ss.oh1.values):+.3f}")

if __name__ == '__main__':
    main()
