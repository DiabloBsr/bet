# -*- coding: utf-8 -*-
"""
_wf_calibration5.py — finalisation :
  X. accuracy : frontiere wr_train>=0.77 + regles union, eval OOS
  Y. steam : distribution temporelle des picks (caveat clustering)
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
    df['expected_start'] = pd.to_datetime(df['expected_start'])
    df['cap2'] = pd.to_datetime(df['cap2'])
    df = df.sort_values('expected_start').reset_index(drop=True)
    df['out'] = np.where(df.score_a > df.score_b, 'H', np.where(df.score_a < df.score_b, 'A', 'D'))
    for p, h, d, a in [('1', 'oh1', 'od1', 'oa1'), ('2', 'oh2', 'od2', 'oa2')]:
        inv = 1/df[h] + 1/df[d] + 1/df[a]
        df[f'ph{p}'] = (1/df[h])/inv; df[f'pd{p}'] = (1/df[d])/inv; df[f'pa{p}'] = (1/df[a])/inv
    df['em'] = df.extra_markets.map(lambda x: json.loads(x) if isinstance(x, str) else (x or {}))
    df['gng_non'] = df.em.map(lambda m: (m.get('G/NG') or {}).get('Non', np.nan))
    return df

def main():
    df = load()
    n = len(df); cut = int(n*0.70)
    df['is_oos'] = df.index >= cut
    train, oos = df[~df.is_oos], df[df.is_oos]
    cov = len(oos)
    print(f"TOTAL={n} train={len(train)} oos={cov}")

    def pickmax(d):
        P = d[['ph1', 'pd1', 'pa1']].values
        idx = P.argmax(axis=1)
        side = np.array(['H', 'D', 'A'])[idx]
        odds = d[['oh1', 'od1', 'oa1']].values[np.arange(len(d)), idx]
        return side, odds, P.max(axis=1)
    s_tr, o_tr, p_tr = pickmax(train); w_tr = (s_tr == train.out.values)
    s_oo, o_oo, p_oo = pickmax(oos); w_oo = (s_oo == oos.out.values)
    g_tr, g_oo = train.gng_non.values, oos.gng_non.values

    print("\n=== X. frontiere accuracy (wr_train>=0.770) + unions ===")
    cands = []
    for t in np.arange(0.60, 0.78, 0.005):
        cands.append((f'p>={t:.3f}', p_tr >= t, p_oo >= t))
        for g in [1.9, 1.95, 2.0, 2.05, 2.1, 2.15, 2.2]:
            cands.append((f'p>={t:.3f} & gngNon<={g}', (p_tr >= t) & (g_tr <= g), (p_oo >= t) & (g_oo <= g)))
    for t_hi in [0.72, 0.74]:
        for t_lo in [0.64, 0.66, 0.68]:
            for g in [1.9, 2.0]:
                cands.append((f'p>={t_hi} | (p>={t_lo} & gngNon<={g})',
                              (p_tr >= t_hi) | ((p_tr >= t_lo) & (g_tr <= g)),
                              (p_oo >= t_hi) | ((p_oo >= t_lo) & (g_oo <= g))))
    results = []
    for lbl, m_tr, m_oo in cands:
        if m_tr.sum() < 150 or m_oo.sum() < 50:
            continue
        wr_t = w_tr[m_tr].mean()
        if wr_t >= 0.770:
            results.append((lbl, int(m_tr.sum()), wr_t, int(m_oo.sum()), w_oo[m_oo].mean(),
                            o_oo[m_oo].mean(), roi(w_oo[m_oo], o_oo[m_oo]), m_oo.sum()/cov))
    # tri : wr_oos>=0.80 d'abord, puis couverture desc
    results.sort(key=lambda r: (0 if r[4] >= 0.80 else 1, -r[7]))
    for r in results[:15]:
        print(f"  {r[0]:36s} n_tr={r[1]:4d} wr_tr={r[2]:.3f} | n_oos={r[3]:4d} wr_oos={r[4]:.3f} "
              f"cote={r[5]:.3f} roi={r[6]:+.3f} couv={r[7]:.1%}")

    print("\n=== Y. steam : distribution temporelle ===")
    mv = df[df.cap2 < df.expected_start].copy()
    mv['drift_h'] = mv.ph2 - mv.ph1
    s = mv[mv.drift_h >= 0.01]
    print("par jour (tous):")
    print(s.groupby([s.expected_start.dt.date, s.is_oos]).size().to_string())
    s18 = s[s.oh1 >= 1.8]
    for name, isoos in [('TRAIN', False), ('OOS', True)]:
        ss = s18[s18.is_oos == isoos]
        w = (ss.out == 'H').values
        print(f"{name} drift>=0.01 & oh1>=1.8 @last: n={len(ss)} wr={w.mean():.3f} "
              f"cote={ss.oh2.mean():.3f} roi={roi(w, ss.oh2.values):+.3f} jours={ss.expected_start.dt.date.nunique()}")

if __name__ == '__main__':
    main()
