# -*- coding: utf-8 -*-
"""
_wf_calibration4.py — iteration 4 :
  U. STEAM robustesse : drift>=+0.01 -> H ; @open vs @last ; bande oh1>=1.8 ;
     stabilite par moitie d'OOS ; concentration par equipe ; drift relatif
  V. ACCURACY finale : grille de regles tunee sur train (WR>=80% & couverture max), eval OOS
  W. Recap DC accuracy (1X) avec WR train/OOS
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
    df['dc_1x'] = df.em.map(lambda m: (m.get('Double Chance') or {}).get('1X', np.nan))
    return df

def stats(lbl, w, o, base=None):
    s = f"{lbl:48s} n={len(w):4d} wr={np.mean(w):.3f} avg_cote={np.mean(o):.3f} roi={roi(w, o):+.3f}"
    if base:
        s += f" couv={len(w)/base:.1%}"
    print(s)

def main():
    df = load()
    n = len(df); cut = int(n*0.70)
    df['is_oos'] = df.index >= cut
    train, oos = df[~df.is_oos], df[df.is_oos]
    cov = len(oos)
    print(f"TOTAL={n} train={len(train)} oos={cov}")

    # =====================================================
    # U. STEAM robustesse
    # =====================================================
    print("\n=== U. STEAM drift_h = ph_last - ph_open, snapshots pre-match ===")
    mv = df[df.cap2 < df.expected_start].copy()
    mv['drift_h'] = mv.ph2 - mv.ph1
    for cond_lbl, cond in [
        ('drift>=+0.01 (tous)', mv.drift_h >= 0.01),
        ('drift>=+0.01 & oh1>=1.8', (mv.drift_h >= 0.01) & (mv.oh1 >= 1.8)),
        ('drift>=+0.01 & oh1>=2.0', (mv.drift_h >= 0.01) & (mv.oh1 >= 2.0)),
        ('drift>=+0.015 & oh1>=1.8', (mv.drift_h >= 0.015) & (mv.oh1 >= 1.8)),
    ]:
        for name, isoos in [('TRAIN', False), ('OOS', True)]:
            s = mv[cond & (mv.is_oos == isoos)]
            if len(s) == 0:
                continue
            w = (s.out == 'H').values
            print(f"-- {cond_lbl} [{name}] --")
            stats('   @open', w, s.oh1.values)
            stats('   @last (execution realiste)', w, s.oh2.values)
    # stabilite : OOS coupe en 2
    s = mv[(mv.drift_h >= 0.01) & mv.is_oos].sort_values('expected_start')
    h = len(s)//2
    for half, ss in [('OOS-1ere moitie', s.iloc[:h]), ('OOS-2e moitie', s.iloc[h:])]:
        w = (ss.out == 'H').values
        stats(f'   {half} @last', w, ss.oh2.values)
    # concentration equipes
    s = mv[(mv.drift_h >= 0.01)]
    print("   repartition team_a (tous):", s.team_a.value_counts().head(6).to_dict())
    print("   nb saisons/jours distincts:", s.expected_start.dt.date.nunique(), "jours")
    # drift relatif en % de cote
    mv['rdrift'] = (mv.oh1 - mv.oh2) / mv.oh1  # baisse de cote home
    for thr in [0.03, 0.05]:
        for name, isoos in [('TRAIN', False), ('OOS', True)]:
            s = mv[(mv.rdrift >= thr) & (mv.is_oos == isoos)]
            if len(s) < 10:
                continue
            w = (s.out == 'H').values
            stats(f'rdrift>={thr:.2f} [{name}] @last', w, s.oh2.values)
    # cote de l'autre cote : steam away ?
    mv['drift_a'] = mv.pa2 - mv.pa1
    for name, isoos in [('TRAIN', False), ('OOS', True)]:
        s = mv[(mv.drift_a >= 0.01) & (mv.is_oos == isoos)]
        if len(s) < 10:
            continue
        w = (s.out == 'A').values
        stats(f'drift_a>=+0.01 [{name}] -> A @last', w, s.oa2.values)

    # =====================================================
    # V. ACCURACY : selection de regle sur train -> OOS
    # =====================================================
    print("\n=== V. ACCURACY : regles candidates (tune train, eval OOS) ===")
    def pickmax(d):
        P = d[['ph1', 'pd1', 'pa1']].values
        idx = P.argmax(axis=1)
        side = np.array(['H', 'D', 'A'])[idx]
        odds = d[['oh1', 'od1', 'oa1']].values[np.arange(len(d)), idx]
        return side, odds, P.max(axis=1)
    s_tr, o_tr, p_tr = pickmax(train); w_tr = (s_tr == train.out.values)
    s_oo, o_oo, p_oo = pickmax(oos); w_oo = (s_oo == oos.out.values)
    g_tr, g_oo = train.gng_non.values, oos.gng_non.values
    d_tr, d_oo = train.od1.values, oos.od1.values

    cands = []
    for t in np.arange(0.60, 0.80, 0.01):
        cands.append((f'p>={t:.2f}', p_tr >= t, p_oo >= t))
        for g in [1.9, 2.0, 2.1, 2.2]:
            cands.append((f'p>={t:.2f} & gngNon<={g}', (p_tr >= t) & (g_tr <= g), (p_oo >= t) & (g_oo <= g)))
        for dd in [3.5, 3.7, 4.0]:
            cands.append((f'p>={t:.2f} & coteNul>={dd}', (p_tr >= t) & (d_tr >= dd), (p_oo >= t) & (d_oo >= dd)))
    results = []
    for lbl, m_tr, m_oo in cands:
        if m_tr.sum() < 150 or m_oo.sum() < 50:
            continue
        wr_t = w_tr[m_tr].mean()
        if wr_t >= 0.78:
            results.append((lbl, int(m_tr.sum()), wr_t, int(m_oo.sum()), w_oo[m_oo].mean(),
                            o_oo[m_oo].mean(), roi(w_oo[m_oo], o_oo[m_oo]), m_oo.sum()/cov))
    results.sort(key=lambda r: -r[3])  # par couverture OOS desc
    print("regles avec WR_train>=0.78 (triees par couverture OOS):")
    for r in results[:12]:
        print(f"  {r[0]:28s} n_tr={r[1]:4d} wr_tr={r[2]:.3f} | n_oos={r[3]:4d} wr_oos={r[4]:.3f} "
              f"cote={r[5]:.3f} roi={r[6]:+.3f} couv={r[7]:.1%}")

    # =====================================================
    # W. DC 1X recap train/OOS
    # =====================================================
    print("\n=== W. DC 1X (accuracy play) train vs OOS ===")
    for lim in [1.5, 1.8, 2.0]:
        for name, d in [('TRAIN', train), ('OOS', oos)]:
            s = d[(d.oh1 <= lim) & d.dc_1x.notna()]
            w = (s.out != 'A').values
            stats(f'1X home<={lim} [{name}]', w, s.dc_1x.values, base=len(d))

if __name__ == '__main__':
    main()
