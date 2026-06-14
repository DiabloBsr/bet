# -*- coding: utf-8 -*-
"""WF2 - goals iteration 2 : audit du signal 'value all-time home WR vs cote'.

Questions :
  A. Significance (SE analytique + bootstrap) du ROI de bet-home quand wr_home_alltime - ph >= thr
  B. Calibration marche : realized home WR vs devig ph par bucket de ph -> bias home-longshot cote-only ?
  C. Le WR all-time separe-t-il ENCORE au sein d'un meme bucket de cote ? (au-dela de la cote)
  D. Coherence train / OOS-moitie1 / OOS-moitie2 ; symetrie (edge negatif -> ROI pire ?)
  E. Regle cote-only equivalente (bet home si oh dans [x,y]) : ROI comparable ?
"""
import sys, json, math
from collections import defaultdict
from datetime import datetime
sys.path.insert(0, '.')
import numpy as np
from scraper.config import load_settings
from sqlalchemy import create_engine, text

SEP = "=" * 78
rng = np.random.default_rng(42)
def parse_dt(s):
    return datetime.fromisoformat(str(s).replace('Z', ''))

eng = create_engine(load_settings().db_url)
with eng.connect() as c:
    rows = c.execute(text("""
        select e.id, cast(e.round_info as int) rd, e.team_a, e.team_b, e.expected_start,
               r.score_a, r.score_b, os.odds_home, os.odds_draw, os.odds_away
        from events e
        join results r on r.event_id = e.id
        join (select event_id, min(id) mid from odds_snapshots group by event_id) f on f.event_id = e.id
        join odds_snapshots os on os.id = f.mid
        where cast(e.round_info as int) >= 1
        order by e.expected_start, e.id
    """)).fetchall()
seen = {}
for r in rows:
    k = (r[2], r[3], str(r[4]))
    if k not in seen:
        seen[k] = r
rows = sorted(seen.values(), key=lambda r: (str(r[4]), r[0]))
matches = []
for r in rows:
    oh, od, oa = r[7], r[8], r[9]
    if not oh or not od or not oa or oh <= 1.0 or od <= 1.0 or oa <= 1.0:
        continue
    inv = 1/oh + 1/od + 1/oa
    matches.append({'home': r[2], 'away': r[3], 't': parse_dt(r[4]), 'sa': int(r[5]), 'sb': int(r[6]),
                    'oh': oh, 'od': od, 'oa': oa, 'ph': (1/oh)/inv,
                    'y': 0 if r[5] > r[6] else (2 if r[5] < r[6] else 1)})

# expanding all-time home WR (no leakage, groupes par timestamp)
home_hist = defaultdict(lambda: [0, 0])  # team -> [wins, n]
i = 0
while i < len(matches):
    j = i
    while j < len(matches) and matches[j]['t'] == matches[i]['t']:
        j += 1
    for m in matches[i:j]:
        w, k = home_hist[m['home']]
        m['wr_h'] = (w/k) if k >= 20 else None
        m['n_h'] = k
    for m in matches[i:j]:
        home_hist[m['home']][1] += 1
        if m['y'] == 0: home_hist[m['home']][0] += 1
    i = j

n = len(matches)
cut = int(n*0.70)
TRAIN, OOS = matches[:cut], matches[cut:]
print(f"n={n} train={len(TRAIN)} oos={len(OOS)}")

def strat_stats(sel, label):
    if not sel:
        print(f"{label}: n=0"); return
    nb = len(sel)
    wins = sum(1 for m in sel if m['y'] == 0)
    rets = np.array([m['oh'] if m['y'] == 0 else 0.0 for m in sel])
    roi = rets.mean() - 1
    se = rets.std(ddof=1)/math.sqrt(nb)
    # bootstrap
    bs = np.array([rng.choice(rets, nb, replace=True).mean() - 1 for _ in range(4000)])
    p_neg = (bs <= 0).mean()
    avg_ph = np.mean([m['ph'] for m in sel])
    wr = wins/nb
    se_wr = math.sqrt(avg_ph*(1-avg_ph)/nb)
    z_wr = (wr - avg_ph)/se_wr
    print(f"{label}: n={nb} WR={wr:.4f} vs ph_devig={avg_ph:.4f} (z_WR={z_wr:+.2f}) "
          f"ROI={roi:+.4f} (SE={se:.4f}, z={roi/se:+.2f}, P(ROI<=0)boot={p_neg:.3f}) avg_cote={np.mean([m['oh'] for m in sel]):.2f}")

print(); print(SEP); print("A+D. STRATEGIE edge = wr_home_alltime - ph >= thr : train vs OOS (et sous-moities OOS)"); print(SEP)
half = len(OOS)//2
for thr in (0.05, 0.10, 0.15):
    for label, pool in [("TRAIN ", TRAIN), ("OOS   ", OOS), ("OOS-h1", OOS[:half]), ("OOS-h2", OOS[half:])]:
        sel = [m for m in pool if m['wr_h'] is not None and (m['wr_h'] - m['ph']) >= thr]
        strat_stats(sel, f"thr={thr:.2f} {label}")
    # symetrie : edge tres negatif
    sel_neg = [m for m in OOS if m['wr_h'] is not None and (m['wr_h'] - m['ph']) <= -thr]
    strat_stats(sel_neg, f"thr={thr:.2f} OOS edge<=-{thr:.2f} (controle symetrique)")
    print()

print(); print(SEP); print("B. CALIBRATION MARCHE : realized home WR vs ph devig par bucket (OOS, tous matchs)"); print(SEP)
buckets = [(0, .15), (.15, .2), (.2, .25), (.25, .3), (.3, .4), (.4, .5), (.5, .6), (.6, .7), (.7, 1.01)]
for lo, hi in buckets:
    sel = [m for m in OOS if lo <= m['ph'] < hi]
    if len(sel) < 30: continue
    wr = sum(1 for m in sel if m['y'] == 0)/len(sel)
    ph = np.mean([m['ph'] for m in sel])
    rets = np.array([m['oh'] if m['y'] == 0 else 0.0 for m in sel])
    roi = rets.mean()-1
    se_wr = math.sqrt(ph*(1-ph)/len(sel))
    print(f"ph [{lo:.2f},{hi:.2f}): n={len(sel):>4} ph_moy={ph:.3f} WR_real={wr:.3f} (z={(wr-ph)/se_wr:+.2f}) ROI_bet_home={roi:+.4f}")

print(); print(SEP); print("C. AU SEIN d'un bucket de cote : WR all-time separe-t-il encore ? (OOS)"); print(SEP)
# bucket de ph ou le signal opere (ph<0.35 : home outsider/mid) ; split par wr_h median
for lo, hi in [(0.0, 0.25), (0.25, 0.35), (0.35, 0.50)]:
    pool = [m for m in OOS if m['wr_h'] is not None and lo <= m['ph'] < hi]
    if len(pool) < 80: continue
    med = float(np.median([m['wr_h'] for m in pool]))
    hi_grp = [m for m in pool if m['wr_h'] >= med]
    lo_grp = [m for m in pool if m['wr_h'] < med]
    print(f"\nbucket ph [{lo},{hi}) n={len(pool)} mediane wr_h={med:.3f}")
    strat_stats(hi_grp, f"  wr_h >= med")
    strat_stats(lo_grp, f"  wr_h <  med")

print(); print(SEP); print("E. REGLE COTE-ONLY equivalente : bet home si oh dans [x,y] (OOS)"); print(SEP)
for lo, hi in [(2.5, 4.0), (3.0, 5.0), (3.5, 5.5), (4.0, 6.5), (3.0, 7.0), (2.5, 10.0)]:
    sel = [m for m in OOS if lo <= m['oh'] <= hi]
    if len(sel) < 40: continue
    strat_stats(sel, f"oh in [{lo},{hi}]")

print(); print(SEP); print("E2. EV-rule fondamentale : bet home si wr_h * oh >= seuil (OOS)"); print(SEP)
for s in (1.0, 1.1, 1.2, 1.3):
    sel = [m for m in OOS if m['wr_h'] is not None and m['wr_h']*m['oh'] >= s]
    if len(sel) < 40:
        print(f"seuil={s}: n={len(sel)} skip"); continue
    strat_stats(sel, f"EV>={s:.1f}")

print("\nFIN.")
