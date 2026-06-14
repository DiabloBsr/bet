# -*- coding: utf-8 -*-
"""Diagnostics complementaires pour la verification adversariale de F2:
   - distribution par jour des bets/wins du pool test (50-100%)
   - bootstrap par CLUSTER (jour) au lieu de par bet
   - sensibilite aux bornes de fenetres (schemas alternatifs)
"""
import sys, json
sys.path.insert(0, '.')
from collections import defaultdict
import importlib.util
import numpy as np

spec = importlib.util.spec_from_file_location(
    "v", "scripts/_wf_verify_F2___Total__quipe_domicile___3_5_sur_hom.py")
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)

ev_raw, bets_raw, ev_clean, bets_clean = v.build()
n_ev = len(ev_clean)

# pool test 50-100%
i_half = int(n_ev * 0.50)
pool = [b for b in bets_clean if b[1] >= i_half]
print(f"\npool test 50-100%: {v.fmt(v.met(pool))}")

print("\n--- bets/wins par jour (pool test 50-100%, CLEAN) ---")
by_day = defaultdict(lambda: [0, 0, 0.0])
for b in pool:
    day = str(b[4])[:10]
    by_day[day][0] += 1
    by_day[day][1] += b[3]
    by_day[day][2] += b[3] * (b[2] - 1) - (1 - b[3])
for day in sorted(by_day):
    n, w, pnl = by_day[day]
    print(f"  {day}: n={n:3d} wins={w:2d} pnl={pnl:+7.1f}u")

# --- cluster bootstrap (by day) ---------------------------------------------
days = sorted(by_day)
pnl_by_day = {d: [] for d in days}
for b in pool:
    pnl_by_day[str(b[4])[:10]].append(b[3] * (b[2] - 1) - (1 - b[3]))
rng = np.random.default_rng(11)
D = len(days)
boots = []
for _ in range(10000):
    pick = rng.integers(0, D, D)
    s = []
    for i in pick:
        s += pnl_by_day[days[i]]
    boots.append(np.mean(s))
boots = np.array(boots)
print(f"\ncluster-bootstrap (jour) P(roi<=0) = {(boots <= 0).mean():.3f}  "
      f"(n jours={D})")

# --- date ranges of the 3 imposed windows ------------------------------------
print("\n--- bornes temporelles des fenetres imposees (events CLEAN) ---")
# need dates per event index: rebuild quickly
rows = v.load_rows()
dates = []
for (eid, ri, es, oh, oa, em, sa, sb, hta, htb, gj) in rows:
    if isinstance(em, str):
        try:
            em = json.loads(em)
        except json.JSONDecodeError:
            continue
    if not isinstance(em, dict):
        continue
    if v.clean_home_score(sa, sb, hta, htb, gj) is None:
        continue
    dates.append(str(es)[:16])
for a, b in ((0.50, 2/3), (2/3, 5/6), (5/6, 1.0)):
    i1, i2 = int(n_ev*a), int(n_ev*b)
    print(f"  test {a*100:5.1f}-{b*100:5.1f}% : {dates[i1]} -> {dates[i2-1]}  "
          f"({i2-i1} events)")
print(f"  dataset complet : {dates[0]} -> {dates[-1]} ({n_ev} events)")

# --- alternative walk-forward schemes (sensitivity) ---------------------------
print("\n--- sensibilite: autres schemas de fenetres (CLEAN) ---")
schemes = {
    "4 fenetres 50/62.5/75/87.5/100": [(0.500, 0.625), (0.625, 0.750),
                                        (0.750, 0.875), (0.875, 1.000)],
    "3 fenetres decalees 55/70/85/100": [(0.55, 0.70), (0.70, 0.85),
                                          (0.85, 1.00)],
    "5 fenetres 50->100 pas 10%": [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8),
                                    (0.8, 0.9), (0.9, 1.0)],
}
for name, wins_ in schemes.items():
    rois = []
    for a, b in wins_:
        i1, i2 = int(n_ev*a), int(n_ev*b)
        te = [x for x in bets_clean if i1 <= x[1] < i2]
        m = v.met(te)
        rois.append(f"{m['roi']*100:+.0f}%(n={m['n']})")
    print(f"  {name}: " + "  ".join(rois))

# --- exclude the single hottest day, recompute pooled test --------------------
print("\n--- pool test 50-100% en EXCLUANT chaque jour a tour de role ---")
for d_ex in days:
    sub = [b for b in pool if str(b[4])[:10] != d_ex]
    m = v.met(sub)
    print(f"  sans {d_ex}: {v.fmt(m)}")
