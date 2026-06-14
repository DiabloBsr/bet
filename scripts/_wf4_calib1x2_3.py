# -*- coding: utf-8 -*-
"""WF4 calibration 1X2 — passe 3: deviations vs null MARGE-NORMALISE pooled 9 ligues.

H0 standard (q=1/cote) teste ROI>0. Ici on teste la CALIBRATION:
H0': q_i = (1/cote_i) / sum_j(1/cote_j du match) — proba implicite renormalisee
(marge 1X2 plate 6% repartie proportionnellement, prouve dans ENGINE_MODEL).
Sous H0', ROI attendu = -marge/(1+marge) ~ -5.66%. Une cellule avec deviation
significative POSITIVE = le simulateur sur-realise cette zone (accuracy > implicite
a cote conservee); il faut +6pp de ROI au-dessus pour etre profitable en mise reelle.

Tests:
T1: position-level pooled-9 (H/D/A, cotes 1.8-5.0) — la grande image.
T2: buckets fins 0.25 x position pooled-9 vs H0' — liste cellules p<=0.01,
    puis replication 8035-test vs nouvelles ligues pour celles-ci.
T3: deficit de nuls: freq reelle des nuls vs implicite normalisee, par bucket de cote nul,
    8035 vs nouvelles ligues separement.
Sortie: exports/wf4_calib1x2_margnorm.json
"""
import sys, json, math
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from scipy.stats import norm
from scraper.config import load_settings

OUT = "exports/wf4_calib1x2_margnorm.json"
L8035 = "InstantLeague-8035"
NEW_LEAGUES = ["InstantLeague-8036", "InstantLeague-8037", "InstantLeague-8042",
               "InstantLeague-8043", "InstantLeague-8044", "InstantLeague-8056",
               "InstantLeague-8060", "InstantLeague-8065"]

eng = create_engine(load_settings().db_url)
with eng.connect() as c:
    df = pd.read_sql(text("""
        SELECT e.id AS event_id, e.competition, e.expected_start,
               r.score_a, r.score_b, r.ht_score_a, r.ht_score_b, r.goals_json,
               o.odds_home, o.odds_draw, o.odds_away
        FROM events e
        JOIN results r ON r.event_id = e.id
        JOIN odds_snapshots o ON o.id = (
            SELECT MIN(o2.id) FROM odds_snapshots o2 WHERE o2.event_id = e.id)
    """), c)

with open("exports/corrupted_events.json", encoding="utf-8") as f:
    corr = json.load(f)
df = df[~df["event_id"].isin(set(int(k) for k in corr["events"].keys()))].copy()

def goals_ok(row):
    if row.ht_score_a is not None and row.ht_score_b is not None:
        if row.ht_score_a > row.score_a or row.ht_score_b > row.score_b:
            return False
    gj = row.goals_json
    if gj:
        try:
            g = json.loads(gj)
            if isinstance(g, list) and len(g) > 0 and len(g) != row.score_a + row.score_b:
                return False
        except Exception:
            pass
    return True

df = df[df.apply(goals_ok, axis=1)].copy()
df = df.dropna(subset=["odds_home", "odds_draw", "odds_away", "score_a", "score_b"])
df["outcome"] = np.where(df.score_a > df.score_b, "H",
                np.where(df.score_a < df.score_b, "A", "D"))
df["booksum"] = 1.0 / df.odds_home + 1.0 / df.odds_draw + 1.0 / df.odds_away
df["expected_start"] = pd.to_datetime(df["expected_start"])
print(f"clean rows: {len(df)}; booksum mean={df.booksum.mean():.4f} std={df.booksum.std():.4f}")

POSCOL = {"H": "odds_home", "D": "odds_draw", "A": "odds_away"}
n_tests = 0

def eval_norm(sub, pos):
    """Deviation vs H0' q_i = (1/cote)/booksum. Rapporte aussi ROI reel (mise 1u)."""
    n = len(sub)
    if n == 0:
        return dict(n=0)
    odds = sub[POSCOL[pos]].values.astype(float)
    bs = sub["booksum"].values
    qn = (1.0 / odds) / bs                      # implicite normalisee
    win = (sub["outcome"] == pos).values
    k = int(win.sum())
    profit = np.where(win, odds - 1.0, -1.0)
    roi = float(profit.mean() * 100)
    # z sur le nombre de victoires sous H0' (Poisson-binomiale ~ normale)
    mu = float(qn.sum()); var = float((qn * (1 - qn)).sum())
    z = (k - mu) / math.sqrt(var) if var > 0 else 0.0
    pv = float(2 * (1 - norm.cdf(abs(z))))
    # ROI attendu sous H0'
    roi0 = float(np.mean(qn * odds - 1.0) * 100)
    return dict(n=n, wins=k, wr=round(k / n, 4), implied_norm=round(mu / n, 4),
                dev_pp=round((k - mu) / n * 100, 2), z=round(z, 3), pvalue=round(pv, 6),
                roi_pct=round(roi, 2), roi_expected_H0=round(roi0, 2),
                avg_odds=round(float(odds.mean()), 3))

d35 = df[df.competition == L8035].sort_values("expected_start")
cut = int(len(d35) * 0.70)
test35 = d35.iloc[cut:]
dnew = df[df.competition.isin(NEW_LEAGUES)]

out = {"meta": dict(n_pooled9=len(df), n_8035=len(d35), n_test35=len(test35), n_new=len(dnew))}

# ===== T1: position-level pooled-9, cotes 1.8-5.0 =====
print("\n===== T1 position-level (1.8-5.0) pooled-9 =====")
out["T1"] = {}
for pos, col in POSCOL.items():
    sub = df[(df[col] >= 1.8) & (df[col] < 5.0)]
    r = eval_norm(sub, pos); n_tests += 1
    out["T1"][pos] = r
    print(f"{pos}: n={r['n']} wr={r['wr']} implied_norm={r['implied_norm']} dev={r['dev_pp']}pp "
          f"z={r['z']} p={r['pvalue']} | ROI reel={r['roi_pct']}% (attendu H0' {r['roi_expected_H0']}%)")

# ===== T2: buckets fins 0.25 pooled-9 vs H0' =====
print("\n===== T2 buckets 0.25 pooled-9 vs H0' (cellules p<=0.01) =====")
EDGES = [round(1.5 + 0.25 * i, 2) for i in range(19)]
t2 = []
for pos, col in POSCOL.items():
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        sub = df[(df[col] >= lo) & (df[col] < hi)]
        r = eval_norm(sub, pos); n_tests += 1
        r.update(pos=pos, lo=lo, hi=hi)
        t2.append(r)
out["T2_scan"] = t2
sig = [r for r in t2 if r.get("n", 0) >= 100 and r.get("pvalue", 1) <= 0.01]
out["T2_sig_replication"] = []
for r in sig:
    pos, col, lo, hi = r["pos"], POSCOL[r["pos"]], r["lo"], r["hi"]
    r35 = eval_norm(test35[(test35[col] >= lo) & (test35[col] < hi)], pos)
    rnw = eval_norm(dnew[(dnew[col] >= lo) & (dnew[col] < hi)], pos)
    rec = dict(cell=r, test_8035=r35, new=rnw)
    out["T2_sig_replication"].append(rec)
    print(f"SIG {pos} [{lo},{hi}): pooled dev={r['dev_pp']}pp p={r['pvalue']} roi={r['roi_pct']}% | "
          f"8035-test dev={r35.get('dev_pp')}pp (n={r35.get('n')}) | new dev={rnw.get('dev_pp')}pp (n={rnw.get('n')})")
if not sig:
    print("aucune cellule p<=0.01")

# ===== T3: deficit de nuls par bucket de cote nul, 8035 vs new =====
print("\n===== T3 nuls: reel vs implicite normalisee =====")
out["T3"] = {}
DEDGES = [(3.0, 3.5), (3.5, 4.0), (4.0, 4.5), (4.5, 5.0), (5.0, 6.0), (6.0, 8.0), (8.0, 15.0)]
for scope, dd in [("8035", d35), ("new8", dnew)]:
    rows = []
    for lo, hi in DEDGES:
        sub = dd[(dd.odds_draw >= lo) & (dd.odds_draw < hi)]
        r = eval_norm(sub, "D"); n_tests += 1
        r.update(lo=lo, hi=hi)
        rows.append(r)
        if r["n"] > 0:
            print(f"{scope} D [{lo},{hi}): n={r['n']} wr={r['wr']} impl={r['implied_norm']} "
                  f"dev={r['dev_pp']}pp p={r['pvalue']} roi={r['roi_pct']}%")
    out["T3"][scope] = rows
# agregat global nuls
for scope, dd in [("8035", d35), ("new8", dnew), ("pooled9", df)]:
    r = eval_norm(dd, "D"); n_tests += 1
    out["T3"][f"all_draws_{scope}"] = r
    print(f"{scope} TOUS nuls: n={r['n']} wr={r['wr']} impl={r['implied_norm']} dev={r['dev_pp']}pp z={r['z']} p={r['pvalue']}")

out["meta"]["n_tests_scanned"] = n_tests
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(f"\nn_tests={n_tests}; written {OUT}")
