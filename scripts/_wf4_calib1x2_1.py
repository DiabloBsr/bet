# -*- coding: utf-8 -*-
"""WF4 — Calibration 1X2 mid-odds (1.8-5.0+) pooled 9 ligues.

Methodologie:
- Cote d'OUVERTURE = snapshot MIN(o.id) par event_id.
- Walk-forward 70/30 par expected_start sur 8035 (train scan -> test eval).
- Replication des candidats sur pooled nouvelles ligues (8 ligues).
- Garde-fou corruption nouvelles ligues: ht<=ft + goals_json len == score_a+score_b si present.
- ROI = mise 1u a la cote offerte. p-value = binomtest(k, n, p0=mean(1/odds), greater).

Sortie: exports/wf4_calib1x2.json
"""
import sys, json, os
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from scipy.stats import binomtest
from scraper.config import load_settings

OUT = "exports/wf4_calib1x2.json"
NEW_LEAGUES = ["InstantLeague-8036", "InstantLeague-8037", "InstantLeague-8042",
               "InstantLeague-8043", "InstantLeague-8044", "InstantLeague-8056",
               "InstantLeague-8060", "InstantLeague-8065"]
L8035 = "InstantLeague-8035"

eng = create_engine(load_settings().db_url)

# ---------- load ----------
with eng.connect() as c:
    df = pd.read_sql(text("""
        SELECT e.id AS event_id, e.competition, e.expected_start, e.round_info,
               r.score_a, r.score_b, r.ht_score_a, r.ht_score_b, r.goals_json,
               o.odds_home, o.odds_draw, o.odds_away
        FROM events e
        JOIN results r ON r.event_id = e.id
        JOIN odds_snapshots o ON o.id = (
            SELECT MIN(o2.id) FROM odds_snapshots o2 WHERE o2.event_id = e.id)
    """), c)

print(f"raw rows: {len(df)}")

# exclure corrompus (couvre 8035)
with open("exports/corrupted_events.json", encoding="utf-8") as f:
    corr = json.load(f)
corrupted_ids = set(int(k) for k in corr["events"].keys())
df = df[~df["event_id"].isin(corrupted_ids)].copy()

# garde-fou maison (toutes ligues): HT > FT impossible ; goals_json incoherent
def goals_ok(row):
    if row.ht_score_a is not None and row.ht_score_b is not None:
        if row.ht_score_a > row.score_a or row.ht_score_b > row.score_b:
            return False
    gj = row.goals_json
    if gj:
        try:
            g = json.loads(gj)
            if isinstance(g, list) and len(g) > 0:
                if len(g) != row.score_a + row.score_b:
                    return False
        except Exception:
            pass
    return True

mask = df.apply(goals_ok, axis=1)
print(f"guard removed: {(~mask).sum()}")
df = df[mask].copy()
df = df.dropna(subset=["odds_home", "odds_draw", "odds_away", "score_a", "score_b"])
df["outcome"] = np.where(df.score_a > df.score_b, "H",
                np.where(df.score_a < df.score_b, "A", "D"))
df["expected_start"] = pd.to_datetime(df["expected_start"])
print(df.groupby("competition").size())

POSITIONS = {"H": "odds_home", "D": "odds_draw", "A": "odds_away"}

# buckets fins largeur 0.25 de 1.50 a 6.00 (couvre la zone prior home 5-6)
EDGES = [round(1.50 + 0.25 * i, 2) for i in range(19)]  # 1.50 .. 6.00
BUCKETS = list(zip(EDGES[:-1], EDGES[1:]))

def eval_bets(sub, pos, col):
    """sub: df rows; bet 1u on pos at sub[col]. Returns dict stats."""
    n = len(sub)
    if n == 0:
        return dict(n=0)
    win = (sub["outcome"] == pos)
    k = int(win.sum())
    odds = sub[col].values
    profit = np.where(win, odds - 1.0, -1.0)
    roi = float(profit.sum() / n * 100)
    p0 = float(np.mean(1.0 / odds))  # break-even win prob
    side = "greater" if roi >= 0 else "less"
    pv = float(binomtest(k, n, p0, alternative=side).pvalue)
    return dict(n=n, wins=k, wr=round(k / n, 4), roi_pct=round(roi, 2),
                avg_odds=round(float(odds.mean()), 3), p0=round(p0, 4),
                pvalue=round(pv, 6))

def scan(dframe, label):
    rows = []
    for pos, col in POSITIONS.items():
        for lo, hi in BUCKETS:
            sub = dframe[(dframe[col] >= lo) & (dframe[col] < hi)]
            st = eval_bets(sub, pos, col)
            st.update(pos=pos, lo=lo, hi=hi, scope=label)
            rows.append(st)
    return rows

# ---------- walk-forward 8035 ----------
d35 = df[df.competition == L8035].sort_values("expected_start").reset_index(drop=True)
cut = int(len(d35) * 0.70)
train, test = d35.iloc[:cut], d35.iloc[cut:]
print(f"8035 clean={len(d35)} train={len(train)} ({train.expected_start.min()} -> {train.expected_start.max()}) "
      f"test={len(test)} ({test.expected_start.min()} -> {test.expected_start.max()})")

train_scan = scan(train, "8035-train")
n_tests = len([r for r in train_scan if r.get("n", 0) > 0])

# candidats: train n>=100, ROI>=+4%
cands = [r for r in train_scan if r.get("n", 0) >= 100 and r.get("roi_pct", -99) >= 4.0]
print(f"\nscanned cells (non-empty): {n_tests}; candidates: {len(cands)}")
for r in cands:
    print(f"  CAND {r['pos']} [{r['lo']},{r['hi']}) train n={r['n']} wr={r['wr']} roi={r['roi_pct']}% p={r['pvalue']}")

dnew = df[df.competition.isin(NEW_LEAGUES)]
results = []
for r in cands:
    pos, col, lo, hi = r["pos"], POSITIONS[r["pos"]], r["lo"], r["hi"]
    te = eval_bets(test[(test[col] >= lo) & (test[col] < hi)], pos, col)
    nw = eval_bets(dnew[(dnew[col] >= lo) & (dnew[col] < hi)], pos, col)
    per_league = {}
    for lg in NEW_LEAGUES:
        dl = dnew[dnew.competition == lg]
        per_league[lg.split("-")[1]] = eval_bets(dl[(dl[col] >= lo) & (dl[col] < hi)], pos, col)
    results.append(dict(pos=pos, lo=lo, hi=hi, train=r, test_8035=te,
                        pooled_new=nw, per_league_new=per_league))
    print(f"\n== {pos} [{lo},{hi}) ==")
    print(f"  train: n={r['n']} wr={r['wr']} roi={r['roi_pct']}% p={r['pvalue']}")
    print(f"  TEST 8035: n={te.get('n')} wr={te.get('wr')} roi={te.get('roi_pct')}% p={te.get('pvalue')}")
    print(f"  NEW pooled: n={nw.get('n')} wr={nw.get('wr')} roi={nw.get('roi_pct')}% p={nw.get('pvalue')}")

# scan complet aussi sur test et new (pour calibration globale / sanity, PAS pour selection)
full = dict(
    meta=dict(n_8035=len(d35), n_train=len(train), n_test=len(test),
              n_new_pooled=len(dnew), n_tests_scanned=n_tests,
              bucket_width=0.25, range=[1.50, 6.00],
              corrupted_excluded=len(corrupted_ids), guard_removed=int((~mask).sum())),
    train_scan=train_scan,
    candidates=results,
    test_scan=scan(test, "8035-test"),
    new_scan=scan(dnew, "pooled-newleagues"),
)
os.makedirs("exports", exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(full, f, ensure_ascii=False, indent=1)
print(f"\nwritten {OUT}")
