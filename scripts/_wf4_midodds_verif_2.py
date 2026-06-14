# -*- coding: utf-8 -*-
"""Verification adversariale du finding RETRAIT mid-odds (_wf4_calib1x2_2.py).

Checks:
 1. MIN(o.id) == plus ancien captured_at par event (look-ahead?)
 2. Structure corrupted_events.json + exclusion effective
 3. Distribution segments par ligue + part round=-1 (J0)
 4. Spot-check settlement 1X2
 5. Cote away > 15 tronquees par [5,15) ?
 6. Recompute conservateur P2 (claim le plus fort) :
    - 8035 pooled complet (in-sample du claim d'origine)
    - splits alternatifs 50/50 et 80/20
    - sous-periodes temporelles sur test+new combines
    - bootstrap CI 10k sur le ROI combine
Sortie: exports/wf4_midodds_verify_checks.json
"""
import sys, json, math
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from scraper.config import load_settings

rng = np.random.default_rng(42)
eng = create_engine(load_settings().db_url)
out = {}

# ---- 1. look-ahead: MIN(id) vs MIN(captured_at)
with eng.connect() as c:
    bad = c.execute(text("""
        WITH firstid AS (
            SELECT event_id, MIN(id) AS mid FROM odds_snapshots GROUP BY event_id),
        firstts AS (
            SELECT event_id, MIN(captured_at) AS mts FROM odds_snapshots GROUP BY event_id)
        SELECT COUNT(*) FROM firstid f
        JOIN odds_snapshots o ON o.id = f.mid
        JOIN firstts t ON t.event_id = f.event_id
        WHERE o.captured_at > t.mts
    """)).scalar()
    n_ev = c.execute(text("SELECT COUNT(DISTINCT event_id) FROM odds_snapshots")).scalar()
    # cote d'ouverture posterieure au coup d'envoi ?
    late_open = c.execute(text("""
        SELECT COUNT(*) FROM events e
        JOIN odds_snapshots o ON o.id = (
            SELECT MIN(o2.id) FROM odds_snapshots o2 WHERE o2.event_id = e.id)
        WHERE o.captured_at > e.expected_start
    """)).scalar()
print(f"1. MIN(id) plus tardif que MIN(captured_at): {bad}/{n_ev} events; opening snapshot APRES expected_start: {late_open}")
out["lookahead"] = dict(minid_vs_mints_bad=int(bad), n_events=int(n_ev), opening_after_start=int(late_open))

# ---- charge le meme dataset que le script attaque
with eng.connect() as c:
    df = pd.read_sql(text("""
        SELECT e.id AS event_id, e.competition, e.expected_start, e.round_info,
               r.score_a, r.score_b, o.odds_home, o.odds_draw, o.odds_away, o.captured_at
        FROM events e
        JOIN results r ON r.event_id = e.id
        JOIN odds_snapshots o ON o.id = (
            SELECT MIN(o2.id) FROM odds_snapshots o2 WHERE o2.event_id = e.id)
    """), c)
corr = json.load(open("exports/corrupted_events.json", encoding="utf-8"))
corrupted_ids = set(int(k) for k in corr["events"].keys())
n_before = len(df)
df = df[~df["event_id"].isin(corrupted_ids)].copy()
print(f"2. corrupted file: {len(corrupted_ids)} ids, exclus du dataset: {n_before - len(df)}")
out["corrupted"] = dict(ids_in_file=len(corrupted_ids), excluded_rows=n_before - len(df))

df = df.dropna(subset=["odds_home", "odds_draw", "odds_away", "score_a", "score_b"])
df["outcome"] = np.where(df.score_a > df.score_b, "H", np.where(df.score_a < df.score_b, "A", "D"))
df["expected_start"] = pd.to_datetime(df["expected_start"])
MAX_ROUND = {"InstantLeague-8035": 38, "InstantLeague-8036": 38, "InstantLeague-8037": 38,
             "InstantLeague-8042": 34, "InstantLeague-8043": 34, "InstantLeague-8044": 34,
             "InstantLeague-8056": 70, "InstantLeague-8060": 46, "InstantLeague-8065": 94}
df["round"] = pd.to_numeric(df["round_info"], errors="coerce").fillna(-1).astype(int)
df["max_round"] = df["competition"].map(MAX_ROUND)
df["rfrac"] = np.where(df["round"] >= 1, df["round"] / df["max_round"], np.nan)
def seg_of(r):
    if np.isnan(r): return "J0"
    if r <= 3/38: return "DS"
    if r <= 12/38: return "MS_early"
    if r <= 25/38: return "MS_mid"
    return "late"
df["seg"] = df["rfrac"].apply(seg_of)

# ---- 3. segments par ligue + J0
tab = df.groupby(["competition", "seg"]).size().unstack(fill_value=0)
print("3. segments par ligue:\n", tab)
out["seg_by_league"] = {str(k): {s: int(v) for s, v in row.items()} for k, row in tab.iterrows()}
# round_info au-dela de max_round ? (mapping max_round faux -> rfrac>1 -> late)
over = df[(df["round"] >= 1) & (df["rfrac"] > 1.0)].groupby("competition").size()
print("   rounds > max_round:", dict(over))
out["rounds_over_max"] = {str(k): int(v) for k, v in over.items()}

# ---- 4. spot-check settlement
sample = df.sample(5, random_state=1)[["event_id", "score_a", "score_b", "outcome"]]
print("4. spot settlement:\n", sample.to_string(index=False))

# ---- 5. cotes away > 15 en MS_mid (tronquees par le bucket [5,15)) ?
msa = df[(df.seg == "MS_mid") & (df.odds_away >= 15)]
print(f"5. MS_mid away odds >=15 (exclues du bucket [5,15)): {len(msa)}, max odds_away global={df.odds_away.max()}")
out["odds_truncation"] = dict(msmid_away_ge15=int(len(msa)), max_odds_away=float(df.odds_away.max()))

# ---- 6. recompute conservateur P2 (MS_mid, away >= 5)
def roi_stats(sub):
    n = len(sub)
    if n == 0: return dict(n=0)
    odds = sub["odds_away"].values.astype(float)
    win = (sub["outcome"] == "A").values
    profit = np.where(win, odds - 1.0, -1.0)
    tot = float(profit.sum())
    q0 = 1.0 / odds
    var0 = q0 * (odds - 1.0) ** 2 + (1 - q0)
    sd = math.sqrt(float(var0.sum()))
    from scipy.stats import norm
    z = tot / sd if sd > 0 else 0
    return dict(n=n, wins=int(win.sum()), roi_pct=round(tot / n * 100, 2),
                avg_odds=round(float(odds.mean()), 2), pvalue=round(float(2 * (1 - norm.cdf(abs(z)))), 5))

L = "InstantLeague-8035"
NEW = [k for k in MAX_ROUND if k != L]
bets = df[(df.seg == "MS_mid") & (df.odds_away >= 5.0) & (df.odds_away < 15.0)].copy()
b35 = bets[bets.competition == L].sort_values("expected_start").reset_index(drop=True)
bnew = bets[bets.competition.isin(NEW)].copy()
print("\n6. P2 (MS_mid A>=5) recompute conservateur")
print("   8035 POOLED complet (perimetre du claim d'origine):", roi_stats(b35))
for frac in (0.5, 0.7, 0.8):
    cut = b35.expected_start.quantile(frac)
    te = b35[b35.expected_start >= cut]
    print(f"   8035 TEST split {int(frac*100)}/{100-int(frac*100)}:", roi_stats(te))
print("   NEW pooled:", roi_stats(bnew))
out["p2_8035_pooled"] = roi_stats(b35)
out["p2_new_pooled"] = roi_stats(bnew)

# sous-periodes temporelles sur l'echantillon hors-train (test 8035 + new)
cut70 = b35.expected_start.quantile(0.70)
oos = pd.concat([b35[b35.expected_start >= cut70], bnew]).sort_values("expected_start").reset_index(drop=True)
print("   OOS combine:", roi_stats(oos))
out["p2_oos_combined"] = roi_stats(oos)
qs = oos.expected_start.quantile([0.25, 0.5, 0.75]).values
parts = [oos[oos.expected_start < qs[0]], oos[(oos.expected_start >= qs[0]) & (oos.expected_start < qs[1])],
         oos[(oos.expected_start >= qs[1]) & (oos.expected_start < qs[2])], oos[oos.expected_start >= qs[2]]]
out["p2_oos_quarters"] = []
for i, p in enumerate(parts):
    s = roi_stats(p)
    out["p2_oos_quarters"].append(s)
    print(f"   OOS quart {i+1}:", s)

# bootstrap 10k sur le ROI OOS combine
odds = oos["odds_away"].values.astype(float)
profit = np.where((oos["outcome"] == "A").values, odds - 1.0, -1.0)
idx = rng.integers(0, len(profit), size=(10000, len(profit)))
rois = profit[idx].mean(axis=1) * 100
ci = np.percentile(rois, [2.5, 50, 97.5])
p_neg = float((rois <= 0).mean())
print(f"   bootstrap ROI OOS: mediane={ci[1]:.2f}% CI95=[{ci[0]:.2f}, {ci[2]:.2f}] P(ROI<=0)={p_neg:.3f}")
out["p2_bootstrap"] = dict(median=round(float(ci[1]), 2), lo=round(float(ci[0]), 2),
                           hi=round(float(ci[2]), 2), p_roi_le_0=p_neg)

json.dump(out, open("exports/wf4_midodds_verify_checks.json", "w", encoding="utf-8"), indent=1)
print("\nwritten exports/wf4_midodds_verify_checks.json")
