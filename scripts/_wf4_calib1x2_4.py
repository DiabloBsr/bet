# -*- coding: utf-8 -*-
"""WF4 calibration 1X2 — passe 4: verifications finales.

V1: stabilite temporelle du deficit de nuls 8035 (train vs test WF 70/30),
    et zoom bucket 3.5-4.0 (le plus deviant) sur test seul.
V2: fragilite du candidat G_A_3.0-3.5_MS_early: cellules voisines (cote +/-0.5,
    segments adjacents, toutes-saisons) + detail par ligue + p combinee.
Sortie: exports/wf4_calib1x2_checks.json
"""
import sys, json, math
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from scipy.stats import norm
from scraper.config import load_settings

OUT = "exports/wf4_calib1x2_checks.json"
L8035 = "InstantLeague-8035"
NEW_LEAGUES = ["InstantLeague-8036", "InstantLeague-8037", "InstantLeague-8042",
               "InstantLeague-8043", "InstantLeague-8044", "InstantLeague-8056",
               "InstantLeague-8060", "InstantLeague-8065"]
MAX_ROUND = {"InstantLeague-8035": 38, "InstantLeague-8036": 38, "InstantLeague-8037": 38,
             "InstantLeague-8042": 34, "InstantLeague-8043": 34, "InstantLeague-8044": 34,
             "InstantLeague-8056": 70, "InstantLeague-8060": 46, "InstantLeague-8065": 94}

eng = create_engine(load_settings().db_url)
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
df["round"] = pd.to_numeric(df["round_info"], errors="coerce").fillna(-1).astype(int)
df["rfrac"] = np.where(df["round"] >= 1, df["round"] / df["competition"].map(MAX_ROUND), np.nan)

def seg_of(rf):
    if np.isnan(rf): return "J0"
    if rf <= 3.0 / 38: return "DS"
    if rf <= 12.0 / 38: return "MS_early"
    if rf <= 25.0 / 38: return "MS_mid"
    return "late"
df["seg"] = df["rfrac"].apply(seg_of)

POSCOL = {"H": "odds_home", "D": "odds_draw", "A": "odds_away"}

def eval_norm(sub, pos):
    n = len(sub)
    if n == 0: return dict(n=0)
    odds = sub[POSCOL[pos]].values.astype(float)
    qn = (1.0 / odds) / sub["booksum"].values
    win = (sub["outcome"] == pos).values
    k = int(win.sum())
    profit = np.where(win, odds - 1.0, -1.0)
    mu = float(qn.sum()); var = float((qn * (1 - qn)).sum())
    z = (k - mu) / math.sqrt(var) if var > 0 else 0.0
    # ROI z-test vs break-even (q=1/odds)
    q0 = 1.0 / odds
    var0 = q0 * (odds - 1.0) ** 2 + (1.0 - q0)
    zroi = float(profit.sum()) / math.sqrt(float(var0.sum())) if n else 0.0
    return dict(n=n, wins=k, wr=round(k / n, 4), implied_norm=round(mu / n, 4),
                dev_pp=round((k - mu) / n * 100, 2), z=round(z, 3),
                pvalue=round(float(2 * (1 - norm.cdf(abs(z)))), 6),
                roi_pct=round(float(profit.mean() * 100), 2),
                z_roi=round(zroi, 3), p_roi=round(float(2 * (1 - norm.cdf(abs(zroi)))), 6),
                avg_odds=round(float(odds.mean()), 3))

d35 = df[df.competition == L8035].sort_values("expected_start")
cut = int(len(d35) * 0.70)
tr35, te35 = d35.iloc[:cut], d35.iloc[cut:]
dnew = df[df.competition.isin(NEW_LEAGUES)]
out = {}

print("===== V1 deficit nuls 8035: stabilite temporelle =====")
out["V1"] = {}
for name, dd in [("train", tr35), ("test", te35), ("full", d35)]:
    r = eval_norm(dd, "D")
    out["V1"][name] = r
    print(f"8035-{name} TOUS nuls: n={r['n']} wr={r['wr']} impl={r['implied_norm']} dev={r['dev_pp']}pp z={r['z']} p={r['pvalue']}")
for name, dd in [("train", tr35), ("test", te35)]:
    sub = dd[(dd.odds_draw >= 3.5) & (dd.odds_draw < 4.0)]
    r = eval_norm(sub, "D")
    out["V1"][f"{name}_3.5-4.0"] = r
    print(f"8035-{name} D[3.5,4.0): n={r['n']} wr={r['wr']} impl={r['implied_norm']} dev={r['dev_pp']}pp p={r['pvalue']} roi={r['roi_pct']}%")
# contrepartie: ou va la masse manquante des nuls sur 8035 ? H et A full
for pos in ["H", "A"]:
    r = eval_norm(d35, pos)
    out["V1"][f"full_{pos}"] = r
    print(f"8035-full {pos}: n={r['n']} dev={r['dev_pp']}pp z={r['z']} p={r['pvalue']}")

print("\n===== V2 fragilite G_A_3.0-3.5_MS_early =====")
out["V2"] = {}
def cellrep(label, pos, lo, hi, seg):
    col = POSCOL[pos]
    def f(dd):
        s = dd[(dd[col] >= lo) & (dd[col] < hi)]
        if seg: s = s[s.seg == seg]
        return s
    tr, te, nw = eval_norm(f(tr35), pos), eval_norm(f(te35), pos), eval_norm(f(dnew), pos)
    comb = eval_norm(pd.concat([f(te35), f(dnew)]), pos)
    rec = dict(train=tr, test=te, new=nw, comb=comb)
    out["V2"][label] = rec
    print(f"{label}: train roi={tr.get('roi_pct')} (n={tr.get('n')}) | test roi={te.get('roi_pct')} (n={te.get('n')}) | "
          f"new roi={nw.get('roi_pct')} (n={nw.get('n')}) | comb roi={comb.get('roi_pct')} n={comb.get('n')} "
          f"z_roi={comb.get('z_roi')} p_roi={comb.get('p_roi')} dev={comb.get('dev_pp')}pp")
cellrep("A_3.0-3.5_MS_early(cible)", "A", 3.0, 3.5, "MS_early")
cellrep("A_2.5-3.0_MS_early(voisin-cote)", "A", 2.5, 3.0, "MS_early")
cellrep("A_3.5-4.0_MS_early(voisin-cote)", "A", 3.5, 4.0, "MS_early")
cellrep("A_3.0-3.5_DS(voisin-seg)", "A", 3.0, 3.5, "DS")
cellrep("A_3.0-3.5_MS_mid(voisin-seg)", "A", 3.0, 3.5, "MS_mid")
cellrep("A_3.0-3.5_toutes-saisons", "A", 3.0, 3.5, None)
cellrep("H_3.0-3.5_MS_early(miroir)", "H", 3.0, 3.5, "MS_early")
# detail par ligue pour la cellule cible
col = POSCOL["A"]
det = []
for lg in NEW_LEAGUES:
    s = dnew[(dnew.competition == lg) & (dnew[col] >= 3.0) & (dnew[col] < 3.5) & (dnew.seg == "MS_early")]
    r = eval_norm(s, "A"); r["lg"] = lg.split("-")[1]
    det.append(r)
    if r.get("n", 0) > 0:
        print(f"  {r['lg']}: n={r['n']} wr={r.get('wr')} roi={r.get('roi_pct')}%")
out["V2"]["per_league_target"] = det

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(f"\nwritten {OUT}")
