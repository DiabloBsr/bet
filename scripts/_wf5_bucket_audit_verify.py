# -*- coding: utf-8 -*-
"""WF5 VERIFY — contre-verification du verdict OVERFIT sur COTE_EDGES.

Variante durcie de _wf5_bucket_audit.py :
  - exclut AUSSI les events dont le snapshot d'ouverture est posterieur au resultat
    (captured_at >= finished_at, odds suspectes / re-listing)
  - recalcule corr orig<->recent, cellules |z|>=2, portefeuilles edges+/traps
  - test z exact du portefeuille EDGES+ recent contre (a) 0 et (b) l'attente declaree,
    avec la vraie variance par pari (pnl par ligne conservee)

Sortie : exports/wf5_verify.json. LECTURE SEULE sur la DB.
"""
import sys, json, math
sys.path.insert(0, ".")
import pandas as pd
from sqlalchemy import create_engine, text
from scraper.config import load_settings
from scraper.strategy_engine import label_segment, COTE_EDGES

SPLIT_DATE = "2026-06-08"
LEAGUE = "InstantLeague-8035"
BUCKETS = [
    ("favori_extreme_1.0_1.3", 1.00, 1.30), ("favori_solide_1.3_1.5", 1.30, 1.50),
    ("favori_modere_1.5_1.8", 1.50, 1.80), ("leger_favori_1.8_2.2", 1.80, 2.20),
    ("equilibre_2.2_2.7", 2.20, 2.70), ("non_favori_2.7_3.5", 2.70, 3.50),
    ("underdog_3.5_5", 3.50, 5.00), ("long_shot_5plus", 5.00, 50.0),
]

def bucket_of(c):
    for name, lo, hi in BUCKETS:
        if lo <= c < hi:
            return name
    return None

eng = create_engine(load_settings().db_url)
bad = set(int(k) for k in json.load(open("exports/corrupted_events.json"))["events"].keys())

q = f"""
SELECT ev.id AS event_id, CAST(ev.round_info AS INT) AS journee,
       os.odds_home, os.odds_away, r.score_a, r.score_b,
       substr(r.finished_at,1,10) AS fin_date,
       CASE WHEN os.captured_at >= r.finished_at THEN 1 ELSE 0 END AS post_result
FROM events ev
JOIN results r ON r.event_id = ev.id
JOIN odds_snapshots os ON os.id = (
    SELECT MIN(os2.id) FROM odds_snapshots os2 WHERE os2.event_id = ev.id)
WHERE ev.competition = '{LEAGUE}'
  AND r.score_a IS NOT NULL AND r.score_b IS NOT NULL
  AND ev.round_info GLOB '[0-9]*' AND CAST(ev.round_info AS INT) BETWEEN 1 AND 38
  AND os.odds_home IS NOT NULL AND os.odds_away IS NOT NULL
"""
df = pd.read_sql(text(q), eng)
df = df[~df.event_id.isin(bad)]
n_before = len(df)
df = df[df.post_result == 0].copy()
print(f"matches: {n_before} -> {len(df)} after excluding post-result openings")
df["segment"] = df.journee.map(label_segment)
df["period"] = (df.fin_date >= SPLIT_DATE).map({False: "orig", True: "recent"})
print(df.period.value_counts().to_dict())

rows = []
for t in df.itertuples(index=False):
    for side, cote, win in (("home", t.odds_home, t.score_a > t.score_b),
                            ("away", t.odds_away, t.score_b > t.score_a)):
        b = bucket_of(float(cote))
        if b:
            rows.append((t.period, t.segment, side, b, float(cote),
                         (cote - 1.0) if win else -1.0))
bets = pd.DataFrame(rows, columns=["period", "segment", "side", "bucket", "cote", "pnl"])

# ---- grille + correlation orig/recent (n>=50) ----
stats = bets.groupby(["segment", "side", "bucket", "period"]).pnl.agg(["count", "mean"]).unstack()
cf = stats.dropna()
cf = cf[(cf[("count", "orig")] >= 50) & (cf[("count", "recent")] >= 50)]
ro, rr = cf[("mean", "orig")], cf[("mean", "recent")]
print(f"\ncells n>=50 both: {len(cf)}  pearson={ro.corr(rr):.3f}  spearman={ro.corr(rr, method='spearman'):.3f}")
print(f"same sign: {int(((ro*rr)>0).sum())}/{len(cf)}")

# ---- cellules |z|>=2 en orig (n>=30) et leur sort en recent ----
sig = []
g_orig = bets[bets.period == "orig"].groupby(["segment", "side", "bucket"]).pnl
for key, g in bets[bets.period == "orig"].groupby(["segment", "side", "bucket"]):
    n = len(g)
    if n < 30:
        continue
    m, s = g.pnl.mean(), g.pnl.std(ddof=1)
    z = m / (s / math.sqrt(n)) if s > 0 else 0
    if abs(z) >= 2:
        gr = bets[(bets.period == "recent") & (bets.segment == key[0]) &
                  (bets.side == key[1]) & (bets.bucket == key[2])]
        mr = gr.pnl.mean() if len(gr) else float("nan")
        sig.append((key, n, m * 100, z, len(gr), mr * 100))
n30 = sum(1 for _, g in bets[bets.period == "orig"].groupby(["segment", "side", "bucket"]) if len(g) >= 30)
print(f"\norig cells n>=30: {n30}, |z|>=2: {len(sig)} (expected ~{0.05*n30:.1f})")
inv = 0
for key, n, roi, z, nr, roir in sig:
    flip = (roi * roir < 0)
    inv += flip
    print(f"  {key} orig n={n} roi={roi:+.1f}% z={z:+.2f} | recent n={nr} roi={roir:+.1f}% {'INVERSE' if flip else 'same-sign'}")
print(f"sign flips in recent: {inv}/{len(sig)}")

# ---- portefeuille EDGES+ / TRAPS declares, test z exact sur recent ----
def in_cell(b, e, seg):
    return (b.segment == seg) & (b.side == e["side"]) & \
           (b.cote >= e["min"]) & (b.cote < e["max"])

res = {}
for label, want_trap in (("edges_plus", False), ("traps", True)):
    mask = pd.Series(False, index=bets.index)
    decl_roi = pd.Series(0.0, index=bets.index)
    for seg, cells in COTE_EDGES.items():
        for name, e in cells.items():
            if (e["edge"] < 0) == want_trap:
                m = in_cell(bets, e, seg)
                mask |= m
                decl_roi[m] = e["roi"]
    for per in ("orig", "recent"):
        sel = bets[mask & (bets.period == per)]
        n = len(sel)
        roi = sel.pnl.mean()
        sd = sel.pnl.std(ddof=1)
        se = sd / math.sqrt(n)
        z0 = roi / se
        exp_decl = decl_roi[sel.index].mean()
        z_decl = (roi - exp_decl) / se
        res[f"{label}_{per}"] = dict(n=int(n), roi_pct=round(roi*100, 2),
                                     se_pct=round(se*100, 2), z_vs_0=round(z0, 2),
                                     declared_exp_pct=round(exp_decl*100, 2),
                                     z_vs_declared=round(z_decl, 2))
        print(f"{label} [{per}]: n={n} roi={roi*100:+.2f}% se={se*100:.2f}pp "
              f"z_vs_0={z0:+.2f} | declared_exp={exp_decl*100:+.1f}% z_vs_declared={z_decl:+.2f}")

base = bets.groupby("period").pnl.agg(["count", "mean"])
print("\nbaseline:", {p: (int(r["count"]), round(r["mean"]*100, 2)) for p, r in base.iterrows()})

out = {"n_matches": int(len(df)), "corr_n50": {"n": len(cf), "pearson": round(float(ro.corr(rr)), 3),
       "spearman": round(float(ro.corr(rr, method='spearman')), 3)},
       "sig_cells": [{"cell": list(k), "n_o": n, "roi_o": round(r, 1), "z_o": round(z, 2),
                      "n_r": nr, "roi_r": round(rr_, 1)} for k, n, r, z, nr, rr_ in sig],
       "portfolios": res,
       "baseline": {p: {"n": int(r["count"]), "roi_pct": round(r["mean"]*100, 2)} for p, r in base.iterrows()}}
json.dump(out, open("exports/wf5_verify.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("written exports/wf5_verify.json")
