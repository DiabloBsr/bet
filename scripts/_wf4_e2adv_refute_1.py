# -*- coding: utf-8 -*-
"""WF4 adversarial re-verification of the ANTI-FINDING on E2 (extreme favorite 1.10-1.20).

Independent code path (fresh SQL, no cache reuse). READ-ONLY on DB.
Checks:
  1. Fresh extraction: opening snapshot = MIN(id) per event, corrupted excluded,
     integrity guards identical in spirit to _wf4_crossleague_1.py.
  2. Reproduce era stats (8035-old / 8035-recent / pooled-new / pooled-newera).
  3. Timing audit: captured_at vs expected_start for opening snapshots (look-ahead?),
     split performance by before/after to test leakage sensitivity.
  4. Per-league breakdown of the new era (dominance check).
  5. Sub-period splits of the new era (quarters by expected_start).
  6. Boundary sensitivity: [1.05,1.25],[1.09,1.21],[1.10,1.15),[1.15,1.20],[1.12,1.18].
  7. Bootstrap 10k CI on pooled-new ROI.
  8. POWER: P(observing <= wins | old-era edge persists, wr=0.856 / 0.8674).
  9. Closing-odds variant (MAX(id) snapshot) on the new era.
Output: exports/wf4_e2adv_refute.json
"""
import sys, json, math
sys.path.insert(0, ".")
import numpy as np
from scipy.stats import binom, binomtest, norm
from sqlalchemy import create_engine, text
from scraper.config import load_settings

rng = np.random.default_rng(42)
REF = "InstantLeague-8035"
LEAGUES = ["InstantLeague-8035", "InstantLeague-8036", "InstantLeague-8037",
           "InstantLeague-8042", "InstantLeague-8043", "InstantLeague-8044",
           "InstantLeague-8056", "InstantLeague-8060", "InstantLeague-8065"]
NEWWIN = "2026-06-12 00:00:00"

eng = create_engine(load_settings().db_url)
corrupted = set(int(k) for k in json.load(
    open("exports/corrupted_events.json", encoding="utf-8"))["events"].keys())

INL = "(" + ",".join("'" + l + "'" for l in LEAGUES) + ")"
SQL_OPEN = """
SELECT e.id, e.competition, e.expected_start, o.captured_at,
       r.score_a, r.score_b, r.ht_score_a, r.ht_score_b, r.goals_json,
       o.odds_home, o.odds_draw, o.odds_away
FROM events e
JOIN results r ON r.event_id = e.id
JOIN (SELECT event_id, MIN(id) AS mid FROM odds_snapshots GROUP BY event_id) m
     ON m.event_id = e.id
JOIN odds_snapshots o ON o.id = m.mid
WHERE e.competition IN %s
""" % INL
SQL_CLOSE = SQL_OPEN.replace("MIN(id) AS mid", "MAX(id) AS mid")

def extract(sql):
    evs = []
    with eng.connect() as c:
        for r in c.execute(text(sql)):
            d = dict(r._mapping)
            if d["id"] in corrupted:
                continue
            sa, sb = d["score_a"], d["score_b"]
            ha, hb = d["ht_score_a"], d["ht_score_b"]
            if sa is None or sb is None:
                continue
            if ha is not None and hb is not None and (ha > sa or hb > sb):
                continue
            gj = d["goals_json"]
            if gj:
                try:
                    gl = json.loads(gj)
                    if isinstance(gl, list) and len(gl) > 0 and len(gl) != sa + sb:
                        continue
                except Exception:
                    pass
            oh, od, oa = d["odds_home"], d["odds_draw"], d["odds_away"]
            if not oh or not od or not oa or oh <= 1 or od <= 1 or oa <= 1:
                continue
            s = 1 / oh + 1 / od + 1 / oa
            evs.append({"id": d["id"], "lg": d["competition"],
                        "ts": str(d["expected_start"]), "cap": str(d["captured_at"]),
                        "sa": sa, "sb": sb, "oh": oh, "od": od, "oa": oa,
                        "ph": (1 / oh) / s, "pa": (1 / oa) / s, "margin": s - 1})
    return evs

events = extract(SQL_OPEN)
print("fresh clean events (opening):", len(events))
from collections import Counter
print(Counter(e["lg"] for e in events))

def fav_bets(evs, lo=1.10, hi=1.20):
    bets = []
    for e in evs:
        if e["oh"] <= e["oa"]:
            o, w, pdev = e["oh"], e["sa"] > e["sb"], e["ph"]
        else:
            o, w, pdev = e["oa"], e["sb"] > e["sa"], e["pa"]
        if lo <= o <= hi:
            bets.append((o, bool(w), pdev, e))
    return bets

def stats(tag, bets, ref_wr_list=(0.856, 0.8674)):
    n = len(bets)
    if n == 0:
        print(f"  {tag:28s} n=0")
        return {"tag": tag, "n": 0}
    wins = sum(1 for _, w, _, _ in bets if w)
    wr = wins / n
    exp = float(np.mean([p for _, _, p, _ in bets]))
    profits = np.array([o * w - 1 for o, w, _, _ in bets])
    roi = float(profits.mean())
    se = float(profits.std() / math.sqrt(n)) if n > 1 else 1.0
    p_be = 2 * norm.sf(abs(roi / se)) if se > 0 else 1.0
    p_cal = binomtest(wins, n, exp).pvalue if n >= 5 else 1.0
    # power: prob of seeing <= wins if the OLD edge were real
    pows = {f"p_if_wr={r}": float(binom.cdf(wins, n, r)) for r in ref_wr_list}
    avg_o = float(np.mean([o for o, _, _, _ in bets]))
    r = {"tag": tag, "n": n, "wins": wins, "wr": round(wr, 4),
         "exp_devig": round(exp, 4), "roi_pct": round(100 * roi, 2),
         "avg_odds": round(avg_o, 4), "breakeven_wr": round(1 / avg_o, 4),
         "p_vs_devig": float(p_cal), "p_roi_vs_0": float(p_be), **{k: round(v, 6) for k, v in pows.items()}}
    print(f"  {tag:28s} n={n:5d} wr={wr:.4f} exp={exp:.4f} roi={100*roi:+.2f}% "
          f"p_cal={p_cal:.3g} p_roi={p_be:.3g} " +
          " ".join(f"{k}:{v:.4g}" for k, v in r.items() if k.startswith("p_if")))
    return r

out = {}
print("\n== 1. ERA REPRODUCTION (fresh extraction, opening odds) ==")
g_old = [e for e in events if e["lg"] == REF and e["ts"] < NEWWIN]
g_rec = [e for e in events if e["lg"] == REF and e["ts"] >= NEWWIN]
g_new = [e for e in events if e["lg"] != REF]
g_newera = g_rec + g_new
out["era"] = [stats("8035-old", fav_bets(g_old)),
              stats("8035-recent", fav_bets(g_rec)),
              stats("POOLED-ALL-NEW", fav_bets(g_new)),
              stats("POOLED-NEWERA", fav_bets(g_newera))]

print("\n== 2. TIMING AUDIT (opening snapshot vs expected_start) ==")
def tparse(s):
    from datetime import datetime
    for f in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s[:26], f)
        except ValueError:
            pass
    return None
dts = []
for e in events:
    a, b = tparse(e["cap"]), tparse(e["ts"])
    if a and b:
        e["lead_min"] = (b - a).total_seconds() / 60.0
        dts.append(e["lead_min"])
dts = np.array(dts)
out["timing"] = {"n": len(dts),
                 "pct_capture_before_start": round(float((dts > 0).mean()) * 100, 2),
                 "lead_minutes_quantiles": {q: round(float(np.percentile(dts, q)), 1)
                                            for q in (1, 5, 25, 50, 75, 95, 99)}}
print(" ", out["timing"])
bets_ne = fav_bets(g_newera)
before = [b for b in bets_ne if b[3].get("lead_min", 0) > 0]
after = [b for b in bets_ne if b[3].get("lead_min", 0) <= 0]
out["timing_split"] = [stats("newera-captured-BEFORE", before),
                       stats("newera-captured-AFTER", after)]

print("\n== 3. PER-LEAGUE (new era) ==")
out["per_league"] = [stats(lg.replace("InstantLeague-", ""), fav_bets([e for e in g_new if e["lg"] == lg]))
                     for lg in LEAGUES if lg != REF]

print("\n== 4. SUB-PERIODS (new era, quartiles of expected_start) ==")
ts_sorted = sorted(e["ts"] for e in g_newera)
qs = [ts_sorted[int(len(ts_sorted) * f)] for f in (0.25, 0.5, 0.75)]
subs = [("Q1", lambda e: e["ts"] < qs[0]), ("Q2", lambda e: qs[0] <= e["ts"] < qs[1]),
        ("Q3", lambda e: qs[1] <= e["ts"] < qs[2]), ("Q4", lambda e: e["ts"] >= qs[2])]
out["subperiods"] = [stats("newera-" + t, fav_bets([e for e in g_newera if f(e)])) for t, f in subs]

print("\n== 5. BOUNDARY SENSITIVITY (new era) ==")
out["bounds"] = [stats(f"newera[{lo}-{hi}]", fav_bets(g_newera, lo, hi))
                 for lo, hi in [(1.05, 1.25), (1.09, 1.21), (1.10, 1.1499),
                                (1.15, 1.20), (1.12, 1.18)]]

print("\n== 6. BOOTSTRAP 10k ROI CI (POOLED-ALL-NEW + NEWERA) ==")
for tag, grp in [("POOLED-ALL-NEW", g_new), ("NEWERA", g_newera)]:
    profits = np.array([o * w - 1 for o, w, _, _ in fav_bets(grp)])
    bs = np.array([rng.choice(profits, len(profits), replace=True).mean()
                   for _ in range(10000)])
    ci = [round(100 * float(np.percentile(bs, q)), 2) for q in (2.5, 97.5)]
    out["bootstrap_" + tag] = {"n": len(profits), "roi_pct": round(100 * float(profits.mean()), 2),
                               "ci95_pct": ci, "p_boot_roi_pos": round(float((bs > 0).mean()), 5)}
    print(f"  {tag}: roi={out['bootstrap_'+tag]['roi_pct']}% CI95={ci} P(roi>0)={out['bootstrap_'+tag]['p_boot_roi_pos']}")

print("\n== 7. CLOSING-ODDS VARIANT (MAX(id) snapshot, new era) ==")
events_c = extract(SQL_CLOSE)
gc_new = [e for e in events_c if e["lg"] != REF]
gc_rec = [e for e in events_c if e["lg"] == REF and e["ts"] >= NEWWIN]
out["closing"] = [stats("close-POOLED-ALL-NEW", fav_bets(gc_new)),
                  stats("close-NEWERA", fav_bets(gc_new + gc_rec)),
                  stats("close-8035-old", fav_bets([e for e in events_c if e["lg"] == REF and e["ts"] < NEWWIN]))]

# draws vs losses among favorite failures (sanity)
fails = [(e["sa"], e["sb"]) for o, w, p, e in fav_bets(g_newera) if not w]
out["newera_fail_split"] = {"n_fail": len(fails),
                            "draws": sum(1 for a, b in fails if a == b),
                            "outright_losses": sum(1 for a, b in fails if a != b)}
print("\nfail split (new era):", out["newera_fail_split"])

with open("exports/wf4_e2adv_refute.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("\nsaved exports/wf4_e2adv_refute.json")
