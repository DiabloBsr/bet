# -*- coding: utf-8 -*-
"""ADVERSARIAL refutation attempt of WF4 finding: FTTS '1' home fav <=1.50, championships only.

Fully independent re-implementation from the DB (no reuse of _wf4_cl_*.json caches).
Attack vectors:
  A. fresh pull (scraper live since 19:00 cache -> forward holdout on new event ids)
  B. settlement cross-validation of first-goal team vs HT scores
  C. duplicate fixtures audit inside the bet set
  D. strict pre-kickoff (fetched_at < expected_start) subset
  E. sub-period splits + per-league + cluster bootstrap by (league, expected_start)
  F. pooled champ+cup (cost of the post-hoc cup carve-out)
  G. survivorship sensitivity: settle guard-excluded events as LOSSES (worst case)
READ-ONLY on DB.
"""
import sys, json, math
sys.path.insert(0, ".")
import numpy as np
from datetime import datetime
from scipy.stats import norm
from sqlalchemy import create_engine, text
from scraper.config import load_settings

CHAMP = {"InstantLeague-8036", "InstantLeague-8037", "InstantLeague-8042",
         "InstantLeague-8043", "InstantLeague-8044"}
CUP = {"InstantLeague-8056", "InstantLeague-8060", "InstantLeague-8065"}
REF = "InstantLeague-8035"
ALL = sorted(CHAMP | CUP | {REF})
NEWWIN = "2026-06-12 00:00:00"

eng = create_engine(load_settings().db_url)
out = {}

with open("exports/corrupted_events.json", "r", encoding="utf-8") as f:
    corrupted = set(int(k) for k in json.load(f)["events"].keys())

# cache ids (to isolate the post-cache forward sample)
with open("exports/_wf4_cl_events.json", "r", encoding="utf-8") as f:
    cache_ids = set(e["id"] for e in json.load(f))
print(f"cache (19:00) had {len(cache_ids)} events")

SQL = """
SELECT e.id, e.competition, e.expected_start, e.team_a, e.team_b,
       r.score_a, r.score_b, r.ht_score_a, r.ht_score_b, r.goals_json,
       o.odds_home, o.odds_draw, o.odds_away, o.extra_markets, o.captured_at
FROM events e
JOIN results r ON r.event_id = e.id
JOIN (SELECT event_id, MIN(id) AS mid FROM odds_snapshots GROUP BY event_id) m
     ON m.event_id = e.id
JOIN odds_snapshots o ON o.id = m.mid
WHERE e.competition IN ({})
""".format(",".join("'" + l + "'" for l in ALL))

rows = []
with eng.connect() as c:
    for r in c.execute(text(SQL)):
        rows.append(dict(r._mapping))
print("raw rows (fresh pull):", len(rows))

# sanity: MIN(id) == earliest fetched_at? sample check on events with >1 snapshot
with eng.connect() as c:
    bad_order = c.execute(text("""
        SELECT COUNT(*) FROM (
          SELECT event_id, MIN(id) mid FROM odds_snapshots GROUP BY event_id HAVING COUNT(*)>1
        ) m
        JOIN odds_snapshots o1 ON o1.id = m.mid
        WHERE EXISTS (SELECT 1 FROM odds_snapshots o2
                      WHERE o2.event_id = m.event_id AND o2.id > m.mid
                        AND o2.captured_at < o1.captured_at)
    """)).scalar()
print("events where a later snapshot id has EARLIER fetched_at:", bad_order)
out["minid_vs_fetchedat_violations"] = int(bad_order)


def parse_ts(s):
    s = str(s)
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None


def first_team_and_check(gj_raw, sa, sb, ha, hb):
    """Return (first_team, ht_contradiction_flag)."""
    if sa + sb == 0:
        return "None", False
    if not gj_raw:
        return None, False
    try:
        gl = json.loads(gj_raw)
    except Exception:
        return None, False
    if not isinstance(gl, list) or len(gl) == 0:
        return None, False
    gl_sorted = sorted(gl, key=lambda g: (g["minute"], g["homeScore"] + g["awayScore"]))
    first = gl_sorted[0]
    ft = first["team"]
    # cross-check vs HT when HT pins down the first scorer
    contra = False
    if ha is not None and hb is not None:
        gh45 = sum(1 for g in gl if g["minute"] <= 45 and g["team"] == "Home")
        ga45 = sum(1 for g in gl if g["minute"] <= 45 and g["team"] == "Away")
        if (gh45, ga45) != (ha, hb):
            contra = True  # goals_json incoherent with recorded HT
        elif ha >= 1 and hb == 0 and first["minute"] <= 45 and ft != "Home":
            contra = True
        elif hb >= 1 and ha == 0 and first["minute"] <= 45 and ft != "Away":
            contra = True
    return ft, contra


events = []
guard_excl = {"corrupt": 0, "ht_gt_ft": 0, "gj_mismatch": 0, "no_odds": 0}
gj_mismatch_events = []   # for survivorship worst-case
for r in rows:
    if r["id"] in corrupted:
        guard_excl["corrupt"] += 1
        continue
    sa, sb = r["score_a"], r["score_b"]
    ha, hb = r["ht_score_a"], r["ht_score_b"]
    if sa is None or sb is None:
        continue
    oh = r["odds_home"]
    if not oh or not r["odds_draw"] or not r["odds_away"] or oh <= 1:
        guard_excl["no_odds"] += 1
        continue
    bad = False
    if ha is not None and hb is not None and (ha > sa or hb > sb):
        guard_excl["ht_gt_ft"] += 1
        bad = True
    gj = r["goals_json"]
    mism = False
    if gj and not bad:
        try:
            gl = json.loads(gj)
            if isinstance(gl, list) and len(gl) > 0 and len(gl) != sa + sb:
                guard_excl["gj_mismatch"] += 1
                mism = True
        except Exception:
            pass
    ev = {"id": r["id"], "league": r["competition"], "ts": str(r["expected_start"]),
          "team_a": r["team_a"], "team_b": r["team_b"],
          "sa": sa, "sb": sb, "ha": ha, "hb": hb, "oh": oh,
          "gj": gj, "xm": r["extra_markets"], "fetched": str(r["captured_at"])}
    if bad:
        continue
    if mism:
        gj_mismatch_events.append(ev)
        continue
    events.append(ev)
print("clean events:", len(events), "| excl:", guard_excl)

# ---- build bets ----
def make_bet(e):
    """Return (odds, won, e) or None if not a qualifying/settleable bet."""
    if e["oh"] > 1.5:
        return None
    try:
        f2 = json.loads(e["xm"]).get("FTTS") or {}
    except Exception:
        return None
    o = f2.get("1")
    if not o or o <= 1 or o >= 99.5:
        return None
    ft, contra = first_team_and_check(e["gj"], e["sa"], e["sb"], e["ha"], e["hb"])
    if ft is None:
        return None
    return (o, ft == "Home", contra, e)


def stats(bets, label, nt=[0]):
    if len(bets) == 0:
        print(f"  {label}: n=0")
        return None
    n = len(bets)
    wr = sum(1 for b in bets if b[1]) / n
    prof = np.array([b[0] * b[1] - 1 for b in bets])
    roi = float(prof.mean())
    se = prof.std() / math.sqrt(n) if n > 1 else float("inf")
    pv = 2 * norm.sf(abs(roi / se)) if se > 0 else 1.0
    r = {"label": label, "n": n, "wr": round(wr, 4), "roi_pct": round(100 * roi, 2),
         "avg_odds": round(float(np.mean([b[0] for b in bets])), 3), "p_two_sided": float(pv)}
    print(f"  {label:55s} n={n:5d} wr={wr:.4f} roi={100*roi:+6.2f}% odds={r['avg_odds']:.3f} p={pv:.4g}")
    return r

groups = {
    "champs-new (5 ligues)": lambda e: e["league"] in CHAMP,
    "8035-recent (>=06-12)": lambda e: e["league"] == REF and e["ts"] >= NEWWIN,
    "8035-old (<06-12)": lambda e: e["league"] == REF and e["ts"] < NEWWIN,
    "NEW-ERA champs pooled (claim)": lambda e: e["league"] in CHAMP or (e["league"] == REF and e["ts"] >= NEWWIN),
    "cups pooled": lambda e: e["league"] in CUP,
    "EVERYTHING new-era (champs+cups, no carve-out)": lambda e: (e["league"] in CHAMP or e["league"] in CUP
                                                                 or (e["league"] == REF and e["ts"] >= NEWWIN)),
}

all_bets = []
for e in events:
    b = make_bet(e)
    if b:
        all_bets.append(b)
print(f"\ntotal qualifying bets (all leagues): {len(all_bets)}")
n_contra = sum(1 for b in all_bets if b[2])
print(f"settlement HT-contradictions inside bet set: {n_contra}/{len(all_bets)}")
out["settlement_ht_contradictions"] = {"n_contra": n_contra, "n_bets": len(all_bets)}

print("\n== A. group ROI (fresh pull, independent code) ==")
res = {}
for g, sel in groups.items():
    res[g] = stats([b for b in all_bets if sel(b[3])], g)
out["groups"] = res

print("\n== A2. per-league ==")
perl = {}
for l in ALL:
    perl[l] = stats([b for b in all_bets if b[3]["league"] == l], l)
out["per_league"] = perl

# ---- B. forward holdout: events NOT in the 19:00 cache (scraped after) ----
print("\n== B. forward holdout (event not in 19:00 cache) ==")
fwd_champ = [b for b in all_bets if b[3]["id"] not in cache_ids
             and (b[3]["league"] in CHAMP or (b[3]["league"] == REF and b[3]["ts"] >= NEWWIN))]
out["forward_holdout_champs"] = stats(fwd_champ, "post-cache champs (pure forward)")
fwd_cup = [b for b in all_bets if b[3]["id"] not in cache_ids and b[3]["league"] in CUP]
out["forward_holdout_cups"] = stats(fwd_cup, "post-cache cups")

# ---- C. duplicates inside the claim bet set ----
print("\n== C. duplicate fixtures audit (claim set: new-era champs) ==")
claim = [b for b in all_bets if b[3]["league"] in CHAMP
         or (b[3]["league"] == REF and b[3]["ts"] >= NEWWIN)]
seen = {}
dups = 0
dup_same_outcome = 0
for b in claim:
    e = b[3]
    k = (e["league"], e["team_a"], e["team_b"])
    t = parse_ts(e["ts"])
    if k in seen:
        for (t2, w2) in seen[k]:
            if t and t2 and abs((t - t2).total_seconds()) < 1800:
                dups += 1
                if w2 == b[1]:
                    dup_same_outcome += 1
                break
    seen.setdefault(k, []).append((t, b[1]))
print(f"  bets sharing (league,teams) within 30min of another bet: {dups} (same outcome: {dup_same_outcome})")
out["dups_in_claim_set"] = {"n_dup_pairs_30min": dups, "same_outcome": dup_same_outcome}

# ---- D. strict pre-kickoff ----
print("\n== D. strict pre-kickoff opening snapshot (fetched_at < expected_start) ==")
strict = []
lag_counts = {}
for b in claim:
    e = b[3]
    t_start, t_fetch = parse_ts(e["ts"]), parse_ts(e["fetched"])
    if t_start is None or t_fetch is None:
        continue
    lag = (t_fetch - t_start).total_seconds() / 60.0
    lag_counts[int(math.floor(lag))] = lag_counts.get(int(math.floor(lag)), 0) + 1
    if t_fetch < t_start:
        strict.append(b)
out["lag_minutes_hist_claimset"] = dict(sorted(lag_counts.items())[:25])
print("  lag(min) hist:", dict(sorted(lag_counts.items())[:15]))
out["strict_prekickoff"] = stats(strict, "claim set, strictly pre-kickoff")
# lag <= +1 min (odds fetched at most 1 min after scheduled start)
near = [b for b in claim if parse_ts(b[3]["fetched"]) and parse_ts(b[3]["ts"])
        and (parse_ts(b[3]["fetched"]) - parse_ts(b[3]["ts"])).total_seconds() <= 60]
out["lag_le_1min"] = stats(near, "claim set, lag <= +1 min")

# ---- E. sub-periods + bootstrap ----
print("\n== E. sub-period splits (claim set, by expected_start) ==")
claim_sorted = sorted(claim, key=lambda b: b[3]["ts"])
k = len(claim_sorted) // 3
out["tercile_1"] = stats(claim_sorted[:k], "tercile 1 (earliest)")
out["tercile_2"] = stats(claim_sorted[k:2 * k], "tercile 2")
out["tercile_3"] = stats(claim_sorted[2 * k:], "tercile 3 (latest)")
half = len(claim_sorted) // 2
out["half_1"] = stats(claim_sorted[:half], "half 1")
out["half_2"] = stats(claim_sorted[half:], "half 2")

rng = np.random.default_rng(42)
prof = np.array([b[0] * b[1] - 1 for b in claim])
boot = np.array([rng.choice(prof, size=len(prof), replace=True).mean() for _ in range(20000)])
ci = np.percentile(boot, [2.5, 5, 50, 95, 97.5])
print(f"  iid bootstrap ROI CI claim set: 2.5%={100*ci[0]:+.2f}% 5%={100*ci[1]:+.2f}% "
      f"med={100*ci[2]:+.2f}% 95%={100*ci[3]:+.2f}% 97.5%={100*ci[4]:+.2f}%  P(ROI<=0)={float((boot<=0).mean()):.5f}")
out["bootstrap_iid"] = {"ci": [round(100 * x, 2) for x in ci], "p_le_0": float((boot <= 0).mean())}

# cluster bootstrap by (league, expected_start) = simulation round
clusters = {}
for b in claim:
    clusters.setdefault((b[3]["league"], b[3]["ts"]), []).append(b[0] * b[1] - 1)
keys = list(clusters.values())
cboot = []
for _ in range(8000):
    idx = rng.integers(0, len(keys), size=len(keys))
    s = [x for i in idx for x in keys[i]]
    cboot.append(float(np.mean(s)))
cboot = np.array(cboot)
cci = np.percentile(cboot, [2.5, 5, 50, 95, 97.5])
print(f"  cluster bootstrap (by league+start, {len(keys)} clusters): "
      f"2.5%={100*cci[0]:+.2f}% med={100*cci[2]:+.2f}% 97.5%={100*cci[4]:+.2f}% P(ROI<=0)={float((cboot<=0).mean()):.5f}")
out["bootstrap_cluster"] = {"n_clusters": len(keys), "ci": [round(100 * x, 2) for x in cci],
                            "p_le_0": float((cboot <= 0).mean())}

# ---- G. survivorship worst case ----
print("\n== G. survivorship worst-case ==")
# 1) guard-excluded (gj mismatch) qualifying events settled as LOSSES
wc = list(claim)
n_added_loss = 0
for e in gj_mismatch_events:
    if e["league"] in CHAMP or (e["league"] == REF and e["ts"] >= NEWWIN):
        if e["oh"] <= 1.5:
            try:
                f2 = json.loads(e["xm"]).get("FTTS") or {}
                o = f2.get("1")
                if o and 1 < o < 99.5:
                    wc.append((o, False, False, e))
                    n_added_loss += 1
            except Exception:
                pass
# 2) qualifying but unsettleable (no goals_json, total>=1) as LOSSES
n_unsettle = 0
for e in events:
    if not (e["league"] in CHAMP or (e["league"] == REF and e["ts"] >= NEWWIN)):
        continue
    if e["oh"] > 1.5:
        continue
    try:
        f2 = json.loads(e["xm"]).get("FTTS") or {}
        o = f2.get("1")
        if not o or o <= 1 or o >= 99.5:
            continue
    except Exception:
        continue
    ft, _ = first_team_and_check(e["gj"], e["sa"], e["sb"], e["ha"], e["hb"])
    if ft is None:
        wc.append((o, False, False, e))
        n_unsettle += 1
print(f"  added as losses: gj-mismatch={n_added_loss}, unsettleable={n_unsettle}")
out["worst_case"] = stats(wc, "claim set + all ambiguous settled as LOSS")
out["worst_case_added"] = {"gj_mismatch_losses": n_added_loss, "unsettleable_losses": n_unsettle}

with open("exports/wf4_adv_ftts_refute.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1, default=str)
print("\nsaved exports/wf4_adv_ftts_refute.json")
