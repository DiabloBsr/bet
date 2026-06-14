# -*- coding: utf-8 -*-
# ADVERSARIAL VERIFY of finding "conditional round N -> N+1 strategies are DEAD"
# (script under audit: scripts/_wf4_roundstruct_8_power_cond.py, section E)
# 1) exact reproduction of section E from the same pickle (deterministic, no RNG)
# 2) test-period baseline (the original compared cond-test vs FULL-sample baseline)
# 3) cluster bootstrap by round for the headline cell ge3favlost_dog_test
# 4) alternative temporal splits 50/60/70/80
# 5) sub-period (quarters) stability of the conditional dog ROI
# 6) DB spot-check: pkl odds == true opening odds (MIN snapshot id), corrupted excluded
# 7) feasibility: when is round N result known vs round N+1 kickoff
# Output -> exports/wf4_refute_cond.json   (READ-ONLY DB)
import sys, json, pickle
sys.path.insert(0, ".")
import numpy as np
from collections import defaultdict
from datetime import datetime

rng = np.random.default_rng(123)
out = {}

recs = pickle.load(open("scripts/_wf4_roundstruct_data.pkl", "rb"))
for r in recs:
    imp = np.array([1 / r["oh"], 1 / r["od"], 1 / r["oa"]])
    fair = imp / imp.sum()
    res = 0 if r["sa"] > r["sb"] else (1 if r["sa"] == r["sb"] else 2)
    fav = 0 if r["oh"] <= r["oa"] else 2
    r["res"] = res
    r["fav"] = fav
    r["p_fav"] = fair[fav]
    r["x_fav"] = 1.0 if res == fav else 0.0
    r["o_fav"] = r["oh"] if fav == 0 else r["oa"]
    r["o_dog"] = r["oa"] if fav == 0 else r["oh"]
    r["x_dog"] = 1.0 if (res != 1 and res != fav) else 0.0

L35 = [r for r in recs if r["comp"] == "InstantLeague-8035"]

def ts(s):
    return datetime.fromisoformat(s).timestamp()

# est format sanity (lexicographic ordering == chronological?)
ests_all = sorted(set(r["est"] for r in L35))
mono = all(ts(ests_all[i]) <= ts(ests_all[i + 1]) for i in range(len(ests_all) - 1))
out["est_lexicographic_ok"] = bool(mono)
out["est_examples"] = ests_all[:3]

g35 = defaultdict(list)
for r in L35:
    g35[r["est"]].append(r)
ests = sorted(g35.keys())

pairs = []
for i in range(len(ests) - 1):
    if ts(ests[i + 1]) - ts(ests[i]) <= 600 and len(g35[ests[i]]) >= 5 and len(g35[ests[i + 1]]) >= 5:
        pairs.append((ests[i], ests[i + 1]))
out["n_pairs"] = len(pairs)

gaps = [ts(ests[i + 1]) - ts(ests[i]) for i in range(len(ests) - 1)]
out["round_gap_seconds"] = dict(median=float(np.median(gaps)),
                                p10=float(np.percentile(gaps, 10)),
                                p90=float(np.percentile(gaps, 90)))

# ---------- 1) exact reproduction of section E ----------
split_t = sorted(r["est"] for r in L35)[int(len(L35) * 0.7)]
out["split_t_70"] = split_t

def run_cond(cond, side, flt):
    stakes = wins = 0
    pnl = 0.0
    odds_sum = 0.0
    bets_by_round = defaultdict(list)
    for (e1, e2) in pairs:
        if not flt(e2):
            continue
        if cond(g35[e1]):
            for r in g35[e2]:
                o = r["o_fav"] if side == "fav" else r["o_dog"]
                x = r["x_fav"] if side == "fav" else r["x_dog"]
                stakes += 1
                odds_sum += o
                b = (o - 1) if x > 0.5 else -1.0
                pnl += b
                wins += 1 if x > 0.5 else 0
                bets_by_round[e2].append(b)
    if not stakes:
        return None, bets_by_round
    return dict(n=stakes, wr=wins / stakes, roi=pnl / stakes, avg_odds=odds_sum / stakes), bets_by_round

CONDS = {"ge3favlost": lambda v: sum(1 - r["x_fav"] for r in v) >= 3,
         "ge5favlost": lambda v: sum(1 - r["x_fav"] for r in v) >= 5,
         "le1favlost": lambda v: sum(1 - r["x_fav"] for r in v) <= 1}

repro = {}
for cn, cond in CONDS.items():
    for side in ["fav", "dog"]:
        for sn, flt in [("full", lambda e: True), ("test", lambda e: e > split_t)]:
            res, _ = run_cond(cond, side, flt)
            if res:
                repro[f"{cn}_{side}_{sn}"] = res
out["repro_E"] = repro
claim = dict(n=1730, wr=0.229, roi=0.0265, avg_odds=5.35)
cell = repro.get("ge3favlost_dog_test", {})
out["repro_matches_claim"] = bool(cell and cell["n"] == claim["n"]
                                  and abs(cell["roi"] - claim["roi"]) < 0.001)
print("REPRO ge3favlost_dog_test:", cell)

# ---------- 2) baselines: full / test / pair-universe-test ----------
def baseline(rl, side):
    n = 0; pnl = 0.0; odds_sum = 0.0; w = 0
    for r in rl:
        o = r["o_fav"] if side == "fav" else r["o_dog"]
        x = r["x_fav"] if side == "fav" else r["x_dog"]
        n += 1; odds_sum += o
        pnl += (o - 1) if x > 0.5 else -1.0
        w += 1 if x > 0.5 else 0
    return dict(n=n, wr=w / n, roi=pnl / n, avg_odds=odds_sum / n)

test_recs = [r for r in L35 if r["est"] > split_t]
pair_e2_test = set(e2 for (_, e2) in pairs if e2 > split_t)
universe_test = [r for r in L35 if r["est"] in pair_e2_test]
out["baseline"] = {
    "dog_full": baseline(L35, "dog"),
    "dog_test": baseline(test_recs, "dog"),
    "dog_pair_universe_test": baseline(universe_test, "dog"),
    "fav_full": baseline(L35, "fav"),
    "fav_test": baseline(test_recs, "fav"),
}
for k, v in out["baseline"].items():
    print(f"BASELINE {k}: n={v['n']} roi={v['roi']*100:+.2f}% wr={v['wr']:.3f}")

# ---------- 3) cluster bootstrap by round, headline cell ----------
_, bets_by_round = run_cond(CONDS["ge3favlost"], "dog", lambda e: e > split_t)
rounds = list(bets_by_round.values())
nR = len(rounds)
out["n_trigger_rounds_test"] = nR
flat = np.concatenate([np.array(b) for b in rounds])
B = 10000
boot = np.empty(B)
for i in range(B):
    idx = rng.integers(0, nR, nR)
    sel = np.concatenate([rounds[j] for j in idx])
    boot[i] = sel.mean()
roi_obs = flat.mean()
base_test_roi = out["baseline"]["dog_pair_universe_test"]["roi"]
out["cluster_bootstrap_ge3dog_test"] = dict(
    roi=float(roi_obs),
    se=float(boot.std()),
    ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
    p_roi_le_0=float(np.mean(boot <= 0)),
    p_roi_le_universe_baseline=float(np.mean(boot <= base_test_roi)),
)
print("CLUSTER BOOT ge3dog_test: roi={:+.2f}% se={:.2f}% ci95=[{:+.2f}%,{:+.2f}%] P(<=0)={:.3f}".format(
    roi_obs * 100, boot.std() * 100,
    np.percentile(boot, 2.5) * 100, np.percentile(boot, 97.5) * 100,
    np.mean(boot <= 0)))

# null check: same conditioning but on SHUFFLED round labels (kills any real signal)
perm_roi = np.empty(2000)
e1list = [e1 for (e1, _) in pairs]
e2list = [e2 for (_, e2) in pairs]
for i in range(2000):
    e1p = rng.permutation(e1list)
    pnl = 0.0; n = 0
    for e1, e2 in zip(e1p, e2list):
        if e2 <= split_t:
            continue
        if CONDS["ge3favlost"](g35[e1]):
            for r in g35[e2]:
                pnl += (r["o_dog"] - 1) if r["x_dog"] > 0.5 else -1.0
                n += 1
    perm_roi[i] = pnl / n if n else 0.0
out["perm_null_ge3dog_test"] = dict(
    mean=float(perm_roi.mean()), sd=float(perm_roi.std()),
    p_two_sided=float((1 + np.sum(np.abs(perm_roi - perm_roi.mean())
                                  >= abs(roi_obs - perm_roi.mean()))) / (len(perm_roi) + 1)))
print("PERM NULL: mean={:+.2f}% sd={:.2f}% p={:.3f}".format(
    perm_roi.mean() * 100, perm_roi.std() * 100, out["perm_null_ge3dog_test"]["p_two_sided"]))

# ---------- 4) alternative temporal splits ----------
alt = {}
for frac in [0.5, 0.6, 0.7, 0.8]:
    st = sorted(r["est"] for r in L35)[int(len(L35) * frac)]
    res, _ = run_cond(CONDS["ge3favlost"], "dog", lambda e: e > st)
    bl = baseline([r for r in L35 if r["est"] > st], "dog")
    alt[f"split_{int(frac*100)}"] = dict(cond=res, baseline_dog=bl,
                                         edge=res["roi"] - bl["roi"] if res else None)
    print(f"SPLIT {frac}: cond roi={res['roi']*100:+.2f}% (n={res['n']}) vs base {bl['roi']*100:+.2f}% (n={bl['n']})")
out["alt_splits"] = alt

# ---------- 5) quarters of the full sample ----------
qs = {}
srt = sorted(r["est"] for r in L35)
bounds = [srt[0], srt[len(srt)//4], srt[len(srt)//2], srt[3*len(srt)//4], srt[-1]]
for q in range(4):
    lo, hi = bounds[q], bounds[q + 1]
    res, _ = run_cond(CONDS["ge3favlost"], "dog",
                      lambda e, lo=lo, hi=hi: (lo <= e <= hi) if q == 3 else (lo <= e < hi))
    bl = baseline([r for r in L35 if (bounds[q] <= r["est"] <= bounds[q+1] if q == 3
                                      else bounds[q] <= r["est"] < bounds[q+1])], "dog")
    qs[f"Q{q+1}"] = dict(cond=res, baseline_dog_roi=bl["roi"],
                         edge=(res["roi"] - bl["roi"]) if res else None)
    if res:
        print(f"Q{q+1}: cond roi={res['roi']*100:+.2f}% (n={res['n']}) vs base {bl['roi']*100:+.2f}%")
out["quarters"] = qs

# ---------- 6) DB spot-check: opening odds + corruption ----------
from scraper.config import load_settings
from sqlalchemy import create_engine, text
eng = create_engine(load_settings().db_url)
with open("exports/corrupted_events.json", "r", encoding="utf-8") as f:
    CORRUPT = set(int(k) for k in json.load(f)["events"].keys())
n_corrupt_in_pkl = sum(1 for r in recs if r["id"] in CORRUPT)
out["corrupt_ids_in_pkl"] = n_corrupt_in_pkl

sample = [L35[i] for i in rng.choice(len(L35), 10, replace=False)]
mismatch = []
with eng.connect() as conn:
    for r in sample:
        row = conn.execute(text(
            "SELECT odds_home, odds_draw, odds_away FROM odds_snapshots "
            "WHERE event_id=:eid ORDER BY id ASC LIMIT 1"), dict(eid=r["id"])).fetchone()
        if row is None or abs(float(row[0]) - r["oh"]) > 1e-9 or \
           abs(float(row[1]) - r["od"]) > 1e-9 or abs(float(row[2]) - r["oa"]) > 1e-9:
            mismatch.append(r["id"])
    # 7) feasibility: result availability of round N vs kickoff of N+1
    rows = conn.execute(text("""
        SELECT ev.expected_start, r.finished_at FROM events ev
        JOIN results r ON r.event_id = ev.id
        WHERE ev.competition='InstantLeague-8035'
        ORDER BY ev.id DESC LIMIT 400""")).fetchall()
lag = []
for est, fin in rows:
    try:
        lag.append(datetime.fromisoformat(fin).timestamp() - ts(est))
    except Exception:
        pass
out["opening_odds_mismatch_ids"] = mismatch
if lag:
    out["result_lag_after_kickoff_sec"] = dict(
        n=len(lag), median=float(np.median(lag)),
        p10=float(np.percentile(lag, 10)), p90=float(np.percentile(lag, 90)))
    print("RESULT LAG after own kickoff (sec): median={:.0f} p10={:.0f} p90={:.0f}".format(
        np.median(lag), np.percentile(lag, 10), np.percentile(lag, 90)))
print("opening-odds mismatches:", mismatch, "| corrupt ids in pkl:", n_corrupt_in_pkl)

with open("exports/wf4_refute_cond.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, default=str)
print("done")
