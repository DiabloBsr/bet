# ADVERSARIAL VERIFY of finding "edge #3 Total de buts = 1 est MORT"
# Independent re-extraction from DB (no pkl), semantics + look-ahead + dup checks,
# sub-periods, alternative splits, bootstrap, rolling windows vs historical +3.5% (n=1758).
# READ-ONLY DB. Output -> exports/wf4_totalsrefute2.json
import sys, json, math
sys.path.insert(0, ".")
import numpy as np
from scipy.stats import norm
from scraper.config import load_settings
from sqlalchemy import create_engine, text

e = create_engine(load_settings().db_url)
rng = np.random.default_rng(42)

with open("exports/corrupted_events.json", "r", encoding="utf-8") as f:
    CORRUPT = set(int(k) for k in json.load(f)["events"].keys())

OUT = {}

with e.connect() as conn:
    # --- check duplicate results / snapshots ---
    dup_res = conn.execute(text(
        "SELECT COUNT(*) FROM (SELECT event_id FROM results GROUP BY event_id HAVING COUNT(*)>1)"
    )).scalar()
    OUT["dup_results_events"] = int(dup_res)

    rows = conn.execute(text("""
        SELECT ev.id, ev.competition, ev.expected_start,
               r.score_a, r.score_b,
               o.odds_home, o.odds_draw, o.odds_away, o.extra_markets, o.captured_at
        FROM events ev
        JOIN results r ON r.event_id = ev.id
        JOIN odds_snapshots o ON o.event_id = ev.id
        WHERE o.id = (SELECT MIN(o2.id) FROM odds_snapshots o2 WHERE o2.event_id = ev.id)
          AND ev.competition = 'InstantLeague-8035'
    """)).fetchall()
print(f"raw 8035 rows: {len(rows)}")

# dedupe guard (in case of dup results rows)
seen = set()
recs = []
n_corrupt = n_dupe = n_lookahead = 0
for (eid, comp, start, sa, sb, oh, od, oa, xm, fts) in rows:
    if eid in CORRUPT:
        n_corrupt += 1
        continue
    if eid in seen:
        n_dupe += 1
        continue
    seen.add(eid)
    if sa is None or sb is None:
        continue
    try:
        xmd = json.loads(xm) if xm else {}
    except Exception:
        xmd = {}
    totx = xmd.get("Total de buts") or {}
    recs.append(dict(eid=eid, start=str(start), fetched=str(fts), sa=sa, sb=sb,
                     tot=sa + sb, oh=oh, od=od, oa=oa, totx=totx))

# look-ahead check: opening snapshot fetched AFTER expected_start?
la_count = sum(1 for r in recs if r["fetched"] and r["start"] and r["fetched"] > r["start"])
OUT["n_8035_events"] = len(recs)
OUT["n_corrupt_excluded"] = n_corrupt
OUT["n_dupe_rows"] = n_dupe
OUT["opening_snapshot_after_kickoff"] = int(la_count)
print(f"clean events: {len(recs)} | corrupt excl: {n_corrupt} | dup rows: {n_dupe} | opening-after-kickoff: {la_count}")

# --- market semantics: keys of "Total de buts", margin, monotonicity ---
from collections import Counter
keycnt = Counter()
margins = []
for r in recs:
    if r["totx"]:
        keycnt.update(r["totx"].keys())
        try:
            margins.append(sum(1.0 / float(v) for v in r["totx"].values() if float(v) > 1))
        except Exception:
            pass
OUT["totx_key_counts"] = dict(keycnt)
OUT["totx_margin_mean"] = float(np.mean(margins)) if margins else None
print("totx keys:", dict(keycnt))
print(f"margin mean: {np.mean(margins):.4f}" if margins else "no margins")

# semantic sanity: implied P(key) should be unimodal peaking near 2-3 goals
imp_by_key = {}
for k in sorted(keycnt, key=lambda x: (len(x), x)):
    vals = [1.0 / float(r["totx"][k]) for r in recs if r["totx"].get(k) and float(r["totx"][k]) > 1]
    if vals:
        imp_by_key[k] = round(float(np.mean(vals)), 4)
OUT["totx_implied_by_key"] = imp_by_key
print("implied by key:", imp_by_key)

# real frequency of each total vs implied (settlement-free cross-check)
freq = Counter(min(r["tot"], 9) for r in recs)
tot_freq = {str(k): round(freq.get(k, 0) / len(recs), 4) for k in range(0, 8)}
OUT["real_total_freq"] = tot_freq
print("real total freq:", tot_freq)

# --- the bet: Total de buts = 1, opening odds ---
def roi_stats(bets):
    if not bets:
        return dict(n=0)
    r = np.array([(o - 1) if w else -1.0 for w, o in bets])
    n = len(r); roi = float(r.mean())
    wr = float(np.mean([w for w, _ in bets])); ao = float(np.mean([o for _, o in bets]))
    se = r.std(ddof=1) / math.sqrt(n)
    p = 2 * (1 - norm.cdf(abs(roi) / se)) if se > 0 else 1.0
    return dict(n=n, wr=round(wr, 4), roi_pct=round(roi * 100, 2), odds=round(ao, 3), p=round(float(p), 5))

def getbets(sub):
    out = []
    for r in sub:
        v = r["totx"].get("1")
        if v is None:
            continue
        try:
            o = float(v)
        except Exception:
            continue
        if 1 < o < 100:
            out.append(((r["tot"] == 1), o, r["start"]))
    return out

bets_all = getbets(recs)
bets_all.sort(key=lambda b: b[2])
OUT["full_8035"] = roi_stats([(w, o) for w, o, _ in bets_all])
print("FULL 8035 (independent re-extract):", OUT["full_8035"])

# ROI decomposition: naive wr*odds-1 vs actual (covariance win/odds)
wr = np.mean([w for w, o, _ in bets_all]); ao = np.mean([o for w, o, _ in bets_all])
won_odds = np.mean([o for w, o, _ in bets_all if w])
OUT["decomp"] = dict(naive_roi_pct=round((wr * ao - 1) * 100, 2),
                     avg_odds_all=round(float(ao), 3), avg_odds_won=round(float(won_odds), 3))
print("decomp:", OUT["decomp"])

# --- alternative temporal splits ---
n = len(bets_all)
OUT["alt_splits"] = {}
for frac in (0.5, 0.6, 0.8):
    cut = int(frac * n)
    OUT["alt_splits"][f"test_last_{int((1-frac)*100)}pct"] = roi_stats([(w, o) for w, o, _ in bets_all[cut:]])
print("alt splits:", OUT["alt_splits"])

# --- sub-periods: quartiles by time ---
OUT["quartiles"] = {}
for i in range(4):
    chunk = bets_all[i * n // 4:(i + 1) * n // 4]
    OUT["quartiles"][f"Q{i+1}"] = roi_stats([(w, o) for w, o, _ in chunk])
print("quartiles:", OUT["quartiles"])

# --- odds buckets: any positive pocket? ---
OUT["odds_buckets"] = {}
edges = [1, 6, 7, 8, 9, 10, 100]
for lo, hi in zip(edges[:-1], edges[1:]):
    sub = [(w, o) for w, o, _ in bets_all if lo <= o < hi]
    OUT["odds_buckets"][f"[{lo},{hi})"] = roi_stats(sub)
print("odds buckets:", OUT["odds_buckets"])

# --- bootstrap CI (10k) on full ROI ---
r = np.array([(o - 1) if w else -1.0 for w, o, _ in bets_all])
boot = np.array([rng.choice(r, size=len(r), replace=True).mean() for _ in range(10000)])
OUT["bootstrap"] = dict(roi_pct=round(float(r.mean()) * 100, 2),
                        ci2_5=round(float(np.percentile(boot, 2.5)) * 100, 2),
                        ci97_5=round(float(np.percentile(boot, 97.5)) * 100, 2),
                        p_roi_pos=round(float((boot > 0).mean()), 5))
print("bootstrap:", OUT["bootstrap"])

# --- rolling windows of n=1758 (size of the historical OOS claim) ---
W = 1758
rois = []
if len(r) > W:
    cs = np.concatenate([[0.0], np.cumsum(r)])
    rois = (cs[W:] - cs[:-W]) / W
    OUT["rolling_1758"] = dict(n_windows=len(rois),
                               max_roi_pct=round(float(rois.max()) * 100, 2),
                               min_roi_pct=round(float(rois.min()) * 100, 2),
                               pct_windows_ge_3_5=round(float((rois >= 0.035).mean()) * 100, 2),
                               pct_windows_ge_0=round(float((rois >= 0).mean()) * 100, 2))
    print("rolling 1758:", OUT["rolling_1758"])

# --- pooled 9 leagues quick re-check ---
LEAGUES = ["InstantLeague-8035", "InstantLeague-8036", "InstantLeague-8037",
           "InstantLeague-8042", "InstantLeague-8043", "InstantLeague-8044",
           "InstantLeague-8056", "InstantLeague-8060", "InstantLeague-8065"]
with e.connect() as conn:
    rows9 = conn.execute(text("""
        SELECT ev.id, ev.competition, r.score_a, r.score_b, o.extra_markets
        FROM events ev
        JOIN results r ON r.event_id = ev.id
        JOIN odds_snapshots o ON o.event_id = ev.id
        WHERE o.id = (SELECT MIN(o2.id) FROM odds_snapshots o2 WHERE o2.event_id = ev.id)
          AND ev.competition IN ('%s')
    """ % "','".join(LEAGUES))).fetchall()
seen9 = set(); bets9 = []; per_lg = {}
for (eid, comp, sa, sb, xm) in rows9:
    if eid in CORRUPT or eid in seen9 or sa is None or sb is None:
        continue
    seen9.add(eid)
    try:
        totx = (json.loads(xm) or {}).get("Total de buts") or {}
    except Exception:
        continue
    v = totx.get("1")
    if v is None:
        continue
    o = float(v)
    if not (1 < o < 100):
        continue
    bets9.append((sa + sb == 1, o))
    per_lg.setdefault(comp, []).append((sa + sb == 1, o))
OUT["pooled9"] = roi_stats(bets9)
OUT["per_league"] = {c: roi_stats(b) for c, b in sorted(per_lg.items())}
print("pooled9:", OUT["pooled9"])
for c, s in OUT["per_league"].items():
    print(f"  {c}: {s}")

with open("exports/wf4_totalsrefute2.json", "w", encoding="utf-8") as f:
    json.dump(OUT, f, ensure_ascii=False, indent=2)
print("written -> exports/wf4_totalsrefute2.json")
