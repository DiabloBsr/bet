# ADVERSARIAL DESTROY attempt on finding:
# "REFUTATION - Under 3.5 'drama-immunise' sur gros favoris: negatif aussi"
# Independent fresh re-extraction (READ-ONLY DB), no reliance on the finding's pkl
# for ROI (pkl lambdas reused only for speed, with random re-inversion audit).
# Output -> exports/wf4_totalsdestroy.json
import sys, json, pickle, math
sys.path.insert(0, ".")
import numpy as np
from datetime import datetime
from scipy.stats import poisson, norm
from scipy.optimize import least_squares
from collections import Counter, defaultdict
from scraper.config import load_settings
from sqlalchemy import create_engine, text

np.random.seed(777)
e = create_engine(load_settings().db_url)
OUT = {}

with open("exports/corrupted_events.json", "r", encoding="utf-8") as f:
    CORRUPT = set(int(k) for k in json.load(f)["events"].keys())

LEAGUES = ["InstantLeague-8035", "InstantLeague-8036", "InstantLeague-8037",
           "InstantLeague-8042", "InstantLeague-8043", "InstantLeague-8044",
           "InstantLeague-8056", "InstantLeague-8060", "InstantLeague-8065"]
NEW = {"InstantLeague-8036", "InstantLeague-8037", "InstantLeague-8042",
       "InstantLeague-8043", "InstantLeague-8044"}
comps = "(" + ",".join("'" + c + "'" for c in LEAGUES) + ")"

def grp(c):
    return "8035" if c == "InstantLeague-8035" else ("dom-new" if c in NEW else "coupes")

# ---------- sanity: dup results / survivorship ----------
with e.connect() as conn:
    dup_res = conn.execute(text(
        "SELECT COUNT(*) FROM (SELECT event_id FROM results GROUP BY event_id HAVING COUNT(*)>1)"
    )).scalar()
    surv = {}
    for lg in LEAGUES:
        n_odds = conn.execute(text(
            "SELECT COUNT(*) FROM events ev WHERE ev.competition=:c "
            "AND EXISTS (SELECT 1 FROM odds_snapshots o WHERE o.event_id=ev.id)"), dict(c=lg)).scalar()
        n_nores = conn.execute(text(
            "SELECT COUNT(*) FROM events ev WHERE ev.competition=:c "
            "AND EXISTS (SELECT 1 FROM odds_snapshots o WHERE o.event_id=ev.id) "
            "AND NOT EXISTS (SELECT 1 FROM results r WHERE r.event_id=ev.id)"), dict(c=lg)).scalar()
        surv[lg] = dict(with_odds=int(n_odds), no_result=int(n_nores),
                        pct_missing=round(100.0 * n_nores / max(n_odds, 1), 1))
OUT["dup_results_events"] = int(dup_res)
OUT["survivorship_per_league"] = surv
print("dup results:", dup_res)
print("survivorship:", surv)

# ---------- fresh extraction ----------
SQL = f"""
SELECT ev.id, ev.competition, ev.expected_start,
       r.score_a, r.score_b, r.ht_score_a, r.ht_score_b, r.goals_json,
       o.odds_home, o.odds_draw, o.odds_away, o.extra_markets, o.captured_at, o.status
FROM events ev
JOIN results r ON r.event_id = ev.id
JOIN odds_snapshots o ON o.event_id = ev.id
WHERE o.id = (SELECT MIN(o2.id) FROM odds_snapshots o2 WHERE o2.event_id = ev.id)
  AND ev.competition IN {comps}
"""
with e.connect() as conn:
    rows = conn.execute(text(SQL)).fetchall()
print(f"fresh raw rows: {len(rows)}")

with open("exports/wf4_totals_data.pkl", "rb") as f:
    PKL = pickle.load(f)
pkl_lam = {r["eid"]: (r["lh"], r["la"]) for r in PKL}
pkl_ids = set(pkl_lam.keys())
print(f"pkl du finding: {len(PKL)} events")

GMAX = 13
ar = np.arange(GMAX + 1)

def invert_lambdas(oh, od, oa):
    imp = np.array([1 / oh, 1 / od, 1 / oa]); fair = imp / imp.sum()
    def resid(x):
        lh, la = np.exp(x)
        g = np.outer(poisson.pmf(ar, lh), poisson.pmf(ar, la))
        return [np.tril(g, -1).sum() - fair[0], np.trace(g) - fair[1]]
    diff0 = math.log(max(fair[0], 1e-6) / max(fair[2], 1e-6)) * 0.55
    x0 = [math.log(max(0.2, 1.4 + diff0 / 2)), math.log(max(0.2, 1.4 - diff0 / 2))]
    sol = least_squares(resid, x0, xtol=1e-12, ftol=1e-12)
    lh, la = np.exp(sol.x)
    return float(lh), float(la), float(max(abs(v) for v in resid(sol.x)))

data = []
seen = set()
n_corrupt = n_dupe = n_guard = n_badodds = n_settle_mismatch = 0
ou_key_counter = Counter()
for row in rows:
    (eid, comp, exp_start, sa, sb, hta, htb, gj, oh, od, oa, xm, cat, st) = row
    if eid in CORRUPT:
        n_corrupt += 1; continue
    if eid in seen:
        n_dupe += 1; continue
    seen.add(eid)
    if sa is None or sb is None:
        continue
    if hta is not None and htb is not None and (hta > sa or htb > sb):
        n_guard += 1; continue
    gj_len = None
    if gj:
        try:
            gl = json.loads(gj)
            if isinstance(gl, list):
                gj_len = len(gl)
        except Exception:
            pass
    if gj_len is not None and gj_len != sa + sb:
        n_settle_mismatch += 1; continue
    if not oh or not od or not oa or oh <= 1 or od <= 1 or oa <= 1:
        n_badodds += 1; continue
    try:
        xmd = json.loads(xm) if xm else {}
    except Exception:
        xmd = {}
    m = xmd.get("+/-") or {}
    ou_key_counter.update(m.keys())
    data.append(dict(eid=eid, comp=comp, start=str(exp_start), cat=str(cat), status=st,
                     sa=sa, sb=sb, tot=sa + sb, oh=oh, od=od, oa=oa,
                     ou_o=m.get("> 3.5"), ou_u=m.get("< 3.5"), gj_len=gj_len))
print(f"fresh kept {len(data)} | corrupt {n_corrupt} | dupe {n_dupe} | guard {n_guard} | "
      f"badodds {n_badodds} | settle_mismatch {n_settle_mismatch}")
OUT["fresh_extraction"] = dict(kept=len(data), corrupt=n_corrupt, dupe=n_dupe, guard=n_guard,
                               badodds=n_badodds, settle_mismatch=n_settle_mismatch)
OUT["ou_market_keys"] = dict(ou_key_counter)
print("cles marche '+/-':", dict(ou_key_counter))
OUT["snapshot_status_counts"] = dict(Counter(r["status"] for r in data))
print("statuts snapshot ouverture:", OUT["snapshot_status_counts"])

# ---------- lambdas: reuse pkl + audit re-inversion sur echantillon ----------
n_inv = 0
for r in data:
    if r["eid"] in pkl_lam:
        r["lh"], r["la"] = pkl_lam[r["eid"]]
    else:
        lh, la, err = invert_lambdas(r["oh"], r["od"], r["oa"])
        r["lh"], r["la"] = (lh, la) if err <= 1e-5 else (None, None)
        n_inv += 1
print(f"lambdas inverses frais: {n_inv}")
# audit: re-invert 60 random pkl events, compare
idx = np.random.choice([i for i, r in enumerate(data) if r["eid"] in pkl_lam],
                       size=60, replace=False)
max_dev = 0.0
for i in idx:
    r = data[i]
    lh, la, err = invert_lambdas(r["oh"], r["od"], r["oa"])
    max_dev = max(max_dev, abs(lh - r["lh"]), abs(la - r["la"]))
OUT["lambda_audit_max_dev"] = float(max_dev)
print(f"audit lambdas (60 random): max deviation = {max_dev:.2e}")

# ---------- helpers ----------
def roi_stats(bets):
    if not bets:
        return dict(n=0, wr=0.0, roi=0.0, odds=0.0, p=1.0)
    rr = np.array([(o - 1) if w else -1.0 for w, o in bets])
    n = len(rr); roi = float(rr.mean())
    se = rr.std(ddof=1) / math.sqrt(n) if n > 1 else 1e9
    p = float(2 * (1 - norm.cdf(abs(roi) / se))) if se > 0 else 1.0
    return dict(n=n, wr=round(float(np.mean([w for w, _ in bets])), 4),
                roi=round(roi, 5), odds=round(float(np.mean([o for _, o in bets])), 3),
                p=round(p, 6))

def under_bets(sub):
    return [((r["tot"] <= 3), r["ou_u"]) for r in sub if r["ou_u"] and 1 < r["ou_u"] < 100]

# ---------- (1) reproduction headline, fresh data ----------
print("\n=== UNDER 3.5 fresh (independant) ===")
fav_lt = [r for r in data if min(r["oh"], r["oa"]) < 1.3]
fav_le = [r for r in data if min(r["oh"], r["oa"]) <= 1.3]
OUT["fresh_under"] = {
    "pooled9": roi_stats(under_bets(data)),
    "8035": roi_stats(under_bets([r for r in data if r["comp"] == "InstantLeague-8035"])),
    "newleagues": roi_stats(under_bets([r for r in data if r["comp"] != "InstantLeague-8035"])),
    "fav_min_odds_lt_1.3": roi_stats(under_bets(fav_lt)),
    "fav_min_odds_le_1.3": roi_stats(under_bets(fav_le)),
}
for k, s in OUT["fresh_under"].items():
    print(f"{k:22s} n={s['n']:5d} WR={s['wr']:.4f} ROI={s['roi']*100:+.2f}% odds={s['odds']:.3f} p={s['p']:.5f}")

# ---------- (2) look-ahead diagnostics ----------
print("\n=== look-ahead ===")
la_rows = [r for r in data if r["cat"] and r["start"] and r["cat"] >= r["start"]]
OUT["look_ahead"] = {"opening_after_start": len(la_rows),
                     "pct": round(100.0 * len(la_rows) / max(len(data), 1), 2)}
deltas = []
for r in data:
    try:
        f1 = datetime.fromisoformat(r["cat"].replace("Z", "+00:00").split("+")[0].split(".")[0])
        s1 = datetime.fromisoformat(r["start"].split("+")[0].split(".")[0])
        deltas.append((f1 - s1).total_seconds() / 60.0)
    except Exception:
        pass
if deltas:
    qs = np.percentile(deltas, [1, 5, 25, 50, 75, 95, 99])
    OUT["capture_minus_start_min"] = {f"q{q:02d}": round(float(v), 1)
                                      for q, v in zip([1, 5, 25, 50, 75, 95, 99], qs)}
    print("delta capture-start (min) quantiles:", OUT["capture_minus_start_min"])
s_before = roi_stats(under_bets([r for r in data if r not in la_rows]))
# faster: use flag
for r in data:
    r["laf"] = bool(r["cat"] and r["start"] and r["cat"] >= r["start"])
s_after = roi_stats(under_bets([r for r in data if r["laf"]]))
s_before = roi_stats(under_bets([r for r in data if not r["laf"]]))
OUT["under_by_lookahead"] = {"captured_before_start": s_before, "captured_after_start": s_after}
print(f"under captured<start : n={s_before['n']} ROI={s_before['roi']*100:+.2f}% p={s_before['p']:.4f}")
print(f"under captured>=start: n={s_after['n']} ROI={s_after['roi']*100:+.2f}% p={s_after['p']:.4f}")
sf = roi_stats(under_bets([r for r in fav_lt if not r["laf"]]))
OUT["fav13_under_noLA"] = sf
print(f"fav<1.3 under sans rows look-ahead: n={sf['n']} ROI={sf['roi']*100:+.2f}% p={sf['p']:.4f}")

# ---------- (3) settlement cross-check ----------
n_gj = sum(1 for r in data if r["gj_len"] is not None)
OUT["settlement"] = {"with_goals_json_pct": round(100.0 * n_gj / len(data), 1),
                     "excluded_mismatch": n_settle_mismatch}
print("\nsettlement: goals_json dispo", OUT["settlement"])

# ---------- (4) walk-forward 70/30 sur 8035 ----------
print("\n=== walk-forward 8035 70/30 (under) ===")
d35 = sorted([r for r in data if r["comp"] == "InstantLeague-8035"], key=lambda r: (r["start"], r["eid"]))
cut = int(0.7 * len(d35))
tr, te = d35[:cut], d35[cut:]
wf = {"blanket_train": roi_stats(under_bets(tr)),
      "blanket_test": roi_stats(under_bets(te)),
      "fav13_train": roi_stats(under_bets([r for r in tr if min(r["oh"], r["oa"]) < 1.3])),
      "fav13_test": roi_stats(under_bets([r for r in te if min(r["oh"], r["oa"]) < 1.3])),
      "cut_date": d35[cut]["start"]}
OUT["walkforward_8035_under"] = wf
for k, s in wf.items():
    if isinstance(s, dict):
        print(f"{k:14s} n={s['n']:5d} ROI={s['roi']*100:+.2f}% p={s['p']:.4f}")

# ---------- (5) sous-periodes (demi-mois) ----------
print("\n=== under par demi-mois (pooled / fav<1.3) ===")
per = defaultdict(list)
for r in data:
    d = r["start"][:10]
    per[d[:8] + ("A" if d[8:10] < "16" else "B")].append(r)
OUT["subperiods_under"] = {}
for k in sorted(per):
    sp = roi_stats(under_bets(per[k]))
    spf = roi_stats(under_bets([r for r in per[k] if min(r["oh"], r["oa"]) < 1.3]))
    OUT["subperiods_under"][k] = {"pooled": sp, "fav13": spf}
    print(f"{k} pooled n={sp['n']:5d} ROI={sp['roi']*100:+.2f}% | fav13 n={spf['n']:4d} ROI={spf['roi']*100:+.2f}%")

# ---------- (6) bootstrap CI ----------
def boot_ci(bets, B=10000):
    rr = np.array([(o - 1) if w else -1.0 for w, o in bets])
    bs = np.array([rr[np.random.randint(0, len(rr), len(rr))].mean() for _ in range(B)])
    return {"roi": round(float(rr.mean()), 5),
            "ci95": [round(float(np.percentile(bs, 2.5)), 5), round(float(np.percentile(bs, 97.5)), 5)],
            "p_roi_pos": round(float((bs > 0).mean()), 5)}
OUT["bootstrap_fav13_under"] = boot_ci(under_bets(fav_lt))
OUT["bootstrap_blanket_under"] = boot_ci(under_bets(data))
print("\nbootstrap fav<1.3 under:", OUT["bootstrap_fav13_under"])
print("bootstrap blanket under:", OUT["bootstrap_blanket_under"])

# ---------- (7) fresh-OOS: events absents du pkl du finding ----------
oos = [r for r in data if r["eid"] not in pkl_ids]
OUT["fresh_oos_under"] = {"n_events": len(oos), **roi_stats(under_bets(oos))}
OUT["fresh_oos_fav13_under"] = roi_stats(under_bets([r for r in oos if min(r["oh"], r["oa"]) < 1.3]))
print(f"\nfresh-OOS (post-pkl) under: {OUT['fresh_oos_under']}")
print(f"fresh-OOS fav<1.3 under: {OUT['fresh_oos_fav13_under']}")

# ---------- (8) scan poches positives cachees sur l'UNDER ----------
print("\n=== scan poches positives under (seuils finding: n>=150, ROI>=4%, p<=0.01) ===")
scan = []
# (a) seuils favoris
for thr in [1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.5]:
    for g in ["all", "8035", "dom-new", "coupes"]:
        sub = [r for r in data if min(r["oh"], r["oa"]) < thr
               and (g == "all" or grp(r["comp"]) == g)]
        scan.append((f"fav<{thr}_{g}", roi_stats(under_bets(sub))))
# (b) lambda buckets x groupes (edges du finding ET edges decales)
for ed in ([0, 2.2, 2.6, 3.0, 3.4, 99], [0, 2.0, 2.4, 2.8, 3.2, 3.6, 99]):
    for g in ["all", "8035", "dom-new", "coupes"]:
        for i in range(len(ed) - 1):
            sub = [r for r in data if r.get("lh") and ed[i] <= r["lh"] + r["la"] < ed[i + 1]
                   and (g == "all" or grp(r["comp"]) == g)]
            scan.append((f"lam[{ed[i]}-{ed[i+1]})_{g}", roi_stats(under_bets(sub))))
# (c) bandes de cote under offerte
oed = [1.0, 1.15, 1.3, 1.5, 1.7, 2.0, 2.5, 99]
for g in ["all", "8035", "dom-new", "coupes"]:
    for i in range(len(oed) - 1):
        sub = [r for r in data if r["ou_u"] and oed[i] <= r["ou_u"] < oed[i + 1]
               and (g == "all" or grp(r["comp"]) == g)]
        scan.append((f"odds_u[{oed[i]}-{oed[i+1]})_{g}", roi_stats(under_bets(sub))))
scan = [(k, s) for k, s in scan if s["n"] >= 150]
pos = [(k, s) for k, s in scan if s["roi"] > 0]
pos_sig = [(k, s) for k, s in pos if s["roi"] >= 0.04 and s["p"] <= 0.01]
scan_sorted = sorted(scan, key=lambda t: -t[1]["roi"])
OUT["under_pocket_scan"] = {"n_segments_n150": len(scan), "n_positive": len(pos),
                            "n_pass_thresholds": len(pos_sig),
                            "top8": [{"seg": k, **s} for k, s in scan_sorted[:8]]}
print(f"segments (n>=150): {len(scan)} | ROI>0: {len(pos)} | passent seuils finding: {len(pos_sig)}")
for k, s in scan_sorted[:8]:
    print(f"  TOP {k:26s} n={s['n']:5d} ROI={s['roi']*100:+.2f}% p={s['p']:.4f}")

# le 'meilleur segment' du finding avec data fraiche + OOS pur
seg1 = [r for r in data if grp(r["comp"]) == "dom-new" and r.get("lh") and r["lh"] + r["la"] >= 3.4]
OUT["domnew_lam34_under_fresh"] = roi_stats(under_bets(seg1))
OUT["domnew_lam34_under_OOSonly"] = roi_stats(under_bets([r for r in seg1 if r["eid"] not in pkl_ids]))
print("dom-new lam>=3.4 under fresh:", OUT["domnew_lam34_under_fresh"])
print("dom-new lam>=3.4 under OOS-only:", OUT["domnew_lam34_under_OOSonly"])
seg2 = [r for r in data if grp(r["comp"]) == "coupes" and r.get("lh") and 3.0 <= r["lh"] + r["la"] < 3.4]
OUT["coupes_lam3034_under_fresh"] = roi_stats(under_bets(seg2))
print("coupes lam[3.0-3.4) under fresh:", OUT["coupes_lam3034_under_fresh"])

# ---------- (9) coherence marge / fair prob ----------
m = [1 / r["ou_o"] + 1 / r["ou_u"] - 1 for r in data
     if r["ou_o"] and r["ou_u"] and r["ou_o"] > 1 and r["ou_u"] > 1]
OUT["margin_ou35"] = {"mean_pct": round(float(np.mean(m)) * 100, 3),
                      "std_pct": round(float(np.std(m)) * 100, 3), "n": len(m)}
print("\nmarge +/- 3.5:", OUT["margin_ou35"])
# break-even: WR reel vs implied (avec marge) vs fair (sans marge)
ub = under_bets(data)
wr_real = np.mean([w for w, _ in ub])
imp_with_margin = np.mean([1 / o for _, o in ub])
fair = []
for r in data:
    if r["ou_o"] and r["ou_u"] and r["ou_o"] > 1 and r["ou_u"] > 1:
        iu, io = 1 / r["ou_u"], 1 / r["ou_o"]
        fair.append(iu / (iu + io))
OUT["under_calibration"] = {"wr_real": round(float(wr_real), 4),
                            "implied_offered": round(float(imp_with_margin), 4),
                            "implied_fair": round(float(np.mean(fair)), 4)}
print("calibration under:", OUT["under_calibration"])

with open("exports/wf4_totalsdestroy.json", "w", encoding="utf-8") as f:
    json.dump(OUT, f, indent=2, ensure_ascii=False)
print("\nsaved -> exports/wf4_totalsdestroy.json")
