# WF4 TOTALS - ADVERSARIAL re-verification du finding REFUTATION
# "Under 3.5 'drama-immunise' sur gros favoris: negatif aussi"
# READ-ONLY DB. Nouveau fichier. Resultats -> exports/wf4_totalsadv_under.json
# Checks UNDER-side:
# (a) fresh re-extraction (DB a grossi depuis le pkl) -> blanket under, fav<1.3 under
# (b) les 2 'meilleurs' segments du finding avec data fraiche (dom-new lam>=3.4, coupes lam[3.0-3.4))
# (c) look-ahead: under ROI fetched_at<start vs >=start + distribution delta temps
# (d) settlement: goals_json count vs score total
# (e) sous-periodes (demi-mois) du fav<1.3 under
# (f) walk-forward 70/30 8035 du blanket under et fav<1.3 under
# (g) bootstrap CI 95% fav<1.3 under
# (h) fresh-OOS: events absents du pkl du finding
# (i) scan seuils favoris alternatifs (1.2/1.25/1.35/1.4) cherche un POSITIF cache
import sys, json, pickle, math
sys.path.insert(0, ".")
import numpy as np
from scipy.stats import poisson, norm
from scipy.optimize import least_squares
from collections import Counter, defaultdict
from scraper.config import load_settings
from sqlalchemy import create_engine, text

np.random.seed(123)
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
n_corrupt = n_guard = n_badodds = n_settle_mismatch = 0
for row in rows:
    (eid, comp, exp_start, sa, sb, hta, htb, gj, oh, od, oa, xm, fat, snap_status) = row
    if eid in CORRUPT:
        n_corrupt += 1; continue
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
    ou_o, ou_u = m.get("> 3.5"), m.get("< 3.5")
    la_flag = bool(fat and exp_start and str(fat) >= str(exp_start))
    data.append(dict(eid=eid, comp=comp, start=str(exp_start), fat=str(fat),
                     status=snap_status,
                     sa=sa, sb=sb, tot=sa + sb, oh=oh, od=od, oa=oa,
                     ou_o=ou_o, ou_u=ou_u, la_flag=la_flag, gj_len=gj_len))
print(f"fresh kept {len(data)} | corrupt {n_corrupt} | guard {n_guard} | "
      f"badodds {n_badodds} | settle_mismatch {n_settle_mismatch}")
OUT["fresh_extraction"] = dict(kept=len(data), corrupt=n_corrupt, guard=n_guard,
                               badodds=n_badodds, settle_mismatch=n_settle_mismatch)

def roi_stats(bets):
    if not bets:
        return dict(n=0, wr=0.0, roi=0.0, odds=0.0, p=1.0)
    r = np.array([(o - 1) if w else -1.0 for w, o in bets])
    n = len(r); roi = float(r.mean())
    se = r.std(ddof=1) / math.sqrt(n) if n > 1 else 1e9
    p = float(2 * (1 - norm.cdf(abs(roi) / se))) if se > 0 else 1.0
    return dict(n=n, wr=float(np.mean([w for w, _ in bets])), roi=roi,
                odds=float(np.mean([o for _, o in bets])), p=p)

def under_bets(sub):
    return [((r["tot"] <= 3), r["ou_u"]) for r in sub
            if r["ou_u"] and 1 < r["ou_u"] < 100]

# lambdas: reuse pkl, invert nouveaux
n_inv = 0
for r in data:
    if r["eid"] in pkl_lam:
        r["lh"], r["la"] = pkl_lam[r["eid"]]
    else:
        lh, la, err = invert_lambdas(r["oh"], r["od"], r["oa"])
        r["lh"], r["la"] = (lh, la) if err <= 1e-5 else (None, None)
        n_inv += 1
print(f"lambdas inverses frais: {n_inv}")

# ---------- (a) reproduction headline sur data fraiche ----------
print("\n=== UNDER 3.5 fresh ===")
OUT["fresh_under"] = {}
favsub = [r for r in data if min(r["oh"], r["oa"]) < 1.3]
for scope, sub in [("pooled-9", data),
                   ("8035", [r for r in data if r["comp"] == "InstantLeague-8035"]),
                   ("fav<1.3_pooled", favsub)]:
    s = roi_stats(under_bets(sub))
    OUT["fresh_under"][scope] = s
    print(f"{scope:16s} n={s['n']:5d} WR={s['wr']:.4f} ROI={s['roi']*100:+.2f}% "
          f"odds={s['odds']:.3f} p={s['p']:.4f}")

# ---------- (b) les 2 'meilleurs' segments du finding, data fraiche ----------
print("\n=== les 2 meilleurs segments du finding (fresh, n a grossi) ===")
def grp(c):
    return "8035" if c == "InstantLeague-8035" else ("dom-new" if c in NEW else "coupes")
seg1 = [r for r in data if grp(r["comp"]) == "dom-new" and r.get("lh")
        and r["lh"] + r["la"] >= 3.4]
seg2 = [r for r in data if grp(r["comp"]) == "coupes" and r.get("lh")
        and 3.0 <= r["lh"] + r["la"] < 3.4]
s1 = roi_stats(under_bets(seg1)); s2 = roi_stats(under_bets(seg2))
OUT["best_segments_fresh"] = {"dom-new_lam>=3.4_under": s1,
                              "coupes_lam[3.0-3.4)_under": s2}
print(f"dom-new lam>=3.4   under n={s1['n']} ROI={s1['roi']*100:+.2f}% p={s1['p']:.4f}")
print(f"coupes lam[3.0-3.4) under n={s2['n']} ROI={s2['roi']*100:+.2f}% p={s2['p']:.4f}")
# fresh-only part de seg1 (OOS pur vs pkl)
seg1_oos = [r for r in seg1 if r["eid"] not in pkl_ids]
s1o = roi_stats(under_bets(seg1_oos))
OUT["best_segments_fresh"]["dom-new_lam>=3.4_under_OOSonly"] = s1o
print(f"dom-new lam>=3.4 under OOS-only n={s1o['n']} ROI={s1o['roi']*100:+.2f}% p={s1o['p']:.4f}")

# ---------- (c) look-ahead ----------
print("\n=== look-ahead: under par flag fetched_at>=expected_start ===")
la1 = roi_stats(under_bets([r for r in data if r["la_flag"]]))
la0 = roi_stats(under_bets([r for r in data if not r["la_flag"]]))
OUT["look_ahead_split"] = {"fetched_after_start": la1, "fetched_before_start": la0}
print(f"fetched>=start n={la1['n']} ROI={la1['roi']*100:+.2f}% | fetched<start n={la0['n']} ROI={la0['roi']*100:+.2f}%")
# distribution delta minutes (fetched - start) pour diagnostiquer timezone
deltas = []
for r in data[:4000]:
    try:
        from datetime import datetime
        f1 = datetime.fromisoformat(r["fat"].replace("Z", "+00:00").split("+")[0])
        s1_ = datetime.fromisoformat(r["start"].split("+")[0])
        deltas.append((f1 - s1_).total_seconds() / 60.0)
    except Exception:
        pass
if deltas:
    qs = np.percentile(deltas, [5, 25, 50, 75, 95])
    OUT["fetch_minus_start_minutes"] = {"q05": qs[0], "q25": qs[1], "median": qs[2],
                                        "q75": qs[3], "q95": qs[4]}
    print("delta fetched-start minutes quantiles:", [round(q, 1) for q in qs])

# fav<1.3 under SANS les rows look-ahead
sNoLA = roi_stats(under_bets([r for r in favsub if not r["la_flag"]]))
OUT["fav13_under_noLA"] = sNoLA
print(f"fav<1.3 under sans look-ahead rows: n={sNoLA['n']} ROI={sNoLA['roi']*100:+.2f}% p={sNoLA['p']:.4f}")
# statuts des snapshots d'ouverture
OUT["snapshot_status_counts"] = dict(Counter(r["status"] for r in data))
print("statuts snapshots ouverture:", OUT["snapshot_status_counts"])

# ---------- (d) settlement croise: goals_json vs score ----------
n_gj = sum(1 for r in data if r["gj_len"] is not None)
agree = sum(1 for r in data if r["gj_len"] is not None and r["gj_len"] == r["tot"])
OUT["settlement_cross"] = {"with_gj": n_gj, "agree": agree,
                           "pct_agree": round(100.0 * agree / max(n_gj, 1), 2)}
print(f"\nsettlement goals_json: {n_gj} dispo, {agree} == total ({OUT['settlement_cross']['pct_agree']}%)")

# ---------- (e) sous-periodes fav<1.3 under ----------
print("\n=== fav<1.3 under par demi-mois ===")
per = defaultdict(list)
for r in favsub:
    d = r["start"][:10]
    per[d[:8] + ("A" if d[8:10] < "16" else "B")].append(r)
OUT["fav13_subperiods"] = {}
for k in sorted(per):
    s = roi_stats(under_bets(per[k]))
    OUT["fav13_subperiods"][k] = s
    print(f"{k} n={s['n']:5d} ROI={s['roi']*100:+.2f}% p={s['p']:.3f}")

# ---------- (f) walk-forward 8035 ----------
print("\n=== walk-forward 8035 (70/30) under ===")
d35 = sorted([r for r in data if r["comp"] == "InstantLeague-8035"], key=lambda r: r["start"])
cut = int(0.7 * len(d35))
tr, te = d35[:cut], d35[cut:]
wf = {"blanket_train": roi_stats(under_bets(tr)), "blanket_test": roi_stats(under_bets(te)),
      "fav13_train": roi_stats(under_bets([r for r in tr if min(r["oh"], r["oa"]) < 1.3])),
      "fav13_test": roi_stats(under_bets([r for r in te if min(r["oh"], r["oa"]) < 1.3]))}
OUT["walkforward_8035_under"] = wf
for k, s in wf.items():
    print(f"{k:14s} n={s['n']:5d} ROI={s['roi']*100:+.2f}% p={s['p']:.4f}")

# ---------- (g) bootstrap CI fav<1.3 under ----------
bets = under_bets(favsub)
rarr = np.array([(o - 1) if w else -1.0 for w, o in bets])
bs = np.array([rarr[np.random.randint(0, len(rarr), len(rarr))].mean() for _ in range(10000)])
OUT["bootstrap_fav13_under"] = {"roi": float(rarr.mean()),
                                "ci95": [float(np.percentile(bs, 2.5)),
                                         float(np.percentile(bs, 97.5))],
                                "p_roi_pos": float((bs > 0).mean())}
print("\nbootstrap fav<1.3 under:", OUT["bootstrap_fav13_under"])

# blanket under pooled bootstrap
bets_all = under_bets(data)
rall = np.array([(o - 1) if w else -1.0 for w, o in bets_all])
bsa = np.array([rall[np.random.randint(0, len(rall), len(rall))].mean() for _ in range(10000)])
OUT["bootstrap_blanket_under"] = {"roi": float(rall.mean()),
                                  "ci95": [float(np.percentile(bsa, 2.5)),
                                           float(np.percentile(bsa, 97.5))],
                                  "p_roi_pos": float((bsa > 0).mean())}
print("bootstrap blanket under pooled:", OUT["bootstrap_blanket_under"])

# ---------- (h) fresh-OOS global ----------
oos = [r for r in data if r["eid"] not in pkl_ids]
s = roi_stats(under_bets(oos))
OUT["fresh_oos_under"] = {"n_events": len(oos), **s}
print(f"\nfresh-OOS under (events post-pkl): n_events={len(oos)} bets n={s['n']} "
      f"ROI={s['roi']*100:+.2f}% p={s['p']:.4f}")
sf = roi_stats(under_bets([r for r in oos if min(r["oh"], r["oa"]) < 1.3]))
OUT["fresh_oos_fav13_under"] = sf
print(f"fresh-OOS fav<1.3 under: n={sf['n']} ROI={sf['roi']*100:+.2f}% p={sf['p']:.4f}")

# ---------- (i) scan seuils favoris alternatifs : un positif cache ? ----------
print("\n=== scan seuils fav alternatifs (under) ===")
OUT["alt_fav_scan"] = {}
for thr in [1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.5]:
    sub = [r for r in data if min(r["oh"], r["oa"]) < thr]
    s = roi_stats(under_bets(sub))
    OUT["alt_fav_scan"][f"min_odds<{thr}"] = s
    print(f"min_odds<{thr:<5} n={s['n']:5d} ROI={s['roi']*100:+.2f}% p={s['p']:.4f}")
# par groupe pour thr=1.3
for g in ["8035", "dom-new", "coupes"]:
    sub = [r for r in favsub if grp(r["comp"]) == g]
    s = roi_stats(under_bets(sub))
    OUT["alt_fav_scan"][f"fav13_{g}"] = s
    print(f"fav<1.3 {g:8s} n={s['n']:5d} ROI={s['roi']*100:+.2f}% p={s['p']:.4f}")

with open("exports/wf4_totalsadv_under.json", "w", encoding="utf-8") as f:
    json.dump(OUT, f, indent=2, ensure_ascii=False)
print("\nsaved -> exports/wf4_totalsadv_under.json")
