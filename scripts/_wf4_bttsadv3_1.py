# WF4 - BTTS - contre-verification ADVERSARIALE #3 du finding "G/NG efficient, aucun pari positif"
# Verifications independantes (extraction SQL refaite from scratch):
#  (1) look-ahead: captured_at du snapshot MIN(id) vs expected_start ; statuts des snapshots
#  (2) MIN(id) vs MIN(captured_at): le "snapshot d'ouverture" est-il bien le premier ?
#  (3) settlement croise: goals_json (team Home/Away) vs score_a/score_b
#  (4) reproduction independante du ROI global Oui/Non + bootstrap 10k (CI 95%)
#  (5) extension hold-out: events finis APRES l'extraction de l'agent (ids hors de son dataset)
#  (6) sous-periodes: ROI Oui/Non par jour (pooled9) ; split alternatif 50/50 sur 8035
#      pour les 5 cellules selectionnees par le scan de l'agent
#  (7) marge par bande de cote (verif "marge plate 6%")
# Sortie: exports/wf4_bttsadv3.json -- LECTURE SEULE sur la DB.
import sys, json, math

import numpy as np

sys.path.insert(0, ".")
from scraper.config import load_settings
from sqlalchemy import create_engine, text

LEAGUES = [
    "InstantLeague-8035", "InstantLeague-8036", "InstantLeague-8037",
    "InstantLeague-8042", "InstantLeague-8043", "InstantLeague-8044",
    "InstantLeague-8056", "InstantLeague-8060", "InstantLeague-8065",
]
OUT = {}

corrupted = set(int(k) for k in json.load(open("exports/corrupted_events.json", encoding="utf-8"))["events"].keys())
agent_rows = json.load(open("exports/wf4_btts_data.json", encoding="utf-8"))["rows"]
agent_ids = set(r["id"] for r in agent_rows)

e = create_engine(load_settings().db_url)
in_list = ",".join("'%s'" % l for l in LEAGUES)

# ---------------------------------------------------------------- extraction independante
SQL = """
SELECT e.id, e.competition, e.expected_start,
       f.sid, fo.captured_at, fo.status,
       g.first_cap,
       fo.odds_home, fo.odds_draw, fo.odds_away, fo.extra_markets,
       r.score_a, r.score_b, r.goals_json, r.finished_at
FROM events e
JOIN results r            ON r.event_id = e.id
JOIN (SELECT event_id, MIN(id) AS sid FROM odds_snapshots GROUP BY event_id) f
                          ON f.event_id = e.id
JOIN odds_snapshots fo    ON fo.id = f.sid
JOIN (SELECT event_id, MIN(captured_at) AS first_cap FROM odds_snapshots GROUP BY event_id) g
                          ON g.event_id = e.id
WHERE e.competition IN (%s)
""" % in_list
with e.connect() as c:
    raw = c.execute(text(SQL)).fetchall()
    statuses = c.execute(text(
        "SELECT status, COUNT(*) FROM odds_snapshots GROUP BY status")).fetchall()
OUT["snapshot_statuses"] = {str(s): int(n) for s, n in statuses}
print("statuts snapshots:", OUT["snapshot_statuses"])

look_after_start = 0     # snapshot d'ouverture capture APRES expected_start
minid_not_first = 0      # captured_at du MIN(id) > premier captured_at de l'event
settle_checked = settle_mismatch = 0
rows = []
for (eid, comp, est, sid, cap, status, first_cap, oh, od, oa, em,
     sa, sb, gj, fin) in raw:
    if eid in corrupted or sa is None or sb is None or not em \
       or oh is None or od is None or oa is None:
        continue
    try:
        gng = json.loads(em).get("G/NG")
    except Exception:
        continue
    if not gng or "Oui" not in gng or "Non" not in gng:
        continue
    cap_s, est_s, fc_s = str(cap), str(est), str(first_cap)
    if cap_s > est_s:
        look_after_start += 1
    if cap_s > fc_s:
        minid_not_first += 1
    # settlement croise goals_json
    if gj:
        try:
            gl = json.loads(gj)
            if isinstance(gl, list) and (gl or (sa + sb) == 0):
                ga = sum(1 for g in gl if isinstance(g, dict) and str(g.get("team", "")).lower().startswith("home"))
                gb = sum(1 for g in gl if isinstance(g, dict) and str(g.get("team", "")).lower().startswith("away"))
                settle_checked += 1
                if (ga, gb) != (sa, sb):
                    settle_mismatch += 1
        except Exception:
            pass
    rows.append({"id": eid, "comp": comp, "start": est_s, "fin": str(fin),
                 "o_yes": float(gng["Oui"]), "o_no": float(gng["Non"]),
                 "sa": int(sa), "sb": int(sb), "win": (sa > 0) and (sb > 0)})

OUT["lookahead"] = {"n": len(rows), "opening_captured_after_start": look_after_start,
                    "minid_not_first_captured": minid_not_first}
OUT["settlement"] = {"checked": settle_checked, "mismatch": settle_mismatch}
print("extraction independante: n=%d ; ouverture capturee apres start: %d ; MIN(id) pas premier captured_at: %d"
      % (len(rows), look_after_start, minid_not_first))
print("settlement goals_json vs score: %d verifies, %d mismatches" % (settle_checked, settle_mismatch))

o_yes = np.array([r["o_yes"] for r in rows])
o_no = np.array([r["o_no"] for r in rows])
win = np.array([r["win"] for r in rows])
start = np.array([r["start"] for r in rows])
ids = np.array([r["id"] for r in rows])
ret_y = o_yes * win - 1
ret_n = o_no * (~win) - 1


def tstat(r):
    if len(r) < 2:
        return 0.0, 1.0
    se = r.std(ddof=1) / math.sqrt(len(r))
    if se == 0:
        return 0.0, 1.0
    t = r.mean() / se
    return float(t), float(2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2)))))


# ---------------------------------------------------------------- (4) ROI global + bootstrap
rng = np.random.default_rng(42)
def boot_ci(r, k=10000):
    n = len(r)
    means = np.empty(k)
    for i in range(k):
        means[i] = r[rng.integers(0, n, n)].mean()
    return [round(float(np.percentile(means, 2.5)) * 100, 2),
            round(float(np.percentile(means, 97.5)) * 100, 2)]

ty, py = tstat(ret_y); tn, pn = tstat(ret_n)
OUT["global"] = {
    "n": len(rows), "wr_yes": round(float(win.mean()), 4),
    "avg_o_yes": round(float(o_yes.mean()), 3),
    "roi_yes_pct": round(float(ret_y.mean()) * 100, 2), "p_yes": float(py),
    "ci95_yes": boot_ci(ret_y),
    "roi_no_pct": round(float(ret_n.mean()) * 100, 2), "p_no": float(pn),
    "ci95_no": boot_ci(ret_n),
}
print("\nGLOBAL indep: n=%d wr=%.4f ROIyes=%.2f%% CI95=%s | ROIno=%.2f%% CI95=%s" % (
    OUT["global"]["n"], OUT["global"]["wr_yes"], OUT["global"]["roi_yes_pct"],
    OUT["global"]["ci95_yes"], OUT["global"]["roi_no_pct"], OUT["global"]["ci95_no"]))

# ---------------------------------------------------------------- (5) hold-out post-extraction
new_mask = ~np.isin(ids, list(agent_ids))
nh = int(new_mask.sum())
if nh > 0:
    ty2, py2 = tstat(ret_y[new_mask]); tn2, pn2 = tstat(ret_n[new_mask])
    OUT["holdout_post_extraction"] = {
        "n": nh, "wr_yes": round(float(win[new_mask].mean()), 4),
        "roi_yes_pct": round(float(ret_y[new_mask].mean()) * 100, 2), "p_yes": float(py2),
        "roi_no_pct": round(float(ret_n[new_mask].mean()) * 100, 2), "p_no": float(pn2),
        "avg_o_yes": round(float(o_yes[new_mask].mean()), 3),
    }
    print("HOLD-OUT (events absents du dataset agent): n=%d wr=%.4f ROIyes=%+.2f%% (p=%.3f) ROIno=%+.2f%% (p=%.3f)" % (
        nh, win[new_mask].mean(), ret_y[new_mask].mean() * 100, py2,
        ret_n[new_mask].mean() * 100, pn2))
else:
    OUT["holdout_post_extraction"] = {"n": 0}
    print("HOLD-OUT: aucun event nouveau depuis l'extraction de l'agent")

# ---------------------------------------------------------------- (6a) sous-periodes par jour
days = np.array([s[:10] for s in start])
udays = sorted(set(days))
day_roi = []
for d in udays:
    m = days == d
    if m.sum() < 100:
        continue
    day_roi.append((d, int(m.sum()), float(ret_y[m].mean()) * 100, float(ret_n[m].mean()) * 100))
pos_y = sum(1 for d in day_roi if d[2] > 0)
pos_n = sum(1 for d in day_roi if d[3] > 0)
OUT["daily"] = {"n_days": len(day_roi), "days_roi_yes_pos": pos_y, "days_roi_no_pos": pos_n,
                "detail": [{"day": d, "n": n, "roi_yes": round(r1, 2), "roi_no": round(r2, 2)}
                           for d, n, r1, r2 in day_roi]}
print("\njours (n>=100): %d ; ROIyes>0: %d ; ROIno>0: %d" % (len(day_roi), pos_y, pos_n))
for d, n, r1, r2 in day_roi:
    print("  %s n=%5d ROIyes=%+6.2f%% ROIno=%+6.2f%%" % (d, n, r1, r2))

# ---------------------------------------------------------------- (6b) split alternatif 50/50 8035
lz = np.load("exports/wf4_btts_lambdas.npz")
assert list(lz["ids"]) == [r["id"] for r in agent_rows]
lh_a, la_a = lz["lh"], lz["la"]
ag_o_yes = np.array([r["o_yes"] for r in agent_rows])
ag_win = np.array([(r["sa"] > 0) and (r["sb"] > 0) for r in agent_rows])
ag_ret_y = ag_o_yes * ag_win - 1
ag_comp = np.array([r["comp"] for r in agent_rows])
ag_start = np.array([r["start"] for r in agent_rows])
gap = np.abs(lh_a - la_a)
mn = np.minimum(lh_a, la_a)
mu = lh_a + la_a
is35 = ag_comp == "InstantLeague-8035"
s35 = np.sort(ag_start[is35])
CELLS = {
    "gapXodds:g060_090|o1.7-1.9": (gap >= 0.60) & (gap < 0.90) & (ag_o_yes >= 1.7) & (ag_o_yes < 1.9),
    "gapXminlam:g060_090|mn0.8-1.0": (gap >= 0.60) & (gap < 0.90) & (mn >= 0.8) & (mn < 1.0),
    "gapXmu:g060_090|mu2.4-2.8": (gap >= 0.60) & (gap < 0.90) & (mu >= 2.4) & (mu < 2.8),
    "gapXmu:g060_090|mu<2.4": (gap >= 0.60) & (gap < 0.90) & (mu < 2.4),
    "gapXminlam:g060_090|mn<0.8": (gap >= 0.60) & (gap < 0.90) & (mn < 0.8),
}
OUT["alt_split_5050_8035"] = {}
print("\nsplit alternatif 50/50 sur 8035 (cellules du scan, side=yes):")
cut50 = s35[int(0.50 * len(s35))]
tr50 = is35 & (ag_start < cut50)
te50 = is35 & (ag_start >= cut50)
for cname, cm in CELLS.items():
    res = {}
    for sname, sm in (("train50", tr50), ("test50", te50), ("full8035", is35),
                      ("fullpooled", np.ones(len(agent_rows), bool))):
        m = cm & sm
        t, p = tstat(ag_ret_y[m])
        res[sname] = {"n": int(m.sum()),
                      "roi_pct": round(float(ag_ret_y[m].mean()) * 100, 2) if m.sum() else None,
                      "p": round(p, 4)}
    OUT["alt_split_5050_8035"][cname] = res
    print("  %-32s train50 n=%4d roi=%+6.2f%% | test50 n=%4d roi=%+6.2f%% | full8035 n=%4d roi=%+6.2f%% | pooled n=%5d roi=%+6.2f%%" % (
        cname, res["train50"]["n"], res["train50"]["roi_pct"],
        res["test50"]["n"], res["test50"]["roi_pct"],
        res["full8035"]["n"], res["full8035"]["roi_pct"],
        res["fullpooled"]["n"], res["fullpooled"]["roi_pct"]))

# ---------------------------------------------------------------- (7) marge par bande de cote
s2 = 1 / o_yes + 1 / o_no
OUT["margin_by_band"] = {}
print("\nmarge G/NG par bande de o_yes:")
for lo, hi in ((1.0, 1.5), (1.5, 1.8), (1.8, 2.2), (2.2, 2.8), (2.8, 99)):
    m = (o_yes >= lo) & (o_yes < hi)
    if m.sum() == 0:
        continue
    OUT["margin_by_band"]["%.1f-%.1f" % (lo, hi)] = {
        "n": int(m.sum()), "margin_mean": round(float((s2[m] - 1).mean()), 4),
        "margin_std": round(float((s2[m] - 1).std()), 4)}
    print("  o_yes [%.1f,%.1f): n=%5d marge=%.4f std=%.4f" % (
        lo, hi, m.sum(), (s2[m] - 1).mean(), (s2[m] - 1).std()))

with open("exports/wf4_bttsadv3.json", "w", encoding="utf-8") as f:
    json.dump(OUT, f, indent=1)
print("\nOK -> exports/wf4_bttsadv3.json")
