# -*- coding: utf-8 -*-
"""WF4 cross-league part 5 — hardening.

1. Score distribution of the excluded corrupted events (audit asymmetry check).
2. Old-vs-recent 8035 composite chi2 on score categories (+ verified-goals-only).
3. 8035-recent vs POOLED-ALL-NEW: same regime?
4. FTTS '1' by FTTS-odds bucket per group (champ vs cup at matched odds).
"""
import sys, json, math
sys.path.insert(0, ".")
import numpy as np
from scipy.stats import chi2_contingency, norm
from sqlalchemy import create_engine, text
from scraper.config import load_settings

LEAGUES = ["InstantLeague-8035", "InstantLeague-8036", "InstantLeague-8037",
           "InstantLeague-8042", "InstantLeague-8043", "InstantLeague-8044",
           "InstantLeague-8056", "InstantLeague-8060", "InstantLeague-8065"]
CHAMP = {"InstantLeague-8036", "InstantLeague-8037", "InstantLeague-8042",
         "InstantLeague-8043", "InstantLeague-8044"}
CUP = {"InstantLeague-8056", "InstantLeague-8060", "InstantLeague-8065"}
REF = "InstantLeague-8035"
NEWWIN = "2026-06-12 00:00:00"

with open("exports/_wf4_cl_events.json", "r", encoding="utf-8") as f:
    events = json.load(f)
with open("exports/_wf4_cl_extra.json", "r", encoding="utf-8") as f:
    extra = json.load(f)
for e in events:
    e["total"] = e["sa"] + e["sb"]

n_tests = 0
out = {}
eng = create_engine(load_settings().db_url)

# ---------- 1. corrupted events score distribution ----------
with open("exports/corrupted_events.json", "r", encoding="utf-8") as f:
    corr_ids = [int(k) for k in json.load(f)["events"].keys()]
with eng.connect() as c:
    rows = c.execute(text(
        "SELECT r.score_a, r.score_b FROM results r WHERE r.event_id IN (" +
        ",".join(str(i) for i in corr_ids) + ")")).fetchall()
from collections import Counter
cs = Counter(f"{a}-{b}" for a, b in rows)
n00 = cs.get("0-0", 0)
print(f"1. corrupted excluded with results: {len(rows)}; 0-0: {n00} "
      f"({100*n00/max(len(rows),1):.1f}%); draws: "
      f"{sum(v for k, v in cs.items() if k.split('-')[0]==k.split('-')[1])}; "
      f"top: {cs.most_common(8)}")
out["corrupted_scores"] = {"n": len(rows), "n_00": n00, "top": cs.most_common(12)}

# ---------- 2 & 3. regime chi2 on score categories ----------
def cat(e):
    s = (e["sa"], e["sb"])
    if s == (0, 0):
        return "0-0"
    if s in ((1, 0), (0, 1)):
        return "1-0/0-1"
    if s in ((2, 1), (1, 2)):
        return "2-1/1-2"
    if e["sa"] == e["sb"]:
        return "draw-other"
    return "other"
CATS = ["0-0", "1-0/0-1", "2-1/1-2", "draw-other", "other"]

def regime_chi2(name, evsA, evsB, labelA, labelB):
    global n_tests
    ta = Counter(cat(e) for e in evsA)
    tb = Counter(cat(e) for e in evsB)
    M = np.array([[ta.get(c, 0) for c in CATS], [tb.get(c, 0) for c in CATS]])
    M = M[:, M.sum(axis=0) > 0]
    chi2, p, dof, _ = chi2_contingency(M)
    n_tests += 1
    na, nb = sum(ta.values()), sum(tb.values())
    print(f"  {name}: chi2={chi2:.2f} dof={dof} p={p:.4g}")
    print(f"    {labelA} (n={na}): " + " ".join(f"{c}:{ta.get(c,0)/na:.4f}" for c in CATS))
    print(f"    {labelB} (n={nb}): " + " ".join(f"{c}:{tb.get(c,0)/nb:.4f}" for c in CATS))
    return {"name": name, "chi2": round(float(chi2), 2), "p": float(p),
            "nA": na, "nB": nb,
            "A": {c: ta.get(c, 0) for c in CATS}, "B": {c: tb.get(c, 0) for c in CATS}}

print("\n2. REGIME TESTS (score categories)")
old35 = [e for e in events if e["league"] == REF and e["ts"] < NEWWIN]
rec35 = [e for e in events if e["league"] == REF and e["ts"] >= NEWWIN]
allnew = [e for e in events if e["league"] != REF]
res2 = []
res2.append(regime_chi2("8035 old vs recent", old35, rec35, "old", "recent"))
# verified goals_json only (corruption-immune for total>=1 cells): drop 0-0 col
with eng.connect() as c:
    gjmap = {r[0]: r[1] for r in c.execute(text("SELECT event_id, goals_json FROM results"))}
def verified(e):
    raw = gjmap.get(e["id"])
    if not raw:
        return False
    try:
        gl = json.loads(raw)
        return isinstance(gl, list) and len(gl) == e["total"] and e["total"] >= 1
    except Exception:
        return False
oldv = [e for e in old35 if verified(e)]
recv = [e for e in rec35 if verified(e)]
res2.append(regime_chi2("8035 old vs recent (verified goals, total>=1)",
                        oldv, recv, "old-verif", "rec-verif"))
res2.append(regime_chi2("8035-recent vs ALL-NEW", rec35, allnew, "8035rec", "allnew"))
res2.append(regime_chi2("8035-old vs ALL-NEW", old35, allnew, "8035old", "allnew"))
# old RAW (corrupted restored, only those with odds) vs recent
with eng.connect() as c:
    raw_corr = c.execute(text(
        "SELECT r.score_a, r.score_b FROM results r "
        "WHERE r.event_id IN (" + ",".join(str(i) for i in corr_ids) + ") "
        "AND EXISTS (SELECT 1 FROM odds_snapshots o WHERE o.event_id=r.event_id)")).fetchall()
corr_evs = [{"sa": a, "sb": b} for a, b in raw_corr]
res2.append(regime_chi2("8035 old-RAW(+corrupted) vs recent", old35 + corr_evs, rec35,
                        "old-raw", "recent"))
# mu composition sanity for old vs recent
mo = np.mean([e["lh"] + e["la"] for e in old35 if e["lh"]])
mr = np.mean([e["lh"] + e["la"] for e in rec35 if e["lh"]])
print(f"  mu sanity: old={mo:.3f} recent={mr:.3f}")
out["regime_tests"] = res2

# ---------- 3b. forensics: are old corrupted FT=0-0 events' goals_json corroborated by HT? ----------
print("\n3b. FORENSICS")
with eng.connect() as c:
    fr = c.execute(text(
        "SELECT r.event_id, r.score_a, r.score_b, r.ht_score_a, r.ht_score_b, r.goals_json "
        "FROM results r WHERE r.event_id IN (" + ",".join(str(i) for i in corr_ids) + ")")).fetchall()
n_00gj, n_ht_match, n_ht_00 = 0, 0, 0
for eid, sa, sb, ha, hb, raw in fr:
    if sa == 0 and sb == 0 and raw:
        try:
            gl = json.loads(raw)
        except Exception:
            continue
        if not isinstance(gl, list) or len(gl) == 0:
            continue
        n_00gj += 1
        gh = sum(1 for g in gl if g["minute"] <= 45 and g["team"] == "Home")
        ga = sum(1 for g in gl if g["minute"] <= 45 and g["team"] == "Away")
        if ha is not None and (gh, ga) == (ha, hb):
            n_ht_match += 1
        if (ha, hb) == (0, 0):
            n_ht_00 += 1
print(f"  corrupted FT=0-0 with non-empty goals_json: {n_00gj}; "
      f"HT matches goals_json-derived HT: {n_ht_match}; HT recorded as 0-0: {n_ht_00}")
out["forensics_corrupted00"] = {"n_00_with_goals": n_00gj,
                                "ht_matches_goals": n_ht_match, "ht_00": n_ht_00}
# null vs empty goals_json for 0-0 events by era / league group
with eng.connect() as c:
    gj00 = {r[0]: r[1] for r in c.execute(text(
        "SELECT event_id, goals_json FROM results WHERE score_a=0 AND score_b=0"))}
def gj_kind(eid):
    raw = gj00.get(eid)
    if raw is None:
        return "null"
    try:
        gl = json.loads(raw)
        return "empty-list" if isinstance(gl, list) and len(gl) == 0 else "nonempty"
    except Exception:
        return "unparseable"
for g, evs in [("8035-old", old35), ("8035-recent", rec35), ("ALL-NEW", allnew)]:
    kinds = Counter(gj_kind(e["id"]) for e in evs if e["sa"] == 0 and e["sb"] == 0)
    print(f"  0-0 goals_json kinds {g}: {dict(kinds)}")
    out[f"gj00_{g}"] = dict(kinds)
# null goals_json rate among matches WITH goals (feed-missing base rate) by era
with eng.connect() as c:
    allgj = c.execute(text(
        "SELECT r.event_id, r.goals_json FROM results r JOIN events e ON e.id=r.event_id "
        "WHERE r.score_a + r.score_b >= 1")).fetchall()
nullset = {eid for eid, raw in allgj if raw is None}
for g, evs in [("8035-old", old35), ("8035-recent", rec35), ("ALL-NEW", allnew)]:
    tot = [e for e in evs if e["total"] >= 1]
    nn = sum(1 for e in tot if e["id"] in nullset)
    print(f"  null-goals_json rate among total>=1 {g}: {nn}/{len(tot)} = {nn/len(tot):.4f}")
    out[f"gjnull_rate_{g}"] = nn / len(tot)

# ---------- 4. FTTS by odds bucket ----------
print("\n4. FTTS '1' (home 1X2 odds<=1.5) BY FTTS-ODDS BUCKET")
for e in events:
    x = extra.get(str(e["id"]))
    if x:
        try:
            d = json.loads(x) if isinstance(x, str) else x
            f2 = d.get("FTTS") or {}
            if f2.get("1") and 1 < f2["1"] < 99.5:
                e["ftts1_odds"] = f2["1"]
        except Exception:
            pass
def first_team(e):
    raw = gjmap.get(e["id"])
    if e["total"] == 0:
        return "None"
    if not raw:
        return None
    try:
        gl = json.loads(raw)
        if not isinstance(gl, list) or len(gl) == 0:
            return None
        first = sorted(gl, key=lambda x: (x["minute"], x["homeScore"] + x["awayScore"]))[0]
        return first["team"]
    except Exception:
        return None
res4 = []
for g, sel in [("8035-old", lambda e: e["league"] == REF and e["ts"] < NEWWIN),
               ("8035-recent", lambda e: e["league"] == REF and e["ts"] >= NEWWIN),
               ("CHAMP-NEW", lambda e: e["league"] in CHAMP),
               ("CUP", lambda e: e["league"] in CUP)]:
    for (a, b) in [(1.0, 1.30), (1.30, 1.45), (1.45, 1.70)]:
        bets = []
        for e in events:
            if not sel(e) or e["oh"] > 1.5 or "ftts1_odds" not in e:
                continue
            o = e["ftts1_odds"]
            if not (a <= o < b):
                continue
            ft = first_team(e)
            if ft is None:
                continue
            bets.append((o, ft == "Home"))
        if len(bets) < 50:
            continue
        n = len(bets)
        wr = sum(w for _, w in bets) / n
        profits = np.array([o * w - 1 for o, w in bets])
        roi = float(profits.mean())
        se = profits.std() / math.sqrt(n)
        pv = 2 * norm.sf(abs(roi / se)) if se > 0 else 1.0
        n_tests += 1
        r = {"group": g, "odds_b": f"[{a}-{b})", "n": n, "wr": round(wr, 4),
             "roi": round(roi, 4), "p": float(pv),
             "avg_odds": round(float(np.mean([o for o, _ in bets])), 3)}
        res4.append(r)
        print(f"  {g:12s} ftts[{a}-{b}) n={n:5d} wr={wr:.4f} roi={roi:+.4f} p={pv:.3g}")
out["ftts_buckets"] = res4

# FTTS champ-new walk-forwardish: per-league consistency
print("  per-league FTTS '1' (home<=1.5):")
res4b = []
for l in sorted(CHAMP | CUP):
    bets = []
    for e in events:
        if e["league"] != l or e["oh"] > 1.5 or "ftts1_odds" not in e:
            continue
        ft = first_team(e)
        if ft is None:
            continue
        bets.append((e["ftts1_odds"], ft == "Home"))
    if len(bets) < 50:
        continue
    n = len(bets)
    profits = np.array([o * w - 1 for o, w in bets])
    roi = float(profits.mean())
    se = profits.std() / math.sqrt(n)
    pv = 2 * norm.sf(abs(roi / se)) if se > 0 else 1.0
    n_tests += 1
    r = {"league": l, "n": n, "roi": round(roi, 4), "p": float(pv)}
    res4b.append(r)
    print(f"    {l}: n={n} roi={roi:+.4f} p={pv:.3g}")
out["ftts_per_league"] = res4b

out["n_tests_part5"] = n_tests
with open("exports/wf4_univrefute_part5_rerun.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("\nsaved exports/wf4_univrefute_part5_rerun.json; n_tests part5 =", n_tests)
