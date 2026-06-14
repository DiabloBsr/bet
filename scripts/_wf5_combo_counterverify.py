# -*- coding: utf-8 -*-
"""WF5 — Contre-verification independante du verdict SCORE_COMBO_GOLD=MORT.

Code path independant de _wf5_pair_gold_audit.py :
 1. recompute OOS hit/ROI par requete SQL fraiche par paire (filtre ligue + corrupted + cutoff)
 2. spot-check settlement (scores + cotes CS, pnl a la main) sur 8 matchs
 3. sanity orientation score "a-b" (favori 1X2 vs cotes CS home/away)
 4. borne ORACLE: meilleurs 2 scores refit sur l'OOS lui-meme (plafond atteignable)
 5. marge du marche CS (sum 1/cote)
 6. deux modeles de mise: 0.5u sur chaque score (1u total) ET 1u sur chaque (2u total)
 7. collision noms d'equipes entre ligues (le filtre competition est-il indispensable?)
 8. derive temporelle: hit par semaine OOS

Sortie: exports/wf5_combo_counterverify.json. LECTURE SEULE.
"""
import sys, json, math
from collections import Counter, defaultdict

sys.path.insert(0, ".")
from scraper.config import load_settings
from sqlalchemy import create_engine, text
from scraper.team_gold_data import SCORE_COMBO_GOLD

LEAGUE = "InstantLeague-8035"
CUTOFF = "2026-06-06"
engine = create_engine(load_settings().db_url)
corrupted = set(int(k) for k in json.load(open("exports/corrupted_events.json"))["events"].keys())

out = {}

# ---- 7. collision de noms d'equipes entre ligues -----------------------------
teams = sorted({t for k in SCORE_COMBO_GOLD for t in k})
with engine.connect() as c:
    rows = c.execute(text(
        "SELECT DISTINCT competition, team_a FROM events WHERE team_a IN ({})".format(
            ",".join("'{}'".format(t.replace("'", "''")) for t in teams))
    )).fetchall()
leagues_with_same_teams = sorted({r[0] for r in rows})
out["team_name_collision"] = {
    "leagues_sharing_team_names": leagues_with_same_teams,
    "filter_required": len(leagues_with_same_teams) > 1,
}

# ---- 1. recompute OOS par paire (SQL fraiche) ---------------------------------
SQL_PAIR = text("""
SELECT e.id, r.score_a, r.score_b, r.finished_at
FROM events e
JOIN results r ON r.event_id = e.id
WHERE e.competition = :lg AND e.team_a = :ta AND e.team_b = :tb
  AND r.finished_at >= :cut AND r.score_a IS NOT NULL
ORDER BY r.finished_at
""")
per_pair, all_matches = [], []
with engine.connect() as c:
    for (ta, tb), d in SCORE_COMBO_GOLD.items():
        rows = c.execute(SQL_PAIR, {"lg": LEAGUE, "ta": ta, "tb": tb, "cut": CUTOFF}).fetchall()
        ms = [dict(id=r[0], sa=r[1], sb=r[2], fin=str(r[3]), pair=(ta, tb),
                   targets=[d["top1"], d["top2"]]) for r in rows if r[0] not in corrupted]
        h = sum(1 for m in ms if "{}-{}".format(m["sa"], m["sb"]) in m["targets"])
        per_pair.append(dict(pair="{} v {}".format(ta, tb), n=len(ms), hits=h,
                             targets=d["top1"] + "+" + d["top2"], combo_is=d["combo"]))
        all_matches += ms

n_tot = sum(p["n"] for p in per_pair)
h_tot = sum(p["hits"] for p in per_pair)
out["recompute_oos"] = dict(n=n_tot, hits=h_tot,
                            hit_rate=round(h_tot / n_tot, 4) if n_tot else None,
                            per_pair=per_pair)

# ---- cotes CS d'ouverture pour ces events (chunks) ----------------------------
ids = sorted(m["id"] for m in all_matches)
cs_by_ev = {}
CH = 300
with engine.connect() as c:
    for i in range(0, len(ids), CH):
        chunk = ids[i:i + CH]
        rs = c.execute(text(
            "SELECT f.event_id, os.extra_markets FROM (SELECT event_id, MIN(id) sid "
            "FROM odds_snapshots WHERE event_id IN ({}) GROUP BY event_id) f "
            "JOIN odds_snapshots os ON os.id = f.sid".format(",".join(map(str, chunk)))
        )).fetchall()
        for ev, raw in rs:
            try:
                em = json.loads(raw) if raw else {}
            except Exception:
                em = {}
            cs_by_ev[ev] = em.get("Score exact") or {}

# ---- 5. marge du marche CS -----------------------------------------------------
margins = []
for ev, cs in cs_by_ev.items():
    if len(cs) >= 10:
        margins.append(sum(1.0 / v for v in cs.values() if v and v > 1.0))
out["cs_market_margin"] = dict(n=len(margins),
                               avg_book=round(sum(margins) / len(margins), 4) if margins else None)

# ---- 6. ROI deux modeles de mise ----------------------------------------------
pnl_split, pnl_full = [], []   # 0.5u chaque (1u total) / 1u chaque (2u total, normalise /2)
miss_odds = 0
for m in all_matches:
    cs = cs_by_ev.get(m["id"], {})
    cotes = [cs.get(t) for t in m["targets"]]
    if not all(cotes):
        miss_odds += 1
        continue
    res = "{}-{}".format(m["sa"], m["sb"])
    raw = [((c - 1.0) if res == t else -1.0) for t, c in zip(m["targets"], cotes)]
    pnl_split.append(sum(x * 0.5 for x in raw))          # 1u total
    pnl_full.append(sum(raw) / 2.0)                      # identique math., garde les 2 noms
def stats(p):
    n = len(p)
    if not n:
        return dict(n=0)
    mean = sum(p) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in p) / max(n - 1, 1))
    return dict(n=n, roi=round(mean, 4), z=round(mean / (sd / math.sqrt(n)), 2) if sd else None)
out["roi_models"] = dict(split_half_unit=stats(pnl_split), missing_odds=miss_odds)

# pari 1u UNIQUEMENT top1 (residu top1?) et 1u UNIQUEMENT top2
for label, idx in (("top1_only", 0), ("top2_only", 1)):
    pl = []
    for m in all_matches:
        cs = cs_by_ev.get(m["id"], {})
        c_ = cs.get(m["targets"][idx])
        if not c_:
            continue
        res = "{}-{}".format(m["sa"], m["sb"])
        pl.append((c_ - 1.0) if res == m["targets"][idx] else -1.0)
    out["roi_models"][label] = stats(pl)

# ---- 3. sanity orientation: favori 1X2 vs cotes CS ----------------------------
SQL_OR = text("""
SELECT e.id, os.odds_home, os.odds_away, os.extra_markets
FROM events e
JOIN (SELECT event_id, MIN(id) sid FROM odds_snapshots GROUP BY event_id) f ON f.event_id = e.id
JOIN odds_snapshots os ON os.id = f.sid
WHERE e.competition = :lg AND os.odds_home <= 1.45
LIMIT 60
""")
ok_or, bad_or = 0, 0
with engine.connect() as c:
    for ev, oh, oa, raw in c.execute(SQL_OR, {"lg": LEAGUE}).fetchall():
        try:
            cs = (json.loads(raw) or {}).get("Score exact") or {}
        except Exception:
            cs = {}
        if "2-0" in cs and "0-2" in cs:
            # gros favori home => 2-0 doit coter moins que 0-2
            ok_or += cs["2-0"] < cs["0-2"]
            bad_or += cs["2-0"] >= cs["0-2"]
out["score_orientation_check"] = dict(ok=ok_or, bad=bad_or)

# ---- 2. spot-check settlement (8 premiers matchs avec cotes) -------------------
spots = []
for m in all_matches[:50]:
    cs = cs_by_ev.get(m["id"], {})
    if all(cs.get(t) for t in m["targets"]) and len(spots) < 8:
        res = "{}-{}".format(m["sa"], m["sb"])
        spots.append(dict(event=m["id"], pair="{} v {}".format(*m["pair"]), fin=m["fin"],
                          score=res, targets=m["targets"],
                          cotes=[cs[t] for t in m["targets"]],
                          hit=res in m["targets"]))
out["settlement_spot_checks"] = spots

# ---- 4. borne ORACLE refit sur OOS ---------------------------------------------
oracle_hits, oracle_n = 0, 0
by_pair = defaultdict(list)
for m in all_matches:
    by_pair[m["pair"]].append("{}-{}".format(m["sa"], m["sb"]))
oracle_rows = []
for pair, scores in by_pair.items():
    cnt = Counter(scores)
    top2 = [s for s, _ in cnt.most_common(2)]
    h = sum(1 for s in scores if s in top2)
    oracle_hits += h; oracle_n += len(scores)
    oracle_rows.append(dict(pair="{} v {}".format(*pair), n=len(scores),
                            oracle_top2="+".join(top2), oracle_hit=round(h / len(scores), 3)))
out["oracle_refit_oos"] = dict(n=oracle_n, hit_rate=round(oracle_hits / oracle_n, 4) if oracle_n else None,
                               note="plafond: top2 choisis SUR l'OOS lui-meme (triche volontaire)",
                               per_pair=oracle_rows)

# ---- 8. derive temporelle -------------------------------------------------------
byday = defaultdict(lambda: [0, 0])
for m in all_matches:
    day = m["fin"][:10]
    byday[day][1] += 1
    byday[day][0] += "{}-{}".format(m["sa"], m["sb"]) in m["targets"]
out["hit_by_day"] = {d: dict(hits=v[0], n=v[1]) for d, v in sorted(byday.items())}

with open("exports/wf5_combo_counterverify.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)

print(json.dumps({k: v for k, v in out.items() if k not in ("recompute_oos", "oracle_refit_oos", "hit_by_day")},
                 ensure_ascii=False, indent=1))
print("recompute_oos:", out["recompute_oos"]["n"], "hits", out["recompute_oos"]["hits"],
      "rate", out["recompute_oos"]["hit_rate"])
print("oracle_refit_oos:", out["oracle_refit_oos"]["hit_rate"])
print("hit_by_day:", {d: "{}/{}".format(v["hits"], v["n"]) for d, v in out["hit_by_day"].items()})
