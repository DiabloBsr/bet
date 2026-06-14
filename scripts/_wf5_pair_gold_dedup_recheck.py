# -*- coding: utf-8 -*-
"""WF5 — Contre-verification SCORE_DOMINANT_GOLD avec DEDUPLICATION.

Decouverte: 3130 matchs finis 2026-06-10 -> 2026-06-12 ont des lignes events/results
dupliquees (scores identiques) — artefact du scraping multi-ligues de la nuit.
L'audit original (scripts/_wf5_pair_gold_audit.py) les compte 2x => n gonfle, z gonfles.

Ici: dedup par (team_a, team_b, expected_start) en gardant MIN(event.id),
puis recalcul hit rate / ROI / z pour SCORE_DOMINANT_GOLD et SCORE_COMBO_GOLD.
Sortie: exports/wf5_pair_gold_dedup_recheck.json. LECTURE SEULE.
"""
import sys, json, math
from collections import defaultdict, Counter

sys.path.insert(0, ".")
from scraper.config import load_settings
from sqlalchemy import create_engine, text
from scraper.team_gold_data import SCORE_DOMINANT_GOLD, SCORE_COMBO_GOLD

LEAGUE = "InstantLeague-8035"
CUTOFF = "2026-06-06"
engine = create_engine(load_settings().db_url)
corrupted = set(int(k) for k in json.load(open("exports/corrupted_events.json"))["events"].keys())

SQL = """
SELECT e.id, e.team_a, e.team_b, e.expected_start, r.score_a, r.score_b, r.finished_at
FROM events e
JOIN results r ON r.event_id = e.id
JOIN (SELECT event_id, MIN(id) AS sid FROM odds_snapshots GROUP BY event_id) f
     ON f.event_id = e.id
JOIN odds_snapshots os ON os.id = f.sid
WHERE e.competition = :lg AND r.finished_at >= :cut
      AND os.odds_home IS NOT NULL AND os.odds_draw IS NOT NULL AND os.odds_away IS NOT NULL
ORDER BY r.finished_at
"""
with engine.connect() as c:
    rows = c.execute(text(SQL), {"lg": LEAGUE, "cut": CUTOFF}).fetchall()

raw = [dict(id=r[0], ta=r[1], tb=r[2], start=r[3], sa=r[4], sb=r[5], fin=r[6])
       for r in rows if r[0] not in corrupted and r[4] is not None]

# dedup: garder le premier event.id par (ta, tb, expected_start)
seen, OOS = {}, []
for m in sorted(raw, key=lambda x: x["id"]):
    k = (m["ta"], m["tb"], m["start"])
    if k in seen:
        continue
    seen[k] = m["id"]
    OOS.append(m)
OOS.sort(key=lambda x: str(x["fin"]))
n_dups_removed = len(raw) - len(OOS)
print(f"OOS brut={len(raw)}  dedup={len(OOS)}  doublons retires={n_dups_removed}")

n_oos = len(OOS)
score_freq = Counter(f"{m['sa']}-{m['sb']}" for m in OOS)
OOS_pair = defaultdict(list)
for m in OOS:
    OOS_pair[(m["ta"], m["tb"])].append(m)

# extra_markets (snapshot d'ouverture) pour les events retenus des paires flaggees
need = sorted({m["id"] for key in (set(SCORE_DOMINANT_GOLD) | set(SCORE_COMBO_GOLD))
               for m in OOS_pair.get(key, [])})
em = {}
CH = 400
with engine.connect() as c:
    for i in range(0, len(need), CH):
        chunk = need[i:i + CH]
        rs = c.execute(text(
            "SELECT f.event_id, os.extra_markets FROM (SELECT event_id, MIN(id) sid "
            "FROM odds_snapshots WHERE event_id IN ({}) GROUP BY event_id) f "
            "JOIN odds_snapshots os ON os.id = f.sid".format(",".join(map(str, chunk)))
        )).fetchall()
        for ev, rawem in rs:
            try:
                d = json.loads(rawem) if rawem else {}
            except Exception:
                d = {}
            em[ev] = d.get("Score exact") or {}

def wilson_z(p_obs, p0, n):
    if not n or p0 <= 0 or p0 >= 1:
        return None
    return (p_obs - p0) / math.sqrt(p0 * (1 - p0) / n)

def roi_stats(pnls):
    n = len(pnls)
    if n == 0:
        return dict(n=0, roi=None, z=None)
    mean = sum(pnls) / n
    var = sum((x - mean) ** 2 for x in pnls) / max(n - 1, 1)
    sd = math.sqrt(var)
    z = mean / (sd / math.sqrt(n)) if sd > 0 else None
    return dict(n=n, roi=round(mean, 4), z=round(z, 2) if z else None)

def audit_scores(table, label, combo=False):
    hits, tot, pnls = 0, 0, []
    is_rate_w, base_w = 0.0, 0.0
    imp_sum, imp_n = 0.0, 0
    per = []
    for key, d in table.items():
        ms = OOS_pair.get(key, [])
        n = len(ms)
        targets = [d["top1"], d["top2"]] if combo else [d["score"]]
        h = sum(1 for m in ms if f"{m['sa']}-{m['sb']}" in targets)
        hits += h; tot += n
        is_rate_w += (d["combo"] if combo else d["rate"]) * n
        base_w += (sum(score_freq.get(t, 0) for t in targets) / n_oos) * n
        for m in ms:
            cs = em.get(m["id"]) or {}
            cotes = [cs.get(t) for t in targets]
            if all(cotes):
                imp_sum += sum(1 / c for c in cotes); imp_n += 1
                res = f"{m['sa']}-{m['sb']}"
                pnl = sum((c - 1) if res == t else -1.0 for t, c in zip(targets, cotes))
                pnls.append(pnl / len(targets))
        per.append(dict(pair=f"{key[0]} v {key[1]}", target="+".join(targets), n_oos=n,
                        hit_oos=round(h / n, 3) if n else None))
    rs = roi_stats(pnls)
    wr = hits / tot if tot else None
    base = base_w / tot if tot else None
    imp = imp_sum / imp_n if imp_n else None
    return dict(label=label, n_oos=tot, hits=hits,
                hit_rate_oos=round(wr, 4) if wr is not None else None,
                hit_rate_is_weighted=round(is_rate_w / tot, 4) if tot else None,
                baseline_global_weighted=round(base, 4) if base else None,
                market_implied_avg=round(imp, 4) if imp else None,
                z_vs_baseline=round(wilson_z(wr, base, tot), 2) if (wr is not None and base) else None,
                z_vs_implied=round(wilson_z(wr, imp, tot), 2) if (wr is not None and imp) else None,
                roi_oos=rs["roi"], roi_n=rs["n"], roi_z=rs["z"], per_pair=per)

rep = {"cutoff": CUTOFF, "n_oos_raw": len(raw), "n_oos_dedup": n_oos,
       "dups_removed": n_dups_removed, "families": {}}
rep["families"]["SCORE_DOMINANT_GOLD"] = audit_scores(SCORE_DOMINANT_GOLD, "SCORE_DOMINANT_GOLD dedup")
rep["families"]["SCORE_COMBO_GOLD"] = audit_scores(SCORE_COMBO_GOLD, "SCORE_COMBO_GOLD dedup", combo=True)

with open("exports/wf5_pair_gold_dedup_recheck.json", "w", encoding="utf-8") as f:
    json.dump(rep, f, ensure_ascii=False, indent=1)

for name, fam in rep["families"].items():
    print(f"\n{name}: n_oos={fam['n_oos']} hits={fam['hits']} hit={fam['hit_rate_oos']} "
          f"is_claim={fam['hit_rate_is_weighted']} base={fam['baseline_global_weighted']} "
          f"implied={fam['market_implied_avg']} z_base={fam['z_vs_baseline']} "
          f"z_implied={fam['z_vs_implied']} roi={fam['roi_oos']} (n={fam['roi_n']}, z={fam['roi_z']})")
print("\nJSON: exports/wf5_pair_gold_dedup_recheck.json")
