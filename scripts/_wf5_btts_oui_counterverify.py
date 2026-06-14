# -*- coding: utf-8 -*-
"""WF5 — CONTRE-VERIFICATION independante du verdict 'BTTS_OUI_GOLD = MORT'.

Re-implementation from scratch (sans reutiliser audit_btts):
 1. Integrite: doublons results / snapshots, format finished_at, settlement
    croise score_a/score_b vs goals_json.
 2. OOS (>= cutoff) pour les 20 paires BTTS_OUI_GOLD: hit/ROI avec et sans
    filtre min_cote_h (semantique production: odds_home >= min_cote_h).
 3. Baseline marche: flat-bet G/NG Oui sur TOUS les matchs OOS de la ligue
    (marge du marche) -> le signal bat-il le hasard ?
 4. Recalcul in-sample des 20 paires (effet de selection).
 5. Sensibilite cutoff: 2026-06-06 / 2026-06-07 / 2026-06-08.
Sortie: exports/wf5_btts_oui_counterverify.json. LECTURE SEULE.
"""
import sys, json, math
from collections import defaultdict

sys.path.insert(0, ".")
from scraper.config import load_settings
from sqlalchemy import create_engine, text
from scraper.team_gold_data import BTTS_OUI_GOLD

LEAGUE = "InstantLeague-8035"
IS_N = 3225

engine = create_engine(load_settings().db_url)
corrupted = set(int(k) for k in json.load(open("exports/corrupted_events.json"))["events"].keys())

# ---------------------------------------------------------------- integrite
with engine.connect() as c:
    dup_res = c.execute(text(
        "SELECT COUNT(*) FROM (SELECT event_id FROM results GROUP BY event_id HAVING COUNT(*)>1)"
    )).scalar()
    n_res = c.execute(text(
        "SELECT COUNT(*) FROM results r JOIN events e ON e.id=r.event_id WHERE e.competition=:lg"),
        {"lg": LEAGUE}).scalar()
print(f"[integrite] results dupliques (toutes ligues): {dup_res} ; results {LEAGUE}: {n_res}")

# ---------------------------------------------------------------- load (colonnes minimales)
SQL = """
SELECT e.id, e.team_a, e.team_b, r.score_a, r.score_b, r.finished_at,
       os.odds_home
FROM events e
JOIN results r ON r.event_id = e.id
JOIN (SELECT event_id, MIN(id) AS sid FROM odds_snapshots GROUP BY event_id) f
     ON f.event_id = e.id
JOIN odds_snapshots os ON os.id = f.sid
WHERE e.competition = :lg
ORDER BY r.finished_at
"""
with engine.connect() as c:
    rows = c.execute(text(SQL), {"lg": LEAGUE}).fetchall()

matches = [dict(id=r[0], ta=r[1], tb=r[2], sa=r[3], sb=r[4], fin=str(r[5]), oh=r[6])
           for r in rows
           if r[0] not in corrupted and r[3] is not None and r[4] is not None and r[6]]
print(f"matchs propres: {len(matches)} ; fin format ex: {matches[0]['fin']!r} -> {matches[-1]['fin']!r}")
bad_fmt = sum(1 for m in matches if len(m["fin"]) < 10 or m["fin"][4] != "-")
print(f"finished_at au format non ISO: {bad_fmt}")

IS = matches[:IS_N]

# ------------------------------------------------- extra markets G/NG : OOS global (chunks)
CUT0 = "2026-06-06"
oos_all = [m for m in matches if m["fin"] >= CUT0]
ids = sorted(m["id"] for m in oos_all)
gng = {}   # event_id -> (oui, non)
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
            if not raw:
                continue
            try:
                g = (json.loads(raw).get("G/NG") or {})
            except Exception:
                continue
            if "Oui" in g and "Non" in g:
                try:
                    gng[ev] = (float(g["Oui"]), float(g["Non"]))
                except Exception:
                    pass
print(f"OOS>={CUT0}: {len(oos_all)} matchs, G/NG dispo pour {len(gng)}")

# ------------------------------------------------- settlement croise via goals_json (picks)
pick_pairs = set(BTTS_OUI_GOLD)
pick_ids = [m["id"] for m in oos_all if (m["ta"], m["tb"]) in pick_pairs]
mismatch, checked = 0, 0
with engine.connect() as c:
    for i in range(0, len(pick_ids), CH):
        chunk = pick_ids[i:i + CH]
        rs = c.execute(text(
            "SELECT event_id, score_a, score_b, goals_json FROM results WHERE event_id IN ({})"
            .format(",".join(map(str, chunk))))).fetchall()
        for ev, sa, sb, gj in rs:
            if not gj:
                continue
            try:
                goals = json.loads(gj)
            except Exception:
                continue
            if not isinstance(goals, list):
                continue
            ga = sum(1 for g in goals if isinstance(g, dict) and g.get("team") in ("a", "home", "A"))
            gb = sum(1 for g in goals if isinstance(g, dict) and g.get("team") in ("b", "away", "B"))
            if ga + gb == 0 and (sa + sb) > 0:
                continue  # goals_json vide/non parsable pour ce schema -> ignore
            checked += 1
            if (ga, gb) != (sa, sb):
                mismatch += 1
print(f"[settlement] goals_json vs score: {checked} verifies, {mismatch} mismatches")

# ------------------------------------------------- evaluation principale
def wilson_ci(h, n, z=1.96):
    if n == 0:
        return None
    p = h / n
    d = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / d
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(ctr - w, 4), round(ctr + w, 4))

def evaluate(ms, use_filter):
    picks = []
    for m in ms:
        key = (m["ta"], m["tb"])
        if key not in BTTS_OUI_GOLD:
            continue
        if use_filter and m["oh"] < BTTS_OUI_GOLD[key]["min_cote_h"]:
            continue
        if m["id"] not in gng:
            continue
        oui = gng[m["id"]][0]
        win = m["sa"] >= 1 and m["sb"] >= 1
        picks.append(((oui - 1) if win else -1.0, win, oui))
    n = len(picks)
    if n == 0:
        return dict(n=0)
    hits = sum(1 for p in picks if p[1])
    pnl = [p[0] for p in picks]
    roi = sum(pnl) / n
    mean = roi
    sd = math.sqrt(sum((x - mean) ** 2 for x in pnl) / max(n - 1, 1))
    return dict(n=n, hits=hits, hit_rate=round(hits / n, 4), ci95=wilson_ci(hits, n),
                avg_oui=round(sum(p[2] for p in picks) / n, 3),
                breakeven=round(1 / (sum(p[2] for p in picks) / n), 4),
                roi=round(roi, 4),
                roi_z=round(mean / (sd / math.sqrt(n)), 2) if sd > 0 else None)

report = {"settlement_checked": checked, "settlement_mismatch": mismatch,
          "dup_results": dup_res, "cutoffs": {}}

for cut in ("2026-06-06", "2026-06-07", "2026-06-08"):
    oos = [m for m in oos_all if m["fin"] >= cut]
    # baseline marche: flat bet Oui sur TOUT
    flat = []
    bts_h = 0
    for m in oos:
        if m["id"] in gng:
            win = m["sa"] >= 1 and m["sb"] >= 1
            bts_h += win
            flat.append((gng[m["id"]][0] - 1) if win else -1.0)
    nf = len(flat)
    flat_roi = sum(flat) / nf if nf else None
    rep = dict(
        n_oos_league=len(oos),
        baseline_btts=round(bts_h / nf, 4) if nf else None,
        flat_bet_oui_all=dict(n=nf, roi=round(flat_roi, 4) if nf else None),
        signal_with_filter=evaluate(oos, True),
        signal_no_filter=evaluate(oos, False),
    )
    report["cutoffs"][cut] = rep
    print(f"\n=== cutoff {cut} (n ligue={len(oos)}) ===")
    print(f"  baseline BTTS={rep['baseline_btts']}  flat-Oui ALL: n={nf} roi={rep['flat_bet_oui_all']['roi']}")
    print(f"  AVEC filtre : {rep['signal_with_filter']}")
    print(f"  SANS filtre : {rep['signal_no_filter']}")

# ------------------------------------------------- recalcul in-sample (effet de selection)
is_pair = defaultdict(list)
for m in IS:
    is_pair[(m["ta"], m["tb"])].append(m)
tot, hits = 0, 0
per = {}
for key, d in BTTS_OUI_GOLD.items():
    ms = is_pair.get(key, [])
    h = sum(1 for m in ms if m["sa"] >= 1 and m["sb"] >= 1)
    tot += len(ms); hits += h
    per[f"{key[0]} v {key[1]}"] = dict(n_is_recalc=len(ms), hit_is_recalc=round(h / len(ms), 3) if ms else None,
                                       n_hard=d["n"], rate_hard=d["rate"])
print(f"\n[in-sample recalc] 20 paires: hit IS recalcule = {hits}/{tot} = {hits/tot:.4f} "
      f"(hardcode pondere ~0.879) -> OOS ~0.58")
report["in_sample_recalc"] = dict(n=tot, hits=hits, rate=round(hits / tot, 4), per_pair=per)

with open("exports/wf5_btts_oui_counterverify.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=1)
print("\nJSON: exports/wf5_btts_oui_counterverify.json")
