# -*- coding: utf-8 -*-
"""WF5 recheck — contre-verification independante du verdict PAIR_AWAY_GOLD=MORT.

Verifie sans reutiliser le code de _wf5_pair_gold_audit.py :
 1. settlement : score vs goals_json (echantillon), away win = score_b > score_a
 2. filtre competition + exclusions corrupted + doublons d'events
 3. cotes d'ouverture = MIN(id) et fetched_at <= expected_start (pas de leak post-KO)
 4. fenetre conservatrice (finished_at >= 2026-06-06) ET fenetre large (matches[3225:])
 5. per-pair OOS + bootstrap du ROI (10k resamples) pour l'IC
Sortie: exports/wf5_pair_away_gold_recheck.json
"""
import sys, json, math, random
from collections import defaultdict

sys.path.insert(0, ".")
from scraper.config import load_settings
from sqlalchemy import create_engine, text
from scraper.team_gold_data import PAIR_AWAY_GOLD

LEAGUE = "InstantLeague-8035"
engine = create_engine(load_settings().db_url)
corrupted = set(int(k) for k in json.load(open("exports/corrupted_events.json"))["events"].keys())
print(f"corrupted ids charges: {len(corrupted)}")

pairs = set(PAIR_AWAY_GOLD.keys())

# ---- 1. tous les matchs finis de la ligue, cotes d'ouverture, ordre finished_at
SQL = """
SELECT e.id, e.team_a, e.team_b, e.expected_start, r.score_a, r.score_b, r.finished_at,
       os.odds_home, os.odds_draw, os.odds_away, os.captured_at, os.status
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

all_rows = [dict(id=r[0], ta=r[1], tb=r[2], xs=r[3], sa=r[4], sb=r[5], fin=r[6],
                 oh=r[7], od=r[8], oa=r[9], ofetch=r[10], ostatus=r[11]) for r in rows]
clean = [m for m in all_rows if m["id"] not in corrupted
         and m["sa"] is not None and m["sb"] is not None
         and m["oh"] and m["od"] and m["oa"]]
print(f"finis ligue={len(all_rows)}  propres+cotes={len(clean)}")

# ---- 2. doublons (meme paire + meme expected_start)
seen, dups = {}, []
for m in clean:
    k = (m["ta"], m["tb"], str(m["xs"]))
    if k in seen:
        dups.append((seen[k], m["id"]))
    seen[k] = m["id"]
print(f"doublons (paire+expected_start): {len(dups)} -> {dups[:5]}")

# ---- 3. leak check : snapshot d'ouverture posterieur au coup d'envoi ?
leaks = [m["id"] for m in clean if m["ofetch"] and m["xs"] and str(m["ofetch"]) > str(m["xs"])]
print(f"snapshots d'ouverture APRES expected_start: {len(leaks)} / {len(clean)}")
if leaks[:5]:
    print("  exemples:", leaks[:5])
from collections import Counter as _C
print("status du snapshot d'ouverture:", dict(_C(m["ostatus"] for m in clean)))

# ---- 4. settlement check sur les matchs des paires OR away (goals_json)
pair_all = [m for m in clean if (m["ta"], m["tb"]) in pairs]
ids = [m["id"] for m in pair_all]
gj = {}
CH = 400
with engine.connect() as c:
    for i in range(0, len(ids), CH):
        chunk = ids[i:i + CH]
        rs = c.execute(text(
            "SELECT event_id, score_a, score_b, goals_json FROM results WHERE event_id IN ({})"
            .format(",".join(str(x) for x in chunk)))).fetchall()
        for ev, sa, sb, g in rs:
            gj[ev] = (sa, sb, g)
bad_settle, checked = [], 0
for m in pair_all:
    sa, sb, g = gj[m["id"]]
    if g:
        try:
            goals = json.loads(g)
        except Exception:
            continue
        if isinstance(goals, list):
            ga = sum(1 for x in goals if str(x.get("team", x.get("side", ""))).lower() in ("a", "home", str(m["ta"]).lower()))
            gb = sum(1 for x in goals if str(x.get("team", x.get("side", ""))).lower() in ("b", "away", str(m["tb"]).lower()))
            checked += 1
            if (ga, gb) != (sa, sb):
                bad_settle.append(dict(id=m["id"], score=(sa, sb), from_goals=(ga, gb)))
print(f"settlement via goals_json: {checked} verifies, {len(bad_settle)} incoherents")
for b in bad_settle[:5]:
    print("  ", b)

# ---- 5. mesure OOS, deux fenetres
def measure(ms, label):
    pnls, w = [], 0
    imp = 0.0
    for m in ms:
        win = m["sb"] > m["sa"]
        w += win
        pnls.append((m["oa"] - 1.0) if win else -1.0)
        imp += 1.0 / m["oa"]
    n = len(pnls)
    if n == 0:
        return dict(label=label, n=0)
    mean = sum(pnls) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in pnls) / max(n - 1, 1))
    z = mean / (sd / math.sqrt(n)) if sd > 0 else None
    # bootstrap IC 90%
    random.seed(42)
    boots = []
    for _ in range(10000):
        s = [pnls[random.randrange(n)] for _ in range(n)]
        boots.append(sum(s) / n)
    boots.sort()
    lo, hi = boots[int(0.05 * len(boots))], boots[int(0.95 * len(boots))]
    return dict(label=label, n=n, wins=w, wr=round(w / n, 4),
                implied_wr=round(imp / n, 4), roi=round(mean, 4),
                z=round(z, 2) if z else None,
                roi_ci90=[round(lo, 4), round(hi, 4)],
                p_roi_pos=round(sum(1 for b in boots if b > 0) / len(boots), 3))

CUTOFF = "2026-06-06"
oos_cons = [m for m in pair_all if str(m["fin"]) >= CUTOFF]
frontier_id_order = clean[3225:]
oos_large = [m for m in frontier_id_order if (m["ta"], m["tb"]) in pairs]

res_cons = measure(oos_cons, "conservatrice (fin >= 2026-06-06)")
res_large = measure(oos_large, "large (apres 3225e match propre)")
res_all = measure(pair_all, "TOUT l'historique (IS+OOS, contamine)")

for r in (res_cons, res_large, res_all):
    print(r)

# ---- 6. per-pair OOS (fenetre conservatrice)
bypair = defaultdict(list)
for m in oos_cons:
    bypair[(m["ta"], m["tb"])].append(m)
per = []
for k, d in PAIR_AWAY_GOLD.items():
    ms = bypair.get(k, [])
    n = len(ms)
    w = sum(1 for m in ms if m["sb"] > m["sa"])
    pnl = sum((m["oa"] - 1) if m["sb"] > m["sa"] else -1.0 for m in ms)
    per.append(dict(pair=f"{k[0]} v {k[1]}", n=n, wr_is=d["win"], n_is=d["n"], roi_is=d["roi"],
                    wr_oos=round(w / n, 3) if n else None,
                    roi_oos=round(pnl / n, 3) if n else None))
per.sort(key=lambda x: -(x["n"] or 0))
print("\nper-pair OOS (conservatrice):")
for p in per:
    print(f"  {p['pair']:38s} n={p['n']:3d}  wr_is={p['wr_is']:.3f}(n={p['n_is']})  "
          f"wr_oos={p['wr_oos']}  roi_is=+{p['roi_is']:.0%}  roi_oos={p['roi_oos']}")

# combien de paires restent positives OOS ?
pos = sum(1 for p in per if p["roi_oos"] is not None and p["roi_oos"] > 0)
print(f"\npaires ROI OOS > 0 : {pos}/{len(per)}")

out = dict(corrupted_loaded=len(corrupted), clean=len(clean), dups=len(dups),
           opening_after_kickoff=len(leaks), settle_checked=checked,
           settle_mismatch=len(bad_settle),
           window_conservative=res_cons, window_large=res_large, window_all=res_all,
           per_pair=per, pairs_positive_oos=pos)
with open("exports/wf5_pair_away_gold_recheck.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("\nJSON: exports/wf5_pair_away_gold_recheck.json")
