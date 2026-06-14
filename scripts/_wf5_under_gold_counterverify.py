# -*- coding: utf-8 -*-
"""WF5 — Contre-verification du verdict OVERFIT sur UNDER_GOLD.

Verifie independamment de _wf5_pair_gold_audit.py :
  1. Reconstruction OOS (filtre competition, exclusions corrupted, cotes d'ouverture).
  2. Settlement: score_a+score_b vs goals_json (corruption residuelle ?).
  3. Inventaire EXHAUSTIF des marches totaux dans extra_markets (la ligne U2.5 existe-t-elle
     sous un autre nom ? '+/-' a-t-il d'autres lignes ?).
  4. Bancabilite: pricing du meilleur proxy (Total de buts 0/1/2 combine, lignes +/- voisines),
     marge du marche, ROI realise OOS.
  5. Sensibilite cutoff: 2026-06-05 / 06 / 07 + frontiere 3225e match.
Sortie: exports/wf5_under_gold_counterverify.json. LECTURE SEULE.
"""
import sys, json, math
from collections import Counter, defaultdict

sys.path.insert(0, ".")
from scraper.config import load_settings
from sqlalchemy import create_engine, text
from scraper.team_gold_data import UNDER_GOLD

LEAGUE = "InstantLeague-8035"
engine = create_engine(load_settings().db_url)
corrupted = set(int(k) for k in json.load(open("exports/corrupted_events.json"))["events"].keys())

report = {}

# ---------------------------------------------------------------- 1. load
SQL = """
SELECT e.id, e.team_a, e.team_b, r.score_a, r.score_b, r.finished_at, r.goals_json,
       os.odds_home, os.odds_draw, os.odds_away
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

all_clean = [dict(id=r[0], ta=r[1], tb=r[2], sa=r[3], sb=r[4], fin=str(r[5]), gj=r[6],
                  oh=r[7], od=r[8], oa=r[9])
             for r in rows if r[0] not in corrupted and r[3] is not None
             and r[7] and r[8] and r[9]]
print(f"matchs propres: {len(all_clean)}")
frontier_fin = all_clean[3224]["fin"] if len(all_clean) > 3224 else None
print(f"3225e match propre fini le: {frontier_fin}")

under_pairs = set(UNDER_GOLD.keys())

def settle_check(ms):
    """sa+sb vs len(goals_json) — settlement residuel corrompu ?"""
    bad = []
    for m in ms:
        if m["gj"]:
            try:
                g = json.loads(m["gj"])
                ng = len(g) if isinstance(g, list) else None
            except Exception:
                ng = None
            if ng is not None and ng != m["sa"] + m["sb"]:
                bad.append(dict(id=m["id"], score=f"{m['sa']}-{m['sb']}", goals_json_len=ng))
    return bad

def audit(cutoff):
    oos = [m for m in all_clean if m["fin"] >= cutoff]
    sel = [m for m in oos if (m["ta"], m["tb"]) in under_pairs]
    n = len(sel)
    hits = sum(1 for m in sel if m["sa"] + m["sb"] <= 2)
    base = sum(1 for m in oos if m["sa"] + m["sb"] <= 2) / len(oos)
    wr = hits / n if n else None
    z = (wr - base) / math.sqrt(base * (1 - base) / n) if n else None
    return dict(cutoff=cutoff, n_oos_total=len(oos), n_sel=n, hits=hits,
                hit_rate=round(wr, 4) if wr is not None else None,
                baseline=round(base, 4), z=round(z, 2) if z is not None else None,
                settlement_mismatches=settle_check(sel))

report["sensitivity"] = [audit(c) for c in
                         ["2026-06-05", "2026-06-06", "2026-06-07", frontier_fin]]
for s in report["sensitivity"]:
    print(s["cutoff"], "n_sel=", s["n_sel"], "hit=", s["hit_rate"], "base=", s["baseline"],
          "z=", s["z"], "settle_bad=", len(s["settlement_mismatches"]))

# ---------------------------------------------------------------- 2. marches dispo
CUTOFF = "2026-06-06"
oos = [m for m in all_clean if m["fin"] >= CUTOFF]
sel = [m for m in oos if (m["ta"], m["tb"]) in under_pairs]
sel_ids = sorted(m["id"] for m in sel)
print(f"\nevents UNDER_GOLD OOS: {len(sel_ids)}")

em_by_ev = {}
CH = 300
with engine.connect() as c:
    for i in range(0, len(sel_ids), CH):
        chunk = sel_ids[i:i + CH]
        rs = c.execute(text(
            "SELECT f.event_id, os.extra_markets FROM (SELECT event_id, MIN(id) sid "
            "FROM odds_snapshots WHERE event_id IN ({}) GROUP BY event_id) f "
            "JOIN odds_snapshots os ON os.id = f.sid".format(",".join(str(x) for x in chunk))
        )).fetchall()
        for ev, raw in rs:
            try:
                em_by_ev[ev] = json.loads(raw) if raw else {}
            except Exception:
                em_by_ev[ev] = {}

mkt_names = Counter()
subkeys = defaultdict(Counter)
for ev, em in em_by_ev.items():
    if not isinstance(em, dict):
        continue
    for name, sub in em.items():
        mkt_names[name] += 1
        if isinstance(sub, dict):
            for k in sub:
                subkeys[name][k] += 1

print("\n=== marches presents (events UNDER_GOLD OOS) ===")
for name, cnt in mkt_names.most_common():
    print(f"  {name!r}: {cnt}")
report["markets_present"] = {k: v for k, v in mkt_names.most_common()}

# tous les sous-marches contenant un total / ligne
total_like = {}
for name in mkt_names:
    low = name.lower()
    if any(t in low for t in ["total", "+/-", "but", "over", "under", "2.5", "3.5", "1.5"]):
        total_like[name] = dict(subkeys[name].most_common())
print("\n=== sous-cles des marches 'total-like' ===")
print(json.dumps(total_like, ensure_ascii=False, indent=1))
report["total_like_markets"] = total_like

# ---------------------------------------------------------------- 3. bancabilite
def fnum(x):
    try:
        return float(x)
    except Exception:
        return None

# 3a. ligne +/- : quelles lignes existent, marge ?
pm_lines = Counter()
for ev, em in em_by_ev.items():
    pm = em.get("+/-")
    if isinstance(pm, dict):
        for k in pm:
            pm_lines[k] += 1
report["plus_minus_lines"] = dict(pm_lines.most_common())
print("\nlignes '+/-':", dict(pm_lines.most_common()))

# 3b. proxy 'Total de buts' (exact): U2.5 synthetique = back 0,1,2 (1u reparti).
#     marge du marche complet + ROI realise.
synth = dict(n=0, pnl=[], margins=[], implied_u25=[], hits=0)
for m in sel:
    em = em_by_ev.get(m["id"]) or {}
    tb = em.get("Total de buts")
    if not isinstance(tb, dict):
        continue
    cotes = {k: fnum(v) for k, v in tb.items()}
    c0, c1, c2 = cotes.get("0"), cotes.get("1"), cotes.get("2")
    if not (c0 and c1 and c2):
        continue
    full = [v for v in cotes.values() if v and v > 1.0]
    margin = sum(1 / v for v in full) if full else None
    inv = 1 / c0 + 1 / c1 + 1 / c2
    # mise repartie pour payout identique ~ dutching: stake_i = (1/c_i)/inv, payout = 1/inv
    tot = m["sa"] + m["sb"]
    win = tot <= 2
    payout = (1 / inv) if win else 0.0
    synth["n"] += 1
    synth["hits"] += win
    synth["pnl"].append(payout - 1.0)
    if margin:
        synth["margins"].append(margin)
        synth["implied_u25"].append(inv / margin)  # de-vig proportionnel
n = synth["n"]
if n:
    pnl = synth["pnl"]
    mean = sum(pnl) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in pnl) / max(n - 1, 1))
    z = mean / (sd / math.sqrt(n)) if sd else None
    report["synthetic_u25_dutching"] = dict(
        n=n, hit_rate=round(synth["hits"] / n, 4), roi=round(mean, 4),
        roi_z=round(z, 2) if z else None,
        avg_market_margin=round(sum(synth["margins"]) / len(synth["margins"]), 4) if synth["margins"] else None,
        avg_payout_odds=round(sum(1 / (1 / c) for c in [sum(1/x for x in [1])]) , 4) if False else None,
        avg_implied_u25_devig=round(sum(synth["implied_u25"]) / len(synth["implied_u25"]), 4) if synth["implied_u25"] else None,
    )
    print("\nU2.5 synthetique (dutching Total de buts 0/1/2):")
    print(json.dumps(report["synthetic_u25_dutching"], indent=1))
else:
    report["synthetic_u25_dutching"] = dict(n=0)
    print("\nU2.5 synthetique: aucun event avec 'Total de buts' 0/1/2")

# 3c. si une ligne < 2.5 (ou < 3 / < 2) existe dans '+/-', ROI direct
for line_key, want_max in [("< 2.5", 2), ("< 1.5", 1), ("< 3.5", 3)]:
    pnl, nn, hh = [], 0, 0
    for m in sel:
        em = em_by_ev.get(m["id"]) or {}
        pm = em.get("+/-")
        if isinstance(pm, dict) and line_key in pm:
            c = fnum(pm[line_key])
            if c and c > 1:
                nn += 1
                win = (m["sa"] + m["sb"]) <= want_max
                hh += win
                pnl.append((c - 1) if win else -1.0)
    if nn:
        mean = sum(pnl) / nn
        report.setdefault("pm_line_roi", {})[line_key] = dict(
            n=nn, hit=round(hh / nn, 4), roi=round(mean, 4))
        print(f"ligne '{line_key}': n={nn} hit={hh/nn:.3f} roi={mean:+.4f}")

with open("exports/wf5_under_gold_counterverify.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=1)
print("\nJSON: exports/wf5_under_gold_counterverify.json")
