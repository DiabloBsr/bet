"""WF5 — Audit LIVE du jour (2026-06-12 depuis 06:00 UTC) sur InstantLeague-8035.

Reconstitution mecanique des picks systeme sur chaque match FINI du jour :
  - TIER1_approx : favori (home ou away) cote ouverture <= 1.30 -> 1X2 favori
  - E1           : odds_home <= 1.50 -> FTTS '1' (logique prod exotic_signals.py), settle goals_json
  - E2           : favori in [1.10, 1.20] -> 1X2 favori (logique prod)
  - BTTS_NON     : home_crush odds_home <= 1.30 -> G/NG 'Non'
  - COMBO        : paire dans SCORE_COMBO_GOLD -> 1u top1 + 1u top2 (Score exact)
  - SWEET        : paire dans SCORE_DOMINANT_GOLD -> 1u score dominant

Cotes d'OUVERTURE uniquement (snapshot MIN(id) par event). Lecture seule.
Exclusion des ids de exports/corrupted_events.json.
Sortie : exports/wf5_today_audit.json
"""
import sys, json, math
sys.path.insert(0, ".")
from collections import defaultdict
from sqlalchemy import create_engine, text

from scraper.config import load_settings
from scraper.team_gold_data import SCORE_COMBO_GOLD, SCORE_DOMINANT_GOLD

DAY_START = "2026-06-12 06:00:00"
COMP = "InstantLeague-8035"

eng = create_engine(load_settings().db_url)

# --- corrupted ids ---
with open("exports/corrupted_events.json", encoding="utf-8") as f:
    corrupted = set(int(k) for k in json.load(f)["events"].keys())

# --- 1. matchs finis du jour + cotes d'ouverture (colonnes minimales) ---
SQL = text("""
SELECT ev.id, ev.team_a, ev.team_b, ev.round_info, ev.expected_start,
       r.score_a, r.score_b,
       s.id AS snap_id, s.odds_home, s.odds_draw, s.odds_away
FROM events ev
JOIN results r ON r.event_id = ev.id
JOIN odds_snapshots s ON s.id = (
    SELECT MIN(s2.id) FROM odds_snapshots s2 WHERE s2.event_id = ev.id)
WHERE ev.competition = :comp AND ev.expected_start >= :start
""")
rows = []
with eng.connect() as c:
    for r in c.execute(SQL, {"comp": COMP, "start": DAY_START}):
        if r.id in corrupted:
            continue
        if not r.odds_home or not r.odds_away or r.odds_home < 1.01 or r.odds_away < 1.01:
            continue
        rows.append(dict(eid=r.id, ta=r.team_a, tb=r.team_b, rnd=r.round_info,
                         start=str(r.expected_start)[:16],
                         sa=int(r.score_a), sb=int(r.score_b),
                         snap=r.snap_id, oh=float(r.odds_home),
                         od=float(r.odds_draw or 0), oa=float(r.odds_away)))
print(f"matchs finis du jour (hors corrompus) : {len(rows)}", file=sys.stderr)

# --- 2. selection des picks par famille (avant fetch extra_markets) ---
need_em = {}   # snap_id -> list of (family, what)
picks = defaultdict(list)   # family -> list of dicts

for m in rows:
    fav_home = m["oh"] <= m["oa"]
    fav_odds = m["oh"] if fav_home else m["oa"]
    fav_won = (m["sa"] > m["sb"]) if fav_home else (m["sb"] > m["sa"])

    # TIER1_approx : favori <= 1.30, 1X2
    if fav_odds <= 1.30:
        picks["TIER1_approx"].append(dict(
            m, sel=("1" if fav_home else "2"), cote=fav_odds, win=fav_won))

    # E2 : favori in [1.10, 1.20] (logique prod)
    if 1.10 <= fav_odds <= 1.20:
        picks["E2"].append(dict(
            m, sel=("1" if fav_home else "2"), cote=fav_odds, win=fav_won))

    # E1 : home <= 1.50 -> FTTS '1' (cote depuis extra_markets, settle goals_json)
    if m["oh"] <= 1.50:
        picks["E1"].append(dict(m, sel="FTTS 1", cote=None, win=None))
        need_em.setdefault(m["snap"], []).append(("E1", "FTTS"))

    # BTTS_NON : home_crush <= 1.30 -> G/NG 'Non'
    if m["oh"] <= 1.30:
        win = (m["sa"] == 0) or (m["sb"] == 0)
        picks["BTTS_NON"].append(dict(m, sel="G/NG Non", cote=None, win=win))
        need_em.setdefault(m["snap"], []).append(("BTTS_NON", "G/NG"))

    # COMBO / SWEET : paires gold (ordre home/away exact)
    pair = (m["ta"], m["tb"])
    if pair in SCORE_COMBO_GOLD:
        g = SCORE_COMBO_GOLD[pair]
        picks["COMBO"].append(dict(m, sel=f"{g['top1']}|{g['top2']}",
                                   top1=g["top1"], top2=g["top2"], cote=None, win=None))
        need_em.setdefault(m["snap"], []).append(("COMBO", "Score exact"))
    if pair in SCORE_DOMINANT_GOLD:
        g = SCORE_DOMINANT_GOLD[pair]
        picks["SWEET"].append(dict(m, sel=g["score"], score=g["score"], cote=None, win=None))
        need_em.setdefault(m["snap"], []).append(("SWEET", "Score exact"))

print({k: len(v) for k, v in picks.items()}, file=sys.stderr)

# --- 3. fetch extra_markets par chunks, n'extraire que les cotes utiles ---
em_odds = {}  # snap_id -> {"FTTS1":x, "GNG_NON":y, "SE": {score: cote}}
snap_ids = sorted(need_em.keys())
with eng.connect() as c:
    for i in range(0, len(snap_ids), 200):
        chunk = snap_ids[i:i + 200]
        q = text(f"SELECT id, extra_markets FROM odds_snapshots WHERE id IN ({','.join(map(str, chunk))})")
        for sid, em_raw in c.execute(q):
            try:
                em = json.loads(em_raw) if isinstance(em_raw, str) else (em_raw or {})
            except Exception:
                em = {}
            out = {}
            ftts = em.get("FTTS") or {}
            v = ftts.get("1")
            out["FTTS1"] = float(v) if isinstance(v, (int, float)) and 1.01 <= v <= 50 else None
            gng = em.get("G/NG") or {}
            v = gng.get("Non")
            out["GNG_NON"] = float(v) if isinstance(v, (int, float)) and 1.01 <= v <= 50 else None
            se = em.get("Score exact") or {}
            out["SE"] = {k: float(v) for k, v in se.items() if isinstance(v, (int, float))}
            em_odds[sid] = out

# --- 4. goals_json pour le settle E1 (premier buteur) ---
e1_eids = [p["eid"] for p in picks["E1"]]
first_goal = {}  # eid -> "Home"/"Away"/"None" (aucun but) / None (indeterminable)
with eng.connect() as c:
    for i in range(0, len(e1_eids), 300):
        chunk = e1_eids[i:i + 300]
        q = text(f"SELECT event_id, goals_json, score_a, score_b FROM results WHERE event_id IN ({','.join(map(str, chunk))})")
        for eid, gj, sa, sb in c.execute(q):
            try:
                goals = json.loads(gj) if isinstance(gj, str) else (gj or [])
            except Exception:
                goals = None
            if goals is None:
                first_goal[eid] = None
            elif len(goals) == 0:
                first_goal[eid] = "None" if (sa + sb) == 0 else None
            else:
                g0 = min(goals, key=lambda g: (g.get("minute", 999),
                                               g.get("homeScore", 0) + g.get("awayScore", 0)))
                first_goal[eid] = g0.get("team")

# --- 5. settle + agregats ---
def agg(bets):
    """bets: list of (win:bool, cote:float). flat 1u."""
    n = len(bets)
    if n == 0:
        return dict(n=0, wins=0, wr=None, staked=0, returned=0, pnl=0, roi=None, z=None)
    wins = sum(1 for w, _ in bets if w)
    returned = sum(c for w, c in bets if w)
    pnl_list = [(c - 1) if w else -1.0 for w, c in bets]
    pnl = sum(pnl_list)
    mean = pnl / n
    var = sum((x - mean) ** 2 for x in pnl_list) / max(n - 1, 1)
    se = math.sqrt(var / n) if var > 0 else None
    z = (mean / se) if se else None
    return dict(n=n, wins=wins, wr=round(wins / n, 4), staked=n,
                returned=round(returned, 2), pnl=round(pnl, 2),
                roi=round(pnl / n, 4), z=round(z, 2) if z is not None else None)

report = {"day_start_utc": DAY_START, "competition": COMP,
          "n_matches_today": len(rows), "families": {}}

def split_am_pm(plist):
    am = [p for p in plist if p["start"] < "2026-06-12 12:00"]
    pm = [p for p in plist if p["start"] >= "2026-06-12 12:00"]
    return am, pm

# TIER1_approx & E2 (cote = 1X2 ouverture, toujours dispo)
for fam in ("TIER1_approx", "E2"):
    plist = picks[fam]
    bets = [(p["win"], p["cote"]) for p in plist]
    rep = agg(bets)
    am, pm = split_am_pm(plist)
    rep["am"] = agg([(p["win"], p["cote"]) for p in am])
    rep["pm"] = agg([(p["win"], p["cote"]) for p in pm])
    report["families"][fam] = rep

# E1 : settle premier buteur, cote FTTS '1'
e1_bets, e1_no_cote, e1_unsettled, e1_detail = [], 0, 0, []
for p in picks["E1"]:
    fg = first_goal.get(p["eid"])
    cote = em_odds.get(p["snap"], {}).get("FTTS1")
    if fg is None:
        e1_unsettled += 1
        continue
    win = (fg == "Home")
    p["win"] = win
    if cote is None:
        e1_no_cote += 1
        continue
    p["cote"] = cote
    e1_bets.append((win, cote))
    e1_detail.append(p)
rep = agg(e1_bets)
rep["no_cote_market_absent"] = e1_no_cote
rep["unsettled_goals_json"] = e1_unsettled
am, pm = split_am_pm(e1_detail)
rep["am"] = agg([(p["win"], p["cote"]) for p in am])
rep["pm"] = agg([(p["win"], p["cote"]) for p in pm])
report["families"]["E1"] = rep

# BTTS_NON : cote G/NG 'Non'
bn_bets, bn_no_cote, bn_detail = [], 0, []
for p in picks["BTTS_NON"]:
    cote = em_odds.get(p["snap"], {}).get("GNG_NON")
    if cote is None:
        bn_no_cote += 1
        continue
    p["cote"] = cote
    bn_bets.append((p["win"], cote))
    bn_detail.append(p)
rep = agg(bn_bets)
rep["no_cote_market_absent"] = bn_no_cote
am, pm = split_am_pm(bn_detail)
rep["am"] = agg([(p["win"], p["cote"]) for p in am])
rep["pm"] = agg([(p["win"], p["cote"]) for p in pm])
report["families"]["BTTS_NON"] = rep

# COMBO : 1u sur top1 + 1u sur top2 (lignes cote>=100 = interdites, skip)
combo_bets, combo_skip100, combo_no_market, combo_matches = [], 0, 0, []
for p in picks["COMBO"]:
    se = em_odds.get(p["snap"], {}).get("SE") or {}
    final = f"{p['sa']}-{p['sb']}"
    lines = []
    for sc in (p["top1"], p["top2"]):
        cote = se.get(sc)
        if cote is None:
            combo_no_market += 1
            continue
        if cote >= 100:
            combo_skip100 += 1
            continue
        win = (final == sc)
        lines.append((win, cote, sc))
        combo_bets.append((win, cote))
    combo_matches.append(dict(eid=p["eid"], pair=f"{p['ta']} v {p['tb']}",
                              rnd=p["rnd"], final=final,
                              lines=[(sc, c, w) for w, c, sc in lines]))
rep = agg(combo_bets)
rep["n_matches"] = len(picks["COMBO"])
rep["lines_skipped_cote100"] = combo_skip100
rep["lines_market_absent"] = combo_no_market
report["families"]["COMBO"] = rep

# SWEET : 1u sur le score dominant
sweet_bets, sweet_skip100, sweet_no_market, sweet_matches = [], 0, 0, []
for p in picks["SWEET"]:
    se = em_odds.get(p["snap"], {}).get("SE") or {}
    final = f"{p['sa']}-{p['sb']}"
    cote = se.get(p["score"])
    if cote is None:
        sweet_no_market += 1
        continue
    if cote >= 100:
        sweet_skip100 += 1
        continue
    win = (final == p["score"])
    sweet_bets.append((win, cote))
    sweet_matches.append(dict(eid=p["eid"], pair=f"{p['ta']} v {p['tb']}",
                              rnd=p["rnd"], bet=p["score"], cote=cote,
                              final=final, win=win))
rep = agg(sweet_bets)
rep["n_matches"] = len(picks["SWEET"])
rep["lines_skipped_cote100"] = sweet_skip100
rep["lines_market_absent"] = sweet_no_market
report["families"]["SWEET"] = rep

# attendu vs realise (references prod)
report["expected_refs"] = {
    "TIER1_approx": "WR attendu ~77-80% (implied 1/1.30=76.9% + edge favori), ROI attendu ~0 a +2%",
    "E1": "WR attendu ~75%, ROI +6.8% full / +2.4-4.3% OOS",
    "E2": "WR attendu ~87-90%, ROI +5.45% OOS",
    "BTTS_NON": "proxy home_crush, pas de ref calibree directe",
    "COMBO": "combo historique >=55% (n=8-12, risque overfit)",
    "SWEET": "rate historique 37-62% (n=8-12, risque overfit)",
}
report["combo_matches"] = combo_matches
report["sweet_matches"] = sweet_matches

with open("exports/wf5_today_audit.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=1)

# resume console
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("am", "pm")}
                  for k, v in report["families"].items()}, indent=1))
print("AM/PM:")
for fam, v in report["families"].items():
    if "am" in v:
        print(fam, "AM", v["am"], "| PM", v["pm"])
