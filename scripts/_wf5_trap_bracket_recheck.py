# -*- coding: utf-8 -*-
"""WF5 — Contre-verification du verdict 'BRACKET_TRAP_HOME = MORT'.

Tests:
 A. Re-mesure OOS (cutoff 2026-06-06) du ROI back-home sur les selections trap.
 B. Controle apparie: back home sur les MEMES brackets de cote mais equipes NON-trap.
 C. Baseline globale: back home sur tous les matchs OOS (attendu ~ -6% = marge).
 D. Test de l'hypothese 'trap reel': l'IS claim pondere (~-33%) est-il rejete?
 E. Stabilite: split OOS en 2 moities temporelles.
 F. Sanity: doublons, bornes de brackets, distribution par bracket.

Sortie: exports/wf5_trap_bracket_recheck.json + stdout. LECTURE SEULE.
"""
import sys, json, math
from collections import defaultdict

sys.path.insert(0, ".")
from scraper.config import load_settings
from sqlalchemy import create_engine, text
from scraper.team_gold_data import BRACKET_TRAP_HOME

LEAGUE = "InstantLeague-8035"
CUTOFF = "2026-06-06"

engine = create_engine(load_settings().db_url)
corrupted = set(int(k) for k in json.load(open("exports/corrupted_events.json"))["events"].keys())

SQL = """
SELECT e.id, e.team_a, e.team_b, r.score_a, r.score_b, r.finished_at,
       os.odds_home, os.odds_draw, os.odds_away
FROM events e
JOIN results r ON r.event_id = e.id
JOIN (SELECT event_id, MIN(id) AS sid FROM odds_snapshots GROUP BY event_id) f
     ON f.event_id = e.id
JOIN odds_snapshots os ON os.id = f.sid
WHERE e.competition = :lg AND r.finished_at >= :cut
ORDER BY r.finished_at
"""
with engine.connect() as c:
    rows = c.execute(text(SQL), {"lg": LEAGUE, "cut": CUTOFF}).fetchall()

OOS = [dict(id=r[0], ta=r[1], tb=r[2], sa=r[3], sb=r[4], fin=str(r[5]),
            oh=float(r[6]), od=float(r[7]), oa=float(r[8]))
       for r in rows if r[0] not in corrupted
       and r[3] is not None and r[6] and r[7] and r[8]]
print(f"OOS matchs propres: {len(OOS)}  ({OOS[0]['fin']} -> {OOS[-1]['fin']})")

trap_teams_brackets = list(BRACKET_TRAP_HOME.items())
trap_teams = set(t for (t, _), _ in trap_teams_brackets)
brackets = sorted(set(b for (_, b), _ in trap_teams_brackets))

def pnl_home(m):
    return (m["oh"] - 1.0) if m["sa"] > m["sb"] else -1.0

def stats(pnls):
    n = len(pnls)
    if n == 0:
        return dict(n=0)
    mean = sum(pnls) / n
    var = sum((x - mean) ** 2 for x in pnls) / max(n - 1, 1)
    sd = math.sqrt(var)
    se = sd / math.sqrt(n) if n else 0
    return dict(n=n, roi=round(mean, 4), sd=round(sd, 3), se=round(se, 4),
                z_vs_0=round(mean / se, 2) if se else None)

# --- A. selections trap (et doublons)
trap_sel, seen = [], set()
dup = 0
claim_w, claim_n = 0.0, 0
for (team, (lo, hi)), roi_is in trap_teams_brackets:
    for m in OOS:
        if m["ta"] == team and lo <= m["oh"] < hi:
            if m["id"] in seen:
                dup += 1
            seen.add(m["id"])
            trap_sel.append((m, roi_is))
            claim_w += roi_is
            claim_n += 1
trap_pnls = [pnl_home(m) for m, _ in trap_sel]
A = stats(trap_pnls)
claim_roi = claim_w / claim_n if claim_n else None
# z de l'observe contre l'hypothese 'le trap IS est reel'
se = A["se"]
A["claim_is_weighted"] = round(claim_roi, 4)
A["z_obs_vs_claim"] = round((A["roi"] - claim_roi) / se, 2) if se else None
A["duplicates"] = dup
print("\nA. TRAP selections OOS:", A)

# --- B. controle apparie: memes brackets, equipes non-trap pour CE bracket
ctrl_pnls = []
trap_keys = set((t, b) for (t, b), _ in trap_teams_brackets)
for m in OOS:
    for b in brackets:
        lo, hi = b
        if lo <= m["oh"] < hi:
            if (m["ta"], b) not in trap_keys:
                ctrl_pnls.append(pnl_home(m))
            break  # un match ne compte qu'une fois (brackets se chevauchent: 3.0-100 inclut 4.0+? non, liste)
B = stats(ctrl_pnls)
print("B. CONTROLE memes brackets, non-trap:", B)

# z trap vs controle
if A["n"] and B["n"]:
    sa_, sb_ = A["sd"], B["sd"]
    se_diff = math.sqrt(sa_**2 / A["n"] + sb_**2 / B["n"])
    z_diff = (A["roi"] - B["roi"]) / se_diff
    print(f"   diff trap-controle = {A['roi']-B['roi']:+.4f}  z = {z_diff:+.2f}")

# --- C. baseline globale back home OOS
C = stats([pnl_home(m) for m in OOS])
print("C. BASELINE back tous les homes OOS:", C)

# --- D. par bracket (trap vs controle dans le meme bracket)
print("\nD. Par bracket (trap | controle):")
detail = {}
for b in brackets:
    lo, hi = b
    tp = [pnl_home(m) for m, _ in trap_sel if lo <= m["oh"] < hi]
    cp = [pnl_home(m) for m in OOS if lo <= m["oh"] < hi and (m["ta"], b) not in trap_keys]
    st_t, st_c = stats(tp), stats(cp)
    detail[f"{lo}-{hi}"] = dict(trap=st_t, ctrl=st_c)
    print(f"  {lo}-{hi}: trap n={st_t.get('n')} roi={st_t.get('roi')} | ctrl n={st_c.get('n')} roi={st_c.get('roi')}")

# --- E. stabilite temporelle (2 moities)
half = len(trap_sel) // 2
E1 = stats([pnl_home(m) for m, _ in trap_sel[:half]])
E2 = stats([pnl_home(m) for m, _ in trap_sel[half:]])
print("\nE. moitie 1:", E1, "\n   moitie 2:", E2)

out = dict(n_oos=len(OOS), trap=A, control_same_brackets=B, baseline_all_homes=C,
           per_bracket=detail, half1=E1, half2=E2)
with open("exports/wf5_trap_bracket_recheck.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("\nJSON: exports/wf5_trap_bracket_recheck.json")
