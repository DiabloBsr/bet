# -*- coding: utf-8 -*-
"""WF5 contre-verification independante du verdict PAIR_TRAP_HOME.

Recompte les 4 paires trap en SQL brut (sans passer par le code de l'audit),
verifie: doublons results, orientation du settlement (team_a = home),
exclusions corrupted, fenetre temporelle, et ROI d'un back home OOS.
Sortie: exports/wf5_pair_trap_recheck.json. LECTURE SEULE.
"""
import sys, json, math
sys.path.insert(0, ".")
from scraper.config import load_settings
from sqlalchemy import create_engine, text

LEAGUE = "InstantLeague-8035"
CUTOFF = "2026-06-06"
TRAPS = [("Everton", "London Blues"), ("London Blues", "Manchester Red"),
         ("Spurs", "N. Forest"), ("Spurs", "Burnley")]

engine = create_engine(load_settings().db_url)
corrupted = set(int(k) for k in json.load(open("exports/corrupted_events.json"))["events"].keys())
out = {}

with engine.connect() as c:
    # 1. doublons results / snapshots min par event (risque d'inflation par JOIN)
    dup_res = c.execute(text(
        "SELECT COUNT(*) FROM (SELECT event_id FROM results GROUP BY event_id HAVING COUNT(*)>1)"
    )).scalar()
    dup_ev = c.execute(text(
        "SELECT COUNT(*) FROM (SELECT competition,team_a,team_b,expected_start FROM events "
        "WHERE competition=:lg GROUP BY competition,team_a,team_b,expected_start HAVING COUNT(*)>1)"
    ), {"lg": LEAGUE}).scalar()
    out["dup_results_per_event"] = dup_res
    out["dup_events_same_kickoff"] = dup_ev

    # 2. orientation settlement: si team_a=home, le favori home (cote<1.5) doit gagner ~2/3+
    rows = c.execute(text("""
        SELECT os.odds_home, os.odds_away, r.score_a, r.score_b
        FROM events e
        JOIN results r ON r.event_id=e.id
        JOIN (SELECT event_id, MIN(id) sid FROM odds_snapshots GROUP BY event_id) f ON f.event_id=e.id
        JOIN odds_snapshots os ON os.id=f.sid
        WHERE e.competition=:lg AND r.score_a IS NOT NULL
    """), {"lg": LEAGUE}).fetchall()
    fav_h = [(sa > sb) for oh, oa, sa, sb in rows if oh and oh < 1.5]
    fav_a = [(sb > sa) for oh, oa, sa, sb in rows if oa and oa < 1.5]
    out["sanity_orientation"] = {
        "home_fav_lt1.5": {"n": len(fav_h), "wr": round(sum(fav_h) / len(fav_h), 3)},
        "away_fav_lt1.5": {"n": len(fav_a), "wr": round(sum(fav_a) / len(fav_a), 3)},
    }

    # 3. recompte brut des paires trap (toutes periodes), avec et sans exclusion corrupted
    out["traps"] = []
    pnls = []
    for ta, tb in TRAPS:
        rows = c.execute(text("""
            SELECT e.id, r.finished_at, r.score_a, r.score_b, os.odds_home
            FROM events e
            JOIN results r ON r.event_id=e.id
            JOIN (SELECT event_id, MIN(id) sid FROM odds_snapshots GROUP BY event_id) f ON f.event_id=e.id
            JOIN odds_snapshots os ON os.id=f.sid
            WHERE e.competition=:lg AND e.team_a=:ta AND e.team_b=:tb
              AND r.score_a IS NOT NULL
            ORDER BY r.finished_at
        """), {"lg": LEAGUE, "ta": ta, "tb": tb}).fetchall()
        clean = [r for r in rows if r[0] not in corrupted]
        pre = [r for r in clean if str(r[1]) < CUTOFF]
        oos = [r for r in clean if str(r[1]) >= CUTOFF]
        hw_pre = sum(1 for r in pre if r[2] > r[3])
        hw_oos = sum(1 for r in oos if r[2] > r[3])
        for r in oos:
            if r[4]:
                pnls.append((r[4] - 1) if r[2] > r[3] else -1.0)
        out["traps"].append({
            "pair": f"{ta} v {tb}",
            "n_total_raw": len(rows), "n_corrupted_excl": len(rows) - len(clean),
            "pre_cutoff": {"n": len(pre), "home_wins": hw_pre,
                           "home_wr": round(hw_pre / len(pre), 3) if pre else None},
            "oos": {"n": len(oos), "home_wins": hw_oos,
                    "home_wr": round(hw_oos / len(oos), 3) if oos else None,
                    "first_fin": str(oos[0][1]) if oos else None,
                    "last_fin": str(oos[-1][1]) if oos else None,
                    "detail": [{"id": r[0], "fin": str(r[1]), "score": f"{r[2]}-{r[3]}",
                                "oh": r[4]} for r in oos]},
        })

    # 4. ROI back home OOS sur les 4 paires trap (ce que le block empeche)
    n = len(pnls)
    mean = sum(pnls) / n if n else None
    sd = math.sqrt(sum((x - mean) ** 2 for x in pnls) / max(n - 1, 1)) if n > 1 else None
    out["back_home_oos"] = {"n": n, "roi": round(mean, 4) if n else None,
                            "z": round(mean / (sd / math.sqrt(n)), 2) if sd else None}

    # 5. fenetre: derniere date finie en base + nb matchs finis apres cutoff
    out["last_finished_at"] = str(c.execute(text(
        "SELECT MAX(r.finished_at) FROM results r JOIN events e ON e.id=r.event_id "
        "WHERE e.competition=:lg"), {"lg": LEAGUE}).scalar())

with open("exports/wf5_pair_trap_recheck.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
