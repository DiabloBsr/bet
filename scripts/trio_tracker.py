"""Suivi FORWARD automatique du prédicteur TRIO (V2+V5+marché).

À chaque run :
  1. PRÉDIT tous les rounds à venir captés et STOCKE (table trio_predictions,
     INSERT OR IGNORE par event_key — une prédiction n'est jamais réécrite).
  2. SCORE les prédictions dont le résultat est arrivé (hit Top-1 brut,
     Top-1 calibré, Top-3, 1X2).
  3. RAPPORTE le hit-rate glissant réel.

Lecture seule sur les données ; n'écrit que sa propre table. À planifier
(schtasks, toutes les ~20 min). Sortie ASCII (Task Scheduler-safe).
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)                       # db_url relative -> CWD projet obligatoire
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
if sys.stdout is None:               # pythonw / Task Scheduler
    (ROOT / "data" / "logs").mkdir(parents=True, exist_ok=True)
    _lg = open(ROOT / "data" / "logs" / "tracker.log", "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = _lg
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import pandas as pd
from sqlalchemy import create_engine, text
from scraper.config import load_settings
import predict_trio as pt

DDL = """
CREATE TABLE IF NOT EXISTS trio_predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_key TEXT UNIQUE,
  expected_start TEXT, local_hhmm TEXT, team_a TEXT, team_b TEXT,
  top1 TEXT, top1_cal TEXT, top3 TEXT, x12_pick TEXT,
  actual TEXT, actual_x12 TEXT,
  hit1 INTEGER, hit1_cal INTEGER, hit3 INTEGER, hitx INTEGER,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
)
"""


def predict_upcoming(eng, m5, v2m) -> int:
    """Prédit tous les matchs à venir non encore stockés. Retourne le nb inséré."""
    now = pd.Timestamp.now(tz="UTC")
    up = pd.read_sql(f"""SELECT e.team_a,e.team_b,e.expected_start,o.odds_home oh,o.odds_draw od,
        o.odds_away oa,o.extra_markets,e.id ev FROM events e
        JOIN odds_snapshots o ON o.id=(SELECT MAX(id) FROM odds_snapshots WHERE event_id=e.id)
        LEFT JOIN results r ON r.event_id=e.id
        WHERE r.id IS NULL AND e.expected_start IS NOT NULL AND e.competition='{pt.LG}'""", eng)
    if not len(up):
        return 0
    up["es"] = pd.to_datetime(up.expected_start, utc=True)
    up = up[up.es > now]
    up["local"] = up.es.dt.tz_convert(pt.MADA).dt.strftime("%H:%M")
    up = up.sort_values(["es", "ev"]).drop_duplicates(["team_a", "team_b", "expected_start"])
    n_ins = 0
    with eng.begin() as cx:
        for r in up.itertuples():
            if float(r.oh) <= 1 or float(r.oa) <= 1:
                continue
            key = f"{r.team_a}|{r.team_b}|{r.expected_start}"
            exists = cx.execute(text(
                "SELECT 1 FROM trio_predictions WHERE event_key=:k"), {"k": key}).fetchone()
            if exists:
                continue
            m = pt.predict_one(eng, m5, v2m, r.team_a, r.team_b, r.oh, r.od, r.oa, r.extra_markets)
            x = m["x12"]; pick = ("1", "X", "2")[int(pd.Series(x).idxmax())]
            t1c = m.get("top1_calibre")
            cx.execute(text("""INSERT OR IGNORE INTO trio_predictions
                (event_key, expected_start, local_hhmm, team_a, team_b, top1, top1_cal, top3, x12_pick)
                VALUES (:k,:es,:loc,:ta,:tb,:t1,:t1c,:t3,:xp)"""),
                {"k": key, "es": str(r.expected_start), "loc": r.local,
                 "ta": r.team_a, "tb": r.team_b,
                 "t1": m["consensus_top3"][0][0] if m["consensus_top3"] else None,
                 "t1c": t1c[0] if t1c else None,
                 "t3": ",".join(s for s, _ in m["consensus_top3"]),
                 "xp": pick})
            n_ins += 1
    return n_ins


def settle(eng) -> int:
    """Score les prédictions dont le résultat est arrivé. Retourne le nb scoré.
    Les rounds annoncés puis jamais joués (replanification de la ligue) sont
    marqués VOID après 90 min sans résultat — exclus des stats, comptés en couverture."""
    with eng.begin() as cx:
        cx.execute(text("""UPDATE trio_predictions SET actual='VOID'
            WHERE actual IS NULL
              AND datetime(expected_start) < datetime('now', '-90 minutes')"""))
    rows = pd.read_sql(text("""
        SELECT p.id pid, p.top1, p.top1_cal, p.top3, p.x12_pick, r.score_a sa, r.score_b sb
        FROM trio_predictions p
        JOIN events e ON e.team_a=p.team_a AND e.team_b=p.team_b AND e.expected_start=p.expected_start
        JOIN results r ON r.event_id=e.id
        WHERE p.actual IS NULL AND r.score_a IS NOT NULL"""), eng)
    rows = rows.drop_duplicates("pid")
    n = 0
    with eng.begin() as cx:
        for r in rows.itertuples():
            sa, sb = min(int(r.sa), 6), min(int(r.sb), 6)
            actual = f"{sa}-{sb}"
            ax = "1" if sa > sb else ("X" if sa == sb else "2")
            t3 = (r.top3 or "").split(",")
            cx.execute(text("""UPDATE trio_predictions SET actual=:a, actual_x12=:ax,
                hit1=:h1, hit1_cal=:h1c, hit3=:h3, hitx=:hx WHERE id=:pid"""),
                {"a": actual, "ax": ax, "pid": int(r.pid),
                 "h1": int(actual == r.top1), "h1c": int(actual == (r.top1_cal or "")),
                 "h3": int(actual in t3), "hx": int(ax == r.x12_pick)})
            n += 1
    return n


def report(eng, window=500):
    nv = pd.read_sql(text("SELECT COUNT(*) c FROM trio_predictions WHERE actual='VOID'"), eng)
    d = pd.read_sql(text(f"""SELECT hit1, hit1_cal, hit3, hitx FROM trio_predictions
        WHERE actual IS NOT NULL AND actual!='VOID' ORDER BY id DESC LIMIT {int(window)}"""), eng)
    if not len(d):
        print(f"suivi : aucune prediction scoree pour l'instant ({int(nv.c.iloc[0])} VOID/replanifiees)")
        return
    print(f"SUIVI FORWARD TRIO — {len(d)} predictions scorees "
          f"({int(nv.c.iloc[0])} VOID/replanifiees exclues)")
    print(f"  Top-1 brut    : {100*d.hit1.mean():5.1f}%   (plafond ~11.7%)")
    print(f"  Top-1 calibre : {100*d.hit1_cal.mean():5.1f}%   (plafond ~11.9%)")
    print(f"  Top-3         : {100*d.hit3.mean():5.1f}%   (plafond ~31.6%)")
    print(f"  1X2           : {100*d.hitx.mean():5.1f}%   (plafond ~55%)")


DDL_COMBO = """
CREATE TABLE IF NOT EXISTS combo_suggestions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  round_key TEXT UNIQUE, created_at TEXT,
  target_odds REAL, odds REAL, p_est REAL, legs TEXT,
  hits INTEGER, won INTEGER, pnl REAL, family TEXT DEFAULT 'safe'
)
"""


FAMILIES = {  # famille -> (marchés, min_legs, p_min)
    "safe": (pt.COMBO_MARKETS, 2, 0.45),
    "totals": (pt.TOTALS_MARKETS, 1, 0.20),
}


def log_combo(eng, family: str = "safe") -> int:
    """Fige le combiné conseillé (cote>=3, le plus probable) du PROCHAIN round 8035.
    Sans fit : boards marché uniquement. 1 combiné max par round et par famille."""
    now = pd.Timestamp.now(tz="UTC")
    up = pd.read_sql(f"""SELECT e.team_a,e.team_b,e.expected_start,o.odds_home oh,o.odds_draw od,
        o.odds_away oa,o.extra_markets xm FROM events e
        JOIN odds_snapshots o ON o.id=(SELECT MAX(id) FROM odds_snapshots WHERE event_id=e.id)
        LEFT JOIN results r ON r.event_id=e.id
        WHERE r.id IS NULL AND e.expected_start IS NOT NULL AND e.competition='{pt.LG}'""", eng)
    if not len(up):
        return 0
    up["es"] = pd.to_datetime(up.expected_start, utc=True)
    up = up[up.es > now]
    if not len(up):
        return 0
    up = up.sort_values("es")
    first = up[up.es == up.es.iloc[0]].drop_duplicates(["team_a", "team_b"])
    mkts, min_legs, p_min = FAMILIES[family]
    key = f"{first.expected_start.iloc[0]}|{family}"
    with eng.begin() as cx:
        if cx.execute(text("SELECT 1 FROM combo_suggestions WHERE round_key=:k"), {"k": key}).fetchone():
            return 0
    matches = [{"match": f"{r.team_a} v {r.team_b}",
                "board": pt.market_board(r.xm, r.oh, r.od, r.oa)}
               for r in first.itertuples() if float(r.oh) > 1 and float(r.oa) > 1]
    if len(matches) < 2:
        return 0
    combos = pt.build_combos(matches, 3.0, 3, top=1,
                             markets=mkts, min_legs=min_legs, p_min=p_min)
    if not combos:
        return 0
    c = combos[0]
    legs = []
    for (mn, mkt, s, p, o) in c["legs"]:
        ta, tb = mn.split(" v ")
        legs.append({"ta": ta, "tb": tb, "mkt": mkt, "sel": s, "p": p, "o": o})
    with eng.begin() as cx:
        cx.execute(text("""INSERT OR IGNORE INTO combo_suggestions
            (round_key, created_at, target_odds, odds, p_est, legs, family)
            VALUES (:k,:c,3.0,:o,:p,:l,:f)"""),
            {"k": key, "c": str(now), "o": float(c["odds"]), "p": float(c["p"]),
             "l": json.dumps(legs, ensure_ascii=False), "f": family})
    return 1


def _settle_leg(mkt, sel, sa, sb):
    tot = sa + sb
    res = "1" if sa > sb else ("2" if sb > sa else "X")
    if mkt == "1X2": return int(sel == res)
    if mkt == "Double Chance": return int(res in sel)
    if mkt == "+/-": return int(tot > 3.5) if ">" in sel else int(tot < 3.5)
    if mkt == "G/NG": return int((sa > 0 and sb > 0) == (sel == "Oui"))
    if mkt == "Total de buts":
        n = int(sel)
        return int(tot >= 6) if n == 6 else int(tot == n)
    if mkt == "Multi-Buts":
        if "0, 1 ou 2" in sel: return int(tot <= 2)
        if "1, 2 ou 3" in sel: return int(1 <= tot <= 3)
        if "2, 3 ou 4" in sel: return int(2 <= tot <= 4)
        return int(tot > 4)
    if mkt == "1X2 & Total":
        part, t = sel.split("/")
        return int(part.strip() == res and ((tot > 3.5) if ">" in t else (tot < 3.5)))
    if mkt == "1X2 & G/NG":
        btts = sa > 0 and sb > 0
        if sel.startswith("1 gagne et les deux"): return int(res == "1" and btts)
        if sel.startswith("1 gagne et seulement"): return int(res == "1" and sb == 0)
        if sel.startswith("2 gagne et les deux"): return int(res == "2" and btts)
        if sel.startswith("2 gagne et seulement"): return int(res == "2" and sa == 0)
        if sel.startswith("X et aucun"): return int(sa == 0 and sb == 0)
        if sel.startswith("X et les deux"): return int(res == "X" and btts)
    return None


def settle_combos(eng) -> int:
    rows = pd.read_sql(text("SELECT id, round_key, legs, odds FROM combo_suggestions WHERE won IS NULL"), eng)
    n = 0
    with eng.begin() as cx:
        for r in rows.itertuples():
            es = str(r.round_key).split("|")[0]
            legs = json.loads(r.legs)
            hits, ok, alldone = 0, True, True
            for l in legs:
                res = cx.execute(text(f"""SELECT rr.score_a, rr.score_b FROM events e
                    JOIN results rr ON rr.event_id=e.id
                    WHERE e.competition='{pt.LG}' AND e.team_a=:ta AND e.team_b=:tb
                      AND e.expected_start=:es AND rr.score_a IS NOT NULL LIMIT 1"""),
                    {"ta": l["ta"], "tb": l["tb"], "es": es}).fetchone()
                if not res:
                    alldone = False; break
                h = _settle_leg(l["mkt"], l["sel"], int(res[0]), int(res[1]))
                hits += h or 0
                ok = ok and bool(h)
            if not alldone:
                continue
            won = int(ok)
            cx.execute(text("UPDATE combo_suggestions SET hits=:h, won=:w, pnl=:p WHERE id=:i"),
                       {"h": hits, "w": won, "p": won * float(r.odds) - 1, "i": int(r.id)})
            n += 1
        # rounds replanifiés/jamais joués -> VOID (won=-1, exclus des stats)
        cx.execute(text("""UPDATE combo_suggestions SET won=-1 WHERE won IS NULL
            AND datetime(substr(round_key, 1, 19)) < datetime('now', '-90 minutes')"""))
    return n


def report_combos(eng):
    d = pd.read_sql(text("""SELECT COALESCE(family,'safe') family, p_est, odds, won, pnl
                            FROM combo_suggestions WHERE won >= 0"""), eng)
    if not len(d):
        print("  combos conseilles : rien de regle encore")
        return
    for fam, g in d.groupby("family"):
        print(f"  COMBOS [{fam:<6}] : n={len(g)}  reussite reelle {100*g.won.mean():.1f}% "
              f"vs annoncee {100*g.p_est.mean():.1f}%  |  cote moy {g.odds.mean():.2f}  |  "
              f"ROI {100*g.pnl.mean():+.1f}%")


def main():
    eng = create_engine(load_settings().db_url)
    with eng.begin() as cx:
        cx.execute(text(DDL))
        cx.execute(text(DDL_COMBO))
    with eng.begin() as cx:                      # migration : colonne family
        try:
            cx.execute(text("ALTER TABLE combo_suggestions ADD COLUMN family TEXT DEFAULT 'safe'"))
        except Exception:
            pass
    n_settled = settle(eng)                      # score d'abord (rapide)
    n_cs = settle_combos(eng)
    n_cl = log_combo(eng, "safe") + log_combo(eng, "totals")
    print(f"settle : {n_settled} prediction(s), {n_cs} combo(s) regle(s), {n_cl} combo(s) fige(s)")
    print("fit V5+V2…")
    m5, v2m, nh = pt.fit(eng)
    n_pred = predict_upcoming(eng, m5, v2m)
    print(f"predict : {n_pred} nouveau(x) match(s) stocke(s) (fit {nh})")
    report(eng)
    report_combos(eng)


if __name__ == "__main__":
    main()
