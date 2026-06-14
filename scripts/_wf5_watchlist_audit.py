"""WF5 — AUDIT lecture seule de la watchlist forward (definitions GELEES de
scripts/_signal_watchlist.py). NE TOUCHE PAS data/watchlist_registry.json.

Replique exactement load_dataset / build_prematch_features / apply_signals du
script gele (2026-06-11), avec deux differences NON-CONTAMINANTES :
  - extra_markets n'est charge que pour les events odds_home>=4.0 (RAM),
    le critere de selection de mitps_longshot_global etant identique ;
  - chaque pari porte l'event_id pour une analyse de sensibilite excluant
    exports/corrupted_events.json (le script gele ne les exclut PAS — la ligne
    de reference reste donc la ligne brute, la ligne CLEAN est informative).

Sortie : exports/wf5_watchlist_audit.json + tableau console.
Usage : PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe scripts/_wf5_watchlist_audit.py
"""
from __future__ import annotations
import sys, json, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
from sqlalchemy import create_engine, text
from scraper.config import load_settings

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "exports" / "wf5_watchlist_audit.json"

# ── Definitions GELEES (copie conforme de scripts/_signal_watchlist.py) ──
FROZEN_AT = "2026-06-11T17:00:00"
FROZEN_AT_V2 = "2026-06-12T01:30:00"
V2_SIGNALS = {"value_jitter_pair", "mitps_longshot_global", "follow_drift"}
ALL_SIGNALS = [
    "fade_serie_5plus", "fade_serie_5plus_draw", "sous_regime_rebond",
    "standings_pos_gap5", "standings_pts_gap5", "value_home_vs_alltime",
    "value_jitter_pair", "mitps_longshot_global", "follow_drift",
]


def _z_score(wins: int, n: int, p_implied: float) -> float:
    if n == 0: return 0.0
    p_obs = wins / n
    se = math.sqrt(p_implied * (1 - p_implied) / n)
    return (p_obs - p_implied) / se if se > 0 else 0.0


def load_dataset(engine) -> pd.DataFrame:
    # Identique au script gele, SANS o.extra_markets (charge a part, cible).
    df = pd.read_sql("""
        SELECT e.id, e.round_info, e.team_a, e.team_b, e.expected_start,
               o.odds_home, o.odds_draw, o.odds_away,
               ol.odds_home AS last_h, ol.odds_draw AS last_d, ol.odds_away AS last_a,
               r.score_a, r.score_b, r.ht_score_a, r.ht_score_b
        FROM events e
        JOIN odds_snapshots o ON o.id = (SELECT MIN(id) FROM odds_snapshots WHERE event_id = e.id)
        JOIN odds_snapshots ol ON ol.id = (SELECT MAX(id) FROM odds_snapshots WHERE event_id = e.id)
        JOIN results r ON r.event_id = e.id
        WHERE r.score_a IS NOT NULL AND e.round_info IS NOT NULL AND e.round_info != '0' AND e.competition = 'InstantLeague-8035'
        ORDER BY e.expected_start
    """, engine)
    df["journee"] = pd.to_numeric(df.round_info, errors="coerce")
    df = df.dropna(subset=["journee"]).copy()
    df["journee"] = df.journee.astype(int)
    df["expected_start"] = pd.to_datetime(df.expected_start)
    df = df.drop_duplicates(["team_a", "team_b", "expected_start"]).reset_index(drop=True)
    df["ft"] = np.where(df.score_a > df.score_b, "1",
                np.where(df.score_a == df.score_b, "X", "2"))
    season_id, cur = [], 0
    prev_j = None
    for j in df.journee:
        if prev_j is not None and j < prev_j - 3:
            cur += 1
        season_id.append(cur)
        prev_j = j
    df["season_id"] = season_id
    return df


def load_extra_markets_for_longshots(engine, event_ids: list[int]) -> dict:
    """extra_markets du snapshot d'OUVERTURE, uniquement pour les ids fournis."""
    em_map = {}
    CHUNK = 500
    with engine.connect() as conn:
        for i in range(0, len(event_ids), CHUNK):
            chunk = event_ids[i:i + CHUNK]
            placeholders = ",".join(str(int(x)) for x in chunk)
            rows = conn.execute(text(f"""
                SELECT o.event_id, o.extra_markets
                FROM odds_snapshots o
                WHERE o.id IN (SELECT MIN(id) FROM odds_snapshots
                               WHERE event_id IN ({placeholders}) GROUP BY event_id)
            """)).fetchall()
            for eid, em in rows:
                if isinstance(em, str):
                    try: em = json.loads(em)
                    except Exception: em = {}
                em_map[eid] = em if isinstance(em, dict) else {}
    return em_map


def build_prematch_features(df: pd.DataFrame) -> pd.DataFrame:
    # Copie conforme du script gele.
    alltime, season, rows = {}, {}, []
    for _, m in df.iterrows():
        sid = m.season_id
        fa = season.setdefault((sid, m.team_a), {"wins": 0, "n": 0, "streak": 0, "pts": 0})
        fb = season.setdefault((sid, m.team_b), {"wins": 0, "n": 0, "streak": 0, "pts": 0})
        aa = alltime.setdefault(m.team_a, {"wins": 0, "n": 0})
        ab = alltime.setdefault(m.team_b, {"wins": 0, "n": 0})
        season_teams = {t: s for (s_id, t), s in season.items() if s_id == sid}
        ranked = sorted(season_teams.items(), key=lambda kv: -kv[1]["pts"])
        pos = {t: i + 1 for i, (t, _) in enumerate(ranked)}
        rows.append({
            "idx": m.name,
            "streak_h": fa["streak"], "streak_a": fb["streak"],
            "season_wr_h": fa["wins"] / fa["n"] if fa["n"] else None,
            "season_wr_a": fb["wins"] / fb["n"] if fb["n"] else None,
            "season_n_h": fa["n"], "season_n_a": fb["n"],
            "alltime_wr_h": aa["wins"] / aa["n"] if aa["n"] else None,
            "alltime_wr_a": ab["wins"] / ab["n"] if ab["n"] else None,
            "alltime_n_h": aa["n"], "alltime_n_a": ab["n"],
            "pts_h": fa["pts"], "pts_a": fb["pts"],
            "pos_h": pos.get(m.team_a), "pos_a": pos.get(m.team_b),
        })
        if m.ft == "1":
            fa["wins"] += 1; fa["streak"] = fa["streak"] + 1 if fa["streak"] >= 0 else 1
            fa["pts"] += 3
            fb["streak"] = min(fb["streak"], 0) - 1
            aa["wins"] += 1
        elif m.ft == "2":
            fb["wins"] += 1; fb["streak"] = fb["streak"] + 1 if fb["streak"] >= 0 else 1
            fb["pts"] += 3
            fa["streak"] = min(fa["streak"], 0) - 1
            ab["wins"] += 1
        else:
            fa["streak"] = 0; fb["streak"] = 0
            fa["pts"] += 1; fb["pts"] += 1
        fa["n"] += 1; fb["n"] += 1
        aa["n"] += 1; ab["n"] += 1
    feats = pd.DataFrame(rows).set_index("idx")
    return df.join(feats)


def apply_signals(d: pd.DataFrame, em_map: dict) -> dict:
    # Copie conforme du script gele + champ eid sur chaque pari.
    out = {}

    bets = []
    for _, r in d.iterrows():
        if r.streak_h >= 5:
            bets.append({"eid": r.id, "pick": "2", "cote": r.odds_away, "won": r.ft == "2", "ts": r.expected_start})
        elif r.streak_a >= 5:
            bets.append({"eid": r.id, "pick": "1", "cote": r.odds_home, "won": r.ft == "1", "ts": r.expected_start})
    out["fade_serie_5plus"] = pd.DataFrame(bets)

    bets = []
    for _, r in d.iterrows():
        if r.streak_h >= 5 or r.streak_a >= 5:
            bets.append({"eid": r.id, "pick": "X", "cote": r.odds_draw, "won": r.ft == "X", "ts": r.expected_start})
    out["fade_serie_5plus_draw"] = pd.DataFrame(bets)

    bets = []
    for _, r in d.iterrows():
        if (r.season_n_h >= 8 and r.alltime_n_h >= 30 and r.season_wr_h is not None
                and r.alltime_wr_h is not None and r.season_wr_h <= r.alltime_wr_h - 0.25):
            bets.append({"eid": r.id, "pick": "1", "cote": r.odds_home, "won": r.ft == "1", "ts": r.expected_start})
        if (r.season_n_a >= 8 and r.alltime_n_a >= 30 and r.season_wr_a is not None
                and r.alltime_wr_a is not None and r.season_wr_a <= r.alltime_wr_a - 0.25):
            bets.append({"eid": r.id, "pick": "2", "cote": r.odds_away, "won": r.ft == "2", "ts": r.expected_start})
    out["sous_regime_rebond"] = pd.DataFrame(bets)

    bets = []
    for _, r in d.iterrows():
        if r.journee < 6 or r.pos_h is None or r.pos_a is None: continue
        fav_home = r.odds_home < r.odds_away
        if fav_home and (r.pos_h - r.pos_a) >= 5:
            bets.append({"eid": r.id, "pick": "2", "cote": r.odds_away, "won": r.ft == "2", "ts": r.expected_start})
        elif (not fav_home) and (r.pos_a - r.pos_h) >= 5:
            bets.append({"eid": r.id, "pick": "1", "cote": r.odds_home, "won": r.ft == "1", "ts": r.expected_start})
    out["standings_pos_gap5"] = pd.DataFrame(bets)

    bets = []
    for _, r in d.iterrows():
        if r.journee < 6: continue
        fav_home = r.odds_home < r.odds_away
        if fav_home and (r.pts_a - r.pts_h) >= 5:
            bets.append({"eid": r.id, "pick": "2", "cote": r.odds_away, "won": r.ft == "2", "ts": r.expected_start})
        elif (not fav_home) and (r.pts_h - r.pts_a) >= 5:
            bets.append({"eid": r.id, "pick": "1", "cote": r.odds_home, "won": r.ft == "1", "ts": r.expected_start})
    out["standings_pts_gap5"] = pd.DataFrame(bets)

    bets = []
    for _, r in d.iterrows():
        if r.odds_home >= 2.5 and r.alltime_n_h >= 30 and r.alltime_wr_h is not None and r.alltime_wr_h >= 0.45:
            bets.append({"eid": r.id, "pick": "1", "cote": r.odds_home, "won": r.ft == "1", "ts": r.expected_start})
    out["value_home_vs_alltime"] = pd.DataFrame(bets)

    pair_hist: dict = {}
    bets = []
    for _, r in d.iterrows():
        key = (r.team_a, r.team_b)
        h = pair_hist.get(key)
        if h and h["n"] >= 8:
            for side, cote in (("1", r.odds_home), ("2", r.odds_away)):
                freq = h[side] / h["n"]
                if freq * cote >= 0.98:
                    bets.append({"eid": r.id, "pick": side, "cote": cote, "won": r.ft == side, "ts": r.expected_start})
        if h is None:
            h = pair_hist[key] = {"1": 0, "X": 0, "2": 0, "n": 0}
        h[r.ft] += 1; h["n"] += 1
    out["value_jitter_pair"] = pd.DataFrame(bets)

    bets = []
    for _, r in d.iterrows():
        if r.odds_home < 4.0 or pd.isna(r.ht_score_a): continue
        em = em_map.get(r.id, {})
        mt = em.get("Mi-tps 1X2")
        cote = mt.get("1") if isinstance(mt, dict) else None
        if not isinstance(cote, (int, float)) or cote <= 1.01: continue
        won = int(r.ht_score_a) > int(r.ht_score_b)
        bets.append({"eid": r.id, "pick": "HT 1", "cote": float(cote), "won": won, "ts": r.expected_start})
    out["mitps_longshot_global"] = pd.DataFrame(bets)

    bets = []
    for _, r in d.iterrows():
        if pd.isna(r.last_h) or pd.isna(r.last_a): continue
        try:
            p_open = (1/r.odds_home) / (1/r.odds_home + 1/r.odds_draw + 1/r.odds_away)
            p_last = (1/r.last_h) / (1/r.last_h + 1/r.last_d + 1/r.last_a)
            delta = math.log(p_last/(1-p_last)) - math.log(p_open/(1-p_open))
        except (ValueError, ZeroDivisionError):
            continue
        if abs(delta) < 0.03: continue
        if delta > 0:
            bets.append({"eid": r.id, "pick": "1", "cote": float(r.last_h), "won": r.ft == "1", "ts": r.expected_start})
        else:
            bets.append({"eid": r.id, "pick": "2", "cote": float(r.last_a), "won": r.ft == "2", "ts": r.expected_start})
    out["follow_drift"] = pd.DataFrame(bets)
    return out


def stats(sub: pd.DataFrame) -> dict | None:
    n = len(sub)
    if n == 0: return None
    wins = int(sub.won.sum())
    wr = wins / n
    avg_cote = float(sub.cote.mean())
    roi = float((sub.won * (sub.cote - 1) - (~sub.won.astype(bool))).mean())
    p_impl = float((1 / sub.cote).mean() * 0.93)   # devig approx identique au script gele
    z = _z_score(wins, n, p_impl)
    return {"n": n, "wins": wins, "wr": round(wr, 4), "avg_cote": round(avg_cote, 3),
            "roi": round(roi, 4), "p_implied": round(p_impl, 4), "z": round(z, 3),
            "first_ts": str(sub.ts.min()), "last_ts": str(sub.ts.max())}


def verdict(s: dict | None) -> str:
    if s is None: return "INCONNU (0 pari forward)"
    n, roi, z = s["n"], s["roi"], s["z"]
    if z >= 2.0 and n >= 80 and roi > 0: return "PROMOUVOIR"
    if n >= 80 and (roi <= -0.10 or z <= -1.5): return "DEMOTER"
    return "CONTINUER"


def main():
    settings = load_settings()
    engine = create_engine(settings.db_url)
    corrupted = set(int(k) for k in json.loads(
        (ROOT / "exports" / "corrupted_events.json").read_text(encoding="utf-8"))["events"].keys())

    print(f"WF5 watchlist audit — gels v1={FROZEN_AT} v2={FROZEN_AT_V2}")
    df = load_dataset(engine)
    print(f"Dataset: {len(df):,} matchs ({df.expected_start.min()} -> {df.expected_start.max()})")

    longshot_ids = df.loc[df.odds_home >= 4.0, "id"].astype(int).tolist()
    em_map = load_extra_markets_for_longshots(engine, longshot_ids)
    print(f"extra_markets charges pour {len(em_map):,}/{len(longshot_ids):,} events odds_home>=4.0")

    d = build_prematch_features(df)
    all_bets = apply_signals(d, em_map)

    frozen_v1 = pd.Timestamp(FROZEN_AT)
    frozen_v2 = pd.Timestamp(FROZEN_AT_V2)

    out = {"frozen_at_v1": FROZEN_AT, "frozen_at_v2": FROZEN_AT_V2,
           "dataset_n": len(df), "dataset_max_ts": str(df.expected_start.max()),
           "rules": {"promotion": "z>=2 & n>=80 & roi>0", "demotion": "n>=80 & (roi<=-10% | z<=-1.5)"},
           "signals": {}}

    hdr = f"{'SIGNAL':<26} {'PER':<8} {'n':>5} {'WR':>7} {'cote':>6} {'ROI':>8} {'z':>7}  VERDICT"
    print("\n" + hdr); print("-" * len(hdr))
    for sig in ALL_SIGNALS:
        bets = all_bets.get(sig, pd.DataFrame())
        ft = frozen_v2 if sig in V2_SIGNALS else frozen_v1
        if bets.empty:
            out["signals"][sig] = {"frozen_at": str(ft), "histo": None, "forward": None,
                                   "forward_clean": None, "verdict": "INCONNU (0 pari total)"}
            print(f"{sig:<26} {'—':<8} {0:>5}")
            continue
        histo = stats(bets[bets.ts < ft])
        fwd_df = bets[bets.ts >= ft]
        fwd = stats(fwd_df)
        fwd_clean = stats(fwd_df[~fwd_df.eid.isin(corrupted)])
        v = verdict(fwd)
        out["signals"][sig] = {"frozen_at": str(ft), "histo": histo, "forward": fwd,
                               "forward_clean": fwd_clean, "verdict": v,
                               "n_forward_corrupted": int(fwd_df.eid.isin(corrupted).sum())}
        for per, s in [("HISTO", histo), ("FWD", fwd), ("FWDcln", fwd_clean)]:
            if s is None:
                print(f"{sig:<26} {per:<8} {0:>5}")
            else:
                tail = f"  {v}" if per == "FWD" else ""
                print(f"{sig:<26} {per:<8} {s['n']:>5} {s['wr']*100:>6.1f}% {s['avg_cote']:>6.2f} "
                      f"{s['roi']*100:>+7.1f}% {s['z']:>+7.2f}{tail}")

    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nEcrit: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
