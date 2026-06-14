"""Prédiction COMPACTE multi-rounds : tous les 10 matchs de chaque round à venir.
Par match : pick 1X2, score (Top-3), EDGE TOTAL, signal CHAÎNAGE, BTTS, grosses côtes.
Usage: ./.venv/Scripts/python.exe scripts/_predict_rounds.py [n_rounds]"""
from __future__ import annotations
import sys, json
from datetime import datetime, timedelta, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from sqlalchemy import create_engine
from scraper.config import load_settings
from scraper.predictor_v5 import fit_model_v5, predict_match_v5
from scraper.journee_inference import infer_current_journee
from scraper.exotic_signals import evaluate_exotics
from scraper.market_inversion import invert_markets, apply_sim_deviations, grid_predictions

MG = timezone(timedelta(hours=3))
N_ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 10

def _load(name):
    try: return json.load(open(Path(__file__).resolve().parents[1] / "exports" / name, encoding="utf-8"))
    except Exception: return None
def _band(v, bands):
    for lo, hi, lbl in bands:
        if lo <= v < hi: return lbl
    return None
def _chain(ct, lt, ld, pb):
    if not ct or pb is None: return None
    b = ct["_bands"]; tl=_band(lt,b["tot"]); dl=_band(ld,b["diff"]); bl=_band(pb,b["btts"])
    return ct["cells"].get(f"{tl}|{dl}|{bl}") if (tl and dl and bl) else None

def main():
    s = load_settings(); e = create_engine(s.db_url)
    hist = pd.read_sql("""SELECT e.team_a,e.team_b,o.odds_home,o.odds_draw,o.odds_away,
        r.score_a,r.score_b,r.ht_score_a,r.ht_score_b FROM events e
        JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
        JOIN results r ON r.event_id=e.id WHERE r.ht_score_a IS NOT NULL AND e.competition='InstantLeague-8035'""", e)
    m5 = fit_model_v5(hist, ht_history=hist.copy(), engine=e, form_alpha=0.0)
    chain_tab = _load("chain_table.json")

    now = datetime.now(timezone.utc)
    up = pd.read_sql("""SELECT e.team_a,e.team_b,e.expected_start,e.round_info,
        o.odds_home,o.odds_draw,o.odds_away,o.extra_markets,e.id ev_id FROM events e
        JOIN odds_snapshots o ON o.id=(SELECT MAX(id) FROM odds_snapshots WHERE event_id=e.id)
        LEFT JOIN results r ON r.event_id=e.id
        WHERE r.id IS NULL AND e.expected_start IS NOT NULL AND e.competition='InstantLeague-8035'""", e)
    up["expected_start"] = pd.to_datetime(up.expected_start, utc=True)
    up = up[up.expected_start > now - pd.Timedelta(minutes=3)].copy()
    up["local"] = up.expected_start.dt.tz_convert(MG).dt.strftime("%H:%M")
    up["has_round"] = up.round_info.fillna("0").astype(str).ne("0")
    up = up.sort_values(["has_round", "ev_id"], ascending=False).drop_duplicates(["team_a","team_b","local"])
    up = up.sort_values("expected_start")
    rounds = sorted(up.local.unique())[:N_ROUNDS]
    print(f"now Mada {now.astimezone(MG):%H:%M:%S} — {len(rounds)} rounds × 10 matchs\n")

    for rd in rounds:
        ms = up[up.local == rd]
        print("█"*100)
        print(f"  ⏰ ROUND {rd}  ({len(ms)} matchs)")
        print("█"*100)
        values = []
        for _, m in ms.iterrows():
            oh, od, oa = float(m.odds_home), float(m.odds_draw), float(m.odds_away)
            p5 = predict_match_v5(m5, m.team_a, m.team_b, oh, od, oa, extra_markets=m.extra_markets)
            top5 = p5.get("top5_scores_enriched") or []
            top3 = " ".join(f"{sc}({p*100:.0f}%)" for sc, p in top5[:3]) if top5 else "?"
            pick = p5.get("primary_pick","—"); pp = (p5.get("primary_p") or 0)*100
            inv = invert_markets(oh, od, oa, m.extra_markets)
            lt, ld = inv.lam_h+inv.lam_a, inv.lam_h-inv.lam_a
            gp = grid_predictions(apply_sim_deviations(inv.lam_h, inv.lam_a, "cells"), top_k=3)
            btts = gp["btts_oui"]*100
            # EDGE TOTAL
            if lt < 2.45: edge = f"Under3.5 ~76%"
            elif lt >= 3.13: edge = f"Over2.5 ~72%"
            else: edge = "—"
            # CHAÎNAGE
            ch = _chain(chain_tab, lt, ld, gp["btts_oui"])
            chs = (f"{ch['score']} | Top3 {'+'.join(ch['top3'])} {ch['top3_cum']*100:.0f}% | {ch['ou']} {ch['ou_rate']*100:.0f}%") if ch else "—"
            # grosses côtes (exotics non-segmentés fiables)
            try:
                jr = infer_current_journee(e, m.expected_start) or 8
            except Exception:
                jr = 8
            exo = evaluate_exotics(jr, oh, oa, extra_markets=m.extra_markets, journee_reliable=False)
            big = [f"{x.signal_id}:{x.market} {x.selection} @{x.cote:.2f}" for x in exo if x.cote]
            cote_pick = {"1":oh,"X":od,"2":oa}.get(pick)

            sc1 = top5[0][0] if top5 else "?"
            mn = f"{m.team_a} v {m.team_b}"
            edge_s = {"Under3.5 ~76%":"⬇U3.5", "Over2.5 ~72%":"⬆O2.5", "—":"·"}[edge]
            ch_s = (f"CH:{ch['score']}/{ch['ou'].replace('2.5','')}{ch['ou_rate']*100:.0f}") if ch else ""
            big_s = ("💰" + " ".join(big)) if big else ""
            print(f"  {mn:<34} [{oh:>5.2f}/{od:>4.2f}/{oa:>5.2f}] {pick}{pp:>3.0f}% │ {sc1:>3} B{lt:.1f} {edge_s:<6} {ch_s:<14} {big_s}")
            for x in exo:
                if x.cote and x.cote >= 3.0:
                    values.append(f"{mn}: {x.market} '{x.selection}' @{x.cote:.2f} ({x.signal_id})")
        # synthèse values du round
        if values:
            print(f"\n  💰 GROSSES CÔTES VALUE du round {rd} :")
            for v in values: print(f"     • {v}")
        print()
    return 0

if __name__ == "__main__":
    sys.exit(main())
