"""Matchs à 3 BUTS du round courant, avec scores exacts V5 ET V2 côte à côte.
Standalone (ne touche pas _predict_one_round.py). Auto-cible le prochain round.
Usage: ./.venv/Scripts/python.exe scripts/_predict3_v2v5.py [HH:MM]"""
from __future__ import annotations
import sys, json
from datetime import datetime, timedelta, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from sqlalchemy import create_engine
from scraper.config import load_settings
from scraper.predictor_v5 import fit_model_v5, predict_match_v5
from scraper.score_predictor_v2 import ScorePredictorV2
from scraper.journee_inference import infer_current_journee
from scraper.market_inversion import invert_markets, apply_sim_deviations, grid_predictions, total_distribution

MG = timezone(timedelta(hours=3))
ROOT = Path(__file__).resolve().parents[1]
ct = json.load(open(ROOT/"exports"/"chain_table.json", encoding="utf-8"))
def _bnd(v, bs):
    for lo, hi, l in bs:
        if lo <= v < hi: return l
    return None
def _chain(lt, ld, pb):
    b = ct["_bands"]; tl=_bnd(lt,b["tot"]); dl=_bnd(ld,b["diff"]); bl=_bnd(pb,b["btts"])
    return ct["cells"].get(f"{tl}|{dl}|{bl}") if (tl and dl and bl) else None

def main():
    s = load_settings(); e = create_engine(s.db_url); now = datetime.now(timezone.utc)
    hist = pd.read_sql("""SELECT e.team_a,e.team_b,o.odds_home,o.odds_draw,o.odds_away,
        r.score_a,r.score_b,r.ht_score_a,r.ht_score_b FROM events e
        JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
        JOIN results r ON r.event_id=e.id WHERE r.ht_score_a IS NOT NULL AND e.competition='InstantLeague-8035'""", e)
    m5 = fit_model_v5(hist, ht_history=hist.copy(), engine=e, form_alpha=0.0)
    v2 = ScorePredictorV2(e)
    up = pd.read_sql("""SELECT e.team_a,e.team_b,e.expected_start,e.round_info,o.odds_home oh,o.odds_draw od,
        o.odds_away oa,o.extra_markets,e.id ev FROM events e
        JOIN odds_snapshots o ON o.id=(SELECT MAX(id) FROM odds_snapshots WHERE event_id=e.id)
        LEFT JOIN results r ON r.event_id=e.id
        WHERE r.id IS NULL AND e.expected_start IS NOT NULL AND e.competition='InstantLeague-8035'""", e)
    up["es"] = pd.to_datetime(up.expected_start, utc=True); up = up[up.es > now - pd.Timedelta(minutes=3)]
    up["local"] = up.es.dt.tz_convert(MG).dt.strftime("%H:%M")
    up = up.sort_values(["es","ev"]).drop_duplicates(["team_a","team_b","local"])
    rounds = sorted(up.local.unique())
    TARGET = sys.argv[1] if len(sys.argv) > 1 else (rounds[1] if len(rounds) > 1 else rounds[0])
    ms = up[up.local == TARGET]
    print(f"now {now.astimezone(MG):%H:%M} | ROUND {TARGET} — matchs à 3 buts, scores V5 + V2\n")
    print(f"{'match':<31}{'λtot':>5}{'P(3)':>6}  {'V5 Top-3':<24}{'V2 Top-3':<24} chaîne")
    print("-"*104)
    out = []
    for r in ms.itertuples():
        oh,od,oa = float(r.oh),float(r.od),float(r.oa)
        if oh<=1 or oa<=1: continue
        inv = invert_markets(oh,od,oa,r.extra_markets)
        g = apply_sim_deviations(inv.lam_h,inv.lam_a,"cells"); td = total_distribution(g)
        lt, ld = inv.lam_h+inv.lam_a, inv.lam_h-inv.lam_a; p3 = float(td[3]); modal = int(td.argmax())
        if modal != 3: continue  # garder seulement les matchs où total=3 est modal
        p5 = predict_match_v5(m5, r.team_a, r.team_b, oh, od, oa, extra_markets=r.extra_markets)
        top5 = p5.get("top5_scores_enriched") or []
        v5s = " ".join(f"{sc}({p*100:.0f}%)" for sc,p in top5[:3]) if top5 else "?"
        # V2 (même appel que le prédicteur)
        try:
            grid = {sc:p for sc,p in top5}
            for sc in ["0-0","1-0","0-1","1-1","2-0","0-2","2-1","1-2","2-2","3-0","0-3","3-1","1-3"]:
                grid.setdefault(sc,0.01)
            tt=sum(grid.values()); grid={k:v/tt for k,v in grid.items()}
            jrn = int(r.round_info) if str(r.round_info).isdigit() and r.round_info!="0" else (infer_current_journee(e,r.es) or 8)
            v2t = v2.predict(r.team_a,r.team_b,jrn,v5_score_grid=grid,extra_markets=r.extra_markets,odds_h=oh,odds_a=oa,top_n=3)
            v2s = " ".join(f"{sc}({p*100:.0f}%)" for sc,p,_ in v2t[:3])
        except Exception:
            v2s = "?"
        ch = _chain(lt,ld,gp_btts(g)); chs = ch["score"] if ch else "-"
        print(f"{r.team_a+' v '+r.team_b:<31}{lt:>5.1f}{p3*100:>5.0f}%  {v5s:<24}{v2s:<24} {chs}")
        out.append((r.team_a,r.team_b,v5s,v2s,chs))
    if not out:
        print("  (aucun match avec total=3 modal dans ce round)")

def gp_btts(g):
    return grid_predictions(g, top_k=1)["btts_oui"]

if __name__ == "__main__":
    main()
