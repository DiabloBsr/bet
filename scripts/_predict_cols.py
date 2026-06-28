"""Round courant (ou [HH:MM]) — tableau EN COLONNES : Top1 & Top3 de V2, V5, V6,
puis ENSEMBLE Top1/Top3 et Zone. Une ligne par match. Standalone.
Usage: ./.venv/Scripts/python.exe scripts/_predict_cols.py [HH:MM]"""
from __future__ import annotations
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from sqlalchemy import create_engine
from scraper.config import load_settings
from scraper.predictor_v5 import fit_model_v5, predict_match_v5
from scraper.score_predictor_v2 import ScorePredictorV2
from scraper.score_predictor_v6 import predict_score_v6
from scraper.score_ensemble import ensemble_from_raw
from scraper.journee_inference import infer_current_journee
from scraper.market_inversion import invert_markets, apply_sim_deviations, total_distribution

MG = timezone(timedelta(hours=3))
COMMON = ["0-0","1-0","0-1","1-1","2-0","0-2","2-1","1-2","2-2","3-0","0-3","3-1","1-3"]
def t1(l): return l[0][0] if l else "-"
def t3(l): return "/".join(s for s,_ in l[:3]) if l else "-"

def main():
    s=load_settings(); e=create_engine(s.db_url); now=datetime.now(timezone.utc)
    hist=pd.read_sql("""SELECT e.team_a,e.team_b,o.odds_home,o.odds_draw,o.odds_away,
        r.score_a,r.score_b,r.ht_score_a,r.ht_score_b FROM events e
        JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
        JOIN results r ON r.ht_score_a IS NOT NULL AND r.event_id=e.id WHERE e.competition='InstantLeague-8035'""",e)
    m5=fit_model_v5(hist,ht_history=hist.copy(),engine=e,form_alpha=0.0); v2=ScorePredictorV2(e)
    up=pd.read_sql("""SELECT e.team_a,e.team_b,e.expected_start,e.round_info,o.odds_home oh,o.odds_draw od,
        o.odds_away oa,o.extra_markets,e.id ev FROM events e
        JOIN odds_snapshots o ON o.id=(SELECT MAX(id) FROM odds_snapshots WHERE event_id=e.id)
        LEFT JOIN results r ON r.event_id=e.id
        WHERE r.id IS NULL AND e.expected_start IS NOT NULL AND e.competition='InstantLeague-8035'""",e)
    up["es"]=pd.to_datetime(up.expected_start,utc=True); up=up[up.es>now-pd.Timedelta(minutes=3)]
    up["local"]=up.es.dt.tz_convert(MG).dt.strftime("%H:%M")
    up=up.sort_values(["es","ev"]).drop_duplicates(["team_a","team_b","local"])
    rounds=sorted(up.local.unique()); TARGET=sys.argv[1] if len(sys.argv)>1 else (rounds[1] if len(rounds)>1 else rounds[0])
    ms=up[up.local==TARGET]
    print(f"now {now.astimezone(MG):%H:%M} | ROUND {TARGET}\n")
    H=(f"{'match':<26}{'V2_T1':>6}{'V2_Top3':>13}{'V5_T1':>6}{'V5_Top3':>13}"
       f"{'V6_T1':>6}{'V6_Top3':>13}{'ENS_T1':>7}{'ENS_Top3':>13}{'zone':>9}")
    print(H); print("-"*len(H))
    for r in ms.itertuples():
        oh,od,oa=float(r.oh),float(r.od),float(r.oa)
        if oh<=1 or oa<=1: continue
        inv=invert_markets(oh,od,oa,r.extra_markets); g=apply_sim_deviations(inv.lam_h,inv.lam_a,"cells")
        td=total_distribution(g); lt=inv.lam_h+inv.lam_a; mt=int(td.argmax())
        p5=predict_match_v5(m5,r.team_a,r.team_b,oh,od,oa,extra_markets=r.extra_markets)
        v5l=p5.get("top5_scores_enriched") or []
        v6=predict_score_v6(oh,od,oa,r.extra_markets,top_n=20); v6l=v6["top"]
        try:
            grid={sc:p for sc,p in v5l}
            for sc in COMMON: grid.setdefault(sc,0.01)
            tt=sum(grid.values()); grid={k:v/tt for k,v in grid.items()}
            jrn=int(r.round_info) if str(r.round_info).isdigit() and r.round_info!="0" else (infer_current_journee(e,r.es) or 8)
            v2t=v2.predict(r.team_a,r.team_b,jrn,v5_score_grid=grid,extra_markets=r.extra_markets,odds_h=oh,odds_a=oa,top_n=5)
            v2l=[(t[0],t[1]) for t in v2t]
        except Exception: v2t=[]; v2l=[]
        ens=ensemble_from_raw(v5l,v2t if v2l else None,v6,top_n=3)["top"]
        zone=("3buts" if mt==3 and 2.5<=lt<3.05 else ("Under" if lt<2.45 else ("Over" if lt>=3.3 else f"tot{mt}")))
        mn=(r.team_a+" v "+r.team_b)[:25]
        print(f"{mn:<26}{t1(v2l):>6}{t3(v2l):>13}{t1(v5l):>6}{t3(v5l):>13}"
              f"{t1(v6l):>6}{t3(v6l):>13}{ens[0][0] if ens else '-':>7}{t3(ens):>13}{zone:>9}")

if __name__=="__main__":
    main()
