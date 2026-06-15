"""ENSEMBLE V5 + V2 + V6 pour le score exact — backtest honnête OOS.
Compare chaque modèle seul + les blends. Mesure Top1/Top3 sur le TEST.
⚠️ V2 interroge la DB (historique des paires incluant le test) -> potentiellement
fuité ; on le marque. V5 (fit train) et V6 (book devig) sont propres.
Usage: ./.venv/Scripts/python.exe scripts/_score_ensemble.py [n_sample]"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sqlalchemy import create_engine
from scraper.config import load_settings
from scraper.predictor_v5 import fit_model_v5, predict_match_v5
from scraper.score_predictor_v2 import ScorePredictorV2
from scraper.score_predictor_v6 import predict_score_v6
from scraper.journee_inference import infer_current_journee

NS = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
COMMON = ["0-0","1-0","0-1","1-1","2-0","0-2","2-1","1-2","2-2","3-0","0-3","3-1","1-3","3-2","2-3","4-0","0-4"]

s = load_settings(); e = create_engine(s.db_url)
hist = pd.read_sql("""SELECT e.team_a,e.team_b,e.expected_start,o.odds_home,o.odds_draw,o.odds_away,
    o.extra_markets, r.score_a,r.score_b,r.ht_score_a,r.ht_score_b FROM events e
    JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
    JOIN results r ON r.score_a IS NOT NULL AND r.event_id=e.id WHERE e.competition='InstantLeague-8035'
    ORDER BY e.expected_start""", e)
hist = hist.dropna(subset=["score_a"]).reset_index(drop=True)
cut = int(len(hist)*0.70)
train = hist.iloc[:cut]; test = hist.iloc[cut:]
test_ht = train[train.ht_score_a.notna()]
m5 = fit_model_v5(train[train.ht_score_a.notna()].copy(), ht_history=train[train.ht_score_a.notna()].copy(), engine=e, form_alpha=0.0)
v2 = ScorePredictorV2(e)
test = test.sample(min(NS, len(test)), random_state=1).reset_index(drop=True)
print(f"backtest ensemble OOS — n_test={len(test)} (train={cut})\n")

def norm(d):
    t=sum(d.values()); return {k:v/t for k,v in d.items()} if t>0 else {}
def topk(d,k): return [s for s,_ in sorted(d.items(),key=lambda x:-x[1])[:k]]

rec = {m:{"t1":0,"t3":0} for m in ["V5","V2","V6","V5+V6","V5+V2+V6","V2+V6"]}
n=0
for r in test.itertuples():
    oh,od,oa=float(r.odds_home),float(r.odds_draw),float(r.odds_away)
    if oh<=1 or oa<=1: continue
    real=f"{int(r.score_a)}-{int(r.score_b)}"
    # V5
    p5=predict_match_v5(m5,r.team_a,r.team_b,oh,od,oa,extra_markets=r.extra_markets)
    top5=p5.get("top5_scores_enriched") or []
    d5=norm({sc:p for sc,p in top5}) if top5 else {}
    # V6 (book)
    v6=predict_score_v6(oh,od,oa,r.extra_markets,top_n=20)
    d6=norm(dict(v6["top"]))
    # V2 (⚠️ leaky)
    try:
        grid={sc:p for sc,p in top5}
        for sc in COMMON: grid.setdefault(sc,0.01)
        tt=sum(grid.values()); grid={k:v/tt for k,v in grid.items()}
        jrn=infer_current_journee(e,pd.Timestamp(r.expected_start)) or 8
        v2t=v2.predict(r.team_a,r.team_b,jrn,v5_score_grid=grid,extra_markets=r.extra_markets,odds_h=oh,odds_a=oa,top_n=20)
        d2=norm({sc:p for sc,p,_ in v2t})
    except Exception:
        d2={}
    if not d5 or not d6: continue
    n+=1
    def blend(*ds):
        keys=set().union(*[set(d) for d in ds if d]);
        return {k:sum(d.get(k,0) for d in ds if d)/len([d for d in ds if d]) for k in keys}
    cand={"V5":d5,"V2":d2 or d5,"V6":d6,"V5+V6":blend(d5,d6),"V5+V2+V6":blend(d5,d2,d6),"V2+V6":blend(d2 or d6,d6)}
    for m,d in cand.items():
        if real in topk(d,1): rec[m]["t1"]+=1
        if real in topk(d,3): rec[m]["t3"]+=1

print(f"{'modèle':<14}{'Top1':>7}{'Top3':>7}   note")
print("-"*46)
notes={"V5":"propre","V2":"⚠️ fuite (paires)","V6":"propre (book)","V5+V6":"propre","V5+V2+V6":"⚠️ contient V2","V2+V6":"⚠️ contient V2"}
for m in ["V5","V2","V6","V5+V6","V2+V6","V5+V2+V6"]:
    print(f"{m:<14}{rec[m]['t1']/n*100:>6.1f}%{rec[m]['t3']/n*100:>6.1f}%   {notes[m]}")
print(f"\nn={n} | Plafond empirique : Top1 ~12-15% / Top3 ~30-36%.")
print("Lecture : si V5+V6 (propre) ne bat pas V6 seul -> l'ensemble n'aide pas, le book suffit.")
print("Si V5+V2+V6 paraît meilleur, c'est probablement la FUITE de V2, pas un vrai gain.")
