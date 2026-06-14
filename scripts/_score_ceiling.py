"""Plafond d'accuracy score exact par profil de cote (8035, cotes ouverture)."""
import sys, json
sys.path.insert(0, ".")
from sqlalchemy import create_engine
from scraper.config import load_settings
from scraper.analysis_utils import load_corrupted_ids
import pandas as pd, numpy as np

e = create_engine(load_settings().db_url)
corrupted = load_corrupted_ids()  # 473 ids sous ["events"] (fix bug set(json.load(...)))

df = pd.read_sql("""
    SELECT e.id, o.odds_home oh, o.odds_draw od, o.odds_away oa, r.score_a sa, r.score_b sb
    FROM events e
    JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
    JOIN results r ON r.event_id=e.id
    WHERE r.score_a IS NOT NULL AND e.competition='InstantLeague-8035'
""", e)
df = df[~df.id.isin(corrupted)].copy()
df["score"] = df.sa.astype(int).astype(str) + "-" + df.sb.astype(int).astype(str)
df["fav"] = df[["oh","oa"]].min(axis=1)
print(f"n={len(df)}")

bins = [1.0,1.2,1.35,1.5,1.7,2.0,2.4,3.0,99]
df["bucket"] = pd.cut(df.fav, bins)

rows=[]
for b, g in df.groupby("bucket", observed=True):
    if len(g) < 150: continue
    vc = g.score.value_counts(normalize=True)
    rows.append(dict(bucket=str(b), n=len(g),
                     top1_score=vc.index[0], top1=round(vc.iloc[0]*100,1),
                     top3=round(vc.iloc[:3].sum()*100,1),
                     top5=round(vc.iloc[:5].sum()*100,1),
                     top3_scores=" ".join(vc.index[:3])))
out = pd.DataFrame(rows)
print(out.to_string(index=False))

# global pondéré : si on choisit TOUJOURS le score modal empirique du bucket
w = out.n / out.n.sum()
print(f"\nTop1 pondere global (plafond empirique) : {(out.top1*w).sum():.1f}%")
print(f"Top3 pondere global : {(out.top3*w).sum():.1f}%")

# selectivité : si on ne joue QUE les gros favoris
heavy = df[df.fav <= 1.35]
vc = heavy.score.value_counts(normalize=True)
print(f"\nFavoris <=1.35 uniquement (n={len(heavy)}, {len(heavy)/len(df)*100:.0f}% des matchs):")
print(f"  Top1 {vc.iloc[0]*100:.1f}% ({vc.index[0]}) · Top3 {vc.iloc[:3].sum()*100:.1f}% · Top5 {vc.iloc[:5].sum()*100:.1f}%")
