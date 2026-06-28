"""EXPÉRIENCE PLAFOND — construit plusieurs moteurs de score GENUINEMENT DIFFÉRENTS
(uniques, biais inductifs distincts), backteste chacun + l'ensemble, et montre si
ajouter de la diversité fait MONTER l'accuracy ou si ça PLAFONNE (limite de Bayes).
Tous train-only (propre). Split chrono 70/30.
Moteurs :
  BOOK   : cotes score offertes devigées (marché ≈ vraie distribution)
  SIM    : Poisson inversé + déviations RNG (génératif)
  KNN    : k plus proches voisins en (λh,λa) -> distribution empirique des voisins
  BUCKET : P(score | bin de favori × bin de λtot) empirique
  PAIR   : P(score | paire d'équipes exacte) empirique
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scraper.market_inversion import invert_markets, apply_sim_deviations

CSV = Path(__file__).resolve().parents[1]/"exports"/"combokeys_features.csv"
SC = ['0-0','1-0','0-1','1-1','2-0','0-2','2-1','1-2','2-2','3-0','0-3','3-1','1-3','3-2','2-3','4-0','0-4','4-1','1-4']
df = pd.read_csv(CSV).sort_values("expected_start").reset_index(drop=True)
cut = int(len(df)*0.70); tr, te = df.iloc[:cut].copy(), df.iloc[cut:].copy()
print(f"n_train={len(tr)} n_test={len(te)}\n")

def norm(d):
    t=sum(d.values()); return {k:v/t for k,v in d.items() if v>0} if t>0 else {}
def topk(d,k): return [s for s,_ in sorted(d.items(),key=lambda x:-x[1])[:k]]

# --- BOOK ---
def book_d(row):
    d={}
    for s in SC:
        c=row.get(f"off_s_{s}")
        if c and 1<c<99.99: d[s]=1/c
    return norm(d)
# --- SIM ---
def sim_d(row):
    try:
        inv=invert_markets(float(row["oh"]),float(row["od"]),float(row["oa"]),row.get("extra_markets"))
        g=apply_sim_deviations(inv.lam_h,inv.lam_a,"cells"); d={}
        for h in range(g.shape[0]):
            for a in range(g.shape[1]):
                if g[h,a]>0: d[f"{h}-{a}"]=float(g[h,a])
        return norm(d)
    except Exception: return {}
# --- KNN sur (λh,λa) ---
trX=tr[["lam_h","lam_a"]].values; tree=cKDTree(trX); tr_sc=tr.exact_score.values
def knn_d(row,K=400):
    _,idx=tree.query([row["lam_h"],row["lam_a"]],k=K)
    vals,cnt=np.unique(tr_sc[idx],return_counts=True)
    return norm(dict(zip(vals,cnt)))
# --- BUCKET fav×λtot ---
tr["fb"]=pd.cut(tr.fav,[0,1.3,1.6,2.0,2.5,3.5,99]); tr["tb"]=pd.cut(tr.lam_tot,[0,2.2,2.6,3.0,3.5,99])
bucket={}
for (fb,tb),g in tr.groupby(["fb","tb"],observed=True):
    bucket[(fb,tb)]=norm(g.exact_score.value_counts().to_dict())
def bucket_d(row):
    fb=pd.cut([row["fav"]],[0,1.3,1.6,2.0,2.5,3.5,99])[0]; tb=pd.cut([row["lam_tot"]],[0,2.2,2.6,3.0,3.5,99])[0]
    return bucket.get((fb,tb),{})
# --- PAIR exacte (train-only) ---
pair={}
for (a,b),g in tr.groupby(["team_a","team_b"]):
    if len(g)>=5: pair[(a,b)]=norm(g.exact_score.value_counts().to_dict())
def pair_d(row): return pair.get((row["team_a"],row["team_b"]),{})

ENG={"BOOK":book_d,"SIM":sim_d,"KNN":knn_d,"BUCKET":bucket_d,"PAIR":pair_d}
rows=te.to_dict("records")
# pré-calc des distributions
dists={m:[] for m in ENG}; reals=[]
for row in rows:
    if not(row["oh"] and row["oa"]) or row["oh"]<=1: continue
    reals.append(row["exact_score"])
    for m,fn in ENG.items(): dists[m].append(fn(row))

def ev(ds,k):
    h=sum(1 for d,r in zip(ds,reals) if d and r in topk(d,k)); n=sum(1 for d in ds if d); return h/n*100 if n else 0
print(f"{'moteur':<16}{'Top1':>7}{'Top3':>7}{'couv':>7}")
print("-"*40)
for m in ENG: print(f"{m:<16}{ev(dists[m],1):>6.1f}%{ev(dists[m],3):>6.1f}%{sum(1 for d in dists[m] if d)/len(reals)*100:>6.0f}%")

def blend(ms):
    out=[]
    for i in range(len(reals)):
        parts=[norm(dists[m][i]) for m in ms if dists[m][i]]
        if not parts: out.append({}); continue
        keys=set().union(*[set(p) for p in parts])
        out.append({k:sum(p.get(k,0) for p in parts)/len(parts) for k in keys})
    return out
print("\nENSEMBLES :")
for combo in [["BOOK","SIM"],["BOOK","SIM","KNN"],["BOOK","SIM","KNN","BUCKET"],
              ["BOOK","SIM","KNN","BUCKET","PAIR"]]:
    b=blend(combo); print(f"  {'+'.join(combo):<34}{ev(b,1):>6.1f}%{ev(b,3):>6.1f}%")
print("\nPlafond empirique : Top1 ~12-15% / Top3 ~30-36%.")
print("Si chaque moteur ajouté ne fait PAS monter -> on est à la limite de Bayes (le RNG est irréductible).")
