# -*- coding: utf-8 -*-
"""Verification adversariale du finding negatif calib 1X2 mid-odds (script _wf4_calib1x2_3.py)."""
import sys, json, math
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from scipy.stats import norm
from scraper.config import load_settings

L8035 = "InstantLeague-8035"
NEW = ["InstantLeague-8036","InstantLeague-8037","InstantLeague-8042","InstantLeague-8043",
       "InstantLeague-8044","InstantLeague-8056","InstantLeague-8060","InstantLeague-8065"]
eng = create_engine(load_settings().db_url)
with eng.connect() as c:
    # 1) competitions presentes dans le join (collision / hors-9 ?)
    comps = pd.read_sql(text("""
        SELECT e.competition, COUNT(*) n FROM events e
        JOIN results r ON r.event_id=e.id GROUP BY e.competition ORDER BY n DESC"""), c)
    print("== competitions avec resultats =="); print(comps.to_string())
    # 2) MIN(id) vs MIN(captured_at): l'id min est-il bien le snapshot le plus ancien ?
    chk = pd.read_sql(text("""
        SELECT COUNT(*) AS n_bad FROM (
          SELECT event_id, MIN(id) mid FROM odds_snapshots GROUP BY event_id) m
        JOIN odds_snapshots a ON a.id = m.mid
        JOIN odds_snapshots b ON b.event_id = m.event_id AND b.captured_at < a.captured_at"""), c)
    print("\nsnapshots avec captured_at < celui du MIN(id):", int(chk.n_bad[0]))
    # 3) opening snapshot poste-kickoff ?
    late = pd.read_sql(text("""
        SELECT COUNT(*) n_late, SUM(CASE WHEN o.captured_at >= e.expected_start THEN 1 ELSE 0 END) n_post
        FROM events e JOIN results r ON r.event_id=e.id
        JOIN odds_snapshots o ON o.id=(SELECT MIN(o2.id) FROM odds_snapshots o2 WHERE o2.event_id=e.id)"""), c)
    print("events finis:", int(late.n_late[0]), "| opening snapshot >= expected_start:", int(late.n_post[0]))
    df = pd.read_sql(text("""
        SELECT e.id AS event_id, e.competition, e.expected_start,
               r.score_a, r.score_b, r.ht_score_a, r.ht_score_b, r.goals_json,
               o.odds_home, o.odds_draw, o.odds_away
        FROM events e JOIN results r ON r.event_id=e.id
        JOIN odds_snapshots o ON o.id=(SELECT MIN(o2.id) FROM odds_snapshots o2 WHERE o2.event_id=e.id)"""), c)

with open("exports/corrupted_events.json", encoding="utf-8") as f:
    corr = json.load(f)
df = df[~df["event_id"].isin(set(int(k) for k in corr["events"].keys()))].copy()
def goals_ok(row):
    if row.ht_score_a is not None and row.ht_score_b is not None:
        if row.ht_score_a > row.score_a or row.ht_score_b > row.score_b: return False
    gj = row.goals_json
    if gj:
        try:
            g = json.loads(gj)
            if isinstance(g, list) and len(g) > 0 and len(g) != row.score_a + row.score_b: return False
        except Exception: pass
    return True
df = df[df.apply(goals_ok, axis=1)].copy()
df = df.dropna(subset=["odds_home","odds_draw","odds_away","score_a","score_b"])
df["outcome"] = np.where(df.score_a>df.score_b,"H",np.where(df.score_a<df.score_b,"A","D"))
df["booksum"] = 1/df.odds_home + 1/df.odds_draw + 1/df.odds_away
df["expected_start"] = pd.to_datetime(df["expected_start"])
hors9 = df[~df.competition.isin([L8035]+NEW)]
print("\nrows clean:", len(df), "| rows HORS des 9 ligues:", len(hors9), sorted(hors9.competition.unique()) if len(hors9) else "")

POSCOL = {"H":"odds_home","D":"odds_draw","A":"odds_away"}
def ev(sub, pos):
    n=len(sub)
    if n==0: return dict(n=0)
    odds=sub[POSCOL[pos]].values.astype(float); bs=sub["booksum"].values
    qn=(1/odds)/bs; win=(sub["outcome"]==pos).values; k=int(win.sum())
    profit=np.where(win,odds-1.0,-1.0); roi=float(profit.mean()*100)
    mu=float(qn.sum()); var=float((qn*(1-qn)).sum())
    z=(k-mu)/math.sqrt(var) if var>0 else 0.0
    return dict(n=n,wr=round(k/n,4),dev_pp=round((k-mu)/n*100,2),z=round(z,2),
                p=round(float(2*(1-norm.cdf(abs(z)))),4),roi=round(roi,2))

# 4) split temporel pooled-9 en 2 moities (stabilite, periode chanceuse ?)
df = df.sort_values("expected_start")
mid = df.expected_start.quantile(0.5)
h1, h2 = df[df.expected_start < mid], df[df.expected_start >= mid]
print("\n== T1 (1.8-5.0) par moitie temporelle pooled-9 ==")
for name, dd in [("H1(ancien)", h1), ("H2(recent)", h2)]:
    for pos,col in POSCOL.items():
        s = dd[(dd[col]>=1.8)&(dd[col]<5.0)]
        r = ev(s,pos); print(f"{name} {pos}: {r}")

# 5) les cellules 'presque positives' du scan original survivent-elles hors echantillon ?
#    split temporel 70/30 pooled-9 (train=scan, test=verite)
cut = df.expected_start.quantile(0.70)
tr, te = df[df.expected_start < cut], df[df.expected_start >= cut]
print("\n== cellules candidates positives: train(70) vs test(30) temporel pooled-9 ==")
CELLS = [("H",3.25,3.5),("H",3.75,4.0),("H",4.0,4.25),("A",3.25,3.5),("A",4.0,4.25),("D",2.5,2.75),("D",3.0,3.25)]
for pos,lo,hi in CELLS:
    col=POSCOL[pos]
    rtr=ev(tr[(tr[col]>=lo)&(tr[col]<hi)],pos); rte=ev(te[(te[col]>=lo)&(te[col]<hi)],pos)
    print(f"{pos} [{lo},{hi}): TRAIN {rtr} | TEST {rte}")

# 6) bootstrap CI du ROI pooled (1.8-5.0, 3 positions empilees, mise 1u)
rows=[]
for pos,col in POSCOL.items():
    s=df[(df[col]>=1.8)&(df[col]<5.0)]
    rows.append(np.where((s["outcome"]==pos).values, s[col].values-1.0, -1.0))
pr=np.concatenate(rows); rng=np.random.default_rng(42)
boot=np.array([rng.choice(pr,len(pr),replace=True).mean() for _ in range(2000)])*100
print(f"\nROI pooled 1.8-5.0 (3 pos, n={len(pr)}): {pr.mean()*100:.2f}%  IC95 boot [{np.percentile(boot,2.5):.2f}, {np.percentile(boot,97.5):.2f}]")

# 7) idem par ligue individuellement (une ligue cachee profitable ?)
print("\n== ROI 1.8-5.0 par ligue (toutes positions) ==")
for comp in [L8035]+NEW:
    dd=df[df.competition==comp]; allp=[]
    for pos,col in POSCOL.items():
        s=dd[(dd[col]>=1.8)&(dd[col]<5.0)]
        allp.append(np.where((s["outcome"]==pos).values, s[col].values-1.0, -1.0))
    a=np.concatenate(allp)
    if len(a)==0: continue
    se=a.std()/math.sqrt(len(a)); zz=(a.mean()+0.0566)/se  # vs -5.66% attendu
    print(f"{comp}: n={len(a)} ROI={a.mean()*100:.2f}% (z vs -5.66%: {zz:.2f})")
