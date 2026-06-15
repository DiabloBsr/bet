"""Étude des GROSSES COTES score-exact (50→100) : quand apparaissent-elles (quels
scores), quand tapent-elles (hit rate réel), et y a-t-il un edge (hit > 1/cote) ?
Cotes pré-match vs résultat = pas de fuite. Split chrono pour vérif OOS."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sqlalchemy import create_engine
from scraper.config import load_settings
from scraper.market_inversion import parse_extra_markets, score_exact_odds

s = load_settings(); e = create_engine(s.db_url)
df = pd.read_sql("""SELECT e.expected_start, o.extra_markets, r.score_a sa, r.score_b sb FROM events e
    JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
    JOIN results r ON r.event_id=e.id WHERE r.score_a IS NOT NULL AND e.competition='InstantLeague-8035'
    AND o.extra_markets IS NOT NULL ORDER BY e.expected_start""", e)
df = df.reset_index(drop=True); cut = int(len(df)*0.70)
print(f"events: {len(df)} (TRAIN {cut} / TEST {len(df)-cut})\n")

# collecte de toutes les cellules (score, cote, hit, split)
cells = []  # (cote, hit, score, is_test)
appear = {}  # score -> [n_offered_high(>=50), n_hit_high]
for i, r in enumerate(df.itertuples()):
    em = parse_extra_markets(r.extra_markets); se = score_exact_odds(em)
    if not se: continue
    real = f"{int(r.sa)}-{int(r.sb)}"; istest = i >= cut
    for sc, cote in se.items():
        if not cote or cote <= 1: continue
        hit = (sc == real)
        cells.append((cote, hit, sc, istest))
        if cote >= 50:
            a = appear.setdefault(sc, [0,0]); a[0]+=1; a[1]+= 1 if hit else 0

C = pd.DataFrame(cells, columns=["cote","hit","score","test"])
print("="*78)
print("HIT-RATE RÉEL vs BREAKEVEN par bande de cote (toutes cellules score-exact)")
print("="*78)
print(f"{'bande cote':<14}{'n cells':>9}{'hits':>7}{'hit%':>8}{'breakeven':>11}{'EV':>8}")
for lo,hi in [(1,5),(5,10),(10,20),(20,30),(30,50),(50,70),(70,90),(90,100)]:
    sub = C[(C.cote>=lo)&(C.cote<hi)]
    if len(sub)==0: continue
    hr = sub.hit.mean(); be = 1/sub.cote.mean(); ev = hr*sub.cote.mean()-1
    print(f"{str(lo)+'-'+str(hi):<14}{len(sub):>9}{int(sub.hit.sum()):>7}{hr*100:>7.2f}%{be*100:>10.2f}%{ev*100:>+7.0f}%")

print("\n"+"="*78)
print("ZOOM grosses cotes (>=50) : OOS (test) pour fiabilité")
print("="*78)
for lo,hi,lbl in [(50,70,"50-70"),(70,90,"70-90"),(90,100,"90-100 (~'cote 100')")]:
    tr=C[(~C.test)&(C.cote>=lo)&(C.cote<hi)]; te=C[(C.test)&(C.cote>=lo)&(C.cote<hi)]
    def line(x):
        if len(x)==0: return "n=0"
        return f"n={len(x)} hit={x.hit.mean()*100:.2f}% be={100/x.cote.mean():.2f}% EV={ (x.hit.mean()*x.cote.mean()-1)*100:+.0f}%"
    print(f"  {lbl:<22} TRAIN[{line(tr)}]  TEST[{line(te)}]")

print("\n"+"="*78)
print("QUELS SCORES portent la cote >=50, et lesquels TAPENT le + (hit/offered)")
print("="*78)
ap = sorted(appear.items(), key=lambda kv: -kv[1][0])
for sc,(n,h) in ap[:18]:
    print(f"  {sc:<6} offert en cote>=50 : {n:>5}x | a tapé {h:>3}x ({h/n*100:.2f}%)  cote-eq~{n/max(h,1):.0f}")

# 'quand ça tape' : profil des hits de grosse cote (>=50)
hits_big = C[(C.cote>=50)&(C.hit)]
print(f"\nQuand une cote>=50 TAPE ({len(hits_big)} cas) — scores réalisés :")
print("  "+" ".join(f"{k}({v})" for k,v in hits_big.score.value_counts().head(10).items()))
