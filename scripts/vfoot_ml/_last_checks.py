"""VÉRIFICATIONS "AU CAS OÙ" — 3 angles structurels jamais testés directement.
1. corrélation INTER-LIGUES (RNG partagé ?) sur données simultanées récentes.
2. effet HEURE RÉELLE (le résultat dépend-il de l'heure du jour ?).
3. PÉRIODICITÉ longue du flux de résultats (cycle caché tous les N rounds ?).
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np, pandas as pd
from sqlalchemy import create_engine, text
from scraper.config import load_settings

eng = create_engine(load_settings().db_url, connect_args={"timeout": 30})
df = pd.read_sql(text("""
    SELECT ev.competition c, ev.expected_start ts, ev.team_a ta, ev.team_b tb,
           r.score_a sa, r.score_b sb
    FROM events ev JOIN results r ON r.event_id=ev.id
    WHERE r.score_a IS NOT NULL AND ev.competition LIKE 'InstantLeague-%'"""), eng)
df = df.drop_duplicates(["c", "ts", "ta", "tb"])
df["tot"] = df.sa + df.sb
df["home"] = (df.sa > df.sb).astype(int)
df["ts"] = pd.to_datetime(df.ts, utc=True)
print(f"{len(df)} matchs, {df.c.nunique()} ligues", flush=True)

# ===== 1. CORRÉLATION INTER-LIGUES (total de buts par round, même timestamp) =====
print("\n=== 1. CORRÉLATION INTER-LIGUES (RNG partagé ?) ===")
rt = df.groupby(["c", "ts"])["tot"].sum().reset_index()          # total buts/round/ligue
piv = rt.pivot_table(index="ts", values="tot", columns="c")
piv = piv.dropna(thresh=2)                                       # timestamps avec >=2 ligues
pairs = 0; strong = 0; cors = []
cols = piv.columns
for i in range(len(cols)):
    for j in range(i+1, len(cols)):
        both = piv[[cols[i], cols[j]]].dropna()
        if len(both) >= 50:
            cc = np.corrcoef(both.iloc[:, 0], both.iloc[:, 1])[0, 1]
            cors.append(cc); pairs += 1
            if abs(cc) > 0.15:
                strong += 1
if cors:
    print(f"  {pairs} paires de ligues (>=50 timestamps communs) | corr moyenne {np.mean(cors):+.4f} "
          f"| |corr| max {max(abs(np.array(cors))):.4f}")
    print(f"  paires à |corr|>0.15 : {strong}  -> "
          f"{'⚠️ à creuser' if strong else 'AUCUNE — ligues indépendantes (pas de RNG partagé)'}")
else:
    print("  pas assez de timestamps communs entre ligues (scraping simultané trop récent).")

# ===== 2. EFFET HEURE RÉELLE =====
print("\n=== 2. EFFET HEURE DU JOUR (Mada) ===")
d = df.copy(); d["hr"] = (d.ts + pd.Timedelta(hours=3)).dt.hour
by_h = d.groupby("hr").agg(n=("tot", "size"), moy=("tot", "mean"), o25=("tot", lambda x: (x > 2.5).mean()),
                           home=("home", "mean"))
by_h = by_h[by_h.n >= 200]
print(f"  buts/match par heure : min {by_h.moy.min():.2f} (h{by_h.moy.idxmin()}) "
      f"max {by_h.moy.max():.2f} (h{by_h.moy.idxmax()}) | écart-type {by_h.moy.std():.3f}")
print(f"  P(over2.5) par heure : min {100*by_h.o25.min():.1f}% max {100*by_h.o25.max():.1f}% "
      f"| écart-type {100*by_h.o25.std():.2f}pp (bruit SE~{100*np.sqrt(.62*.38/by_h.n.mean()):.2f}pp)")
corr_h = np.corrcoef(d.hr, d.tot)[0, 1]
print(f"  corrélation heure <-> buts : {corr_h:+.4f}  -> "
      f"{'⚠️ effet' if abs(corr_h) > 0.03 else 'nul (aucun effet horaire)'}")

# ===== 3. PÉRIODICITÉ LONGUE du flux (par ligue principale, ordre chrono) =====
print("\n=== 3. PÉRIODICITÉ LONGUE (cycle caché du RNG ?) ===")
s = df[df.c == "InstantLeague-8035"].sort_values("ts")["tot"].values.astype(float)
s = s - s.mean()
n = len(s)
acf = []
for lag in (1, 2, 5, 10, 20, 50, 100, 200, 380):
    if n > lag + 50:
        a, b = s[:-lag], s[lag:]
        acf.append((lag, float((a @ b) / (np.sqrt((a@a)*(b@b)) or 1))))
print("  autocorrélation du total de buts à lags longs :")
for lag, v in acf:
    flag = "  <<< pic ?" if abs(v) > 0.05 else ""
    print(f"     lag {lag:>4} : {v:+.4f}{flag}")
mx = max(acf, key=lambda x: abs(x[1]))
print(f"  -> |acf| max = {abs(mx[1]):.4f} (lag {mx[0]}) : "
      f"{'⚠️ cycle à creuser' if abs(mx[1]) > 0.05 else 'aucun cycle — flux sans mémoire même à longue portée'}")
