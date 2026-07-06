"""COMPLÉMENTS D'AUDIT — les pièces du template pas encore faites exactement :
  3.1 TRJ par marché (retour au joueur = 1/overround)
  4.1 FFT / spectral (cycle caché du RNG ?)
  4.3 Détection de RESET (rupture brutale = maj logicielle ?)
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np, pandas as pd
from sqlalchemy import create_engine, text
from scraper.config import load_settings

LG = "InstantLeague-8035"
eng = create_engine(load_settings().db_url, connect_args={"timeout": 30})
df = pd.read_sql(text(f"""
    SELECT ev.expected_start ts, o.odds_home oh, o.odds_draw od, o.odds_away oa,
           o.extra_markets xm, r.score_a sa, r.score_b sb
    FROM events ev JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=ev.id)
    JOIN results r ON r.event_id=ev.id
    WHERE r.score_a IS NOT NULL AND ev.competition='{LG}' AND o.odds_home>1 ORDER BY ev.expected_start"""), eng)
df = df.drop_duplicates(["ts", "sa", "sb"]).reset_index(drop=True)
df["ts"] = pd.to_datetime(df.ts, utc=True); df["tot"] = df.sa + df.sb
n = len(df); print(f"{n} matchs", flush=True)


def gm(xm, pref):
    for k, v in (xm or {}).items():
        if k.replace("\x82", "é").replace("\xe9", "é").startswith(pref):
            return v
    return None


# ===== 3.1 TRJ PAR MARCHÉ (partitions) =====
print("\n=== 3.1 TRJ PAR MARCHÉ (retour au joueur = 1/overround) ===")
mk_keys = {"1X2": None, "+/-": ["> 3.5", "< 3.5"], "G/NG": ["Oui", "Non"],
           "Mi-tps 1X2": ["1", "X", "2"], "Total de buts": [str(k) for k in range(7)],
           "Pair/Impair": ["Pair", "Impair"]}
acc = {m: [] for m in mk_keys}
for r in df.itertuples():
    acc["1X2"].append(1/r.oh + 1/r.od + 1/r.oa)
    try: xm = json.loads(r.xm) if isinstance(r.xm, str) else (r.xm or {})
    except Exception: xm = {}
    for m, keys in mk_keys.items():
        if keys is None: continue
        sel = gm(xm, m)
        if isinstance(sel, dict):
            vals = [1/sel[k] for k in keys if isinstance(sel.get(k), (int, float)) and 1 < sel[k] < 99.99]
            if len(vals) == len(keys):
                acc[m].append(sum(vals))
print(f"  {'marché':<16}{'overround':>11}{'TRJ (retour)':>14}{'marge book':>12}")
for m, arr in acc.items():
    if arr:
        ov = np.mean(arr); trj = 1/ov
        print(f"  {m:<16}{ov:>10.3f}{100*trj:>13.1f}%{100*(1-trj):>11.1f}%")
print("  -> TRJ le + élevé = marge la + faible = 'moins pire' pour parier (mais toujours <100%).")

# ===== 4.1 FFT / SPECTRAL =====
print("\n=== 4.1 FFT — cycle caché dans la série des résultats ? ===")
for name, s in (("total de buts", df.tot.values.astype(float)),
                ("victoire domicile (0/1)", (df.sa > df.sb).astype(float).values)):
    x = s - s.mean()
    F = np.abs(np.fft.rfft(x)); freqs = np.fft.rfftfreq(len(x))
    F[0] = 0
    noise = np.median(F[1:])
    peak = np.argmax(F[1:]) + 1
    ratio = F[peak] / (noise or 1)
    period = 1/freqs[peak] if freqs[peak] > 0 else np.inf
    print(f"  {name:<26} pic à période ~{period:.0f} matchs | amplitude/bruit = {ratio:.2f}x "
          f"-> {'⚠️ CYCLE' if ratio > 6 else 'aucun cycle (bruit blanc)'}")
print("  (bruit blanc = pas de fréquence dominante = RNG sans période exploitable)")

# ===== 4.3 DÉTECTION DE RESET (rupture temporelle) =====
print("\n=== 4.3 RESET DU RNG — le comportement change-t-il brutalement ? ===")
df["day"] = df.ts.dt.date
daily = df.groupby("day").agg(n=("tot", "size"), moy=("tot", "mean"),
                              o25=("tot", lambda x: (x > 2.5).mean()),
                              home=("sa", lambda x: (x > df.loc[x.index, "sb"]).mean()))
daily = daily[daily.n >= 200]
for col, lbl in (("moy", "buts/match"), ("o25", "P(over2.5)"), ("home", "P(dom gagne)")):
    v = daily[col].values
    z = (v - v.mean()) / (v.std() or 1)
    worst = daily.index[np.argmax(np.abs(z))]
    print(f"  {lbl:<14}: {v.min():.3f}..{v.max():.3f} sur {len(v)} jours | "
          f"|z|max {np.abs(z).max():.2f} ({worst}) -> {'⚠️ rupture' if np.abs(z).max() > 4 else 'stable, aucun reset'}")
# différence 1re moitié vs 2e moitié (dérive ?)
mid = len(df)//2
for lbl, a, b in (("buts/match", df.tot[:mid].mean(), df.tot[mid:].mean()),
                  ("P(over2.5)", (df.tot[:mid] > 2.5).mean(), (df.tot[mid:] > 2.5).mean())):
    print(f"  dérive {lbl:<12}: 1re moitié {a:.3f} vs 2e {b:.3f} (écart {abs(a-b):.3f})")
