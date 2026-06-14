"""Axe TEMPOREL — la seule dimension non couverte par (lam_h, lam_a).
Teste : (1) le DRIFT open->close (edge #8 : suivre le mouvement) ajoute-t-il de
l'info au-dela du prix d'ouverture ? (2) la position dans le round (round_info)
influe-t-elle sur le total ?

Lit la DB directement (besoin de plusieurs snapshots). 8035. Sortie console + JSON.
"""
import sys
sys.path.insert(0, ".")
import json
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from scraper.config import load_settings
from scraper.analysis_utils import load_corrupted_ids
from scraper.predictor_v2 import devig

e = create_engine(load_settings().db_url)
corrupted = load_corrupted_ids()

# open (MIN id) + close (MAX id) + resultat, 8035
df = pd.read_sql("""
    SELECT e.id, e.round_info, e.expected_start,
           omin.odds_home oh0, omin.odds_draw od0, omin.odds_away oa0,
           omax.odds_home oh1, omax.odds_draw od1, omax.odds_away oa1,
           r.score_a sa, r.score_b sb
    FROM events e
    JOIN odds_snapshots omin ON omin.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
    JOIN odds_snapshots omax ON omax.id=(SELECT MAX(id) FROM odds_snapshots WHERE event_id=e.id)
    JOIN results r ON r.event_id=e.id
    WHERE r.score_a IS NOT NULL AND e.competition='InstantLeague-8035'
      AND omin.id <> omax.id
      AND omin.odds_home IS NOT NULL AND omax.odds_home IS NOT NULL
""", e)
df = df[~df.id.isin(corrupted)].copy()
print(f"events avec >=2 snapshots (drift mesurable) : {len(df)}")
out = {}
if len(df) > 300:
    ph0 = df.apply(lambda r: devig(r.oh0, r.od0, r.oa0)[0], axis=1)
    ph1 = df.apply(lambda r: devig(r.oh1, r.od1, r.oa1)[0], axis=1)
    df["ph0"] = ph0; df["drift"] = ph1 - ph0
    df["home_win"] = (df.sa > df.sb).astype(int)
    # info incrementale du drift au-dela du prix d'ouverture :
    # dans chaque bin de ph0, comparer home_win rate entre drift+ et drift-
    df["b0"] = pd.cut(df.ph0, [0, 0.3, 0.45, 0.6, 1.0])
    print("\n=== DRIFT open->close : home_win rate par bin de prix d'ouverture ===")
    cells_sig = 0
    for b, g in df.groupby("b0", observed=True):
        up = g[g.drift > 0.01]; dn = g[g.drift < -0.01]
        if len(up) < 40 or len(dn) < 40:
            continue
        pu, pd_ = up.home_win.mean(), dn.home_win.mean()
        p = pd.concat([up.home_win, dn.home_win]).mean()
        se = np.sqrt(p * (1 - p) * (1 / len(up) + 1 / len(dn)))
        z = (pu - pd_) / se if se > 0 else 0
        if abs(z) >= 2.5:
            cells_sig += 1
        print(f"  ph0∈{b}  drift+ home_win {100*pu:.1f}%(n={len(up)}) vs drift- {100*pd_:.1f}%(n={len(dn)})  z={z:+.2f}")
    out["drift_cells_sig"] = cells_sig
    out["drift_verdict"] = "DRIFT_INFORMATIF" if cells_sig >= 2 else "DRIFT_PEU_OU_PAS_INFORMATIF"

# round position -> total de buts
df2 = pd.read_sql("""
    SELECT e.round_info, r.score_a sa, r.score_b sb
    FROM events e JOIN results r ON r.event_id=e.id
    WHERE r.score_a IS NOT NULL AND e.competition='InstantLeague-8035'
      AND e.round_info IS NOT NULL AND e.round_info <> '0'
""", e)
df2 = df2.copy()
df2["tot"] = df2.sa + df2.sb
df2["rnd"] = pd.to_numeric(df2.round_info, errors="coerce")
df2 = df2.dropna(subset=["rnd"])
if len(df2) > 1000:
    corr = np.corrcoef(df2.rnd, df2.tot)[0, 1]
    print(f"\n=== round_info vs total buts : correlation = {corr:+.4f} (n={len(df2)}) ===")
    out["round_total_corr"] = round(float(corr), 4)
    out["round_verdict"] = "ROUND_INFLUENCE" if abs(corr) > 0.05 else "ROUND_SANS_EFFET"

print("\n" + json.dumps(out, ensure_ascii=False))
with open("exports/temporal_test.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
print("ecrit exports/temporal_test.json")
