"""Test d'INFO INCREMENTALE : une famille de marche ajoute-t-elle de l'info
predictive AU-DELA de (lam_tot, lam_diff) ?

Si tous les marches derivent de la grille Poisson(lam_h,lam_a), alors conditionner
sur un signal de marche SUPPLEMENTAIRE (a lam fixe) ne doit RIEN changer a la
distribution du total / score. Ce script le mesure :
  - dans chaque cellule (lam_tot_band x lam_diff_band, n>=200), on coupe les events
    en 2 selon la mediane du signal famille, et on compare P(total>2.5) et le taux
    du score modal entre les 2 moities (z-test diff de proportions).
  - apres correction (Bonferroni sur le nb de cellules), combien de cellules montrent
    une vraie difference ? ~0 => la famille n'ajoute RIEN.

Usage: ./.venv/Scripts/python.exe scripts/_ck_family_test.py --market "FTTS"
"""
import sys
sys.path.insert(0, ".")
import argparse
import json
import numpy as np
import pandas as pd
from scipy.stats import norm
from sqlalchemy import create_engine
from scraper.config import load_settings
from scraper.market_inversion import parse_extra_markets, devig_market, _get_market

# famille -> (nom marche, selection a extraire comme signal, target_sum)
FAMILIES = {
    "Mi-tps 1X2": ("Mi-tps 1X2", "1", 1.0),
    "HT/FT": ("HT/FT", "1/1", 1.0),
    "FTTS": ("FTTS", "1", 1.0),
    "Pair/Impair": ("Pair/Impair", "Pair", 1.0),
    "Double Chance": ("Double Chance", "1X", 2.0),
    "Total equipe domicile": ("Total equipe domicile", ">3.5", 1.0),
    "Minute du premier but": ("Minute du premier but", None, 1.0),
}

ap = argparse.ArgumentParser()
ap.add_argument("--market", default="ALL")
args = ap.parse_args()

feat = pd.read_csv("exports/combokeys_features.csv")[["id", "lam_tot", "lam_diff", "total_goals", "exact_score"]]
e = create_engine(load_settings().db_url)
import sqlalchemy as sa

to_test = list(FAMILIES.items()) if args.market == "ALL" else [(args.market, FAMILIES[args.market])]

# UNE seule passe DB : extraire tous les signaux famille par event
sigcols = {name: [] for name, _ in to_test}
with e.connect() as c:
    res = c.execute(sa.text("""
        SELECT ev.id, o.extra_markets
        FROM events ev JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=ev.id)
        WHERE ev.competition='InstantLeague-8035' AND o.extra_markets IS NOT NULL
    """))
    for eid, em_raw in res:
        em = parse_extra_markets(em_raw)
        for name, (mkt, sel, tsum) in to_test:
            if sel is None:
                continue
            m = _get_market(em, exact=mkt) or _get_market(em, prefix=mkt)
            sig = None
            if isinstance(m, dict):
                dv = devig_market({k: v for k, v in m.items()}, target_sum=tsum)
                key = sel.replace(" ", "")
                for k, p in (dv or {}).items():
                    if str(k).replace(" ", "") == key:
                        sig = float(p); break
            sigcols[name].append((int(eid), sig))

TOT_E = [0, 2.2, 2.6, 3.0, 3.4, 3.8, 99]
DIFF_E = [-9, -0.5, 0, 0.5, 9]


def ztest_prop(x1, n1, x2, n2):
    if n1 < 20 or n2 < 20:
        return 0.0
    p1, p2 = x1 / n1, x2 / n2
    p = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return (p1 - p2) / se if se > 0 else 0.0


def test_family(name, sel):
    sd = pd.DataFrame(sigcols[name], columns=["id", "sig"]).dropna()
    df = feat.merge(sd, on="id", how="inner").dropna(subset=["sig", "lam_tot", "lam_diff"])
    if len(df) < 500:
        return {"market": name, "n": len(df), "verdict": "TROP_PEU"}
    df = df.copy()
    df["bt"] = pd.cut(df.lam_tot, TOT_E); df["bd"] = pd.cut(df.lam_diff, DIFF_E)
    df["o25"] = (df.total_goals > 2).astype(int)
    cells, sig_score, zs = 0, 0, []
    for (bt, bd), g in df.groupby(["bt", "bd"], observed=True):
        if len(g) < 200:
            continue
        med = g.sig.median()
        lo, hi = g[g.sig <= med], g[g.sig > med]
        if len(lo) < 50 or len(hi) < 50:
            continue
        cells += 1
        zs.append(abs(ztest_prop(lo.o25.sum(), len(lo), hi.o25.sum(), len(hi))))
        modal = g.exact_score.value_counts().index[0]
        if abs(ztest_prop((lo.exact_score == modal).sum(), len(lo),
                          (hi.exact_score == modal).sum(), len(hi))) >= 3.0:
            sig_score += 1
    zstar = norm.ppf(1 - 0.05 / (2 * max(cells, 1)))
    sig_o25 = sum(1 for z in zs if z >= zstar)
    verdict = "RIEN_AU_DELA_DE_LAMBDA" if sig_o25 == 0 and sig_score == 0 else "INFO_POSSIBLE"
    return {"market": name, "sel": sel, "n": len(df), "cells": cells, "z_star": round(zstar, 2),
            "max_z_o25": round(max(zs), 2) if zs else 0, "cells_sig_o25": sig_o25,
            "cells_sig_score": sig_score, "verdict": verdict}


results = [test_family(name, fam[1]) for name, fam in to_test]
for r in results:
    print(json.dumps(r, ensure_ascii=False))
with open("exports/family_incremental_test.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=1, ensure_ascii=False)
print("ecrit exports/family_incremental_test.json")
