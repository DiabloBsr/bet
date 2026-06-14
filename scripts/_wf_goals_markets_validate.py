# -*- coding: utf-8 -*-
"""
Validation finale des finalistes (gelés d'après le train de _wf_goals_markets).
Pour chaque finaliste :
  - métriques OOS (n, wins, wr, cote, roi)
  - bootstrap 10k : P(roi_oos <= 0)
  - sensibilité : OOS restreint aux scores NON reconstruits (gj cohérent d'origine)
"""
import sys, json
sys.path.insert(0, '.')
import numpy as np

sys.argv = [sys.argv[0]]
import importlib.util
spec = importlib.util.spec_from_file_location(
    "wfgm", "scripts/_wf_goals_markets.py")
wfgm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wfgm)

import pandas as pd
from sqlalchemy import create_engine, text
from scraper.config import load_settings


# --- rebuild dataset but tag reconstructed rows -----------------------------
def build_with_flags():
    rows = wfgm.load_rows()
    recs = []
    for (eid, ri, _es, oh, od, oa, em, sa, sb, hta, htb, gj) in rows:
        if isinstance(em, str):
            try:
                em = json.loads(em)
            except json.JSONDecodeError:
                continue
        if not isinstance(em, dict):
            continue
        cs = wfgm.clean_score(sa, sb, hta, htb, gj)
        if cs is None:
            continue
        csa, csb, goals = cs
        reconstructed = (csa != sa or csb != sb)
        prof = wfgm.classify_profile(oh, oa)
        seg = wfgm.seg_of(ri)
        for (mkt, sel, odd, won) in wfgm.settle(em, csa, csb, hta, htb, goals):
            recs.append((eid, prof, seg, oh, oa, mkt, sel, odd, won,
                         reconstructed))
    df = pd.DataFrame(recs, columns=["eid", "prof", "seg", "oh", "oa",
                                     "mkt", "sel", "odd", "won", "reco"])
    eids = df["eid"].drop_duplicates().tolist()
    cut = int(len(eids) * 0.70)
    train_ids = set(eids[:cut])
    df["split"] = np.where(df["eid"].isin(train_ids), "train", "oos")
    return df


def obin(d, lo, hi):
    return (d.odd >= lo) & (d.odd < hi)


FINALISTS = [
    ("F1 TotExt>3.5 away_slight MS_mid",
     "Total equipe extérieur > 3.5 quand profil=away_slight (1.6<=oddsA<2.2, oddsH>=2.5) et segment=MS_mid (J13-25)",
     lambda d: (d.mkt == "Total equipe extérieur") & (d.sel == "> 3.5")
     & (d.prof == "away_slight") & (d.seg == "MS_mid")),
    ("F2 TotDom>3.5 home_slight cote5-8",
     "Total equipe domicile > 3.5 quand profil=home_slight (1.6<=oddsH<2.2, oddsA>=2.5) et cote selection dans [5,8)",
     lambda d: (d.mkt == "Total equipe domicile") & (d.sel == "> 3.5")
     & (d.prof == "home_slight") & obin(d, 5.0, 8.0)),
    ("F3 TotalButs=1 home_slight MS_early",
     "Total de buts = 1 (exact) quand profil=home_slight et segment=MS_early (J4-12)",
     lambda d: (d.mkt == "Total de buts") & (d.sel == "1")
     & (d.prof == "home_slight") & (d.seg == "MS_early")),
    ("F4 TotalButs=6 away_slight",
     "Total de buts = 6 quand profil=away_slight (tous segments)",
     lambda d: (d.mkt == "Total de buts") & (d.sel == "6")
     & (d.prof == "away_slight")),
    ("F5 MultiButs>4 away_slight cote5-8",
     "Multi-Buts 'superieur a 4' quand profil=away_slight et cote dans [5,8)",
     lambda d: (d.mkt == "Multi-Buts")
     & (d.sel == "Le total de buts est supérieur à 4")
     & (d.prof == "away_slight") & obin(d, 5.0, 8.0)),
    ("F6 1erBut46-60 home_slight DS",
     "Minute du premier but 46-60 quand profil=home_slight et segment=DS (J1-3)",
     lambda d: (d.mkt == "Minute du premier but") & (d.sel == "46-60")
     & (d.prof == "home_slight") & (d.seg == "DS")),
    ("F6b 1erBut46-60 home_slight DS+MS_early",
     "Minute du premier but 46-60 quand profil=home_slight et segment in (DS, MS_early)",
     lambda d: (d.mkt == "Minute du premier but") & (d.sel == "46-60")
     & (d.prof == "home_slight") & d.seg.isin(["DS", "MS_early"])),
    ("F7 HTBTTS-Oui away_strong MS_early",
     "Les deux equipes marquent / 1ere mi temps = Oui quand oddsA<1.6 (away_strong) et segment=MS_early",
     lambda d: (d.mkt == "Les deux équipes marquent / 1ère mi temps")
     & (d.sel == "Oui") & (d.prof == "away_strong") & (d.seg == "MS_early")),
    ("F8 FTTS=1 home_crush (accuracy)",
     "FTTS = 1 (domicile marque en premier) quand profil=home_crush (oddsH<1.3, oddsA>7)",
     lambda d: (d.mkt == "FTTS") & (d.sel == "1") & (d.prof == "home_crush")),
    ("F8b FTTS=1 home_crush MS_late",
     "FTTS = 1 quand profil=home_crush et segment=MS_late (J26-33)",
     lambda d: (d.mkt == "FTTS") & (d.sel == "1") & (d.prof == "home_crush")
     & (d.seg == "MS_late")),
    ("F9 O/U>3.5 away_slight cote3.5-5",
     "+/- > 3.5 quand profil=away_slight et cote dans [3.5,5)",
     lambda d: (d.mkt == "+/-") & (d.sel == "> 3.5")
     & (d.prof == "away_slight") & obin(d, 3.5, 5.0)),
]


def stats(g):
    n = len(g)
    if n == 0:
        return None
    pnl = (g["won"] * (g["odd"] - 1) - (1 - g["won"])).values
    wins = int(g["won"].sum())
    roi = pnl.mean()
    # bootstrap
    rng = np.random.default_rng(42)
    idx = rng.integers(0, n, size=(10000, n))
    boots = pnl[idx].mean(axis=1)
    p_le0 = (boots <= 0).mean()
    return n, wins, g["won"].mean(), g["odd"].mean(), roi, p_le0


def main():
    df = build_with_flags()
    oo = df[df.split == "oos"]
    tr = df[df.split == "train"]
    print(f"{'finaliste':40s} {'set':12s}  n  wins   wr%   cote   roi%  P(roi<=0)")
    for name, desc, fn in FINALISTS:
        for label, d in (("train", tr), ("OOS", oo),
                         ("OOS-noReco", oo[~oo.reco])):
            g = d[fn(d)]
            s = stats(g)
            if s is None:
                print(f"{name:40s} {label:12s}  n=0")
                continue
            n, w, wr, av, roi, p = s
            print(f"{name:40s} {label:12s} {n:4d} {w:4d} {wr*100:6.1f} "
                  f"{av:6.2f} {roi*100:+7.1f}  {p:.3f}")
        print()


if __name__ == "__main__":
    main()
