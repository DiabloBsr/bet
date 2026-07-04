"""Teste : à COTE égale, l'identité d'une équipe (ou paire, ou domicile/ext)
ajoute-t-elle de l'information sur le résultat ? Sur un RNG calibré, la réponse
DOIT être non (résidus réel-vs-implicite ~ 0, pas de persistance OOS).

Sortie : data/team_study.json (tous les chiffres) + résumé console.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import create_engine, text
from scraper.config import load_settings

LG = "InstantLeague-8035"
OVERROUND = None  # devig proportionnel

_SQL = """
SELECT e.expected_start ts, e.team_a home, e.team_b away,
       o.odds_home oh, o.odds_draw od, o.odds_away oa, r.score_a sa, r.score_b sb
FROM events e
JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
JOIN results r ON r.event_id=e.id
WHERE r.score_a IS NOT NULL AND e.competition=:lg
  AND o.odds_home>1 AND o.odds_draw>1 AND o.odds_away>1
ORDER BY e.expected_start
"""


def bh_fdr(pvals, q=0.10):
    p = np.asarray(pvals); n = len(p); order = np.argsort(p)
    thr = q * (np.arange(1, n + 1)) / n
    passed = p[order] <= thr
    kmax = np.where(passed)[0].max() if passed.any() else -1
    sig = np.zeros(n, bool)
    if kmax >= 0:
        sig[order[:kmax + 1]] = True
    return sig


def main():
    eng = create_engine(load_settings().db_url)
    df = pd.read_sql(text(_SQL), eng, params={"lg": LG})
    print(f"chargé {len(df)} matchs")

    # ---- probas implicites (devig proportionnel) ----
    ih, idr, ia = 1 / df.oh, 1 / df.od, 1 / df.oa
    s = ih + idr + ia
    df["p1"] = ih / s; df["pX"] = idr / s; df["p2"] = ia / s  # implicites home/draw/away
    df["res"] = np.sign(df.sa - df.sb)          # 1 home, 0 nul, -1 away
    df["home_win"] = (df.sa > df.sb).astype(int)
    df["away_win"] = (df.sb > df.sa).astype(int)
    df["tot"] = df.sa + df.sb

    # ---- format long : 1 ligne par (équipe, match) ----
    home = pd.DataFrame({"team": df.home, "opp": df.away, "is_home": 1,
                         "p_win": df.p1, "won": df.home_win, "gf": df.sa, "ga": df.sb,
                         "p_implied_gf": None, "ts": df.ts})
    away = pd.DataFrame({"team": df.away, "opp": df.home, "is_home": 0,
                         "p_win": df.p2, "won": df.away_win, "gf": df.sb, "ga": df.sa,
                         "p_implied_gf": None, "ts": df.ts})
    L = pd.concat([home, away], ignore_index=True)
    L["resid"] = L.won - L.p_win    # actual win - implied win prob

    teams = sorted(L.team.unique())
    out = {"n_matches": int(len(df)), "n_teams": len(teams), "teams": teams}

    # ===== TEST 1 : biais de victoire par équipe (réel - implicite), FDR =====
    rows = []
    for t in teams:
        sub = L[L.team == t]
        n = len(sub); mean_p = sub.p_win.mean(); mean_w = sub.won.mean()
        resid = mean_w - mean_p
        # test binomial : nb de victoires vs proba implicite moyenne
        k = int(sub.won.sum())
        p = stats.binomtest(k, n, mean_p).pvalue
        se = np.sqrt(mean_p * (1 - mean_p) / n)
        rows.append({"team": t, "n": n, "implied_win": round(mean_p, 4),
                     "real_win": round(mean_w, 4), "resid_pp": round(100 * resid, 3),
                     "z": round(resid / se, 2), "p": p})
    t1 = pd.DataFrame(rows)
    t1["fdr_sig"] = bh_fdr(t1.p.values, 0.10)
    out["test1_team_winbias"] = t1.to_dict("records")

    # ===== TEST 2 : persistance OUT-OF-SAMPLE (le test décisif) =====
    cut = df.ts.iloc[len(df) // 2]
    tr, te = L[L.ts < cut], L[L.ts >= cut]
    btr = tr.groupby("team").apply(lambda g: g.won.mean() - g.p_win.mean())
    bte = te.groupby("team").apply(lambda g: g.won.mean() - g.p_win.mean())
    j = pd.concat([btr.rename("train"), bte.rename("test")], axis=1).dropna()
    r_p, p_p = stats.pearsonr(j.train, j.test)
    r_s, p_s = stats.spearmanr(j.train, j.test)
    out["test2_oos_persistence"] = {
        "pearson_r": round(r_p, 4), "pearson_p": round(p_p, 4),
        "spearman_r": round(r_s, 4), "spearman_p": round(p_s, 4),
        "interpretation": "r~0 => biais d'équipe = bruit, AUCUNE info au-dela des cotes",
        "per_team": {t: {"train_pp": round(100 * j.train[t], 3), "test_pp": round(100 * j.test[t], 3)}
                     for t in j.index},
    }

    # ===== TEST 3 : effet domicile/extérieur AU-DELA des cotes =====
    rows = []
    for t in teams:
        h = L[(L.team == t) & (L.is_home == 1)]; a = L[(L.team == t) & (L.is_home == 0)]
        rows.append({"team": t, "home_resid_pp": round(100 * (h.won.mean() - h.p_win.mean()), 3),
                     "away_resid_pp": round(100 * (a.won.mean() - a.p_win.mean()), 3),
                     "diff_pp": round(100 * ((h.won.mean() - h.p_win.mean()) - (a.won.mean() - a.p_win.mean())), 3)})
    out["test3_venue"] = rows

    # ===== TEST 4 : paires équipe vs équipe (résidu vs implicite), top par fréquence =====
    df["pair"] = df.apply(lambda r: " v ".join(sorted([r.home, r.away])), axis=1)
    rows = []
    for pair, g in df.groupby("pair"):
        if len(g) < 150:
            continue
        # résidu du résultat home dans l'orientation stockée
        resid = (g.home_win.mean() - g.p1.mean())
        k = int(g.home_win.sum()); p = stats.binomtest(k, len(g), g.p1.mean()).pvalue
        rows.append({"pair": pair, "n": int(len(g)), "resid_home_pp": round(100 * resid, 3), "p": p})
    t4 = pd.DataFrame(rows)
    if len(t4):
        t4["fdr_sig"] = bh_fdr(t4.p.values, 0.10)
        out["test4_pairs"] = {"n_pairs_tested": int(len(t4)),
                              "n_fdr_sig": int(t4.fdr_sig.sum()),
                              "worst": t4.sort_values("p").head(8).to_dict("records")}

    # ===== TEST 5 : profil EXHAUSTIF de 3 équipes (vs chaque adversaire, dom/ext) =====
    focus = ["Liverpool", "Burnley", "Brighton"]  # un fort, un faible, un moyen
    prof = {}
    for t in focus:
        sub = L[L.team == t]
        per_opp = []
        for opp, g in sub.groupby("opp"):
            per_opp.append({"opp": opp, "n": int(len(g)),
                            "implied_win": round(g.p_win.mean(), 3),
                            "real_win": round(g.won.mean(), 3),
                            "resid_pp": round(100 * (g.won.mean() - g.p_win.mean()), 2)})
        h = sub[sub.is_home == 1]; a = sub[sub.is_home == 0]
        prof[t] = {"n": int(len(sub)),
                   "overall_resid_pp": round(100 * (sub.won.mean() - sub.p_win.mean()), 3),
                   "home_resid_pp": round(100 * (h.won.mean() - h.p_win.mean()), 3),
                   "away_resid_pp": round(100 * (a.won.mean() - a.p_win.mean()), 3),
                   "max_abs_opp_resid_pp": round(max(abs(x["resid_pp"]) for x in per_opp), 2),
                   "per_opponent": sorted(per_opp, key=lambda x: -abs(x["resid_pp"]))[:5]}
    out["test5_deep_profiles"] = prof

    Path("data/team_study.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- résumé console ----
    print("\n===== TEST 1 : biais de victoire par équipe (réel - implicite) =====")
    print(t1[["team", "n", "implied_win", "real_win", "resid_pp", "z", "p", "fdr_sig"]]
          .sort_values("resid_pp").to_string(index=False))
    print(f"\n  équipes FDR-significatives : {int(t1.fdr_sig.sum())}/{len(t1)} "
          f"| résidu max |{t1.resid_pp.abs().max():.2f}|pp")
    o2 = out["test2_oos_persistence"]
    print(f"\n===== TEST 2 : persistance OOS (DÉCISIF) =====")
    print(f"  corrélation biais TRAIN vs TEST : Pearson r={o2['pearson_r']} (p={o2['pearson_p']}) | "
          f"Spearman r={o2['spearman_r']} (p={o2['spearman_p']})")
    if "test4_pairs" in out:
        print(f"\n===== TEST 4 : paires =====  {out['test4_pairs']['n_fdr_sig']}/"
              f"{out['test4_pairs']['n_pairs_tested']} paires FDR-significatives")
    print(f"\n===== TEST 5 : profils =====")
    for t, d in prof.items():
        print(f"  {t}: global {d['overall_resid_pp']:+.2f}pp | dom {d['home_resid_pp']:+.2f} | "
              f"ext {d['away_resid_pp']:+.2f} | pire adversaire |{d['max_abs_opp_resid_pp']:.2f}|pp")
    print("\n-> data/team_study.json écrit.")


if __name__ == "__main__":
    main()
