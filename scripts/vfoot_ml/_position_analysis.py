"""Ré-analyse : la POSITION d'un match dans le round (1..10) porte-t-elle un edge ?

Hypothèse utilisateur : les matchs se regroupent (ex. 3-4-3) avec des distributions
caractéristiques par position, fréquemment similaires d'un round à l'autre.

On distingue DEUX choses :
  (A) la position prédit-elle la distribution BRUTE (résultat/buts) ? -> descriptif
  (B) la position prédit-elle AU-DELÀ des cotes (résidu réalisé-implicite) ? -> EDGE

Seul (B) serait monétisable. (A) sans (B) = le book price déjà la position.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np, pandas as pd
from scipy import stats
from sqlalchemy import create_engine, text
from scraper.config import load_settings

LG = "InstantLeague-8035"
SQL = """
SELECT e.id, e.expected_start ts, o.odds_home oh, o.odds_draw od, o.odds_away oa,
       r.score_a sa, r.score_b sb
FROM events e
JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
JOIN results r ON r.event_id=e.id
WHERE r.score_a IS NOT NULL AND e.competition=:lg
  AND o.odds_home>1 AND o.odds_draw>1 AND o.odds_away>1
ORDER BY e.expected_start, e.id
"""


def drift_analysis(df, recent_n=3000):
    """Détecte un CHANGEMENT RÉCENT de distribution (rumeur de maj du RNG)."""
    print("\n" + "=" * 64)
    print("  RÉ-ANALYSE : la distribution a-t-elle CHANGÉ récemment ? (maj RNG ?)")
    print("=" * 64)
    d = df.sort_values("ts").reset_index(drop=True)
    base, rec = d.iloc[:-recent_n], d.iloc[-recent_n:]
    print(f"  baseline={len(base)} matchs | récent={len(rec)} matchs (derniers)")

    def ztest(a, b):
        pa, pb, na, nb = a.mean(), b.mean(), len(a), len(b)
        p = (a.sum() + b.sum()) / (na + nb)
        se = np.sqrt(p * (1 - p) * (1 / na + 1 / nb))
        z = (pb - pa) / se if se > 0 else 0
        return z, 2 * (1 - stats.norm.cdf(abs(z)))

    print(f"\n  {'métrique':<22}{'baseline':>10}{'récent':>10}{'Δ':>9}{'z':>7}{'p':>8}")
    metrics = [("% victoire dom", (base.result == "1").astype(int), (rec.result == "1").astype(int)),
               ("% nul", (base.result == "X").astype(int), (rec.result == "X").astype(int)),
               ("% Over 2.5", (base.tot > 2.5).astype(int), (rec.tot > 2.5).astype(int)),
               ("% BTTS", ((base.sa > 0) & (base.sb > 0)).astype(int), ((rec.sa > 0) & (rec.sb > 0)).astype(int)),
               ("% score 0-0", ((base.sa == 0) & (base.sb == 0)).astype(int), ((rec.sa == 0) & (rec.sb == 0)).astype(int))]
    for name, a, b in metrics:
        z, p = ztest(a, b)
        flag = "  <-- CHANGÉ" if p < 0.01 else ""
        print(f"  {name:<22}{100*a.mean():>9.2f}%{100*b.mean():>9.2f}%{100*(b.mean()-a.mean()):>+8.2f}{z:>7.2f}{p:>8.4f}{flag}")
    # buts moyens (t-test)
    t, pt = stats.ttest_ind(rec.tot, base.tot)
    print(f"  {'buts moy/match':<22}{base.tot.mean():>10.3f}{rec.tot.mean():>10.3f}{rec.tot.mean()-base.tot.mean():>+9.3f}{t:>7.2f}{pt:>8.4f}"
          f"{'  <-- CHANGÉ' if pt < 0.01 else ''}")

    # LE TEST QUI COMPTE : la calibration a-t-elle décroché ? (résidu récent)
    print("\n  CALIBRATION (réalisé - implicite domicile) — l'EDGE potentiel :")
    print(f"    baseline résidu = {100*base.resid.mean():+.2f}pp | récent résidu = {100*rec.resid.mean():+.2f}pp")
    zr, pr = ztest(base.home_win, rec.home_win)  # approximation
    rr = rec.resid.mean(); se_r = rec.resid.std() / np.sqrt(len(rec))
    zr2 = rr / se_r if se_r > 0 else 0
    print(f"    résidu récent vs 0 : {100*rr:+.2f}pp (z={zr2:.2f}, p={2*(1-stats.norm.cdf(abs(zr2))):.4f})")
    print("    -> si récent décroche de 0 ET pas la baseline = FENÊTRE (le book n'a pas suivi la maj)")

    # point de bascule : rolling over2.5 et buts
    print("\n  Tendance (fenêtres de 1500, du + ancien au + récent) :")
    w = 1500
    for i in range(0, len(d) - w + 1, max(1, (len(d) - w) // 6)):
        seg = d.iloc[i:i + w]
        print(f"    [{str(seg.ts.iloc[0])[:10]} -> {str(seg.ts.iloc[-1])[:10]}] "
              f"buts={seg.tot.mean():.3f} over2.5={100*(seg.tot>2.5).mean():.1f}% "
              f"résidu={100*seg.resid.mean():+.2f}pp")


def main():
    df = pd.read_sql(text(SQL), create_engine(load_settings().db_url), params={"lg": LG})
    # position dans le round = rang par id au sein de chaque ts
    df["pos"] = df.groupby("ts").cumcount() + 1
    df = df[df.pos <= 10]
    inv = 1 / df.oh + 1 / df.od + 1 / df.oa
    df["imp_home"] = (1 / df.oh) / inv
    df["home_win"] = (df.sa > df.sb).astype(int)
    df["tot"] = df.sa + df.sb
    df["resid"] = df.home_win - df.imp_home          # réalisé - implicite (domicile)
    df["result"] = np.where(df.sa > df.sb, "1", np.where(df.sa == df.sb, "X", "2"))

    n_rounds = df.ts.nunique()
    print(f"{len(df)} matchs sur {n_rounds} rounds, positions 1..10\n")

    # ---- (A) DISTRIBUTION BRUTE par position ----
    print("=== (A) DISTRIBUTION BRUTE par position ===")
    print(f"  {'pos':>3} {'n':>5} {'%1':>6} {'%X':>6} {'%2':>6} {'butsTot':>8} {'over2.5':>8} {'cote_dom_moy':>12}")
    rows = []
    for p, g in df.groupby("pos"):
        r1 = 100 * (g.result == "1").mean(); rx = 100 * (g.result == "X").mean(); r2 = 100 * (g.result == "2").mean()
        rows.append((p, len(g), r1, rx, r2, g.tot.mean(), 100 * (g.tot > 2.5).mean(), g.oh.mean()))
        print(f"  {p:>3} {len(g):>5} {r1:>6.1f} {rx:>6.1f} {r2:>6.1f} {g.tot.mean():>8.3f} "
              f"{100*(g.tot>2.5).mean():>8.1f} {g.oh.mean():>12.2f}")

    # ---- (B) RÉSIDU (réalisé - implicite) par position = test d'EDGE ----
    print("\n=== (B) RÉSIDU réalisé-implicite (domicile) par position — l'EDGE ===")
    print(f"  {'pos':>3} {'n':>5} {'implicite%':>11} {'réalisé%':>10} {'résidu_pp':>10} {'z':>7} {'p':>7}")
    pvals, info = [], []
    for p, g in df.groupby("pos"):
        impl = g.imp_home.mean(); real = g.home_win.mean(); resid = real - impl
        se = np.sqrt(impl * (1 - impl) / len(g))
        z = resid / se if se > 0 else 0
        pval = 2 * (1 - stats.norm.cdf(abs(z)))
        pvals.append(pval); info.append((p, len(g), impl, real, resid, z, pval))
        print(f"  {p:>3} {len(g):>5} {100*impl:>11.2f} {100*real:>10.2f} {100*resid:>+10.2f} {z:>7.2f} {pval:>7.4f}")
    # correction de Bonferroni (10 tests)
    sig = [i for i, pv in zip(info, pvals) if pv < 0.05 / 10]
    print(f"\n  -> positions résidu-significatives (Bonferroni 0.005) : "
          f"{[i[0] for i in sig] if sig else 'AUCUNE'}")

    # ---- indépendance position x résultat (chi2) ----
    ct = pd.crosstab(df.pos, df.result)
    chi2, pchi, _, _ = stats.chi2_contingency(ct)
    print(f"  chi2 indépendance position×résultat : p={pchi:.4f} "
          f"({'LIÉ' if pchi < 0.05 else 'indépendant'})")

    # ---- hypothèse 3-4-3 : groupes {1-3},{4-7},{8-10} ----
    print("\n=== hypothèse 3-4-3 (groupes de position) ===")
    df["grp"] = pd.cut(df.pos, [0, 3, 7, 10], labels=["G1(1-3)", "G2(4-7)", "G3(8-10)"])
    for grp, g in df.groupby("grp", observed=True):
        print(f"  {grp}: n={len(g)} | %1={100*(g.result=='1').mean():.1f} "
              f"butsTot={g.tot.mean():.3f} over2.5={100*(g.tot>2.5).mean():.1f} "
              f"| résidu_moy={100*g.resid.mean():+.2f}pp")

    # ---- OOS : le résidu par position persiste-t-il ? ----
    print("\n=== (B-OOS) persistance du résidu par position (train vs test) ===")
    cut = df.ts.iloc[len(df) // 2]
    tr, te = df[df.ts < cut], df[df.ts >= cut]
    rtr = tr.groupby("pos").resid.mean(); rte = te.groupby("pos").resid.mean()
    j = pd.concat([rtr.rename("train"), rte.rename("test")], axis=1).dropna()
    rp, pp = stats.pearsonr(j.train, j.test)
    print(f"  corrélation résidu_position TRAIN vs TEST : Pearson r={rp:.3f} (p={pp:.3f})")
    print("  (r>0 et significatif = effet position RÉEL et exploitable ; r~0 = bruit)")
    print(j.round(4).to_string())

    drift_analysis(df)


if __name__ == "__main__":
    main()
