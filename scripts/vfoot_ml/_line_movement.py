"""Test du MOUVEMENT DE LIGNE : la cote bouge-t-elle VERS le résultat ?

Pour chaque event multi-snapshot : cote OUVERTURE (1er snap) vs CLÔTURE (dernier
snap AVANT le coup d'envoi, anti-fuite). On teste :
  A) le mouvement va-t-il vers le gagnant ? (le book fuit-il le résultat ?)
  B) la clôture est-elle mieux calibrée que l'ouverture ? (closing line value)
  C) EDGE : conditionnel à la clôture, le mouvement prédit-il l'issue ? ROI OOS.
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
WITH snaps AS (
  SELECT o.event_id, o.captured_at, o.odds_home, o.odds_draw, o.odds_away,
         ROW_NUMBER() OVER (PARTITION BY o.event_id ORDER BY o.captured_at ASC) rn_open,
         ROW_NUMBER() OVER (PARTITION BY o.event_id ORDER BY o.captured_at DESC) rn_close
  FROM odds_snapshots o
  JOIN events e ON e.id=o.event_id
  WHERE e.competition=:lg AND o.captured_at < e.expected_start
    AND o.odds_home>1 AND o.odds_draw>1 AND o.odds_away>1
)
SELECT e.expected_start ts, r.score_a sa, r.score_b sb,
       op.odds_home oh_o, op.odds_draw od_o, op.odds_away oa_o,
       cl.odds_home oh_c, cl.odds_draw od_c, cl.odds_away oa_c
FROM events e
JOIN results r ON r.event_id=e.id AND r.score_a IS NOT NULL
JOIN snaps op ON op.event_id=e.id AND op.rn_open=1
JOIN snaps cl ON cl.event_id=e.id AND cl.rn_close=1
WHERE e.competition=:lg AND op.captured_at < cl.captured_at
ORDER BY e.expected_start
"""


def imp(h, d, a):
    inv = 1/h + 1/d + 1/a
    return (1/h)/inv, (1/d)/inv, (1/a)/inv


def main():
    e = create_engine(load_settings().db_url)
    df = pd.read_sql(text(SQL), e, params={"lg": LG})
    print(f"events multi-snapshot avec ouverture != clôture (anti-fuite) : {len(df)}")
    if len(df) < 200:
        print("pas assez de données pour un test fiable.") ; return

    df["sa"] = df.sa.clip(0, 6); df["sb"] = df.sb.clip(0, 6)
    ho, do, ao = imp(df.oh_o, df.od_o, df.oa_o)
    hc, dc, ac = imp(df.oh_c, df.od_c, df.oa_c)
    df["imp_h_o"], df["imp_h_c"] = ho, hc
    df["imp_a_o"], df["imp_a_c"] = ao, ac
    df["move_h"] = hc - ho                      # >0 : domicile a RACCOURCI (plus probable)
    df["move_a"] = ac - ao
    df["home_win"] = (df.sa > df.sb).astype(int)
    df["away_win"] = (df.sb > df.sa).astype(int)

    # ---- A) le mouvement va-t-il vers le gagnant ? ----
    print("\n=== (A) le mouvement va-t-il VERS le résultat ? ===")
    won = df[df.home_win == 1]; lost = df[df.home_win == 0]
    print(f"  move_h moyen quand domicile GAGNE : {won.move_h.mean():+.4f}")
    print(f"  move_h moyen quand domicile PERD  : {lost.move_h.mean():+.4f}")
    t, p = stats.ttest_ind(won.move_h, lost.move_h)
    print(f"  t-test (gagne vs perd) : t={t:.2f} p={p:.4f}  "
          f"{'-> le mouvement FUIT le résultat !' if p < 0.01 else '(pas de fuite nette)'}")

    # ---- B) closing line value : clôture mieux calibrée que l'ouverture ? ----
    def logloss(impv, y):
        impv = np.clip(impv, 1e-6, 1-1e-6)
        return float(-(y*np.log(impv) + (1-y)*np.log(1-impv)).mean())
    print("\n=== (B) closing line value (calibration ouverture vs clôture) ===")
    print(f"  log-loss domicile  ouverture={logloss(df.imp_h_o, df.home_win):.4f}  "
          f"clôture={logloss(df.imp_h_c, df.home_win):.4f}")
    print("  (clôture < ouverture = la clôture est plus informée = mouvement utile)")

    # ---- C) EDGE conditionnel à la clôture + ROI OOS ----
    print("\n=== (C) EDGE : conditionnel à la clôture, le mouvement prédit-il ? (OOS) ===")
    cut = df.ts.iloc[len(df)//2]
    tr, te = df[df.ts < cut], df[df.ts >= cut]
    # stratégie : parier DOMICILE à la cote de CLÔTURE quand le domicile a raccourci de >seuil
    for thr in (0.01, 0.02, 0.03, 0.05):
        sub_tr = tr[tr.move_h > thr]; sub_te = te[te.move_h > thr]
        if len(sub_te) < 50:
            continue
        # résidu = réalisé - implicite_clôture
        res_tr = sub_tr.home_win.mean() - sub_tr.imp_h_c.mean()
        res_te = sub_te.home_win.mean() - sub_te.imp_h_c.mean()
        roi_te = (sub_te.home_win * sub_te.oh_c - 1).mean()
        print(f"  move_h>{thr}: n_test={len(sub_te):>4} | résidu_train={100*res_tr:+5.2f}pp "
              f"résidu_test={100*res_te:+5.2f}pp | ROI_test(clôture)={100*roi_te:+5.2f}%")
    # idem côté extérieur
    print("  -- côté extérieur (parier EXT à la clôture si ext a raccourci) --")
    for thr in (0.02, 0.05):
        sub_te = te[te.move_a > thr]
        if len(sub_te) < 50:
            continue
        res_te = sub_te.away_win.mean() - sub_te.imp_a_c.mean()
        roi_te = (sub_te.away_win * sub_te.oa_c - 1).mean()
        print(f"  move_a>{thr}: n_test={len(sub_te):>4} | résidu_test={100*res_te:+5.2f}pp "
              f"ROI_test={100*roi_te:+5.2f}%")
    print("\n  -> ROI_test>0 significatif = EDGE de mouvement de ligne. Sinon = le mouvement "
          "est déjà dans la cote de clôture (pas exploitable).")


if __name__ == "__main__":
    main()
