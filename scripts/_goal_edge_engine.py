"""MOTEUR TOUS-EN-UN — edge +EV sur les marchés BUTS (Over/Under, Total exact, BTTS).

Teste l'Idée #1 (réalité vs prix) sur les VRAIES cotes offertes (extra_markets),
sur les 9 ligues (Idée #6), avec split OOS, ROI réel, stabilité temporelle (#2)
et simulation de capital Kelly (#4).

Hypothèse : le RNG produit +0,12 but + mode drama ; si le book price les totals
naïvement, les OVER sont sous-pricés -> ROI>0 en pariant Over aux cotes offertes.

Lecture seule. Sortie : data/goal_edge.json + résumé console.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sqlalchemy import create_engine, text
from scraper.config import load_settings

_SQL = """
SELECT e.competition lg, e.expected_start ts, o.extra_markets xm, r.score_a sa, r.score_b sb
FROM events e
JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
JOIN results r ON r.event_id=e.id
WHERE r.score_a IS NOT NULL AND e.competition LIKE 'InstantLeague-%'
  AND o.extra_markets IS NOT NULL AND o.extra_markets NOT IN ('','{}','null')
ORDER BY e.expected_start
"""


def parse_xm(raw):
    try:
        o = json.loads(raw) if isinstance(raw, str) else raw
        return o if isinstance(o, dict) else {}
    except Exception:
        return {}


def get_market(xm, name):
    """Récupère un sous-marché par nom exact ou préfixe (robuste mojibake)."""
    if name in xm:
        return xm[name]
    for k, v in xm.items():
        if k.startswith(name):
            return v
    return None


def roi_stats(hit, odds):
    """ROI = mean(hit*odds - 1). Retourne (n, roi, t)."""
    hit = np.asarray(hit, float); odds = np.asarray(odds, float)
    m = (odds > 1) & np.isfinite(odds)
    hit, odds = hit[m], odds[m]
    if len(hit) < 30:
        return len(hit), float("nan"), float("nan")
    pnl = hit * odds - 1.0
    roi = pnl.mean(); se = pnl.std(ddof=1) / np.sqrt(len(pnl))
    return len(hit), roi, (roi / se if se > 0 else float("nan"))


def build(df):
    """Construit les colonnes de paris depuis extra_markets + résultat."""
    rows = []
    for r in df.itertuples():
        xm = parse_xm(r.xm)
        if not xm:
            continue
        sa, sb = int(r.sa), int(r.sb); tot = sa + sb
        rec = {"lg": r.lg, "ts": r.ts, "tot": tot,
               "over35": int(tot > 3.5), "under35": int(tot < 3.5),
               "btts": int(sa > 0 and sb > 0)}
        # +/- 3.5
        pm = get_market(xm, "+/-")
        if isinstance(pm, dict):
            rec["o_over35"] = pm.get("> 3.5"); rec["o_under35"] = pm.get("< 3.5")
        # G/NG (BTTS)
        gn = get_market(xm, "G/NG")
        if isinstance(gn, dict):
            rec["o_btts_oui"] = gn.get("Oui"); rec["o_btts_non"] = gn.get("Non")
        # Total de buts (cotes par total exact)
        tb = get_market(xm, "Total de buts")
        if isinstance(tb, dict):
            for k, v in tb.items():
                if k.isdigit():
                    rec[f"o_tot{k}"] = v
        rows.append(rec)
    return pd.DataFrame(rows)


def bet_table(d):
    """ROI par type de pari sur un sous-ensemble d."""
    out = {}
    # Over / Under 3.5
    n, roi, t = roi_stats(d.over35, d.get("o_over35")); out["Over3.5"] = (n, roi, t)
    n, roi, t = roi_stats(d.under35, d.get("o_under35")); out["Under3.5"] = (n, roi, t)
    # BTTS
    n, roi, t = roi_stats(d.btts, d.get("o_btts_oui")); out["BTTS_Oui"] = (n, roi, t)
    n, roi, t = roi_stats(1 - d.btts, d.get("o_btts_non")); out["BTTS_Non"] = (n, roi, t)
    # Total exact k
    for k in range(0, 7):
        col = f"o_tot{k}"
        if col in d.columns:
            n, roi, t = roi_stats((d.tot == k).astype(int), d[col])
            out[f"Total={k}"] = (n, roi, t)
    return out


def kelly_sim(d_train, d_test, hit_col, odds_col, frac=0.25):
    """Estime p sur TRAIN, mise Kelly fractionné sur TEST, retourne multiple de capital."""
    m = d_train[odds_col] > 1
    p = float(d_train[m][hit_col].mean())
    dd = d_test[d_test[odds_col] > 1]
    if len(dd) < 30:
        return None
    bank = 1.0; curve = []
    for _, row in dd.iterrows():
        o = float(row[odds_col])
        edge = p * o - 1.0
        f = max(0.0, frac * edge / (o - 1.0))
        stake = bank * f
        bank += stake * (o - 1.0) if row[hit_col] == 1 else -stake
        curve.append(bank)
    return {"p_train": round(p, 4), "n_test": len(dd),
            "final_bank": round(bank, 3), "max_bank": round(max(curve), 3),
            "min_bank": round(min(curve), 3)}


def main():
    eng = create_engine(load_settings().db_url)
    print("chargement extra_markets + résultats…")
    df = pd.read_sql(text(_SQL), eng)
    print(f"{len(df)} events avec marchés + résultat")
    B = build(df)
    print(f"{len(B)} events parsés")
    leagues = sorted(B.lg.unique(), key=lambda L: -len(B[B.lg == L]))

    out = {"n_events": int(len(B)), "leagues": {}}
    print("\n===== ROI PAR LIGUE x TYPE DE PARI (toutes données) =====")
    print(f"{'ligue':<20}{'pari':<12}{'n':>7}{'implied':>9}{'réalisé':>9}{'ROI%':>8}{'t':>7}")
    for L in leagues:
        d = B[B.lg == L]
        tbl = bet_table(d)
        out["leagues"][L] = {"n": int(len(d)), "bets": {}}
        for bet, (n, roi, t) in tbl.items():
            if n < 30 or not np.isfinite(roi):
                continue
            out["leagues"][L]["bets"][bet] = {"n": int(n), "roi_pct": round(100 * roi, 2),
                                              "t": round(float(t), 2)}
            # afficher seulement les paris notables (|ROI|>3% ou Over/BTTS)
            if abs(roi) > 0.03 or bet in ("Over3.5", "BTTS_Oui", "Under3.5"):
                print(f"{L:<20}{bet:<12}{n:>7}{'':>9}{'':>9}{100*roi:>7.2f}{t:>7.2f}")

    # ===== focus Over3.5 : implied vs réalisé + OOS + Kelly, sur la meilleure ligue =====
    print("\n===== OVER 3.5 — détail implied vs réalisé + OOS (split médian) =====")
    over_summary = {}
    for L in leagues:
        d = B[B.lg == L].dropna(subset=["o_over35"])
        if len(d) < 200:
            continue
        impl = (1 / d.o_over35) / (1 / d.o_over35 + 1 / d.o_under35)   # devig
        realized = d.over35.mean()
        cut = d.ts.iloc[len(d) // 2]
        tr, te = d[d.ts < cut], d[d.ts >= cut]
        _, roi_tr, _ = roi_stats(tr.over35, tr.o_over35)
        n_te, roi_te, t_te = roi_stats(te.over35, te.o_over35)
        k = kelly_sim(tr, te, "over35", "o_over35")
        over_summary[L] = {"n": int(len(d)), "implied_over": round(float(impl.mean()), 4),
                           "realized_over": round(float(realized), 4),
                           "gap_pp": round(100 * (realized - impl.mean()), 2),
                           "roi_train_pct": round(100 * roi_tr, 2),
                           "roi_test_pct": round(100 * roi_te, 2), "roi_test_t": round(float(t_te), 2),
                           "kelly": k}
        print(f"  {L:<20} n={len(d):>5} implied={100*impl.mean():5.1f}% réalisé={100*realized:5.1f}% "
              f"gap={100*(realized-impl.mean()):+5.2f}pp | ROI train={100*roi_tr:+5.2f}% "
              f"test={100*roi_te:+5.2f}% (t={t_te:.2f}) | Kelly bank x{k['final_bank'] if k else 'NA'}")
    out["over35_detail"] = over_summary

    Path("data/goal_edge.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n-> data/goal_edge.json écrit.")


if __name__ == "__main__":
    main()
