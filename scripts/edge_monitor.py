"""MOTEUR MULTI — Tier 1 : veille de calibration + dérive, toutes ligues (déterministe).

Pour CHAQUE ligue de la base, audite en lecture seule :
  1) ARBITRAGE   — overround back-all par marché (les 2 jambes valides) ; <1.0 = profit garanti.
  2) DIRECTIONNEL — ROI OOS (Over/Under 3.5, BTTS) sur tout l'historique ET sur la fenêtre récente.
  3) DÉRIVE      — la fenêtre récente bascule-t-elle +EV (book qui décroche de la réalité) ?
  4) NOUVEAUTÉ   — nouvelle ligue jamais auditée.

Ne lève AUCUNE alerte si tout est scellé (l'attendu). Flague seulement un survivant
réel -> c'est LÀ qu'on escalade vers le workflow LLM adverse (Tier 2).

Sortie : data/edge_monitor_report.json + résumé console. Exit code 2 si un FLAG.
Conçu pour tourner en cron. Lecture seule (n'écrit que son rapport).
"""
from __future__ import annotations
import json, sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sqlalchemy import create_engine, text
from scraper.config import load_settings

# ---- seuils (un survivant DOIT les franchir pour flaguer) ----
ROI_FLAG = 0.02         # ROI OOS > +2% -> fenêtre MONÉTISABLE
T_FLAG = 2.0            # t-stat > 2
DRIFT_GAP_FLAG = 0.03   # |réalisé - implicite| récent > 3pp -> ALERTE PRÉCOCE (fenêtre qui s'ouvre)
RECENT_WINDOW = 3000    # taille de la fenêtre "récente" pour la dérive
MIN_N = 300             # n minimal pour juger un marché
REPORT = Path(__file__).resolve().parents[1] / "data" / "edge_monitor_report.json"
HISTORY = Path(__file__).resolve().parents[1] / "data" / "edge_monitor_history.jsonl"

_SQL = """
SELECT e.competition lg, e.expected_start ts,
       o.odds_home oh, o.odds_draw od, o.odds_away oa, o.extra_markets xm,
       r.score_a sa, r.score_b sb
FROM events e
JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
JOIN results r ON r.event_id=e.id
WHERE r.score_a IS NOT NULL AND e.competition LIKE 'InstantLeague-%'
  AND o.extra_markets IS NOT NULL AND o.extra_markets NOT IN ('','{}','null')
ORDER BY e.competition, e.expected_start
"""


def gm(xm, name):
    if name in xm:
        return xm[name]
    for k, v in xm.items():
        if k.startswith(name):
            return v
    return None


def roi(hit, odds):
    hit = np.asarray(hit, float); odds = np.asarray(odds, float)
    m = (odds > 1) & np.isfinite(odds) & np.isfinite(hit)
    hit, odds = hit[m], odds[m]
    if len(hit) < MIN_N:
        return len(hit), float("nan"), float("nan")
    pnl = hit * odds - 1.0
    se = pnl.std(ddof=1) / np.sqrt(len(pnl))
    return len(hit), float(pnl.mean()), (float(pnl.mean() / se) if se > 0 else float("nan"))


def build(df):
    rows = []
    for r in df.itertuples():
        try:
            xm = json.loads(r.xm) if isinstance(r.xm, str) else r.xm
        except Exception:
            continue
        if not isinstance(xm, dict):
            continue
        sa, sb = int(r.sa), int(r.sb); tot = sa + sb
        rec = {"ts": r.ts, "tot": tot, "over35": int(tot > 3.5),
               "under35": int(tot < 3.5), "btts": int(sa > 0 and sb > 0)}
        pm = gm(xm, "+/-")
        if isinstance(pm, dict):
            a, b = pm.get("> 3.5"), pm.get("< 3.5")
            rec["o_over35"] = a if isinstance(a, (int, float)) and a > 1 else np.nan
            rec["o_under35"] = b if isinstance(b, (int, float)) and b > 1 else np.nan
            # arbitrage : SEULEMENT si les 2 jambes valides
            if rec["o_over35"] > 1 and rec["o_under35"] > 1:
                rec["ou_overround"] = 1 / rec["o_over35"] + 1 / rec["o_under35"]
        gn = gm(xm, "G/NG")
        if isinstance(gn, dict):
            o = gn.get("Oui"); n = gn.get("Non")
            rec["o_btts_oui"] = o if isinstance(o, (int, float)) and o > 1 else np.nan
            rec["o_btts_non"] = n if isinstance(n, (int, float)) and n > 1 else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def audit_league(name, d):
    """Retourne un verdict {sealed, flags[], metrics}."""
    d = d.sort_values("ts").reset_index(drop=True)
    flags = []
    # 1) arbitrage réel
    n_arb = int((d["ou_overround"] < 1.0).sum()) if "ou_overround" in d else 0
    if n_arb > 0:
        flags.append(f"ARBITRAGE: {n_arb} events Over/Under back-all<100%")
    # 2) directionnel : full + récent
    bets = [("Over3.5", "over35", "o_over35"), ("Under3.5", "under35", "o_under35"),
            ("BTTS_Oui", "btts", "o_btts_oui"), ("BTTS_Non", lambda x: 1 - x["btts"], "o_btts_non")]
    metrics = {}
    recent = d.tail(RECENT_WINDOW)
    for label, hitc, oddc in bets:
        if oddc not in d.columns:
            continue
        hit_full = (1 - d["btts"]) if label == "BTTS_Non" else d[hitc]
        hit_rec = (1 - recent["btts"]) if label == "BTTS_Non" else recent[hitc]
        nf, rf, tf = roi(hit_full, d[oddc])
        nr, rr, tr = roi(hit_rec, recent[oddc])
        metrics[label] = {"full": [nf, round(100 * rf, 2), round(tf, 2)] if nf >= MIN_N else None,
                          "recent": [nr, round(100 * rr, 2), round(tr, 2)] if nr >= MIN_N else None}
        if nf >= MIN_N and rf > ROI_FLAG and tf > T_FLAG:
            flags.append(f"MONÉTISABLE full {label}: ROI={100*rf:+.2f}% t={tf:.2f}")
        if nr >= MIN_N and rr > ROI_FLAG and tr > T_FLAG:
            flags.append(f"MONÉTISABLE récent {label}: ROI={100*rr:+.2f}% t={tr:.2f} (fenêtre {nr})")

    # --- alerte PRÉCOCE : écart réalisé - implicite (Over3.5) qui s'ouvre ---
    def gap(sub):
        s = sub.dropna(subset=["o_over35", "o_under35"])
        s = s[(s.o_over35 > 1) & (s.o_under35 > 1)]
        if len(s) < MIN_N:
            return None, 0
        impl = ((1 / s.o_over35) / (1 / s.o_over35 + 1 / s.o_under35)).mean()
        return float(s.over35.mean() - impl), len(s)
    gap_full, _ = gap(d)
    gap_recent, n_gap_rec = gap(recent)
    if gap_recent is not None and abs(gap_recent) > DRIFT_GAP_FLAG:
        flags.append(f"DÉRIVE PRÉCOCE Over3.5: écart récent={100*gap_recent:+.2f}pp "
                     f"(vs {100*(gap_full or 0):+.2f}pp historique, n={n_gap_rec}) — fenêtre qui s'ouvre ?")
    return {"n": int(len(d)), "n_arb": n_arb, "sealed": len(flags) == 0,
            "flags": flags, "metrics": metrics,
            "gap_over35_full_pp": round(100 * gap_full, 3) if gap_full is not None else None,
            "gap_over35_recent_pp": round(100 * gap_recent, 3) if gap_recent is not None else None}


def main():
    eng = create_engine(load_settings().db_url)
    df = pd.read_sql(text(_SQL), eng)
    parts = []
    for lg, g in df.groupby("lg"):
        sub = build(g); sub["lg"] = lg
        parts.append(sub)
    B = pd.concat(parts, ignore_index=True)

    prev = {}
    if REPORT.exists():
        try:
            prev = json.loads(REPORT.read_text(encoding="utf-8")).get("leagues", {})
        except Exception:
            prev = {}

    report = {"leagues": {}, "global_flags": []}
    print("=" * 78)
    print("  MOTEUR MULTI — VEILLE DE CALIBRATION (Tier 1, déterministe)")
    print("=" * 78)
    print(f"  {'ligue':<20}{'n':>8}{'arb':>5}  {'Over3.5 ROI':>14}{'récent':>10}  verdict")
    print("  " + "-" * 74)
    any_flag = False
    for lg in sorted(B.lg.unique(), key=lambda L: -len(B[B.lg == L])):
        d = B[B.lg == lg]
        v = audit_league(lg, d)
        report["leagues"][lg] = v
        if lg not in prev:
            v["flags"].append("NOUVELLE LIGUE (jamais auditée)")
            v["sealed"] = False
        o = v["metrics"].get("Over3.5", {})
        of = o.get("full"); orr = o.get("recent")
        ostr = f"{of[1]:+.2f}% (t{of[2]:+.1f})" if of else "  n/a"
        rstr = f"{orr[1]:+.2f}%" if orr else "  n/a"
        verdict = "SCELLÉ" if v["sealed"] else "⚠ FLAG"
        print(f"  {lg:<20}{v['n']:>8}{v['n_arb']:>5}  {ostr:>14}{rstr:>10}  {verdict}")
        if not v["sealed"]:
            any_flag = True
            for f in v["flags"]:
                print(f"        -> {f}")
                report["global_flags"].append(f"{lg}: {f}")

    report["any_flag"] = any_flag
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    # --- journal daté (tendance dans le temps : 1 ligne JSONL par run) ---
    data_clock = str(df["ts"].max()) if len(df) else None
    hist_row = {"run_utc": datetime.now(timezone.utc).isoformat(), "data_clock": data_clock,
                "any_flag": any_flag,
                "leagues": {lg: {"n": v["n"], "n_arb": v["n_arb"],
                                 "gap_recent_pp": v.get("gap_over35_recent_pp"),
                                 "gap_full_pp": v.get("gap_over35_full_pp"),
                                 "over35_roi_recent": (v["metrics"].get("Over3.5", {}).get("recent") or [None, None, None])[1]}
                            for lg, v in report["leagues"].items()}}
    with HISTORY.open("a", encoding="utf-8") as f:
        f.write(json.dumps(hist_row, ensure_ascii=False) + "\n")
    print("  " + "-" * 74)
    if any_flag:
        print(f"  ⚠ {len(report['global_flags'])} SIGNAL(AUX) — escalade Tier 2 recommandée "
              f"(workflow adverse goal-edge-hunt).")
    else:
        print("  ✅ Toutes les ligues SCELLÉES. Aucune action. (re-vérifié, capital protégé.)")
    print(f"  rapport -> {REPORT.name}")
    print("=" * 78)
    sys.exit(2 if any_flag else 0)


if __name__ == "__main__":
    main()
