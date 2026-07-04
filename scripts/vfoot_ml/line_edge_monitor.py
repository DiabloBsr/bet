"""Moniteur auto de l'EDGE MOUVEMENT DE LIGNE — 3 signaux : DOM / EXT / NUL.

Re-teste à chaque run : parier l'issue dont la proba implicite a AUGMENTÉ entre
l'ouverture et la clôture (anti-fuite : clôture = dernier snap avant le start),
à la cote de clôture. Pool des 9 ligues (même fournisseur).
Historise (n, ROI, t-stat par signal×seuil) dans line_edge_history.jsonl.

Exit 2 (ALERTE) si UN signal est CONFIRMÉ : t>2.5 ET OOS même signe ET ROI>+2%.
Sinon exit 0. Lecture seule (n'écrit que son historique).
"""
from __future__ import annotations
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
os.chdir(_ROOT)                      # db_url relative -> CWD projet obligatoire
sys.path.insert(0, str(_ROOT))
if sys.stdout is None:               # pythonw / Task Scheduler
    (_ROOT / "data" / "logs").mkdir(parents=True, exist_ok=True)
    _lg = open(_ROOT / "data" / "logs" / "line_monitor.log", "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = _lg
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np, pandas as pd
from sqlalchemy import create_engine, text
from scraper.config import load_settings

LG = "InstantLeague-%"        # POOL des 9 ligues (même fournisseur/marge)
HIST = _ROOT / "data" / "vfoot_ml" / "line_edge_history.jsonl"
T_CONFIRM = 2.5
ROI_MIN = 0.02
THRESHOLDS = (0.005, 0.01, 0.02)

SQL = """
WITH snaps AS (
  SELECT o.event_id,o.captured_at,o.odds_home,o.odds_draw,o.odds_away,
    ROW_NUMBER() OVER (PARTITION BY o.event_id ORDER BY o.captured_at ASC) ro,
    ROW_NUMBER() OVER (PARTITION BY o.event_id ORDER BY o.captured_at DESC) rc
  FROM odds_snapshots o JOIN events e ON e.id=o.event_id
  WHERE e.competition LIKE :lg AND o.captured_at<e.expected_start
    AND o.odds_home>1 AND o.odds_draw>1 AND o.odds_away>1)
SELECT e.expected_start ts, r.score_a sa, r.score_b sb,
 op.odds_home oh_o, op.odds_draw od_o, op.odds_away oa_o,
 cl.odds_home oh_c, cl.odds_draw od_c, cl.odds_away oa_c
FROM events e JOIN results r ON r.event_id=e.id AND r.score_a IS NOT NULL
JOIN snaps op ON op.event_id=e.id AND op.ro=1
JOIN snaps cl ON cl.event_id=e.id AND cl.rc=1
WHERE e.competition LIKE :lg AND op.captured_at<cl.captured_at ORDER BY e.expected_start
"""


def imps(h, d, a):
    inv = 1/h + 1/d + 1/a
    return (1/h)/inv, (1/d)/inv, (1/a)/inv


def roi_t(pnl):
    pnl = np.asarray(pnl, float)
    if len(pnl) < 30:
        return len(pnl), float("nan"), float("nan")
    se = pnl.std(ddof=1) / np.sqrt(len(pnl))
    return len(pnl), float(pnl.mean()), (float(pnl.mean()/se) if se > 0 else float("nan"))


def main():
    e = create_engine(load_settings().db_url)
    df = pd.read_sql(text(SQL), e, params={"lg": LG})
    ih_c, id_c, ia_c = imps(df.oh_c, df.od_c, df.oa_c)
    ih_o, id_o, ia_o = imps(df.oh_o, df.od_o, df.oa_o)
    df["move_h"], df["move_d"], df["move_a"] = ih_c - ih_o, id_c - id_o, ia_c - ia_o
    df["w_h"] = (df.sa > df.sb).astype(int)
    df["w_d"] = (df.sa == df.sb).astype(int)
    df["w_a"] = (df.sa < df.sb).astype(int)
    cut = df.ts.iloc[len(df)//2] if len(df) else None
    te = df[df.ts >= cut] if cut is not None else df

    SIGNALS = [("DOM", "move_h", "w_h", "oh_c"), ("EXT", "move_a", "w_a", "oa_c"),
               ("NUL", "move_d", "w_d", "od_c")]
    print("=" * 70)
    print("  MONITEUR EDGE MOUVEMENT DE LIGNE — 3 signaux (pool 9 ligues)")
    print("=" * 70)
    print(f"  events anti-fuite (ouverture!=cloture) : {len(df)}")
    print(f"  {'signal':<7}{'seuil':<8}{'n_full':>7}{'ROI_full':>10}{'t_full':>8}{'ROI_oos':>10}{'n_oos':>7}")
    record = {"run_utc": datetime.now(timezone.utc).isoformat(), "n_total": int(len(df)),
              "signals": {}}
    confirmed = []
    for sig, mv, w, oc in SIGNALS:
        record["signals"][sig] = {}
        for thr in THRESHOLDS:
            full = df[df[mv] > thr]
            n, roi, t = roi_t((full[w] * full[oc] - 1).values)
            oos = te[te[mv] > thr]
            no, roio, _ = roi_t((oos[w] * oos[oc] - 1).values)
            same = (roi == roi and roio == roio and np.sign(roi) == np.sign(roio))
            flag = bool(n >= 30 and roi > ROI_MIN and t > T_CONFIRM and same)
            if flag:
                confirmed.append(f"{sig} move>{thr}")
            record["signals"][sig][str(thr)] = {
                "n": n, "roi_full": round(roi, 4) if roi == roi else None,
                "t_full": round(t, 2) if t == t else None,
                "roi_oos": round(roio, 4) if roio == roio else None, "confirmed": flag}
            rs = f"{100*roi:+.2f}%" if roi == roi else "  n/a"
            ts_ = f"{t:+.2f}" if t == t else " n/a"
            ros = f"{100*roio:+.2f}%" if roio == roio else "  n/a"
            print(f"  {sig:<7}>{thr:<7}{n:>7}{rs:>10}{ts_:>8}{ros:>10}{no:>7}"
                  f"{'  <<< CONFIRMÉ' if flag else ''}")
    # projection combinés pour tout signal×seuil à ROI>0 et n>=100
    print("  " + "-" * 66)
    print("  PROJECTION COMBINES (valable SEULEMENT une fois l'edge confirme) :")
    for sig, mv, w, oc in SIGNALS:
        for thr in THRESHOLDS:
            full = df[df[mv] > thr]
            if len(full) < 100:
                continue
            roi = float((full[w] * full[oc] - 1).mean())
            if roi <= 0:
                continue
            hit = float(full[w].mean())
            c2, c3 = (1+roi)**2 - 1, (1+roi)**3 - 1
            print(f"    {sig} move>{thr}: jambe {100*roi:+5.1f}% (hit {100*hit:.0f}%)"
                  f" | x2 {100*c2:+6.1f}% | x3 {100*c3:+6.1f}%")

    record["confirmed"] = confirmed
    HIST.parent.mkdir(parents=True, exist_ok=True)
    with HIST.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print("  " + "-" * 66)
    if confirmed:
        print(f"  🚨 EDGE CONFIRMÉ : {', '.join(confirmed)}")
        print("     -> vérif adverse + décision humaine. NE PAS déployer auto.")
    else:
        nn = record["signals"]["DOM"].get("0.01", {}).get("n", 0)
        print(f"  ⏳ Pas encore confirmé (n[DOM>0.01]={nn}). Le scraper 9-ligues accumule.")
    print(f"  historique -> {HIST.name}")
    print("=" * 70)
    sys.exit(2 if confirmed else 0)


if __name__ == "__main__":
    main()
