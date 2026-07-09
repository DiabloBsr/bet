"""MONITEUR du CYCLE SEEDÉ (théorie en ligne #1) — re-teste chaque jour.

Après 5 unders-2.5 consécutifs (pool 9 ligues) -> pari under 2.5 (Multi-Buts 0,1,2).
Historise ROI OOS + IC95 bootstrap. Exit 2 (ALERTE) si IC95 bas > 0 (edge confirmé
= l'empreinte a dépassé la marge). Sinon exit 0. Sortie ASCII (Task Scheduler-safe).
"""
from __future__ import annotations
import json, os, random, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
if sys.stdout is None:
    (ROOT / "data" / "logs").mkdir(parents=True, exist_ok=True)
    _lg = open(ROOT / "data" / "logs" / "seeded_monitor.log", "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = _lg
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import sqlite3

K = 5
HIST = ROOT / "data" / "vfoot_ml" / "seeded_history.jsonl"
DB = ROOT / "data" / "virtual_sports.db"


def gm(xm, pref):
    for k, v in (xm or {}).items():
        if k.replace("\x82", "e").replace("\xe9", "e").startswith(pref):
            return v
    return None


def main():
    c = sqlite3.connect(DB, timeout=60)
    rows = c.execute("""
        SELECT e.competition comp, e.external_id xid, o.extra_markets xm, r.score_a sa, r.score_b sb
        FROM events e JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
        JOIN results r ON r.event_id=e.id
        WHERE e.competition LIKE 'InstantLeague-%' AND r.score_a IS NOT NULL""").fetchall()
    seqs = {}
    for comp, xid, xm, sa, sb in rows:
        try:
            j = json.loads(xm) if isinstance(xm, str) else (xm or {}); xi = int(xid)
        except Exception:
            continue
        tot = sa + sb
        tt = gm(j, "Total de buts"); imp = None
        if isinstance(tt, dict):
            v = {k: 1/tt[k] for k in [str(x) for x in range(7)]
                 if isinstance(tt.get(k), (int, float)) and 1 < tt[k] < 99.99}
            s = sum(v.values())
            if s and len(v) == 7:
                imp = (v["0"]+v["1"]+v["2"])/s
        mb = gm(j, "Multi-Buts"); o = None
        if isinstance(mb, dict):
            for k, val in mb.items():
                if "0, 1 ou 2" in k and isinstance(val, (int, float)) and 1 < val < 99.99:
                    o = val
        if imp is None:
            continue
        seqs.setdefault(comp, []).append((xi, int(tot <= 2), imp, o))

    ev = []
    for comp, arr in seqs.items():
        arr.sort()
        half = arr[len(arr)//2][0] if arr else 0
        run = 0
        for xi, u, imp, o in arr:
            if run >= K:
                ev.append((u, imp, o, xi >= half))
            run = run + 1 if u == 1 else 0
    te = [e for e in ev if e[3]]
    bets = [(e[0], e[2]) for e in te if e[2]]
    now = datetime.now(timezone.utc).isoformat()
    rec = {"run_utc": now, "k": K, "n_cases": len(ev), "n_oos_bets": len(bets)}
    confirmed = False
    if len(bets) >= 100:
        edge_oos = sum(e[0] for e in te)/len(te) - sum(e[1] for e in te)/len(te)
        pnl = [u*o - 1 for u, o in bets]
        roi = sum(pnl)/len(pnl)
        random.seed(12345)
        boots = sorted(sum(pnl[random.randrange(len(pnl))] for _ in range(len(pnl)))/len(pnl)
                       for _ in range(2000))
        lo, hi = boots[50], boots[1949]
        confirmed = lo > 0
        rec.update({"edge_oos": round(edge_oos, 4), "roi_oos": round(roi, 4),
                    "ci_lo": round(lo, 4), "ci_hi": round(hi, 4), "confirmed": confirmed})
        print(f"[{now[:16]}] cycle seede : n_oos={len(bets)} edge {100*edge_oos:+.2f}pp "
              f"ROI {100*roi:+.1f}% IC95[{100*lo:+.1f},{100*hi:+.1f}] "
              f"-> {'CONFIRME' if confirmed else 'pas d edge'}")
    else:
        print(f"[{now[:16]}] pas assez de cas OOS ({len(bets)})")
    HIST.parent.mkdir(parents=True, exist_ok=True)
    with HIST.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    sys.exit(2 if confirmed else 0)


if __name__ == "__main__":
    main()
