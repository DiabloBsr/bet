"""FILET MAX-PUISSANCE — re-teste toute la batterie de signaux, poolee 9 ligues,
au CANAL LE MOINS CHER, sur les donnees les plus fraiches. Python pur (RAM-safe).

But : "exploiter a max meme l'infime" dans le TEMPS. Le RNG est certifie, mais si
le seed derive un jour et qu'un signal FRANCHIT la marge, ce filet le detecte la
meme semaine (exit 2 = ALERTE). Sinon exit 0.

Chaque hypothese : ROI OOS aux vraies cotes + IC95 analytique + edge implicite.
Correction multiple BH-FDR (q=0.10) pour la liste de veille. Historise en JSONL.
Sortie ASCII (Task Scheduler-safe).
"""
from __future__ import annotations
import json, os, sys
from math import sqrt, erf
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
if sys.stdout is None:
    (ROOT / "data" / "logs").mkdir(parents=True, exist_ok=True)
    _lg = open(ROOT / "data" / "logs" / "max_power_sweep.log", "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = _lg
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import sqlite3

DB = ROOT / "data" / "virtual_sports.db"
HIST = ROOT / "data" / "vfoot_ml" / "max_power_history.jsonl"


def gm(xm, pref):
    for k, v in (xm or {}).items():
        if k.replace("\x82", "e").replace("\xe9", "e").startswith(pref):
            return v
    return None


def ok(x):
    return x if isinstance(x, (int, float)) and 1 < x < 99.99 else None


def roi_ci(pnl):
    """ROI moyen + IC95 analytique (normal, exact au n>1e4)."""
    n = len(pnl)
    if n < 100:
        return None
    m = sum(pnl) / n
    v = sum((x - m) ** 2 for x in pnl) / (n - 1)
    se = sqrt(v / n)
    return m, m - 1.96 * se, m + 1.96 * se, n, m / se if se else 0.0


def bh_fdr(pvals, q=0.10):
    """Benjamini-Hochberg : renvoie l'ensemble des indices declares significatifs."""
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    keep = set()
    for rank, i in enumerate(idx, 1):
        if pvals[i] <= q * rank / m:
            keep = set(idx[:rank])
    return keep


def pnorm_two(z):
    return 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))


def load():
    c = sqlite3.connect(DB, timeout=60)
    rows = c.execute("""
        SELECT e.competition comp, e.external_id xid, o.extra_markets xm,
               r.score_a sa, r.score_b sb, o.odds_home oh, o.odds_draw od, o.odds_away oa,
               r.ht_score_a ha, r.ht_score_b hb
        FROM events e JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
        JOIN results r ON r.event_id=e.id
        WHERE e.competition LIKE 'InstantLeague-%' AND r.score_a IS NOT NULL""").fetchall()
    seqs = {}
    for comp, xid, xm, sa, sb, oh, od, oa, ha, hb in rows:
        try:
            j = json.loads(xm) if isinstance(xm, str) else (xm or {})
            xi = int(xid)
        except Exception:
            continue
        pm = gm(j, "+/-")
        o_o35 = ok(pm.get("> 3.5")) if isinstance(pm, dict) else None
        o_u35 = ok(pm.get("< 3.5")) if isinstance(pm, dict) else None
        mb = gm(j, "Multi-Buts")
        o_u25 = None
        if isinstance(mb, dict):
            for k, val in mb.items():
                if "0, 1 ou 2" in k and ok(val):
                    o_u25 = val
        gg = gm(j, "G/NG")
        o_ng = ok(gg.get("Non")) if isinstance(gg, dict) else None
        o_g = ok(gg.get("Oui")) if isinstance(gg, dict) else None
        tot = sa + sb
        fav = None
        dog = None
        if ok(oh) and ok(oa):
            fav = ("H", ok(oh)) if oh < oa else ("A", ok(oa))
            dog = ("A", ok(oa)) if oh < oa else ("H", ok(oh))   # outsider = cote la + haute
        # --- marche mi-temps : outsider mene a la MT ? + sa cote MT ---
        o_ht_out = None
        ht_out_leads = None
        ht = gm(j, "Mi-tps 1X2")
        if isinstance(ht, dict) and fav and ha is not None and hb is not None:
            out_side = "A" if fav[0] == "H" else "H"      # outsider = pas le favori
            o_ht_out = ok(ht.get("2")) if out_side == "A" else ok(ht.get("1"))
            ht_out_leads = int((hb > ha) if out_side == "A" else (ha > hb))
        seqs.setdefault(comp, []).append({
            "xi": xi, "tot": tot, "u25": int(tot <= 2), "u35": int(tot <= 3),
            "o_o35": o_o35, "o_u35": o_u35, "o_u25": o_u25,
            "btts": int(sa > 0 and sb > 0), "o_ng": o_ng, "o_g": o_g,
            "odd": tot % 2, "fav": fav, "dog": dog,
            "res": ("H" if sa > sb else "A" if sb > sa else "D"),
            "o_ht_out": o_ht_out, "ht_out_leads": ht_out_leads,
        })
    for comp in seqs:
        seqs[comp].sort(key=lambda r: r["xi"])
    return seqs


def main():
    seqs = load()
    ntot = sum(len(v) for v in seqs.values())
    tests = []  # (label, [pnl_oos])

    # -- inconditionnels (canal le moins cher priorise) --
    def collect(pred, oddkey, oos_only=True):
        pnl = []
        for comp, arr in seqs.items():
            half = arr[len(arr) // 2]["xi"] if arr else 0
            for r in arr:
                if oos_only and r["xi"] < half:
                    continue
                o = r.get(oddkey)
                if o:
                    pnl.append(pred(r) * o - 1)
        return pnl

    tests.append(("Under 3.5 global (marge 5.7%)", collect(lambda r: r["u35"], "o_u35")))
    tests.append(("Over 3.5 global (marge 5.7%)", collect(lambda r: 1 - r["u35"], "o_o35")))
    tests.append(("Under 2.5 global (marge 9%)", collect(lambda r: r["u25"], "o_u25")))
    tests.append(("NG (0-0..) global", collect(lambda r: 1 - r["btts"], "o_ng")))
    tests.append(("G (BTTS) global", collect(lambda r: r["btts"], "o_g")))
    tests.append(("Suivre le favori", collect(
        lambda r: int(r["res"] == r["fav"][0]) if r["fav"] else 0, None) or []))

    # favori via ses propres cotes (canal 1X2, marge 5.7%)
    favp = []
    for comp, arr in seqs.items():
        half = arr[len(arr) // 2]["xi"] if arr else 0
        for r in arr:
            if r["xi"] >= half and r["fav"]:
                favp.append(int(r["res"] == r["fav"][0]) * r["fav"][1] - 1)
    tests[-1] = ("Suivre le favori (1X2, 5.7%)", favp)

    # -- CAN outsider cote 6-10 : le pari le MOINS marge de tout Bet261 (~breakeven) --
    #    surveille en permanence : si un jour il passe +EV (IC_bas>0), le RNG a derive.
    canp = []
    can_arr = seqs.get("InstantLeague-8060", [])
    half_can = can_arr[len(can_arr) // 2]["xi"] if can_arr else 0
    for r in can_arr:
        if r["xi"] >= half_can and r.get("dog") and 6.0 <= r["dog"][1] <= 10.0:
            canp.append(int(r["res"] == r["dog"][0]) * r["dog"][1] - 1)
    tests.append(("CAN outsider cote 6-10 (le moins marge)", canp))

    # -- marche mi-temps : outsider mene a la MT (grosse cote) --
    htp = []
    for comp, arr in seqs.items():
        half = arr[len(arr) // 2]["xi"] if arr else 0
        for r in arr:
            if r["xi"] >= half and r.get("o_ht_out") and r.get("ht_out_leads") is not None:
                htp.append(r["ht_out_leads"] * r["o_ht_out"] - 1)
    tests.append(("Outsider mene a la MT (grosse cote)", htp))

    # -- conditionnels : apres k unders/overs -> canal le moins cher (Under 3.5) --
    for K in (3, 5, 7):
        pnl_u, pnl_o = [], []
        for comp, arr in seqs.items():
            half = arr[len(arr) // 2]["xi"] if arr else 0
            run_u = run_o = 0
            for r in arr:
                if run_u >= K and r["xi"] >= half and r["o_u35"]:
                    pnl_u.append(r["u35"] * r["o_u35"] - 1)
                if run_o >= K and r["xi"] >= half and r["o_o35"]:
                    pnl_o.append((1 - r["u35"]) * r["o_o35"] - 1)
                run_u = run_u + 1 if r["u25"] == 1 else 0
                run_o = run_o + 1 if r["tot"] > 2 else 0
        tests.append((f"apres {K} unders -> Under 3.5", pnl_u))
        tests.append((f"apres {K} overs -> Over 3.5", pnl_o))

    # -- parite : apres impair -> impair (canal G/NG proxy: aucun canal direct, on skip ROI) --

    # evaluation
    results, pvals, valid = [], [], []
    for lbl, pnl in tests:
        r = roi_ci(pnl)
        if r is None:
            results.append((lbl, None))
            continue
        m, lo, hi, n, z = r
        results.append((lbl, (m, lo, hi, n, z)))
        pvals.append(pnorm_two(z))
        valid.append(len(results) - 1)

    keep = bh_fdr(pvals, 0.10) if pvals else set()
    # map FDR indices back
    fdr_flags = {}
    for j, ri in enumerate(valid):
        fdr_flags[ri] = j in keep

    now = datetime.now(timezone.utc).isoformat()
    alert = False
    print(f"[{now[:16]}] FILET MAX-PUISSANCE | 9 ligues | {ntot} matchs")
    print(f"{'hypothese':<34}{'n':>7}{'ROI':>9}{'IC95_bas':>10}{'IC95_haut':>10}  flag")
    rec = {"run_utc": now, "n_matchs": ntot, "hyps": []}
    for ri, (lbl, r) in enumerate(results):
        if r is None:
            print(f"{lbl:<34}{'  (n<100)':>36}")
            continue
        m, lo, hi, n, z = r
        profitable = lo > 0            # edge net APRES marge -> exploitable
        watch = fdr_flags.get(ri) and m > 0
        flag = "ALERTE!!" if profitable else ("veille" if watch else "")
        if profitable:
            alert = True
        print(f"{lbl:<34}{n:>7}{100*m:>+8.1f}%{100*lo:>+9.1f}%{100*hi:>+9.1f}%  {flag}")
        rec["hyps"].append({"label": lbl, "n": n, "roi": round(m, 4),
                            "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                            "profitable": profitable, "watch": bool(watch)})
    rec["alert"] = alert
    HIST.parent.mkdir(parents=True, exist_ok=True)
    with HIST.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print("-> " + ("!! UN SIGNAL A FRANCHI LA MARGE — verifier d'urgence !!" if alert
                   else "aucun signal ne franchit la marge (RNG toujours calibre)."))
    sys.exit(2 if alert else 0)


if __name__ == "__main__":
    main()
