"""BILAN DE SANTÉ du système — une commande pour tout voir.

Usage : ./.venv/Scripts/python.exe scripts/health_check.py

Vérifie chaque maillon de l'auto-ajustement :
  1. scraper (fraîcheur des captures)
  2. calibration auto (âge du recalage + résidu)
  3. suivi forward trio (hit-rates vs plafonds -> détecteur de dérive du RNG)
  4. moniteur mouvement de ligne (accumulation vers le verdict n~1000)
  5. paper-trader forward
  6. tâches planifiées Windows
Statuts : [OK] / [!] attention / [X] mort.
"""
from __future__ import annotations
import json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np, pandas as pd
from sqlalchemy import create_engine, text
from scraper.config import load_settings

NOW = pd.Timestamp.now(tz="UTC")


def age_min(ts):
    return (NOW - pd.to_datetime(ts, utc=True)).total_seconds() / 60


def main():
    eng = create_engine(load_settings().db_url)
    print("=" * 72)
    print("  BILAN DE SANTE — VFoot (auto-ajustement & veille)")
    print("=" * 72)

    # 1. scraper
    try:
        c = pd.read_sql(text("SELECT MAX(captured_at) c FROM odds_snapshots"), eng).c.iloc[0]
        a = age_min(c)
        st = "OK" if a < 5 else ("!" if a < 60 else "X")
        print(f"  [{st}] SCRAPER          derniere capture il y a {a:.1f} min "
              f"{'' if a < 5 else '(watchdog relancera sous 1h)' if a < 90 else '(VERIFIER)'}")
    except Exception as exc:
        print(f"  [X] SCRAPER          erreur : {exc}")

    # 2. calibration auto
    try:
        cal = json.loads((ROOT / "data" / "vfoot_ml" / "score_calibration.json").read_text(encoding="utf-8"))
        upd = cal.get("updated_utc")
        if upd:
            days = age_min(upd) / 1440
            st = "OK" if days < 8 else "!"
            print(f"  [{st}] CALIBRATION      recalee il y a {days:.1f} j sur {cal.get('n_matches','?')} matchs "
                  f"(auto chaque dimanche 03:17{'' if days < 8 else ' — EN RETARD'})")
        else:
            print("  [!] CALIBRATION      fichier sans date (sera date au prochain refresh dimanche)")
    except Exception as exc:
        print(f"  [X] CALIBRATION      erreur : {exc}")

    # 3. suivi forward = détecteur de dérive de la DISTRIBUTION
    try:
        d = pd.read_sql(text("""SELECT hit1_cal, hit3, hitx, created_at FROM trio_predictions
            WHERE actual IS NOT NULL AND actual!='VOID' ORDER BY id DESC LIMIT 300"""), eng)
        last = pd.read_sql(text("SELECT MAX(created_at) c FROM trio_predictions"), eng).c.iloc[0]
        a = age_min(last) if last else 1e9
        st = "OK" if a < 45 else ("!" if a < 180 else "X")
        print(f"  [{st}] TRACKER TRIO     derniere prediction il y a {a:.0f} min ; {len(d)} scorees (fenetre)")
        if len(d) >= 50:
            checks = [("Top-1 cal", d.hit1_cal.mean(), 0.119), ("Top-3", d.hit3.mean(), 0.316),
                      ("1X2", d.hitx.mean(), 0.55)]
            drift = []
            for name, obs, ceil in checks:
                se = np.sqrt(ceil * (1 - ceil) / len(d))
                z = (obs - ceil) / se
                flag = "DERIVE" if abs(z) > 3 else "ok"
                if abs(z) > 3:
                    drift.append(name)
                print(f"        {name:<10} {100*obs:5.1f}% vs plafond {100*ceil:.1f}%  (z={z:+.1f}, {flag})")
            if drift:
                print(f"        >>> DERIVE DETECTEE ({', '.join(drift)}) : la distribution du RNG a peut-etre")
                print("        >>> change — la calibration se recalera dimanche ; relancer une chasse est justifie.")
            else:
                print("        -> distribution STABLE : le systeme colle au RNG, rien a ajuster.")
        else:
            print(f"        (encore {50-len(d)} predictions scorees avant le premier verdict de derive)")
    except Exception as exc:
        print(f"  [X] TRACKER TRIO     erreur : {exc}")

    # 4. moniteur ligne
    try:
        hist = (ROOT / "data" / "vfoot_ml" / "line_edge_history.jsonl").read_text(encoding="utf-8").strip().splitlines()
        rec = json.loads(hist[-1])
        n01 = rec.get("by_thr", {}).get("0.01", {}).get("n", "?")
        conf = rec.get("confirmed", False)
        a = age_min(rec["run_utc"]) / 1440
        st = "OK" if a < 2 else "!"
        print(f"  [{st}] LIGNE (edge)     dernier check il y a {a:.1f} j ; n[move>0.01]={n01}/1000 vers le verdict"
              f"{' ; *** CONFIRME ***' if conf else ''}")
    except Exception as exc:
        print(f"  [!] LIGNE (edge)     pas encore d'historique ({type(exc).__name__})")

    # 5. paper-trader
    try:
        p = pd.read_sql(text("""SELECT COUNT(*) n, SUM(CASE WHEN home_win IS NOT NULL THEN 1 ELSE 0 END) s
            FROM line_paper_bets"""), eng)
        b = pd.read_sql(text("SELECT pnl_home FROM line_paper_bets WHERE home_win IS NOT NULL AND move_h>0.01"), eng)
        roi = f", ROI move>0.01 = {100*b.pnl_home.mean():+.1f}% (n={len(b)})" if len(b) else ""
        print(f"  [OK] PAPER-TRADER    {int(p.n.iloc[0])} matchs au registre, {int(p.s.iloc[0] or 0)} regles{roi}")
    except Exception as exc:
        print(f"  [X] PAPER-TRADER    erreur : {exc}")

    # 6. tâches planifiées
    try:
        out = subprocess.run(["schtasks", "/Query", "/FO", "CSV", "/NH"], capture_output=True,
                             text=True, encoding="cp850", errors="replace", timeout=30).stdout or ""
        tasks = [l.split('","')[0].strip('"\\') for l in out.splitlines() if "VFoot" in l]
        st = "OK" if len(tasks) >= 5 else "!"
        print(f"  [{st}] TACHES WINDOWS   {len(tasks)}/5 actives : {', '.join(t.lstrip(chr(92))for t in tasks)}")
    except Exception as exc:
        print(f"  [!] TACHES WINDOWS   verification impossible : {exc}")

    print("=" * 72)
    print("  Auto-ajustement : calibration -> dimanche ; modeles -> re-fit 30 min ;")
    print("  derive RNG -> visible ici (z>3) AVANT toute action. Verdict ligne -> n~1000.")
    print("=" * 72)


if __name__ == "__main__":
    main()
