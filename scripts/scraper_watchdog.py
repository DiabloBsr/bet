"""WATCHDOG du scraper continu — à planifier toutes les heures (Task Scheduler).

Vérifie la fraîcheur des captures en base ; si rien depuis >4 min :
  1. tue les éventuels loops zombies (_scrape_loop dans la cmdline),
  2. relance _scrape_loop.py --interval 45 --n 100000 en DÉTACHÉ sans fenêtre
     (pythonw), stdout -> data/logs/scrape_loop.log.
Anti-doublon par fraîcheur DB : si un loop tourne, on ne fait rien.
"""
from __future__ import annotations
import os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
LOGS = ROOT / "data" / "logs"
LOGS.mkdir(parents=True, exist_ok=True)
if sys.stdout is None:               # lancé par pythonw (Task Scheduler, sans fenêtre)
    _lg = open(LOGS / "watchdog.log", "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = _lg

import pandas as pd
from sqlalchemy import create_engine, text
from scraper.config import load_settings

FRESH_MIN = 4
INTERVAL = 45


def now_utc():
    return datetime.now(timezone.utc)


def main():
    stamp = now_utc().strftime("%Y-%m-%d %H:%M:%S UTC")
    eng = create_engine(load_settings().db_url)
    last = pd.read_sql(text("SELECT MAX(captured_at) c FROM odds_snapshots"), eng).c.iloc[0]
    if last is not None:
        age = (now_utc() - pd.to_datetime(last, utc=True)).total_seconds() / 60
        if age < FRESH_MIN:
            print(f"[{stamp}] scraper VIVANT (derniere capture il y a {age:.1f} min) — rien a faire")
            return
        print(f"[{stamp}] scraper MORT (derniere capture il y a {age:.1f} min) — relance")
    else:
        print(f"[{stamp}] base vide — lancement initial")

    # tue les zombies eventuels (loop bloque sans capturer)
    try:
        out = subprocess.run(
            ["wmic", "process", "where",
             "commandline like '%_scrape_loop%' and name like 'python%'",
             "get", "processid", "/format:value"],
            capture_output=True, text=True, timeout=30).stdout
        for line in out.splitlines():
            if line.startswith("ProcessId="):
                pid = line.split("=")[1].strip()
                if pid:
                    subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True, timeout=15)
                    print(f"  zombie tue : PID {pid}")
    except Exception as exc:
        print(f"  (scan zombies impossible : {exc})")

    pyw = ROOT / ".venv" / "Scripts" / "pythonw.exe"
    py = pyw if pyw.exists() else ROOT / ".venv" / "Scripts" / "python.exe"
    log = open(LOGS / "scrape_loop.log", "a", encoding="utf-8")
    DETACHED = 0x00000008 | 0x00000200 | 0x08000000   # DETACHED | NEW_PROCESS_GROUP | NO_WINDOW
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    p = subprocess.Popen(
        [str(py), str(ROOT / "scripts" / "_scrape_loop.py"),
         "--interval", str(INTERVAL), "--n", "100000"],
        cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
        creationflags=DETACHED, env=env)
    print(f"  relance : PID {p.pid}, interval {INTERVAL}s, log -> data/logs/scrape_loop.log")


if __name__ == "__main__":
    main()
