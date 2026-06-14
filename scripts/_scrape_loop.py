"""Scraper CONTINU : capte chaque round dès qu'il devient imminent (avec toutes ses
cotes + extra_markets) et l'accumule en base. Intervalle < cadence des rounds (~120 s)
pour ne RIEN rater. L'API ne publie que le round imminent → seul ce mode couvre tout.

Usage: ./.venv/Scripts/python.exe scripts/_scrape_loop.py [--interval 80] [--n 300]
"""
import sys, time, argparse
sys.path.insert(0, ".")
from datetime import datetime, timezone, timedelta
from scraper.config import load_settings
from scraper.collector import run_iteration
from scraper.db import init_engine
from scraper.utils import configure_logging

ap = argparse.ArgumentParser()
ap.add_argument("--interval", type=int, default=80, help="secondes entre 2 scrapes (<120)")
ap.add_argument("--n", type=int, default=300, help="nb d'itérations max")
a = ap.parse_args()

s = load_settings()
configure_logging(s.log_level, s.log_file)
init_engine(s)
MG = timezone(timedelta(hours=3))
print(f"scrape continu : interval={a.interval}s, n={a.n} (~{a.interval*a.n/3600:.1f}h)", flush=True)
for i in range(1, a.n + 1):
    t0 = time.time()
    try:
        run_iteration(s)
        ok = "ok"
    except Exception as e:  # noqa: BLE001
        ok = f"ERR {type(e).__name__}: {e}"
    now = datetime.now(timezone.utc).astimezone(MG).strftime("%H:%M:%S")
    print(f"[{i}/{a.n}] {now} {ok} ({time.time()-t0:.1f}s)", flush=True)
    time.sleep(max(0, a.interval - (time.time() - t0)))
