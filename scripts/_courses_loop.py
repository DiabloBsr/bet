"""Boucle de collecte des courses (cotes + arrivées) — léger (urllib + sqlite).

Lancée en fond sur le Space par deploy/start_cloud.sh. Chaque cycle capte les
courses à venir des deux disciplines et récupère l'ordre d'arrivée de celles qui
viennent de se courir. Écrit dans la table `courses` de la base /data.

Usage : python scripts/_courses_loop.py [--interval 60] [--once]
"""
import argparse
import sys
import time
from pathlib import Path

_ICI = Path(__file__).resolve()
sys.path.insert(0, str(_ICI.parent))        # scripts/ -> courses_collect
sys.path.insert(0, str(_ICI.parents[1]))    # racine -> package scraper

from scraper.config import load_settings  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402

import courses_collect as cc  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=60,
                    help="secondes entre deux cycles")
    ap.add_argument("--once", action="store_true", help="un seul cycle puis stop")
    a = ap.parse_args()

    eng = create_engine(load_settings().db_url, connect_args={"timeout": 30})
    while True:
        try:
            neuves, reglees = cc.collect_once(eng)
            print(f"[courses] +{neuves} à venir · {reglees} réglées", flush=True)
        except Exception as exc:               # une panne réseau ne tue pas la boucle
            print(f"[courses] erreur : {type(exc).__name__}: {exc}", flush=True)
        if a.once:
            break
        time.sleep(max(10, a.interval))


if __name__ == "__main__":
    main()
