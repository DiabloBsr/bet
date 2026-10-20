"""Boucle de collecte TT Cup / Czech Liga Pro — données officielles ttcup.com.

Lancée en fond sur le Space par deploy/start_cloud.sh. Chaque cycle récupère le
snapshot hotblock (joignable depuis le datacenter) et ACCUMULE les matchs joués
dans un JSON sur /data. L'historique grandit au fil de la journée et des jours,
ce qui permet au prédicteur de s'entraîner et de mesurer sa fiabilité réelle.

Usage : python scripts/_ttcup_loop.py [--interval 120] [--once] [--path /data/ttcup_history.json]
"""
import argparse
import os
import sys
import time
from pathlib import Path

_ICI = Path(__file__).resolve()
sys.path.insert(0, str(_ICI.parent))        # scripts/ -> ttcup_tt

import ttcup_tt as tt  # noqa: E402


def _defaut_path() -> str:
    """/data si présent (Space), sinon un fichier local."""
    base = "/data" if os.path.isdir("/data") else str(_ICI.parents[1])
    return os.path.join(base, "ttcup_history.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=120,
                    help="secondes entre deux cycles")
    ap.add_argument("--once", action="store_true", help="un seul cycle puis stop")
    ap.add_argument("--path", default=_defaut_path(), help="fichier d'historique JSON")
    a = ap.parse_args()

    while True:
        try:
            total, neufs = tt.collecter_once(a.path)
            print(f"[ttcup] {total} matchs en base (+{neufs} joués ce cycle)", flush=True)
        except Exception as exc:                # une panne réseau ne tue pas la boucle
            print(f"[ttcup] erreur : {type(exc).__name__}: {exc}", flush=True)
        if a.once:
            break
        time.sleep(max(30, a.interval))


if __name__ == "__main__":
    main()
