"""Recalibration automatique quand le RNG dérive.

Jusqu'ici la dérive était seulement SIGNALÉE dans le dashboard : il fallait penser
à relancer la calibration à la main, et entre-temps le moteur affichait des probas
périmées. Ce script ferme la boucle.

Deux capteurs indépendants, l'un sur nos prédictions, l'autre sur le book :

  1. DÉRIVE DE PRÉCISION — le taux de réussite réel des 300 dernières prédictions
     s'écarte de son plafond théorique de plus de 3 écarts-types. C'est le signe
     que la distribution des scores a bougé sous nos pieds.
  2. DÉRIVE DE PRICING — cross_market_check signale une marge sortie de sa bande
     habituelle. C'est le signe que le book a changé de moteur de cotation.

L'un ou l'autre suffit à déclencher `refresh_calibration.py`, qui régénère les
9 tables par ligue. Un délai de garde (24h par défaut) évite qu'un capteur bruyant
ne relance la calibration en boucle.

    python scripts/auto_recalibrate.py [--dry-run] [--force] [--cooldown-h 24]

Sortie ASCII, code retour 0 (rien à faire ou recalibré), 1 (échec).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)                       # db_url relatif -> CWD projet obligatoire
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

if sys.stdout is None:               # pythonw / Planificateur de taches : pas de console
    (ROOT / "data" / "logs").mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(ROOT / "data" / "logs" / "auto_recalib.log",
                                   "a", encoding="utf-8", buffering=1)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ETAT = ROOT / "data" / "vfoot_ml" / "auto_recalib_state.json"
Z_MAX = 3.0
N_MIN = 100

# plafonds theoriques du moteur (bornes de Bayes mesurees, cf. THEORIES_TESTED.md)
PLAFONDS = (("Top-1", "hit1_cal", 0.119), ("Top-3", "hit3", 0.316), ("1X2", "hitx", 0.55))


def _python() -> str:
    """Interpreteur AVEC console : pythonw n'a pas de stdout, les sous-processus
    doivent donc utiliser python.exe pour que leur sortie soit capturable."""
    return sys.executable.replace("pythonw.exe", "python.exe")


def _etat() -> dict:
    try:
        return json.loads(ETAT.read_text(encoding="utf-8"))
    except Exception:
        return {}


def derive_precision() -> tuple[bool, list[str]]:
    """Nos prédictions s'écartent-elles de leur plafond théorique ?"""
    import numpy as np
    import pandas as pd
    from sqlalchemy import create_engine
    from scraper.config import load_settings

    msgs, derive = [], False
    try:
        d = pd.read_sql("""SELECT hit1_cal, hit3, hitx FROM trio_predictions
                           WHERE actual IS NOT NULL AND actual!='VOID'
                           ORDER BY id DESC LIMIT 300""",
                        create_engine(load_settings().db_url))
    except Exception as exc:
        return False, [f"  precision : lecture impossible ({exc})"]
    if len(d) < N_MIN:
        return False, [f"  precision : {len(d)} predictions suivies (<{N_MIN}) -> non concluant"]
    for nom, col, plafond in PLAFONDS:
        obs = float(d[col].mean())
        z = (obs - plafond) / np.sqrt(plafond * (1 - plafond) / len(d))
        flag = abs(z) > Z_MAX
        derive |= flag
        msgs.append(f"  precision {nom:<6} reel {100*obs:5.1f}% vs plafond {100*plafond:4.1f}%"
                    f"  z={z:+5.1f}{'  <<< DERIVE' if flag else ''}")
    return derive, msgs


def derive_pricing() -> tuple[bool, list[str]]:
    """Le book a-t-il changé sa structure de marge ?"""
    try:
        r = subprocess.run([_python(), str(ROOT / "scripts" / "cross_market_check.py"),
                            "--limit", "20000"],
                           capture_output=True, text=True, timeout=1800, cwd=ROOT)
    except Exception as exc:
        return False, [f"  pricing : controle impossible ({exc})"]
    lignes = [l for l in r.stdout.splitlines() if "DERIVE" in l]
    if r.returncode == 0:
        return False, ["  pricing   : marges dans leur bande habituelle"]
    return True, ["  pricing   : ANOMALIE"] + [f"    {l.strip()}" for l in lignes[:6]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="diagnostiquer sans recalibrer")
    ap.add_argument("--force", action="store_true", help="recalibrer quoi qu'il arrive")
    ap.add_argument("--cooldown-h", type=float, default=24.0,
                    help="delai de garde entre deux recalibrations")
    args = ap.parse_args()

    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] controle de derive")
    d1, m1 = derive_precision()
    d2, m2 = derive_pricing()
    for l in m1 + m2:
        print(l)

    if not (d1 or d2 or args.force):
        print("-> aucune derive : rien a faire")
        return 0

    etat = _etat()
    last = etat.get("derniere_recalibration")
    if last and not args.force:
        try:
            age = datetime.now(timezone.utc) - datetime.fromisoformat(last)
            if age < timedelta(hours=args.cooldown_h):
                reste = args.cooldown_h - age.total_seconds() / 3600
                print(f"-> derive detectee mais delai de garde actif ({reste:.1f}h restantes)")
                return 0
        except Exception:
            pass

    raison = [n for n, f in (("precision", d1), ("pricing", d2), ("force", args.force)) if f]
    if args.dry_run:
        print(f"-> RECALIBRERAIT (raison : {', '.join(raison)}) — mode dry-run")
        return 0

    print(f"-> recalibration ({', '.join(raison)})…")
    r = subprocess.run([_python(), str(ROOT / "scripts" / "refresh_calibration.py")],
                       capture_output=True, text=True, timeout=3600, cwd=ROOT)
    print(r.stdout.rstrip() or r.stderr[-2000:])
    if r.returncode != 0:
        print("-> ECHEC de la recalibration")
        return 1

    # la copie embarquee sert l'app en ligne : la laisser perimee la ferait
    # calibrer avec une table morte (bug deja survenu sur streamlit_app.py).
    src = ROOT / "data" / "vfoot_ml" / "score_calibration.json"
    dst = ROOT / "config" / "score_calibration.json"
    try:
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"-> copie embarquee synchronisee : {dst}")
    except Exception as exc:
        print(f"-> AVERTISSEMENT : copie embarquee non synchronisee ({exc})")

    ETAT.parent.mkdir(parents=True, exist_ok=True)
    etat.update({"derniere_recalibration": datetime.now(timezone.utc).isoformat(),
                 "raison": raison})
    ETAT.write_text(json.dumps(etat, indent=1), encoding="utf-8")
    print("-> recalibration terminee")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
