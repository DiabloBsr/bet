"""Déploie l'app sur le Space Hugging Face, puis VÉRIFIE qu'elle tient debout.

Remplace la douzaine d'appels API tapés à la main : un seul commit, puis un contrôle
de stabilité (plusieurs requêtes espacées — un seul 200 ne prouve rien, l'ancien
conteneur peut encore répondre pendant le swap).

Usage:
    export HF_TOKEN=hf_xxx
    python deploy/deploy_hf.py                      # fichiers app par défaut
    python deploy/deploy_hf.py scripts/foo.py ...   # fichiers explicites
    python deploy/deploy_hf.py --no-verify          # déploie sans attendre

Sortie: exit 0 si l'app répond de façon stable, 1 sinon.
"""
from __future__ import annotations

import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPACE = "olivioBsr/vfoot-trio"
URL = "https://olivioBsr-vfoot-trio.hf.space/"
DEFAULT_FILES = [
    "scripts/predict_trio.py",
    "scripts/dashboard_trio.py",
    "scripts/trap_detector.py",
    "scripts/market_ranges.py",
    "deploy/start_cloud.sh",
    # Les corrections par ligue. C'est bien config/ qu'il faut pousser : start_cloud.sh
    # le recopie au boot vers data/vfoot_ml/ (deployer data/ serait ecrase). Sans ce
    # fichier, l'app en ligne afficherait des probas fausses hors anglaise.
    "config/score_calibration.json",
]
BUILD_WAIT = 240      # laisser le temps au rebuild Docker + boot
PROBES = 5            # nb de requêtes de contrôle
PROBE_GAP = 9         # secondes entre deux requêtes


def _probe() -> int | str:
    try:
        req = urllib.request.Request(URL, headers={"User-Agent": "deploy-check"})
        return urllib.request.urlopen(req, timeout=30).getcode()
    except Exception as exc:  # noqa: BLE001 - on veut le code HTTP ou le type d'erreur
        return getattr(exc, "code", type(exc).__name__)


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    verify = "--no-verify" not in sys.argv
    files = args or DEFAULT_FILES

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN absent de l'environnement."); return 1

    missing = [f for f in files if not (ROOT / f).exists()]
    if missing:
        print(f"fichiers introuvables : {missing}"); return 1

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=token)
    ops = [CommitOperationAdd(path_in_repo=f, path_or_fileobj=str(ROOT / f)) for f in files]
    info = api.create_commit(
        SPACE, repo_type="space", operations=ops,
        commit_message=f"deploy: {', '.join(Path(f).name for f in files)}")
    print(f"commit HF : {getattr(info, 'commit_url', info)}")
    for f in files:
        print(f"  + {f}")

    if not verify:
        return 0

    print(f"\nattente du rebuild ({BUILD_WAIT}s)…")
    time.sleep(BUILD_WAIT)
    stage = api.get_space_runtime(SPACE).stage
    codes = []
    for i in range(PROBES):
        codes.append(_probe())
        if i < PROBES - 1:
            time.sleep(PROBE_GAP)
    ok = sum(1 for c in codes if c == 200)
    print(f"stage={stage} | codes={codes} -> {ok}/{PROBES} en 200")
    if ok >= PROBES - 1:
        print("DEPLOIEMENT OK (app stable)")
        return 0
    print("APP INSTABLE — vérifier les logs du Space avant de continuer")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
