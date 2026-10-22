"""Déploie l'app sur le Space Hugging Face, puis VÉRIFIE qu'elle tient debout.

Remplace la douzaine d'appels API tapés à la main : un seul commit, puis un contrôle
de stabilité (plusieurs requêtes espacées — un seul 200 ne prouve rien, l'ancien
conteneur peut encore répondre pendant le swap).

Usage:
    export HF_TOKEN=hf_xxx          # ou HF_TOKEN=... dans le .env a la racine
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

# Le token vit dans le .env (non versionne) comme le reste de la conf. Sans ca,
# il faudrait le reexporter a la main dans chaque shell. Une variable deja
# presente dans l'environnement reste prioritaire (cas CI).
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:  # dotenv absent : on retombe sur l'environnement seul
    pass

SPACE = "olivioBsr/vfoot-trio"
URL = "https://olivioBsr-vfoot-trio.hf.space/"
DEFAULT_FILES = [
    "scripts/predict_trio.py",
    "scripts/dashboard_trio.py",
    "scripts/trap_detector.py",
    "scripts/market_ranges.py",
    "scripts/lire_cotes.py",
    # Importe par dashboard_trio : absent, l'app en ligne meurt a l'import.
    "scripts/forme_pastilles.py",
    # Prediction sport reel par Elo (donnees ESPN), importe par dashboard (20/10).
    "scripts/sport_reel.py",
    # 2e reference foot : Elo multi-saisons depuis openfootball (GitHub, joignable
    # datacenter), melange au pronostic. Importe par dashboard (22/10).
    "scripts/elo_multi.py",
    # 3e signal foot : xG-forme (Understat) EMBARQUEE (Understat bloque le Space).
    # Regeneree via le navigateur, voir tools/maj_xg.py. 22/10.
    "scripts/xg_form.py",
    "scripts/data/xg_form.json",
    # TT Cup / Czech Liga Pro (donnees OFFICIELLES ttcup.com, datacenter-OK, 20/10).
    # Le collecteur _ttcup_loop est lance par start_cloud ; il accumule /data.
    "scripts/ttcup_tt.py",
    "scripts/_ttcup_loop.py",
    # TT Elite Series (donnees OFFICIELLES tt-series.com). Le site renvoie 403 a
    # l'IP du Space (pare-feu WordPress) -> on EMBARQUE les donnees (joueurs +
    # historique), generees en local par tools/maj_tt_elite.py, et le Space lit
    # ce fichier sans appeler tt-series.com. 20/10.
    "scripts/tt_elite.py",
    "scripts/data/tt_elite.json",
    # NB: onglets Courses (courses_bet261/courses_collect/_courses_loop) et Tennis
    # de table (wtt_tt.py) RETIRES de l'app le 20/10 (demande Olivio) : scripts
    # gardes en local mais plus deployes ni lances par start_cloud.
    # scores24_tt.py (TT Elite Series) existe en local mais N'EST PAS deploye :
    # scores24 bloque l'IP du Space (403). Gardé pour un eventuel contournement.
    # Deja sur le Space, identique au depot (controle le 04/10), mais hors
    # liste : une retouche du theme ne serait jamais partie.
    "scripts/ui_theme.py",
    # L OCR de l onglet a cette cote a besoin du binaire tesseract : sans le
    # Dockerfile et les deps, le Space garde son image et la lecture est morte.
    "Dockerfile",
    "requirements.txt",
    "deploy/start_cloud.sh",
    # Les corrections par ligue. C'est bien config/ qu'il faut pousser : start_cloud.sh
    # le recopie au boot vers data/vfoot_ml/ (deployer data/ serait ecrase). Sans ce
    # fichier, l'app en ligne afficherait des probas fausses hors anglaise.
    "config/score_calibration.json",
    "config/over25_calibration.json",
    "config/totals_calibration.json",
    "config/ou25_calibration.json",
    "config/marches_calibration.json",
    # Sans ce fichier, l'onglet « Que jouer ? » afficherait les mi-temps en
    # BRUT : 46,9 % annonces pour 42,1 % touches sur le 1X2 de 2e periode.
    "config/mitemps_calibration.json",
    "config/minute_table.json",
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
        print("HF_TOKEN absent : le renseigner dans .env ou l'exporter."); return 1

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
