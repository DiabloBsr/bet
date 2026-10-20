"""Régénère le bundle de données TT Elite (scripts/data/tt_elite.json).

tt-series.com renvoie 403 à l'IP datacenter du Space : on NE peut PAS récupérer
les données en ligne. On les télécharge depuis une IP NON bloquée (poste local,
réseau résidentiel) et on les embarque dans l'app, que le Space lit sans appeler
tt-series.com.

À lancer en local quand on veut rafraîchir l'historique, PUIS redéployer :
    python tools/maj_tt_elite.py
    python deploy/deploy_hf.py
"""
import datetime
import json
import os
import sys

_ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ICI, "..", "scripts"))

import tt_elite as te  # noqa: E402

_OUT = os.path.join(_ICI, "..", "scripts", "data", "tt_elite.json")


def main() -> None:
    print("Téléchargement depuis tt-series.com (IP non bloquée requise)…", flush=True)
    js = te.joueurs()
    if not js:
        print("ÉCHEC : tt-series.com injoignable —", te.diagnostic())
        sys.exit(1)
    hist = te.historique(js)
    os.makedirs(os.path.dirname(_OUT), exist_ok=True)
    bundle = {"genere_le": datetime.date.today().isoformat(),
              "joueurs": js, "matchs": hist}
    with open(_OUT, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, separators=(",", ":"))
    mo = os.path.getsize(_OUT) / 1e6
    print(f"OK : {len(js)} joueurs, {len(hist)} matchs -> {_OUT} ({mo:.1f} Mo)")
    print("Pense à redéployer : python deploy/deploy_hf.py")


if __name__ == "__main__":
    main()
