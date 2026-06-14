"""Utilitaires partages pour les scripts d'analyse (eviter de redupliquer les bugs)."""
from __future__ import annotations

import json
from pathlib import Path

# Racine du repo = parent du dossier scraper/
_REPO_ROOT = Path(__file__).resolve().parent.parent
_CORRUPTED_PATH = _REPO_ROOT / "exports" / "corrupted_events.json"


def load_corrupted_ids(path: str | Path | None = None) -> set[int]:
    """Renvoie l'ensemble des event_id corrompus a EXCLURE de toute analyse.

    Le fichier exports/corrupted_events.json est un DICT :
        {"generated_at":..., "n_corrupted":473, "events": {"50":[...], "187":[...], ...}}
    Les 473 ids vivent sous la cle "events" (cles = strings).

    Bug historique (NE PAS reproduire) : `set(json.load(open(...)))` renvoie les cles
    de premier niveau ({"generated_at","events",...}), et comme df.id sont des entiers
    le filtre `~df.id.isin(corrupted)` n'excluait RIEN.
    """
    p = Path(path) if path is not None else _CORRUPTED_PATH
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    events = data.get("events", {}) if isinstance(data, dict) else {}
    out: set[int] = set()
    for k in events:
        try:
            out.add(int(k))
        except (ValueError, TypeError):
            continue
    return out
