"""xG-forme du football (3e signal) — données Understat EMBARQUÉES.

La FORME basée sur les xG (buts attendus) est plus prédictive que la forme sur
résultats : elle capte comment une équipe a VRAIMENT joué, en filtrant la chance.
Mesuré le 22/10/2026 (backtest walk-forward, 5 ligues, données Understat) :
remplacer la forme-résultats par la xG-forme gagne de +0,5 à +4,7 pts (≈ +2 en
moyenne). Vrai signal supplémentaire, le seul mesuré positif après Poisson et ML.

Understat ne sert ses données qu'à un vrai navigateur (requêtes brutes bloquées,
et l'IP du Space l'est aussi) → on EMBARQUE la xG-forme (ratio xG/(xG+xGA) des 6
derniers matchs par équipe), récupérée via le navigateur intégré puis livrée.
Régénération : tools/maj_xg.py (voir sa notice). Mappage des noms Understat↔ESPN
par normalisation (même schéma qu'elo_multi).
"""
from __future__ import annotations

import json
import os
import re
import unicodedata

_BUNDLE = os.path.join(os.path.dirname(__file__), "data", "xg_form.json")

# couverture : les 5 grandes ligues domestiques (pas la Ligue des Champions).
LIGUES = {"soccer/eng.1", "soccer/esp.1", "soccer/ita.1", "soccer/ger.1", "soccer/fra.1"}

_STOP = {"fc", "cf", "afc", "sc", "ac", "as", "ssc", "rc", "cd", "ud", "fsv",
         "tsg", "vfl", "vfb", "bsc", "calcio", "club", "de", "the", "us", "ca",
         "aj", "ogc", "rcd", "sd", "sv", "bv", "if", "1", "1913", "05", "04"}


def _norm(nom: str) -> str:
    n = unicodedata.normalize("NFKD", str(nom or ""))
    n = "".join(c for c in n if not unicodedata.combining(c)).lower()
    toks = [t for t in re.split(r"[^a-z0-9]+", n) if t and t not in _STOP]
    return " ".join(toks)


def _bundle(path_fichier: str | None = None) -> dict:
    try:
        with open(path_fichier or _BUNDLE, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (FileNotFoundError, ValueError, OSError):
        return {}


def date_donnees(path_fichier: str | None = None) -> str | None:
    return _bundle(path_fichier).get("genere_le")


def _apparier(tokens_espn: frozenset, cands: list):
    """Rapproche un nom ESPN (ensemble de tokens) d'un nom Understat. Exact
    d'abord ; sinon sous-ensemble UNIQUE dans un sens ou l'autre (« Brighton » ⊂
    « Brighton Hove Albion », « Leeds » ⊂ « Leeds United »). Ambiguïté → None (on
    évite ainsi de confondre « Manchester City » et « Manchester United »)."""
    for toks, v in cands:
        if toks == tokens_espn:
            return v
    subs = [v for toks, v in cands
            if toks and (toks <= tokens_espn or tokens_espn <= toks)]
    return subs[0] if len(subs) == 1 else None


def xform_pour(path: str, noms_espn, path_fichier: str | None = None) -> dict:
    """{nom ESPN: xG-forme} pour les équipes `noms_espn` d'une ligue. {} si la
    ligue n'est pas couverte ou le bundle est absent."""
    if path not in LIGUES:
        return {}
    table = (_bundle(path_fichier).get("xform") or {}).get(path) or {}
    if not table:
        return {}
    cands = [(frozenset(_norm(k).split()), v) for k, v in table.items()]
    out = {}
    for nom in noms_espn:
        v = _apparier(frozenset(_norm(nom).split()), cands)
        if v is not None:
            out[nom] = v
    return out
