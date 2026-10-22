"""Elo MULTI-SAISONS du football, calculé depuis openfootball (2e référence).

Mon Elo « maison » (sport_reel) n'apprend que sur ~220 jours (une saison ESPN).
openfootball (github.com/openfootball/football.json, hébergé sur GitHub donc
joignable depuis le datacenter du Space) fournit les résultats de PLUSIEURS
saisons. On en calcule un Elo à MÉMOIRE LONGUE (report entre saisons), qu'on
mappe sur les noms d'équipes ESPN et qu'on mélange au pronostic.

Mesuré le 22/10/2026 (backtest walk-forward, matchs décisifs des 5 ligues) :
l'Elo multi-saisons bat l'Elo « saison » de +1 à +6 pts ; combiné à la forme,
gain net sur 4 ligues sur 5 (≈ +1,5 en moyenne). Référence INDÉPENDANTE de mon
Elo ESPN, donc vraiment additive. ClubElo (l'idéal) ayant fermé son accès
gratuit, openfootball est la seule source gratuite fiable (voir mémoire
sources-foot-force).
"""
from __future__ import annotations

import json
import re
import unicodedata
import urllib.request

_H = {"User-Agent": "Mozilla/5.0"}
_BASE = "https://raw.githubusercontent.com/openfootball/football.json/master"

# chemin ESPN -> code openfootball (5 ligues domestiques ; pas la Ligue des
# Champions, non couverte par openfootball).
LIGUES = {
    "soccer/eng.1": "en.1", "soccer/esp.1": "es.1", "soccer/ita.1": "it.1",
    "soccer/ger.1": "de.1", "soccer/fra.1": "fr.1",
}
# saisons prises en compte (report Elo de l'une à l'autre). De la plus ancienne
# à la plus récente ; la saison en cours est incluse si disponible.
SAISONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26", "2026-27"]

# mots à retirer pour rapprocher « Arsenal FC » (openfootball) de « Arsenal »
# (ESPN). Mappage vérifié 20/20 sur l'EPL le 22/10.
_STOP = {"fc", "cf", "afc", "sc", "ac", "as", "ssc", "rc", "cd", "ud", "fsv",
         "tsg", "vfl", "vfb", "bsc", "calcio", "club", "de", "the", "us", "ca",
         "aj", "ogc", "rcd", "sd", "sv", "bv", "if", "1"}


def _norm(nom: str) -> str:
    n = unicodedata.normalize("NFKD", str(nom or ""))
    n = "".join(c for c in n if not unicodedata.combining(c)).lower()
    toks = [t for t in re.split(r"[^a-z0-9]+", n) if t and t not in _STOP]
    return " ".join(toks)


def _gj(url: str, timeout: float = 20.0, essais: int = 2):
    for _ in range(essais):
        try:
            req = urllib.request.Request(url, headers=_H)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read().decode("utf-8", "replace"))
            if isinstance(d, dict) and isinstance(d.get("matches"), list):
                return d
        except Exception:
            continue
    return None


def _att(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def elo_ligue(of_code: str, k: float = 20.0, home_adv: float = 60.0,
              timeout: float = 20.0) -> dict:
    """Elo multi-saisons {nom_normalisé: note} d'une ligue openfootball. Report
    entre saisons : nouvelle note = 0,75·ancienne + 0,25·1500 (régression vers la
    moyenne). {} si rien n'est joignable (le modèle retombe alors sur l'ESPN)."""
    r: dict = {}
    vu = False
    for s in SAISONS:
        d = _gj(f"{_BASE}/{s}/{of_code}.json", timeout)
        if not d:
            continue
        if vu:                                   # report inter-saisons
            r = {t: 0.75 * v + 0.25 * 1500 for t, v in r.items()}
        vu = True
        for m in d["matches"]:
            if not isinstance(m, dict):
                continue
            sc = m.get("score")
            ft = sc.get("ft") if isinstance(sc, dict) else None
            h, a = _norm(m.get("team1")), _norm(m.get("team2"))
            if not isinstance(ft, (list, tuple)) or len(ft) < 2 or not h or not a:
                continue
            try:
                hs, as_ = int(ft[0]), int(ft[1])
            except (TypeError, ValueError, IndexError):
                continue
            rh, ra = r.get(h, 1500.0), r.get(a, 1500.0)
            e = _att(rh + home_adv, ra)
            res = 1.0 if hs > as_ else (0.0 if hs < as_ else 0.5)
            r[h] = rh + k * (res - e)
            r[a] = ra - k * (res - e)
    return r if vu else {}


def ratings_pour(path: str, noms_espn, timeout: float = 20.0) -> dict:
    """{nom ESPN: note Elo multi-saisons} pour les équipes `noms_espn` d'une
    ligue (chemin ESPN). {} si la ligue n'est pas couverte ou source injoignable."""
    of_code = LIGUES.get(path)
    if not of_code:
        return {}
    r = elo_ligue(of_code, timeout=timeout)
    if not r:
        return {}
    out = {}
    for nom in noms_espn:
        v = r.get(_norm(nom))
        if v is not None:
            out[nom] = v
    return out
