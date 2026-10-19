"""Courses virtuelles Bet261 (Sporty-Tech) — PALIER 1, lecture des cotes en direct.

Deux disciplines : Dashing Derby (chevaux, catégorie 8006) et Platinum Hounds
(lévriers, 8005). Pour chaque course à venir : les partants numérotés et leur
cote « Vainqueur », d'où une probabilité DÉVIGÉE de victoire (marge du book
retirée).

⚠️ Les RÉSULTATS (ordre d'arrivée / gagnant) NE SONT PAS exposés publiquement
par l'API (le flux ne liste que les courses à venir ; voir la mémoire
`racing-api-bet261`). Donc PAS d'historique, de forme ni de backtest ici : comme
le football virtuel, c'est du RNG, on ne fait que lire proprement les cotes.
"""
from __future__ import annotations

import json
import urllib.request

# Catégories Bet261 (vérifiées 19/10/2026 sur bet261.mg/virtual/category/{id}).
CATEGORIES = {"Dashing Derby": 8006, "Platinum Hounds": 8005}  # chevaux / lévriers
DISCIPLINE = {8006: "🏇 Chevaux", 8005: "🐕 Lévriers"}

_BASE = "https://hg-event-api-prod.sporty-tech.net/api/events"
# L'API Sporty-Tech répond aux GET simples avec Origin/Referer bet261 (comme le
# scraper football). Pas d'authentification pour les cotes (seul le direct vidéo
# exige un compte). On reste sur urllib.request (stdlib) comme scraper/collector :
# httpx n'est PAS dans l'image du Space.
_HEADERS = {
    "Origin": "https://bet261.mg",
    "Referer": "https://bet261.mg/",
    "User-Agent": "Mozilla/5.0",
}


def _get_json(url: str, timeout: float):
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310 (https constant)
        return json.loads(resp.read().decode("utf-8"))


def _devig(runners: list) -> list:
    """Probabilité DÉVIGÉE de victoire : p_i = (1/cote_i) / Σ(1/cote_j).

    La somme des probabilités implicites (1/cote) dépasse 1 à cause de la marge
    du book ; on renormalise pour qu'elle fasse 1. Partants sans cote valable
    ignorés. Retour trié par probabilité décroissante (favori en tête)."""
    inv = [(r, 1.0 / r["odds"]) for r in runners if r.get("odds")]
    tot = sum(v for _, v in inv)
    out = [{**r, "p_impl": v, "p_devig": (v / tot) if tot else 0.0} for r, v in inv]
    out.sort(key=lambda x: x["p_devig"], reverse=True)
    return out


def parse_events(data) -> list:
    """JSON de `/events` → liste de courses normalisées, triées par heure.

    `data` est un dict index→event (ou une liste). Chaque course rendue :
    {id, name (piste), start (ISO UTC), runners:[{num, odds}]} à partir du
    marché « Vainqueur ». Les événements sans marché ou sans partant coté sont
    écartés."""
    races = list(data.values()) if isinstance(data, dict) else (data or [])
    out = []
    for e in races:
        if not isinstance(e, dict):
            continue
        bts = e.get("eventBetTypes") or []
        vainq = next((b for b in bts if b.get("name") == "Vainqueur"),
                     bts[0] if bts else None)
        if not vainq:
            continue
        runners = []
        for it in vainq.get("eventBetTypeItems") or []:
            if it.get("active") is False:
                continue
            try:
                num = int(it.get("shortName"))
            except (TypeError, ValueError):
                continue
            od = it.get("odds")
            if isinstance(od, (int, float)) and od > 0:
                runners.append({"num": num, "odds": float(od)})
        if not runners:
            continue
        out.append({"id": e.get("id"), "name": e.get("name"),
                    "start": e.get("expectedStart"), "end": e.get("expectedEnd"),
                    "runners": runners})
    out.sort(key=lambda r: r.get("start") or "")
    return out


def course_cotes(race: dict) -> dict:
    """Course enrichie : partants triés par proba dévigée + `favori` en tête."""
    rr = _devig(race.get("runners") or [])
    return {**race, "runners": rr, "favori": rr[0] if rr else None}


def fetch_courses(category_id: int, length: int = 21, timeout: float = 20.0) -> list:
    """Courses À VENIR d'une catégorie (8006 chevaux / 8005 lévriers), triées
    par heure, chacune passée par `course_cotes`. Lève en cas d'erreur réseau."""
    u = (f"{_BASE}?eventCategoryIds={int(category_id)}&offset=0"
         f"&length={int(length)}&fetchEventBetTypesMode=1&l=fr")
    return [course_cotes(c) for c in parse_events(_get_json(u, timeout))]
