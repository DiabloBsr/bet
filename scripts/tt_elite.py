"""Pronostics TT Elite Series — données OFFICIELLES tt-series.com.

Le « TT Elite Series » des bookmakers = la division Elite du circuit International
TT Series (ELITE Series Sp. z o.o., Pologne). Site officiel **tt-series.com** :
WordPress + LiteSpeed, **SANS Cloudflare**, donc joignable depuis l'IP datacenter du
Space (contrairement à scores24/Sofascore qui renvoient 403). API JSON d'un plugin
maison (namespace `tt/v1`) :

- `GET /wp-json/tt/v1/zawodnicy` → ~480 joueurs {id, nazwa (« Nom Prénom »), elo
  (classement officiel), poziom (division)}.
- `GET /wp-json/tt/v1/mecze-excel?players=ID1,ID2,…` (param `players` OBLIGATOIRE) →
  matchs TERMINÉS {id, player1_id, player2_id, match_date, winner_id}. Historique
  complet 2022 → aujourd'hui. On interroge par paquets d'ids (≤ 50) puis on
  déduplique par id de match.

Le JSON ne donne QUE le vainqueur (pas le score en manches) et PAS les matchs à
venir → on en fait un **prédicteur de duel** : l'utilisateur choisit deux joueurs,
on prédit avec le moteur forme + face-à-face + Elo de sport_reel (comme le tennis :
victoire/défaite, terrain neutre). Entraîné sur ~20 000 matchs réels → fiabilité
MESURABLE. Backtest walk-forward (20/10) : 56 % tous picks, ~61 % à conf ≥ 60 %,
~65 % à conf ≥ 65 %, ~69 % à conf ≥ 70 % (base pile ou face 50 %).
"""
from __future__ import annotations

import json
import urllib.request

import sport_reel as _sr

_H = {"User-Agent": "Mozilla/5.0"}
BASE = "https://www.tt-series.com/wp-json/tt/v1"
_LOT = 50          # ids par requête (au-delà de ~200 l'endpoint ne rend plus une liste)


def _get(url: str, timeout: float = 40.0):
    req = urllib.request.Request(url, headers=_H)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def joueurs(timeout: float = 40.0) -> dict:
    """{id (str): {'nom', 'elo' (float|None), 'div'}}. {} si injoignable."""
    out: dict = {}
    try:
        for j in _get(f"{BASE}/zawodnicy", timeout) or []:
            pid = str(j.get("id"))
            nom = j.get("nazwa")
            if not pid or not nom:
                continue
            try:
                elo = float(j.get("elo"))
            except (TypeError, ValueError):
                elo = None
            out[pid] = {"nom": nom, "elo": elo, "div": j.get("poziom")}
    except Exception:
        return {}
    return out


def _matchs_bruts(ids: list, timeout: float = 40.0) -> dict:
    """{id_match: brut} pour tous les ids (par paquets de _LOT), dédupliqué."""
    vus: dict = {}
    for i in range(0, len(ids), _LOT):
        lot = ids[i:i + _LOT]
        try:
            for m in _get(f"{BASE}/mecze-excel?players=" + ",".join(lot), timeout) or []:
                if m.get("id"):
                    vus[m["id"]] = m
        except Exception:
            continue          # un paquet qui échoue ne casse pas la collecte
    return vus


def historique(js: dict | None = None, timeout: float = 40.0) -> list:
    """Tous les matchs TERMINÉS au format sport_reel {id, date, home, away, hs,
    as_, done} (hs/as_ = 1/0 selon le vainqueur, pas de score en manches), triés
    par date. `js` = dict joueurs (évite un 2e fetch) ; sinon récupéré ici."""
    noms = js or joueurs(timeout)
    if not noms:
        return []
    bruts = _matchs_bruts(list(noms), timeout)
    out = []
    for m in bruts.values():
        a, b, w = str(m.get("player1_id")), str(m.get("player2_id")), str(m.get("winner_id"))
        if a not in noms or b not in noms or w not in (a, b):
            continue
        out.append({"id": m["id"], "date": (m.get("match_date") or "")[:10],
                    "home": noms[a]["nom"], "away": noms[b]["nom"],
                    "hs": 1 if w == a else 0, "as_": 0 if w == a else 1, "done": True})
    out.sort(key=lambda x: x["date"])
    return out


def pronostic(hist: list, nom_a: str, nom_b: str) -> dict:
    """Pronostic d'un duel (forme + face-à-face + Elo), terrain neutre, sans nul."""
    elo = _sr.Elo(home_adv=0).entrainer(hist)
    return _sr.pronostic_forme(hist, nom_a, nom_b, nuls=False, home_adv=0, elo=elo)


def taux_fiabilite(hist: list, maxi: int = 1500, warmup: int = 200) -> dict | None:
    """Backtest walk-forward honnête (forme + Elo) sur l'historique."""
    return _sr.taux_reussite(hist, nuls=False, home_adv=0, maxi=maxi, warmup=warmup)


def analyse(p: dict) -> str:
    return _sr.analyse_match(p)
