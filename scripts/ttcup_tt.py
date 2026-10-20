"""Pronostics tennis de table TT Cup / Czech Liga Pro — données OFFICIELLES ttcup.com.

Percée 20/10 : les agrégateurs (scores24, Sofascore, aiscore, Flashscore) sont tous
protégés par Cloudflare et bloquent l'IP datacenter du Space (403). Le SITE OFFICIEL
ttcup.com, lui, est un simple serveur Django SANS Cloudflare : il répond en HTTP 200
depuis un datacenter. On peut donc prédire ces ligues EN LIGNE, gratuitement, sans
proxy ni API payante.

Source : https://ttcup.com/tournaments/ajax/tournament/home/hotblock/ (en-tête
X-Requested-With). Un appel = snapshot de tous les tournois actifs du jour, avec pour
chacun une grille round-robin : chaque cellule « a:b » = manches gagnées (joueur de la
ligne : joueur de la colonne). « 0:0 » = match programmé non joué (à prédire). La
diagonale est vide.

Comme un seul snapshot ne contient que les matchs DÉJÀ joués du jour (peu le matin, la
grille se remplit dans la journée), un collecteur (_ttcup_loop.py) poll régulièrement
et ACCUMULE l'historique dans /data ; le modèle (forme + Elo, réutilisé de sport_reel)
s'entraîne dessus. Honnêteté : ces ligues rapides sont très aléatoires — la fiabilité
réelle se mesure par backtest (taux_fiabilite) et n'est affichée que telle quelle.
"""
from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timezone

import sport_reel as _sr

_H = {"User-Agent": "Mozilla/5.0", "X-Requested-With": "XMLHttpRequest"}
HOTBLOCK_URL = "https://ttcup.com/tournaments/ajax/tournament/home/hotblock/"


def _get(url: str, timeout: float = 25.0):
    req = urllib.request.Request(url, headers=_H)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch_hotblock(timeout: float = 25.0) -> list:
    """Snapshot brut des tournois actifs (liste de dicts). [] si injoignable."""
    try:
        d = _get(HOTBLOCK_URL, timeout)
        return d if isinstance(d, list) else []
    except Exception:
        return []


def _iso(periode: str) -> str:
    """'20.10.2026' -> '2026-10-20' (pour trier). Rend la chaîne telle quelle si
    le format diffère."""
    try:
        return datetime.strptime(periode, "%d.%m.%Y").strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return periode or ""


def _sets(valeur: str):
    """'2:3' -> (2, 3) ; None si illisible ou non joué ('0:0' / '' géré ailleurs)."""
    try:
        a, b = valeur.split(":")
        return int(a), int(b)
    except (ValueError, AttributeError):
        return None


def parse_snapshot(data: list) -> list:
    """Déplie les grilles round-robin en matchs {id, date, home, away, hs, as_,
    done, hall}. Chaque paire n'est comptée qu'UNE fois (triangle supérieur par
    ordre). done=True si joué (ex. '2:3'), done=False si programmé ('0:0')."""
    out = []
    for t in data or []:
        pk = t.get("pk")
        hall = t.get("hall_hint") or t.get("auto_hall_hint")
        date = _iso(t.get("period") or t.get("date_first_day") or "")
        rows = (t.get("table") or {}).get("tournament_table") or []
        # ordre (1..N) -> nom du joueur
        noms = {}
        for r in rows:
            p = r.get("player") or {}
            if p.get("order") and p.get("full_name"):
                noms[p["order"]] = p["full_name"]
        for r in rows:
            p = r.get("player") or {}
            oi = p.get("order")
            if not oi:
                continue
            cells = r.get("cells") or []
            for idx, c in enumerate(cells):
                oj = idx + 1                      # la colonne idx = joueur d'ordre idx+1
                if oj <= oi:                      # triangle supérieur : une paire = un match
                    continue
                if not isinstance(c, dict) or c.get("is_empty"):
                    continue
                val = c.get("value")
                home, away = noms.get(oi), noms.get(oj)
                if not home or not away:
                    continue
                mid = f"{pk}:{oi}:{oj}:{date}"
                if val == "0:0":
                    out.append({"id": mid, "date": date, "home": home, "away": away,
                                "hs": None, "as_": None, "done": False, "hall": hall})
                else:
                    s = _sets(val)
                    if s is None:
                        continue
                    out.append({"id": mid, "date": date, "home": home, "away": away,
                                "hs": s[0], "as_": s[1], "done": True, "hall": hall})
    return out


def fusionner(ancienne: list, nouveaux: list) -> list:
    """Fusionne l'historique accumulé et un nouveau snapshot, dédupliqué par id.
    Un match qui passe de programmé à joué est mis à jour (la version jouée gagne)."""
    par_id = {}
    for m in ancienne or []:
        par_id[m["id"]] = m
    for m in nouveaux or []:
        anc = par_id.get(m["id"])
        if anc is None or (m.get("done") and not anc.get("done")):
            par_id[m["id"]] = m
    return sorted(par_id.values(), key=lambda m: (m.get("date") or "", m.get("id")))


def matchs_joues(matchs: list) -> list:
    return [m for m in matchs if m.get("done") and m.get("hs") is not None]


def matchs_a_venir(matchs: list) -> list:
    return [m for m in matchs if not m.get("done") and m.get("home") and m.get("away")]


def pronostics(historique: list, fixtures: list) -> list:
    """Pronostics forme + face-à-face + Elo (moteur sport_reel) pour les matchs à
    venir. Tennis de table : pas de nul, terrain neutre (home_adv=0)."""
    return _sr.pronostics_a_venir(historique, fixtures, nuls=False, home_adv=0,
                                  kind="team")


def taux_fiabilite(historique: list) -> dict | None:
    """Backtest HONNÊTE (walk-forward) sur l'historique accumulé. None tant qu'il
    n'y a pas assez de matchs — on ne montre pas un taux calculé sur 3 matchs."""
    return _sr.taux_reussite(historique, nuls=False, home_adv=0)


def analyse(p: dict) -> str:
    return _sr.analyse_match(p)


def charger_disque(path: str) -> list:
    """Historique accumulé depuis un fichier JSON. [] si absent ou illisible."""
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, list) else []
    except (FileNotFoundError, ValueError, OSError):
        return []


def sauver_disque(path: str, matchs: list) -> None:
    """Écrit l'historique (écriture atomique via un fichier temporaire)."""
    import os
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(matchs, f, ensure_ascii=False)
    os.replace(tmp, path)


def collecter_once(path: str) -> tuple:
    """Un cycle : fetch hotblock → parse → fusionne avec le disque → sauve.
    Rend (total_en_base, nouveaux_joues_ce_cycle)."""
    snap = parse_snapshot(fetch_hotblock())
    ancienne = charger_disque(path)
    deja_joues = {m["id"] for m in ancienne if m.get("done")}
    fus = fusionner(ancienne, snap)
    sauver_disque(path, fus)
    nouveaux = sum(1 for m in fus if m.get("done") and m["id"] not in deja_joues)
    return len(fus), nouveaux


def charger_live(historique_accumulee: list | None = None) -> dict:
    """Pipeline complet : fetch hotblock → parse → fusionne avec l'historique
    fournie → rend {hist, fixtures, pronos, taux, halls}. Tout en ligne."""
    snap = parse_snapshot(fetch_hotblock())
    tout = fusionner(historique_accumulee or [], snap)
    hist = matchs_joues(tout)
    fx = matchs_a_venir(snap)                      # on ne prédit que les matchs du jour
    pronos = pronostics(hist, fx) if fx else []
    return {"tout": tout, "hist": hist, "fixtures": fx, "pronos": pronos,
            "taux": taux_fiabilite(hist), "halls": sorted({m.get("hall") for m in snap if m.get("hall")})}
