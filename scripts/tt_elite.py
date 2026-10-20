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
import re
import unicodedata
import urllib.request

import sport_reel as _sr

# En-têtes « navigateur » complets : certains pare-feux WordPress (et l'IP
# datacenter du Space) rejettent un User-Agent générique. On imite un vrai Chrome.
_H = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
    "Referer": "https://www.tt-series.com/",
}
BASE = "https://www.tt-series.com/wp-json/tt/v1"
_LOT = 50          # ids par requête (au-delà de ~200 l'endpoint ne rend plus une liste)


def _get(url: str, timeout: float = 40.0):
    req = urllib.request.Request(url, headers=_H)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def diagnostic(timeout: float = 20.0) -> str:
    """Teste l'accès à l'API et rend un message lisible (statut HTTP ou erreur) —
    pour afficher POURQUOI le chargement a échoué au lieu d'un silencieux « 0 »."""
    import urllib.error
    try:
        d = _get(f"{BASE}/zawodnicy", timeout)
        return f"OK ({len(d) if isinstance(d, list) else '?'} joueurs)"
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code} {e.reason}"
    except Exception as e:                       # réseau, TLS, timeout…
        return f"{type(e).__name__}: {e}"


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


# ---------------------------------------------------------------------------
# Rapprochement de noms + pronostics d'une LISTE de matchs collée (Bet261).
# Bet261 affiche « Surname, First » (virgule, nom en premier) ; tt-series stocke
# « Surname First ». On normalise (minuscule, sans accent/ponctuation) et on
# rapproche par nom de famille + prénom (ou son initiale « P. »).
# ---------------------------------------------------------------------------
def _norm(s: str) -> list:
    s = unicodedata.normalize("NFKD", str(s or ""))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return [t for t in re.split(r"[^a-z0-9]+", s) if t]


def _index_noms(js: dict) -> dict:
    """{nom_canonique: [tokens normalisés]} pour tous les joueurs."""
    return {v["nom"]: _norm(v["nom"]) for v in js.values()}


def trouver_joueur(index: dict, texte: str) -> str | None:
    """Meilleur nom canonique correspondant à `texte` (ex. 'Glanowski, Waldemar'
    ou 'Kurek, P.'). None si aucune correspondance fiable."""
    q = _norm(texte)
    if not q:
        return None
    qs = set(q)
    best, best_score = None, 0
    for nom, toks in index.items():
        ts = set(toks)
        if qs == ts:
            return nom                      # correspondance exacte
        commun = qs & ts
        if not commun:
            # tolère l'initiale du prénom : 'kurek p' vs 'kurek pawel'
            if toks and q and toks[0] == q[0] and any(
                    t.startswith(q[-1]) or q[-1].startswith(t) for t in toks[1:]):
                score = 1
            else:
                continue
        else:
            score = len(commun)
            # bonus si le nom de famille (1er token) coïncide
            if toks and q and toks[0] == q[0]:
                score += 1
        if score > best_score:
            best, best_score = nom, score
    return best if best_score >= 2 else None


def _deux_cotes(ligne: str):
    """Coupe une ligne de match en deux noms. Sépare sur ' - ' / ' – ' / ' vs '."""
    for sep in (" - ", " – ", " — ", " vs ", " VS ", " v "):
        if sep in ligne:
            a, b = ligne.split(sep, 1)
            return a.strip(), b.strip()
    return None


_RE_HEURE = re.compile(r"^\s*(\d{1,2})\s*[:hH]\s*(\d{2})\b[\s.\-–—]*")


def _extraire_heure(ligne: str):
    """Détache une heure en tête de ligne ('11:10 …' ou '11h10 …') : rend
    ('HH:MM' ou None, reste de la ligne)."""
    m = _RE_HEURE.match(ligne)
    if m:
        return f"{int(m.group(1)):02d}:{m.group(2)}", ligne[m.end():].strip()
    return None, ligne


def pronostics_liste(hist: list, js: dict, texte: str) -> tuple:
    """Pour un bloc de matchs collé (un par ligne, « [HH:MM] JoueurA - JoueurB »),
    rend (pronos, inconnus) : un pronostic par match reconnu (avec son heure si
    présente) + les lignes non reconnues. Entraîne l'Elo UNE fois."""
    index = _index_noms(js)
    elo = _sr.Elo(home_adv=0).entrainer(hist)
    pronos, inconnus = [], []
    for brute in (texte or "").splitlines():
        ligne = brute.strip()
        if not ligne:
            continue
        heure, reste = _extraire_heure(ligne)
        duo = _deux_cotes(reste)
        if not duo:
            inconnus.append(ligne)
            continue
        na, nb = trouver_joueur(index, duo[0]), trouver_joueur(index, duo[1])
        if not na or not nb or na == nb:
            inconnus.append(ligne)
            continue
        pr = _sr.pronostic_forme(hist, na, nb, nuls=False, home_adv=0, elo=elo)
        pr["time"] = heure
        pronos.append(pr)
    return pronos, inconnus
