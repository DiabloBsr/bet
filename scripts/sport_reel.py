"""Prédiction de sport RÉEL par Elo — données ESPN (gratuit, sans clé).

Le sport réel n'est PAS du RNG : il y a un vrai signal (force des équipes, forme,
avantage du terrain). Modèle **Elo** (notes d'équipes mises à jour match après
match, avantage terrain, nul au foot) → probabilités de victoire calibrées.

⚠️ Honnêteté : des probas correctes, proches des bookmakers, mais le modèle ne
BAT PAS le marché de façon rentable (les books sont efficaces). C'est fait pour
prédire et comprendre, pas pour gagner contre le bookmaker.

MVP : football réel (⚽), plusieurs ligues. L'API ESPN étant uniforme, c'est
réplicable à NFL / NBA / NHL (ajouter une entrée dans SPORTS).
"""
from __future__ import annotations

import json
import math
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

_H = {"User-Agent": "Mozilla/5.0"}
_BASE = "https://site.api.espn.com/apis/site/v2/sports/{path}/scoreboard"

# label affiché -> {path ESPN, nul possible, avantage terrain (pts Elo), kind}.
#   kind="team"   : event.competitions[0].competitors[] (équipes, score chiffré).
#   kind="tennis" : event.groupings[].competitions[] (joueurs 1v1, flag `winner`,
#                   terrain neutre -> home_adv 0, pas de nul).
# L'avantage terrain diffère par sport (ordres de grandeur façon FiveThirtyEight).
SPORTS = {
    "⚽ Premier League (Angleterre)": {"path": "soccer/eng.1", "nuls": True, "home_adv": 60, "kind": "team"},
    "⚽ La Liga (Espagne)":           {"path": "soccer/esp.1", "nuls": True, "home_adv": 60, "kind": "team"},
    "⚽ Serie A (Italie)":            {"path": "soccer/ita.1", "nuls": True, "home_adv": 60, "kind": "team"},
    "⚽ Bundesliga (Allemagne)":      {"path": "soccer/ger.1", "nuls": True, "home_adv": 60, "kind": "team"},
    "⚽ Ligue 1 (France)":            {"path": "soccer/fra.1", "nuls": True, "home_adv": 60, "kind": "team"},
    "⚽ Ligue des Champions":         {"path": "soccer/uefa.champions", "nuls": True, "home_adv": 55, "kind": "team"},
    "🏈 NFL (football américain)":    {"path": "football/nfl", "nuls": False, "home_adv": 48, "kind": "team"},
    "🏀 NBA (basket)":                {"path": "basketball/nba", "nuls": False, "home_adv": 100, "kind": "team"},
    "🏒 NHL (hockey)":                {"path": "hockey/nhl", "nuls": False, "home_adv": 50, "kind": "team"},
    "🎾 Tennis ATP (simple messieurs)": {"path": "tennis/atp", "nuls": False, "home_adv": 0, "kind": "tennis", "slug": "mens-singles"},
    "🎾 Tennis WTA (simple dames)":      {"path": "tennis/wta", "nuls": False, "home_adv": 0, "kind": "tennis", "slug": "womens-singles"},
}


# Regroupement des ligues par SPORT (pour le « top 3 du jour par sport »).
GROUPES = ["⚽ Football", "🏈 NFL", "🏀 NBA", "🏒 NHL", "🎾 Tennis"]


def groupe_de(cfg: dict) -> str:
    """Sport d'appartenance d'une entrée SPORTS (les 6 ligues de foot → Football,
    ATP+WTA → Tennis, etc.)."""
    if cfg.get("kind") == "tennis":
        return "🎾 Tennis"
    p = cfg.get("path", "")
    if p.startswith("soccer"):
        return "⚽ Football"
    return {"football/nfl": "🏈 NFL", "basketball/nba": "🏀 NBA",
            "hockey/nhl": "🏒 NHL"}.get(p, p)


def _get(url: str, timeout: float = 20.0):
    req = urllib.request.Request(url, headers=_H)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _completed(comp: dict) -> bool:
    return bool(((comp.get("status") or {}).get("type") or {}).get("completed"))


def _parse_team(data: dict) -> list:
    """Sports d'équipe : event.competitions[0].competitors[] (score chiffré)."""
    out = []
    for e in (data or {}).get("events", []):
        comp = (e.get("competitions") or [{}])[0]
        cs = comp.get("competitors") or []
        home = next((c for c in cs if c.get("homeAway") == "home"), None)
        away = next((c for c in cs if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue

        def _nm(c):
            return (c.get("team") or {}).get("displayName")

        def _sc(c):
            try:
                return int(c.get("score"))
            except (TypeError, ValueError):
                return None

        out.append({"id": comp.get("id") or e.get("id"), "date": e.get("date"),
                    "home": _nm(home), "away": _nm(away),
                    "hs": _sc(home), "as_": _sc(away), "done": _completed(comp)})
    return out


def _parse_tennis(data: dict, slug: str | None = None) -> list:
    """Tennis (1v1) : event(tournoi).groupings[].competitions[](matchs). Pas de
    score chiffré exposé → le résultat vient du flag `winner` (hs/as_ = 1/0).
    Terrain neutre : les deux joueurs sont mis en home/away arbitrairement, mais
    l'Elo tennis tourne avec home_adv=0, donc l'ordre n'influe pas.

    `slug` filtre le TABLEAU (ex 'mens-singles') : les flux ATP et WTA d'ESPN
    renvoient le même tournoi avec TOUS les tableaux (simples + doubles + mixte),
    donc sans filtre on mélangerait hommes/femmes et on polluerait avec les
    doubles. Par défaut (slug None) : tous les tableaux de SIMPLE."""
    out = []
    for e in (data or {}).get("events", []):
        for gr in e.get("groupings") or []:
            g_slug = (gr.get("grouping") or {}).get("slug") or ""
            if slug:
                if g_slug != slug:
                    continue
            elif "singles" not in g_slug:     # défaut : simples seulement
                continue
            for comp in gr.get("competitions") or []:
                cs = comp.get("competitors") or []
                if len(cs) != 2:
                    continue

                def _nm(c):
                    return (c.get("athlete") or {}).get("displayName")

                a, b = cs[0], cs[1]
                na, nb = _nm(a), _nm(b)
                if not na or not nb:
                    continue
                done = _completed(comp)
                hs = as_ = None
                if done:
                    # qui a gagné : flag `winner` sur l'un des deux.
                    if a.get("winner"):
                        hs, as_ = 1, 0
                    elif b.get("winner"):
                        hs, as_ = 0, 1
                    else:
                        continue          # terminé mais sans vainqueur clair -> ignoré
                out.append({"id": comp.get("id"), "date": comp.get("date") or e.get("date"),
                            "home": na, "away": nb, "hs": hs, "as_": as_, "done": done})
    return out


def parse_games(data: dict, kind: str = "team", slug: str | None = None) -> list:
    """Dispatcher : → [{id, date, home, away, hs, as_, done}] selon le type."""
    return _parse_tennis(data, slug) if kind == "tennis" else _parse_team(data)


def _jour(path: str, d: datetime, timeout: float, kind: str, slug: str | None) -> list:
    try:
        u = _BASE.format(path=path) + "?dates=" + d.strftime("%Y%m%d") + "&limit=300"
        return parse_games(_get(u, timeout), kind, slug)
    except Exception:
        return []          # un jour qui échoue ne casse pas la collecte


def fetch_range(path: str, debut: datetime, fin: datetime,
                kind: str = "team", slug: str | None = None, timeout: float = 20.0) -> list:
    """ESPN ne gère pas les plages (400) : on interroge JOUR PAR JOUR, EN
    PARALLÈLE (sinon ~0,5 s × N jours), on concatène, déduplique par id de match
    (ou (home, away, jour) à défaut), et on trie par date croissante."""
    jours = []
    d = debut
    while d <= fin:
        jours.append(d)
        d += timedelta(days=1)
    with ThreadPoolExecutor(max_workers=16) as ex:
        lots = list(ex.map(lambda x: _jour(path, x, timeout, kind, slug), jours))
    jeux, vus = [], set()
    for lot in lots:
        for g in lot:
            k = g.get("id") or (g["home"], g["away"], (g["date"] or "")[:10])
            if k not in vus:
                vus.add(k)
                jeux.append(g)
    jeux.sort(key=lambda g: g["date"] or "")
    return jeux


class Elo:
    """Elo d'équipes. `home_adv` en points ajoutés au domicile ; `k` vitesse
    d'ajustement. Le nul (foot) est modélisé à part : sa part décroît avec
    l'écart de niveau."""

    def __init__(self, k: float = 20.0, home_adv: float = 60.0, base: float = 1500.0):
        self.k, self.home_adv, self.base = k, home_adv, base
        self.r: dict = {}

    def rating(self, t: str) -> float:
        return self.r.get(t, self.base)

    @staticmethod
    def _attendu(ra: float, rb: float) -> float:
        """Proba (hors nul) que A batte B selon l'écart de notes."""
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    def proba(self, home: str, away: str, nuls: bool = True) -> dict:
        ra = self.rating(home) + self.home_adv
        rb = self.rating(away)
        p_core = self._attendu(ra, rb)        # domicile vs extérieur, hors nul
        if not nuls:
            return {"home": p_core, "draw": 0.0, "away": 1.0 - p_core}
        # part de nul : ~28% quand les niveaux sont égaux, décroît avec l'écart.
        p_draw = max(0.06, 0.28 * math.exp(-abs(ra - rb) / 200.0))
        p_home = p_core * (1.0 - p_draw)
        p_away = (1.0 - p_core) * (1.0 - p_draw)
        s = p_home + p_draw + p_away
        return {"home": p_home / s, "draw": p_draw / s, "away": p_away / s}

    def maj(self, home: str, away: str, hs: int, as_: int) -> None:
        s = 1.0 if hs > as_ else (0.0 if hs < as_ else 0.5)   # vu du domicile
        e = self._attendu(self.rating(home) + self.home_adv, self.rating(away))
        delta = self.k * (s - e)
        self.r[home] = self.rating(home) + delta
        self.r[away] = self.rating(away) - delta

    def entrainer(self, games: list) -> "Elo":
        for g in games:
            if g.get("done") and g.get("hs") is not None and g.get("as_") is not None:
                self.maj(g["home"], g["away"], g["hs"], g["as_"])
        return self


def _auj() -> datetime:
    return datetime.now(timezone.utc)


def historique(path: str, jours: int = 160, kind: str = "team",
               slug: str | None = None) -> list:
    """Matchs TERMINÉS des `jours` derniers jours (pour entraîner l'Elo)."""
    fin = _auj()
    jeux = fetch_range(path, fin - timedelta(days=jours), fin, kind, slug)
    return [g for g in jeux if g["done"] and g["hs"] is not None and g["as_"] is not None]


def a_venir(path: str, jours: int = 10, kind: str = "team",
            slug: str | None = None) -> list:
    """Matchs À VENIR (non terminés) des `jours` prochains jours. En tennis,
    on ne garde que ceux dont les DEUX joueurs sont déjà connus."""
    debut = _auj()
    jeux = fetch_range(path, debut, debut + timedelta(days=jours), kind, slug)
    return [g for g in jeux if not g["done"] and g["home"] and g["away"]]


# ---------------------------------------------------------------------------
# PRONOSTIC par FORME + FACE-À-FACE (sans aucune cote) — Olivio, 20/10.
# Le pronostic ne regarde QUE les résultats : la forme récente de chaque camp
# et leur historique de confrontations directes.
# ---------------------------------------------------------------------------

def _vu_de(g: dict, team: str):
    """Résultat du match `g` vu de `team` : ('V'|'N'|'D', écart de buts)."""
    gf, ga = (g["hs"], g["as_"]) if g["home"] == team else (g["as_"], g["hs"])
    if gf > ga:
        return "V", gf - ga
    if gf < ga:
        return "D", gf - ga
    return "N", 0


def forme(passes: list, team: str, n: int = 6) -> dict:
    """Forme récente d'une équipe/joueur sur ses `n` derniers matchs TERMINÉS
    (parmi `passes`, déjà chronologiques). `score` = taux de réussite (V=1,
    N=0,5, D=0). `seq` = du plus récent au plus ancien."""
    rec = [g for g in passes if g["home"] == team or g["away"] == team][-n:]
    seq, pts = [], 0.0
    for g in rec:
        r, _ = _vu_de(g, team)
        seq.append(r)
        pts += 1.0 if r == "V" else (0.5 if r == "N" else 0.0)
    return {"seq": "".join(reversed(seq)), "n": len(rec),
            "score": pts / len(rec) if rec else 0.5}


def face_a_face(passes: list, a: str, b: str, n: int = 6) -> dict:
    """Confrontations directes récentes entre `a` et `b` (max `n`). `score_a` =
    taux de réussite de `a` dans ces face-à-face (None si aucun)."""
    h = [g for g in passes if {g["home"], g["away"]} == {a, b}][-n:]
    va = vb = nul = 0
    for g in h:
        r, _ = _vu_de(g, a)
        va += r == "V"
        vb += r == "D"
        nul += r == "N"
    tot = va + vb + nul
    return {"va": va, "vb": vb, "nul": nul, "tot": tot,
            "score_a": (va + 0.5 * nul) / tot if tot else None}


def pronostic_forme(passes: list, home: str, away: str, nuls: bool,
                    home_adv: float = 0.0, n_forme: int = 6, n_h2h: int = 6) -> dict:
    """Pronostic d'un match à partir de la FORME récente + du FACE-À-FACE, sans
    aucune cote. Rend probas, `pick` (home/draw/away) et la confiance."""
    fh = forme(passes, home, n_forme)
    fa = forme(passes, away, n_forme)
    eps = 0.1
    # Cœur (hors nul) : part de forme du domicile, + l'avantage terrain traduit
    # en probabilité via la même courbe que l'Elo.
    p_ratio = (fh["score"] + eps) / (fh["score"] + fa["score"] + 2 * eps)
    bonus = Elo._attendu(1500 + home_adv, 1500) - 0.5
    p_core = min(0.97, max(0.03, p_ratio + bonus))
    h2h = face_a_face(passes, home, away, n_h2h)
    if h2h["score_a"] is not None and h2h["tot"] >= 2:
        # le face-à-face pèse ~35 % quand il existe (au moins 2 confrontations).
        p_core = min(0.97, max(0.03, 0.65 * p_core + 0.35 * h2h["score_a"]))
    if nuls:
        # le nul est d'autant plus probable que les deux camps sont proches.
        p_draw = max(0.06, 0.30 * (1.0 - 2.0 * abs(p_core - 0.5)))
        ph, pd_, pa = p_core * (1 - p_draw), p_draw, (1 - p_core) * (1 - p_draw)
        s = ph + pd_ + pa
        p = {"home": ph / s, "draw": pd_ / s, "away": pa / s}
        pick = max(p, key=p.get)
    else:
        p = {"home": p_core, "draw": 0.0, "away": 1.0 - p_core}
        pick = "home" if p_core >= 0.5 else "away"
    return {"home": home, "away": away, "p": p, "pick": pick,
            "conf": max(p.values()), "forme_home": fh, "forme_away": fa, "h2h": h2h}


def pronostics_a_venir(hist: list, fixtures: list, nuls: bool,
                       home_adv: float = 0.0) -> list:
    """Pronostic forme + face-à-face pour chaque match à venir (contexte = tout
    l'historique terminé)."""
    passes = [g for g in hist if g.get("done") and g.get("hs") is not None]
    passes.sort(key=lambda g: g.get("date") or "")
    return [pronostic_forme(passes, g["home"], g["away"], nuls, home_adv)
            for g in fixtures if g.get("home") and g.get("away")]


def taux_reussite(hist: list, nuls: bool, home_adv: float = 0.0,
                  maxi: int = 250, warmup: int = 50) -> dict | None:
    """Backtest HONNÊTE : pour chaque match passé (après un warmup), on prédit
    avec la forme + le face-à-face calculés UNIQUEMENT sur les matchs d'AVANT,
    puis on compare au résultat réel. Rend le taux de réussite + une base de
    comparaison (toujours le domicile). None si trop peu de données."""
    done = sorted([g for g in hist if g.get("done") and g.get("hs") is not None],
                  key=lambda g: g.get("date") or "")
    if len(done) < warmup + 20:
        return None
    sous = done[warmup:]
    if len(sous) > maxi:
        sous = sous[-maxi:]
    ok = tot = ok_base = 0
    for g in sous:
        d = g.get("date") or ""
        passes = [x for x in done if (x.get("date") or "") < d]
        pr = pronostic_forme(passes, g["home"], g["away"], nuls, home_adv)
        reel = "home" if g["hs"] > g["as_"] else ("away" if g["hs"] < g["as_"] else "draw")
        ok += pr["pick"] == reel
        ok_base += reel == "home"
        tot += 1
    return {"ok": ok, "tot": tot, "taux": ok / tot if tot else 0.0,
            "base_domicile": ok_base / tot if tot else 0.0}


def predire(elo: Elo, fixtures: list, nuls: bool = True) -> list:
    """Attache à chaque match à venir les probas du modèle + le favori."""
    out = []
    for g in fixtures:
        p = elo.proba(g["home"], g["away"], nuls)
        if nuls:
            fav = max((("home", p["home"]), ("draw", p["draw"]), ("away", p["away"])),
                      key=lambda kv: kv[1])
        else:
            fav = max((("home", p["home"]), ("away", p["away"])), key=lambda kv: kv[1])
        out.append({**g, "p": p, "favori": fav[0],
                    "rating_home": round(elo.rating(g["home"])),
                    "rating_away": round(elo.rating(g["away"]))})
    return out
