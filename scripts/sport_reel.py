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
    "⚾ MLB (baseball)":               {"path": "baseball/mlb", "nuls": False, "home_adv": 24, "kind": "team"},
    "🎾 Tennis ATP (simple messieurs)": {"path": "tennis/atp", "nuls": False, "home_adv": 0, "kind": "tennis", "slug": "mens-singles"},
    "🎾 Tennis WTA (simple dames)":      {"path": "tennis/wta", "nuls": False, "home_adv": 0, "kind": "tennis", "slug": "womens-singles"},
}


# Regroupement des ligues par SPORT (pour le « top 3 du jour par sport »).
GROUPES = ["⚽ Football", "🏈 NFL", "🏀 NBA", "🏒 NHL", "⚾ MLB", "🎾 Tennis"]


def groupe_de(cfg: dict) -> str:
    """Sport d'appartenance d'une entrée SPORTS (les 6 ligues de foot → Football,
    ATP+WTA → Tennis, etc.)."""
    if cfg.get("kind") == "tennis":
        return "🎾 Tennis"
    p = cfg.get("path", "")
    if p.startswith("soccer"):
        return "⚽ Football"
    return {"football/nfl": "🏈 NFL", "basketball/nba": "🏀 NBA",
            "hockey/nhl": "🏒 NHL", "baseball/mlb": "⚾ MLB"}.get(p, p)


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
                    "hs": _sc(home), "as_": _sc(away), "done": _completed(comp),
                    "lead_home": _leader_cat(home, "points"),
                    "lead_away": _leader_cat(away, "points"),
                    "but_home": _leader_cat(home, "goals"),
                    "but_away": _leader_cat(away, "goals")})
    return out


def _leader_cat(c: dict, cat: str):
    """Meneur d'une équipe dans un match pour une catégorie du scoreboard ESPN
    ('points' en NBA/NHL, 'goals' en NHL) : {'nom', 'val'}. Absent au foot."""
    for l in c.get("leaders") or []:
        if l.get("name") == cat:
            it = (l.get("leaders") or [{}])[0]
            nom = (it.get("athlete") or {}).get("displayName")
            try:
                val = float(it.get("value"))
            except (TypeError, ValueError):
                val = None
            if nom:
                return {"nom": nom, "val": val}
    return None


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
                # Nombre de SETS = longueur des linescores. Nombre de JEUX =
                # somme des jeux gagnés par les deux joueurs sur tous les sets
                # (chaque linescore `value` = jeux gagnés dans ce set).
                # `periods` = best-of (3 ou 5), pour fixer les lignes.
                n_sets = n_games = None
                ls = a.get("linescores") or b.get("linescores")
                if done and ls:
                    n_sets = len(ls)
                    tot, ok = 0, False
                    for c in (a, b):
                        for s in c.get("linescores") or []:
                            try:
                                tot += int(round(float(s.get("value"))))
                                ok = True
                            except (TypeError, ValueError):
                                pass
                    if ok:
                        n_games = tot
                periods = ((comp.get("format") or {}).get("regulation") or {}).get("periods")
                out.append({"id": comp.get("id"), "date": comp.get("date") or e.get("date"),
                            "home": na, "away": nb, "hs": hs, "as_": as_, "done": done,
                            "n_sets": n_sets, "n_games": n_games, "periods": periods})
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


def forme(passes: list, team: str, n: int = 6, lieu: str | None = None) -> dict:
    """Forme récente d'une équipe/joueur sur ses `n` derniers matchs TERMINÉS
    (parmi `passes`, déjà chronologiques). `score` = taux de réussite (V=1,
    N=0,5, D=0). `seq` = du plus récent au plus ancien.

    `lieu` restreint au LIEU du match : 'home' (seulement les matchs joués à
    domicile) ou 'away' (seulement à l'extérieur) — pour mesurer si une équipe
    est forte chez elle ou solide en déplacement. None = tous les matchs."""
    rec = [g for g in passes if g["home"] == team or g["away"] == team]
    if lieu == "home":
        rec = [g for g in rec if g["home"] == team]
    elif lieu == "away":
        rec = [g for g in rec if g["away"] == team]
    rec = rec[-n:]
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


_INJ = "https://site.api.espn.com/apis/site/v2/sports/{path}/injuries"


def blessures_du_sport(path: str, timeout: float = 20.0) -> dict:
    """{équipe (displayName): [{'nom', 'status'}]} depuis l'endpoint injuries
    d'ESPN. Dispo pour NBA/NFL/NHL ; VIDE pour le foot (ESPN ne publie pas) et
    le tennis (un forfait annule le match)."""
    out: dict = {}
    try:
        d = _get(_INJ.format(path=path), timeout)
    except Exception:
        return out
    for grp in d.get("injuries") or []:
        team = grp.get("displayName")
        lst = []
        for it in grp.get("injuries") or []:
            nom = (it.get("athlete") or {}).get("displayName")
            if nom:
                lst.append({"nom": nom, "status": it.get("status")})
        if team and lst:
            out[team] = lst
    return out


_STAND = "https://site.api.espn.com/apis/v2/sports/{path}/standings"


def classement_du_sport(path: str, timeout: float = 20.0) -> dict:
    """{équipe (displayName): position (1 = meilleur)} depuis l'endpoint standings
    d'ESPN. La position vient de l'ORDRE des entrées dans chaque groupe (elles
    sont déjà triées par classement) — robuste même quand le champ 'rank' varie.
    Best-effort : VIDE si indisponible ou illisible (coupes à élimination,
    tennis, early-season sans table). Ne sert qu'en live (le classement ACTUEL
    ne vaut pas pour un match passé : pas utilisable dans le backtest)."""
    out: dict = {}
    try:
        d = _get(_STAND.format(path=path), timeout)
    except Exception:
        return out

    def _groupes(node):
        """Rend la liste des groupes {entries:[…]} où qu'ils soient dans l'arbre
        (structure ESPN variable : racine, children[], ou standings imbriqués)."""
        res = []
        st = node.get("standings")
        if isinstance(st, dict) and st.get("entries"):
            res.append(st)
        for ch in node.get("children") or []:
            res.extend(_groupes(ch))
        return res

    for grp in _groupes(d if isinstance(d, dict) else {}):
        for i, e in enumerate(grp.get("entries") or [], start=1):
            team = (e.get("team") or {}).get("displayName")
            if team and team not in out:          # meilleur groupe gardé
                out[team] = i
    return out


def poids_blessures(liste: list | None) -> float:
    """Poids d'absence d'une équipe : 1,0 par « Out », 0,4 par « Day-To-Day »
    (incertain). Faute de valeur par joueur, c'est un COMPTE pondéré, pas une
    mesure de l'importance réelle des blessés."""
    if not liste:
        return 0.0
    return sum(1.0 if x.get("status") == "Out" else 0.4 for x in liste)


def pronostic_forme(passes: list, home: str, away: str, nuls: bool,
                    home_adv: float = 0.0, n_forme: int = 6, n_h2h: int = 6,
                    inj_home: list | None = None, inj_away: list | None = None,
                    rang_home: int | None = None, rang_away: int | None = None,
                    elo: "Elo | None" = None, elo_ext: dict | None = None,
                    xform: dict | None = None) -> dict:
    """Pronostic d'un match à partir de la FORME récente + de la FORME AU LIEU
    (domicile/extérieur) + du FACE-À-FACE, sans aucune cote. Les BLESSURES et le
    CLASSEMENT (si fournis) décalent modérément la proba.

    Si un `elo` (déjà entraîné sur `passes`) est fourni, la proba finale est le
    MÉLANGE 60 % Elo + 40 % forme : l'Elo accumule le signal sur toute la saison
    en pondérant par la force des adversaires, la forme apporte le court terme.
    Backtest réel (8 ligues) : moyenne et plancher meilleurs que chaque méthode
    seule — surtout au foot, où la forme seule faisait moins bien que « toujours
    domicile ». Rend probas, `pick`, confiance, blessés."""
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
    # Force SELON LE LIEU : est-ce que l'hôte est INHABITUELLEMENT fort chez lui,
    # le visiteur solide en déplacement ? On prend l'ÉCART de chacun à sa propre
    # forme globale (pas le niveau brut : l'avantage terrain moyen est déjà dans
    # `bonus`, le re-compter biaiserait tout vers le domicile). Nudge modéré,
    # plafonné à ±0,03, seulement avec ≥ 4 matchs au lieu concerné. Mesuré sur
    # données réelles : effet dans le bruit (neutre), sans dégrader la précision.
    fh_lieu = forme(passes, home, n_forme, lieu="home")
    fa_lieu = forme(passes, away, n_forme, lieu="away")
    d_home = (fh_lieu["score"] - fh["score"]) if fh_lieu["n"] >= 4 else 0.0
    d_away = (fa_lieu["score"] - fa["score"]) if fa_lieu["n"] >= 4 else 0.0
    if d_home or d_away:
        p_core = min(0.97, max(0.03, p_core + max(-0.03, min(0.03, 0.3 * (d_home - d_away)))))
    # Classement (live seulement) : petit coup de pouce au mieux classé, selon
    # l'écart de rang, plafonné à ±0,06. Non backtesté (le rang actuel ne vaut
    # pas pour un match passé) → ne s'applique que si les deux rangs sont fournis.
    if rang_home and rang_away:
        adjc = max(-0.06, min(0.06, 0.004 * (rang_away - rang_home)))
        p_core = min(0.97, max(0.03, p_core + adjc))
    # Blessures : décale la proba contre l'équipe la plus touchée. 0,04 par unité
    # de poids d'écart, plafonné à ±0,12 (repère, pas une mesure précise).
    pw_h, pw_a = poids_blessures(inj_home), poids_blessures(inj_away)
    if pw_h or pw_a:
        adj = max(-0.12, min(0.12, 0.04 * (pw_a - pw_h)))
        p_core = min(0.97, max(0.03, p_core + adj))
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
    # Mélange avec l'Elo (60 % Elo / 40 % forme) quand un Elo entraîné est fourni.
    if elo is not None:
        pe = elo.proba(home, away, nuls)
        w = 0.6
        p = {k: w * pe.get(k, 0.0) + (1.0 - w) * p[k] for k in p}
        tot = sum(p.values()) or 1.0
        p = {k: v / tot for k, v in p.items()}
        pick = max(p, key=p.get) if nuls else ("home" if p["home"] >= p["away"] else "away")
    # 2e RÉFÉRENCE : Elo MULTI-SAISONS (openfootball). Quand les deux équipes y
    # figurent, on mélange sa proba (35 %) — signal indépendant, mémoire longue.
    if elo_ext and home in elo_ext and away in elo_ext:
        me = Elo(home_adv=home_adv)
        me.r = elo_ext
        px = me.proba(home, away, nuls)
        w = 0.35
        p = {k: w * px.get(k, 0.0) + (1.0 - w) * p[k] for k in p}
        tot = sum(p.values()) or 1.0
        p = {k: v / tot for k, v in p.items()}
        pick = max(p, key=p.get) if nuls else ("home" if p["home"] >= p["away"] else "away")
    # 3e SIGNAL : xG-FORME (Understat). Forme réelle via les buts attendus, plus
    # prédictive que la forme-résultats (mesuré +2 pts moyen). Mélange 30 %.
    if xform and home in xform and away in xform:
        core = 0.5 + (xform[home] - xform[away]) + (Elo._attendu(1500 + home_adv, 1500) - 0.5)
        core = min(0.97, max(0.03, core))
        if nuls:
            pdr = max(0.06, 0.30 * (1.0 - 2.0 * abs(core - 0.5)))
            pxf = {"home": core * (1 - pdr), "draw": pdr, "away": (1 - core) * (1 - pdr)}
        else:
            pxf = {"home": core, "draw": 0.0, "away": 1.0 - core}
        sx = sum(pxf.values()) or 1.0
        pxf = {k: v / sx for k, v in pxf.items()}
        w = 0.30
        p = {k: w * pxf[k] + (1.0 - w) * p[k] for k in p}
        tot = sum(p.values()) or 1.0
        p = {k: v / tot for k, v in p.items()}
        pick = max(p, key=p.get) if nuls else ("home" if p["home"] >= p["away"] else "away")
    return {"home": home, "away": away, "p": p, "pick": pick,
            "conf": max(p.values()), "forme_home": fh, "forme_away": fa, "h2h": h2h,
            "forme_home_dom": fh_lieu, "forme_away_ext": fa_lieu,
            "rang_home": rang_home, "rang_away": rang_away,
            "bless_home": inj_home or [], "bless_away": inj_away or []}


def stats_off(passes: list, team: str, n: int = 8):
    """Moyennes offensive/défensive récentes : (buts/points marqués par match,
    encaissés par match) sur les `n` derniers. (None, None) si aucun."""
    rec = [g for g in passes if g["home"] == team or g["away"] == team][-n:]
    bf = bc = 0.0
    for g in rec:
        gf, ga = (g["hs"], g["as_"]) if g["home"] == team else (g["as_"], g["hs"])
        bf += gf
        bc += ga
    return (bf / len(rec), bc / len(rec)) if rec else (None, None)


def _pois_cdf_le(lam: float, k: int) -> float:
    """P(X ≤ k) pour une loi de Poisson de moyenne `lam`."""
    s, t = 0.0, math.exp(-lam)
    for i in range(k + 1):
        if i:
            t *= lam / i
        s += t
    return s


def marches_buts(passes: list, home: str, away: str, nuls: bool,
                 ligue_avg: float, n: int = 8) -> dict | None:
    """Marchés dérivés des scores (sans cote) : total attendu, puis — au foot —
    « les deux marquent » + plus/moins 2,5 buts (Poisson) ; en NBA/NHL — total
    de points/buts plus/moins la ligne (moyenne récente de la ligue)."""
    bf_h, bc_h = stats_off(passes, home, n)
    bf_a, bc_a = stats_off(passes, away, n)
    if bf_h is None or bf_a is None:
        return None
    lam_h, lam_a = (bf_h + bc_a) / 2.0, (bf_a + bc_h) / 2.0
    tot = lam_h + lam_a
    out = {"exp_total": tot}
    if nuls:                                   # football
        out["btts"] = (1 - math.exp(-lam_h)) * (1 - math.exp(-lam_a))
        out["ligne"] = 2.5
        out["over"] = 1.0 - _pois_cdf_le(tot, 2)
    else:                                      # NBA / NHL : DEMI-ligne près de la moyenne
        ligne = int(ligue_avg) + 0.5 if ligue_avg else tot
        sigma = max(1.0, (ligue_avg or tot) * 0.08)
        out["ligne"] = ligne
        out["over"] = 1.0 / (1.0 + math.exp(-(tot - ligne) / sigma))
    return out


def meilleur_marqueur(passes: list, home: str, away: str, n: int = 10,
                      buts: bool = False) -> dict | None:
    """Meilleur marqueur probable (NBA/NHL) sur les `n` derniers matchs de chaque
    équipe : POINTS (NBA) ou BUTS (NHL, `buts=True`). Chaque `top` porte sa MOY
    et une LIGNE « plus de X » (≈ moyenne arrondie − 0,5). None si pas de donnée."""
    ch, ca = ("but_home", "but_away") if buts else ("lead_home", "lead_away")

    def _agg(team):
        rec = [g for g in passes if g["home"] == team or g["away"] == team][-n:]
        tot: dict = {}
        for g in rec:
            ld = g.get(ch) if g["home"] == team else g.get(ca)
            if ld and ld.get("nom"):
                e = tot.setdefault(ld["nom"], [0, 0.0])
                e[0] += 1
                if ld.get("val") is not None:
                    e[1] += ld["val"]
        if not tot:
            return None
        nom = max(tot, key=lambda k: tot[k][0])
        fois, somme = tot[nom]
        moy = somme / fois if fois else None
        ligne = max(0.5, round(moy) - 0.5) if moy is not None else None
        return {"nom": nom, "fois": fois, "sur": len(rec), "moy": moy, "ligne": ligne}
    mh, ma = _agg(home), _agg(away)
    cand = [x for x in (mh, ma) if x and x.get("moy") is not None]
    return {"home": mh, "away": ma,
            "top": max(cand, key=lambda x: x["moy"]) if cand else None, "buts": buts}


def marche_sets(passes: list, a: str, b: str, periods: int | None = None,
                n: int = 12) -> dict | None:
    """Total de SETS d'un match de tennis : moyenne de sets joués par chaque
    joueur récemment → total attendu, plus/moins la ligne (2,5 en best-of-3,
    3,5 en best-of-5). `n_sets` = longueur des linescores."""
    def _moy(j):
        vals = [g["n_sets"] for g in passes
                if (g["home"] == j or g["away"] == j) and g.get("n_sets")]
        vals = vals[-n:]
        return sum(vals) / len(vals) if vals else None
    ma, mb = _moy(a), _moy(b)
    if ma is None or mb is None:
        return None
    exp = (ma + mb) / 2.0
    ligne = 3.5 if (periods or 3) >= 5 else 2.5
    over = 1.0 / (1.0 + math.exp(-(exp - ligne) / 0.5))
    return {"exp": exp, "ligne": ligne, "over": over}


def marche_jeux(passes: list, a: str, b: str, periods: int | None = None,
                n: int = 12) -> dict | None:
    """Total de JEUX d'un match de tennis : moyenne de jeux joués par chaque
    joueur récemment → total attendu, plus/moins la ligne (22,5 en best-of-3,
    36,5 en best-of-5)."""
    def _moy(j):
        vals = [g["n_games"] for g in passes
                if (g["home"] == j or g["away"] == j) and g.get("n_games")]
        vals = vals[-n:]
        return sum(vals) / len(vals) if vals else None
    ma, mb = _moy(a), _moy(b)
    if ma is None or mb is None:
        return None
    exp = (ma + mb) / 2.0
    ligne = 36.5 if (periods or 3) >= 5 else 22.5
    over = 1.0 / (1.0 + math.exp(-(exp - ligne) / 3.5))
    return {"exp": exp, "ligne": ligne, "over": over}


_SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/{path}/summary?event={eid}"


def _corners_match(path: str, eid, timeout: float) -> dict | None:
    """Corners par équipe d'un match (endpoint summary ESPN, stat 'wonCorners')."""
    try:
        s = _get(_SUMMARY.format(path=path, eid=eid), timeout)
    except Exception:
        return None
    res = {}
    for t in (s.get("boxscore") or {}).get("teams") or []:
        name = (t.get("team") or {}).get("displayName")
        wc = None
        for st in t.get("statistics") or []:
            if st.get("name") == "wonCorners":
                try:
                    wc = float(st.get("displayValue") or st.get("value"))
                except (TypeError, ValueError):
                    wc = None
        if name and wc is not None:
            res[name] = wc
    return res or None


def stats_corners(path: str, jours: int = 45, timeout: float = 20.0,
                  maxi: int = 90) -> dict:
    """Corners moyens (pour / contre) par équipe sur les matchs récents (foot).
    Coûteux : 1 requête `summary` par match → fenêtre courte + plafond `maxi`."""
    fin = _auj()
    games = fetch_range(path, fin - timedelta(days=jours), fin, "team")
    done = [g for g in games if g["done"] and g.get("id")][-maxi:]
    pour: dict = {}
    contre: dict = {}

    def _work(g):
        return g, _corners_match(path, g["id"], timeout)

    with ThreadPoolExecutor(max_workers=12) as ex:
        for g, c in ex.map(_work, done):
            if not c:
                continue
            h, a = g["home"], g["away"]
            if h in c and a in c:
                pour.setdefault(h, []).append(c[h]); contre.setdefault(h, []).append(c[a])
                pour.setdefault(a, []).append(c[a]); contre.setdefault(a, []).append(c[h])
    out = {}
    for t in set(list(pour) + list(contre)):
        f, ag = pour.get(t, []), contre.get(t, [])
        out[t] = {"for": sum(f) / len(f) if f else None,
                  "against": sum(ag) / len(ag) if ag else None, "n": len(f)}
    return out


def marche_corners(sc: dict, home: str, away: str, ligne: float = 9.5) -> dict | None:
    """Total de corners attendu + plus/moins la ligne (9,5 par défaut)."""
    h, a = sc.get(home), sc.get(away)
    if not h or not a or h["for"] is None or a["for"] is None:
        return None
    exp = (h["for"] + a["against"]) / 2 + (a["for"] + h["against"]) / 2
    over = 1.0 / (1.0 + math.exp(-(exp - ligne) / 3.0))
    return {"exp": exp, "ligne": ligne, "over": over}


def unite_du_sport(path: str) -> str:
    """Nom de l'unité du total selon le sport : buts (NHL), runs (MLB), sinon
    points (NBA…). Sert au libellé « Total de … » du Top 3."""
    return {"hockey/nhl": "buts", "baseball/mlb": "runs"}.get(path, "points")


def pronostics_a_venir(hist: list, fixtures: list, nuls: bool,
                       home_adv: float = 0.0, blessures: dict | None = None,
                       kind: str = "team", hockey: bool = False,
                       classement: dict | None = None,
                       unite: str | None = None, elo_ext: dict | None = None,
                       xform: dict | None = None) -> list:
    """Pronostic forme + face-à-face (+ blessures) + marchés additionnels pour
    chaque match à venir. Marchés selon le sport : foot = buts/BTTS ; NBA/NHL =
    total points + meilleur marqueur (buts en hockey) ; tennis = total de SETS."""
    passes = [g for g in hist if g.get("done") and g.get("hs") is not None]
    passes.sort(key=lambda g: g.get("date") or "")
    tots = [g["hs"] + g["as_"] for g in passes]
    ligue_avg = sum(tots) / len(tots) if tots else 0.0
    tennis = kind == "tennis"
    bl = blessures or {}
    cl = classement or {}
    elo = Elo(home_adv=home_adv).entrainer(passes)   # backbone saison entière
    out = []
    for g in fixtures:
        if not (g.get("home") and g.get("away")):
            continue
        pr = pronostic_forme(passes, g["home"], g["away"], nuls, home_adv,
                             inj_home=bl.get(g["home"]), inj_away=bl.get(g["away"]),
                             rang_home=cl.get(g["home"]), rang_away=cl.get(g["away"]),
                             elo=elo, elo_ext=elo_ext, xform=xform)
        pr["date"] = g.get("date")      # date/heure ISO UTC du coup d'envoi
        if tennis:
            pr["sets"] = marche_sets(passes, g["home"], g["away"], g.get("periods"))
            pr["jeux"] = marche_jeux(passes, g["home"], g["away"], g.get("periods"))
        else:
            pr["marches"] = marches_buts(passes, g["home"], g["away"], nuls, ligue_avg)
            if pr["marches"] is not None and not nuls:
                pr["marches"]["unite"] = unite or ("buts" if hockey else "points")
            pr["marqueur"] = (None if nuls else
                              meilleur_marqueur(passes, g["home"], g["away"], buts=hockey))
        out.append(pr)
    return out


def matchs_du_jour(path: str, jour: datetime, kind: str = "team",
                   slug: str | None = None, tz_offset_h: int = 3) -> list:
    """Matchs programmés (non terminés) dont la date LOCALE (UTC+tz_offset_h,
    Madagascar par défaut) tombe le jour `jour`.

    Un match NHL/NBA/MLB joué le soir aux USA a lieu APRÈS MINUIT chez nous : sa
    date calendaire US (celle du programme ESPN) n'est pas la date où l'utilisateur
    le voit. On interroge donc la veille, le jour et le lendemain (dates US), puis
    on filtre sur la date LOCALE — pour que « 21/10 » montre les matchs du 21/10
    à Mada, pas ceux affichés à 02h30 le 22/10."""
    cible = jour.date()
    jeux = fetch_range(path, jour - timedelta(days=1), jour + timedelta(days=1),
                       kind, slug)
    tz = timezone(timedelta(hours=tz_offset_h))
    out = []
    for g in jeux:
        if g["done"] or not (g["home"] and g["away"]):
            continue
        try:
            local = datetime.fromisoformat(str(g.get("date")).replace("Z", "+00:00"))
            if local.astimezone(tz).date() != cible:
                continue
        except (ValueError, TypeError):
            pass          # date illisible : on garde (ne jamais perdre un vrai match)
        out.append(g)
    return out


def analyse_match(p: dict) -> str:
    """Mon analyse en clair d'un pronostic (forme + face-à-face + blessures),
    sans aucune cote. Rend un court texte français."""
    fh, fa, h = p["forme_home"], p["forme_away"], p["h2h"]
    home, away = p["home"], p["away"]
    parts = [f"Forme récente — {home} : {fh['seq'] or '—'} ; {away} : {fa['seq'] or '—'}."]
    if fh["score"] > fa["score"] + 0.12:
        parts.append(f"{home} est nettement mieux lancé.")
    elif fa["score"] > fh["score"] + 0.12:
        parts.append(f"{away} est nettement mieux lancé.")
    else:
        parts.append("Les deux sont dans une forme comparable.")
    # Force selon le lieu (domicile de l'hôte / extérieur du visiteur).
    fhd, fae = p.get("forme_home_dom"), p.get("forme_away_ext")
    if fhd and fhd["n"] >= 3:
        parts.append(f"À domicile, {home} : {fhd['seq']} ({fhd['score'] * 100:.0f}%).")
    if fae and fae["n"] >= 3:
        parts.append(f"À l'extérieur, {away} : {fae['seq']} ({fae['score'] * 100:.0f}%).")
    # Classement (si connu).
    rh, ra = p.get("rang_home"), p.get("rang_away")
    if rh and ra:
        mieux = home if rh < ra else away
        parts.append(f"Classement : {home} {rh}ᵉ, {away} {ra}ᵉ (avantage {mieux}).")
    if h["tot"]:
        if h["va"] > h["vb"]:
            parts.append(f"Face-à-face : {home} mène {h['va']}–{h['vb']} sur {h['tot']}.")
        elif h["vb"] > h["va"]:
            parts.append(f"Face-à-face : {away} mène {h['vb']}–{h['va']} sur {h['tot']}.")
        else:
            parts.append(f"Face-à-face équilibré ({h['va']}–{h['vb']} sur {h['tot']}).")
    else:
        parts.append("Pas de confrontation directe récente.")
    for cote, lst, nom in ((home, p.get("bless_home"), home), (away, p.get("bless_away"), away)):
        outs = [b["nom"] for b in (lst or []) if b.get("status") == "Out"]
        if outs:
            parts.append(f"{nom} privé de {', '.join(outs[:3])}.")
    # Marchés additionnels (totaux, les deux marquent, meilleur marqueur).
    m = p.get("marches")
    if m:
        if "btts" in m:                        # football
            parts.append(f"Les deux marquent : **{'oui' if m['btts'] >= 0.5 else 'non'}** "
                         f"({m['btts'] * 100:.0f}%).")
            sens = "plus" if m["over"] >= 0.5 else "moins"
            parts.append(f"Total buts : **{sens} de {m['ligne']:g}** "
                         f"({max(m['over'], 1 - m['over']) * 100:.0f}%, "
                         f"~{m['exp_total']:.1f} attendus).")
        else:                                  # NBA (points) / NHL (buts)
            sens = "plus" if m["over"] >= 0.5 else "moins"
            unite = m.get("unite", "points")
            parts.append(f"Total de {unite} : **{sens} de {m['ligne']:g}** "
                         f"({max(m['over'], 1 - m['over']) * 100:.0f}%, "
                         f"~{m['exp_total']:.0f} attendus).")
    co = p.get("corners")
    if co:
        sens = "plus" if co["over"] >= 0.5 else "moins"
        parts.append(f"Corners : **{sens} de {co['ligne']:g}** "
                     f"({max(co['over'], 1 - co['over']) * 100:.0f}%, "
                     f"~{co['exp']:.0f} attendus).")
    se = p.get("sets")
    if se:
        sens = "plus" if se["over"] >= 0.5 else "moins"
        parts.append(f"Total de sets : **{sens} de {se['ligne']:g}** "
                     f"({max(se['over'], 1 - se['over']) * 100:.0f}%, "
                     f"~{se['exp']:.1f} attendus).")
    je = p.get("jeux")
    if je:
        sens = "plus" if je["over"] >= 0.5 else "moins"
        parts.append(f"Total de jeux : **{sens} de {je['ligne']:g}** "
                     f"({max(je['over'], 1 - je['over']) * 100:.0f}%, "
                     f"~{je['exp']:.0f} attendus).")
    mq = p.get("marqueur")
    if mq and mq.get("top") and mq["top"].get("moy") is not None:
        t = mq["top"]
        unite = "buts" if mq.get("buts") else "pts"
        ligne = (f" → **plus de {t['ligne']:g} {unite}**"
                 if t.get("ligne") is not None else "")
        parts.append(f"Meilleur marqueur probable : **{t['nom']}** "
                     f"(~{t['moy']:.0f} {unite}/match{ligne}).")
    cible = ("le nul" if p["pick"] == "draw"
             else home if p["pick"] == "home" else away)
    parts.append(f"👉 Mon pronostic : **{cible}** (confiance {p['conf'] * 100:.0f}%).")
    return " ".join(parts)


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
        elo = Elo(home_adv=home_adv).entrainer(passes)
        pr = pronostic_forme(passes, g["home"], g["away"], nuls, home_adv, elo=elo)
        reel = "home" if g["hs"] > g["as_"] else ("away" if g["hs"] < g["as_"] else "draw")
        ok += pr["pick"] == reel
        ok_base += reel == "home"
        tot += 1
    return {"ok": ok, "tot": tot, "taux": ok / tot if tot else 0.0,
            "base_domicile": ok_base / tot if tot else 0.0}


def taux_par_confiance(hist: list, nuls: bool, home_adv: float = 0.0,
                       seuils=(0.50, 0.60, 0.65, 0.70, 0.75),
                       maxi: int = 400, warmup: int = 50) -> list | None:
    """Backtest walk-forward qui mesure la réussite PAR TRANCHE DE CONFIANCE :
    pour chaque seuil, le taux sur les seuls pronostics où le modèle était au
    moins aussi sûr. Idée clé : la moyenne est capée, mais les picks confiants
    sont bien plus fiables — on ne parie que ceux-là. Rend [{seuil, taux, tot}]
    (seuils sans match ignorés), ou None si trop peu de données."""
    done = sorted([g for g in hist if g.get("done") and g.get("hs") is not None],
                  key=lambda g: g.get("date") or "")
    if len(done) < warmup + 20:
        return None
    sous = done[warmup:][-maxi:]
    buckets = {s: [0, 0] for s in seuils}            # seuil -> [ok, tot]
    for g in sous:
        d = g.get("date") or ""
        passes = [x for x in done if (x.get("date") or "") < d]
        elo = Elo(home_adv=home_adv).entrainer(passes)
        pr = pronostic_forme(passes, g["home"], g["away"], nuls, home_adv, elo=elo)
        reel = "home" if g["hs"] > g["as_"] else ("away" if g["hs"] < g["as_"] else "draw")
        bon = pr["pick"] == reel
        for s in seuils:
            if pr["conf"] >= s:
                buckets[s][1] += 1
                buckets[s][0] += bon
    out = [{"seuil": s, "taux": ok / tot, "tot": tot}
           for s, (ok, tot) in sorted(buckets.items()) if tot >= 10]
    return out or None


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
