"""Tennis de table — ligues de paris rapides (Czech Liga Pro, TT Cup, TT Elite
Series, Challenger Series…) via l'API publique de scores24.live, SANS cote.

Ces ligues ne sont pas sur World Table Tennis ni exposées en résultats par
Bet261. scores24.live publie les résultats (joueurs, sets gagnés, score par set,
vainqueur) en JSON, accessible côté serveur. On interroge JOUR PAR JOUR
(l'endpoint n'inline les matchs que par fenêtre courte).

Réutilise le moteur de `sport_reel` (Elo, forme, face-à-face, total de manches).
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import sport_reel as _sr

_H = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
    "Accept": "application/json",
    "Referer": "https://scores24.live/en/table-tennis",
}
_API = "https://scores24.live/rapi/sport/table-tennis/leagues"


def _get_jour(jour: datetime, timeout: float = 25.0) -> list:
    """Blocs ligues+matchs d'un JOUR (fenêtre 22:00 J-1 → 21:59 J, heure du site)."""
    d1 = (jour - timedelta(days=1)).strftime("%Y-%m-%d 22:00:00")
    d2 = jour.strftime("%Y-%m-%d 21:59:59")
    blocs = []
    for is_open in ("true", "false"):
        q = ("date_between[]=" + urllib.parse.quote(d1)
             + "&date_between[]=" + urllib.parse.quote(d2)
             + "&is_bot=false&is_open=" + is_open
             + "&lang=en&match_filter=all&with_live=true")
        try:
            req = urllib.request.Request(_API + "?" + q, headers=_H)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read().decode("utf-8")).get("data") or []
            blocs += [b for b in data if b.get("matches")]
        except Exception:
            pass
    return blocs


def _norm_nom(n: str) -> str:
    """« Nom, Prénom » → « Prénom Nom » (certaines ligues inversent), pour ne pas
    fragmenter un même joueur en deux."""
    if n and "," in n:
        a, b = [x.strip() for x in n.split(",", 1)]
        return (b + " " + a).strip()
    return n


def _parse_match(m: dict, ligue: str) -> dict | None:
    """Un match scores24 → {home, away, hs, as_, done, date, n_sets, n_points,
    periods, ligue}. hs/as_ = manches gagnées (depuis result_score « 2:3 »)."""
    teams = m.get("teams") or []
    if len(teams) != 2:
        return None
    na, nb = _norm_nom(teams[0].get("name")), _norm_nom(teams[1].get("name"))
    if not na or not nb:
        return None
    rs = str(m.get("result_score") or "").split(":")
    if len(rs) != 2:
        return None
    try:
        hs, as_ = int(rs[0]), int(rs[1])
    except ValueError:
        return None
    done = bool(m.get("is_finished")) and hs != as_
    n_sets = pts = 0
    for s in m.get("result_scores") or []:
        if s.get("type") == "FT":
            continue
        val = str(s.get("value") or "").split(":")
        if len(val) != 2:
            continue
        try:
            x, y = int(val[0]), int(val[1])
        except ValueError:
            continue
        if x == 0 and y == 0:
            continue
        n_sets += 1
        pts += x + y
    return {"home": na, "away": nb, "hs": hs, "as_": as_, "done": done,
            "date": m.get("match_date") or "", "n_sets": n_sets or (hs + as_),
            "n_points": pts or None, "periods": None, "ligue": ligue}


def jour(d: datetime) -> list:
    """Tous les matchs d'un jour (toutes ligues TT de scores24), parsés."""
    out = []
    for b in _get_jour(d):
        ligue = (b.get("league") or {}).get("name") or (b.get("league") or {}).get("slug")
        for m in b.get("matches") or []:
            g = _parse_match(m, ligue)
            if g:
                out.append(g)
    return out


def _auj() -> datetime:
    return datetime.now(timezone.utc)


def historique(jours: int = 14, ligue: str | None = None) -> list:
    """Matchs TERMINÉS des `jours` derniers jours (jour par jour, en parallèle).
    `ligue` (nom) filtre une ligue ; None = toutes."""
    fin = _auj()
    dates = [fin - timedelta(days=k) for k in range(jours)]
    with ThreadPoolExecutor(max_workers=8) as ex:
        lots = list(ex.map(jour, dates))
    vus, out = set(), []
    for lot in lots:
        for g in lot:
            if not (g["done"] and g["hs"] is not None):
                continue
            if ligue and g["ligue"] != ligue:
                continue
            k = (g["home"], g["away"], g["date"])
            if k not in vus:
                vus.add(k)
                out.append(g)
    out.sort(key=lambda g: g["date"] or "")
    return out


def a_venir(ligue: str | None = None) -> list:
    """Matchs À VENIR (non terminés) aujourd'hui/demain, avec les 2 joueurs."""
    out, vus = [], set()
    for d in (_auj(), _auj() + timedelta(days=1)):
        for g in jour(d):
            if g["done"] or not (g["home"] and g["away"]):
                continue
            if ligue and g["ligue"] != ligue:
                continue
            k = (g["home"], g["away"], g["date"])
            if k not in vus:
                vus.add(k)
                out.append(g)
    out.sort(key=lambda g: g["date"] or "")
    return out


def ligues_dispo(matches: list) -> list:
    return sorted({g["ligue"] for g in matches if g.get("ligue")})


def joueurs(matches: list) -> list:
    s = set()
    for g in matches:
        s.add(g["home"]); s.add(g["away"])
    return sorted(s)


def _median(xs: list):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else None


# Seuil de CONFIANCE : en dessous, le match est trop serré pour être fiable
# (backtest : sous ~62 % c'est pile ou face ; au-dessus, la TT Elite Series
# réussit ~66 %). On ne propose QUE les pronostics confiants.
SEUIL_CONFIANCE = 0.62


def entrainer_elo(hist: list) -> _sr.Elo:
    """Elo cumulé sur l'historique (pas d'avantage terrain au ping-pong)."""
    passes = sorted([g for g in hist if g.get("done")], key=lambda g: g.get("date") or "")
    return _sr.Elo(home_adv=0).entrainer(passes)


def pronostics_a_venir(hist: list, fixtures: list, elo: _sr.Elo | None = None) -> list:
    """Pronostic ELO (confiance = proba du favori) + forme + face-à-face + total
    de manches, pour chaque match à venir. `confiant` = proba ≥ SEUIL."""
    import math
    passes = sorted([g for g in hist if g.get("done")], key=lambda g: g.get("date") or "")
    med_s = _median([g["n_sets"] for g in passes if g.get("n_sets")]) or 4
    elo = elo or entrainer_elo(passes)
    out = []
    for g in fixtures:
        a, b = g["home"], g["away"]
        p = elo.proba(a, b, nuls=False)
        pr = {"home": a, "away": b, "p": p,
              "pick": "home" if p["home"] >= p["away"] else "away",
              "conf": max(p["home"], p["away"]),
              "forme_home": _sr.forme(passes, a), "forme_away": _sr.forme(passes, b),
              "h2h": _sr.face_a_face(passes, a, b),
              "rating_home": round(elo.rating(a)), "rating_away": round(elo.rating(b)),
              "date": g.get("date"), "ligue": g.get("ligue")}
        pr["confiant"] = pr["conf"] >= SEUIL_CONFIANCE

        def _moy(j):
            v = [x["n_sets"] for x in passes
                 if (x["home"] == j or x["away"] == j) and x.get("n_sets")][-12:]
            return sum(v) / len(v) if v else None
        ma, mb = _moy(a), _moy(b)
        if ma is not None and mb is not None:
            exp = (ma + mb) / 2.0
            pr["sets"] = {"exp": exp, "ligne": med_s + 0.5,
                          "over": 1.0 / (1.0 + math.exp(-(exp - (med_s + 0.5)) / 0.8))}
        else:
            pr["sets"] = None
        out.append(pr)
    out.sort(key=lambda x: -x["conf"])
    return out


def taux_confiants(hist: list, seuil: float = SEUIL_CONFIANCE,
                   maxi: int = 400, warmup: int = 200) -> dict | None:
    """Backtest HONNÊTE sur les seuls picks CONFIANTS (proba Elo ≥ seuil),
    walk-forward : Elo reconstruit match après match."""
    done = sorted([g for g in hist if g.get("done") and g.get("hs") is not None],
                  key=lambda g: g.get("date") or "")
    if len(done) < warmup + 40:
        return None
    elo = _sr.Elo(home_adv=0)
    ok = tot = vus = 0
    for g in done:
        vus += 1
        if vus > warmup:
            p = elo.proba(g["home"], g["away"], nuls=False)
            pf = max(p["home"], p["away"])
            if pf >= seuil:
                pick = "home" if p["home"] >= p["away"] else "away"
                reel = "home" if g["hs"] > g["as_"] else "away"
                ok += pick == reel
                tot += 1
        elo.maj(g["home"], g["away"], g["hs"], g["as_"])
    if not tot:
        return None
    return {"ok": ok, "tot": tot, "taux": ok / tot, "seuil": seuil}
