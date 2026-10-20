"""Tennis de table RÉEL — données World Table Tennis (WTT), SANS cote.

Source trouvée le 20/10 : les résultats OFFICIELS de chaque tournoi sont publiés
en JSON (compressé Brotli) sur le CDN WTT, accessibles côté serveur avec les
en-têtes du site. Chaque match porte les joueurs, les manches gagnées
(overallScores « 3-2 »), le score manche par manche (gameScores) et le format
(best-of). On en tire : vainqueur, nombre de MANCHES, total de POINTS.

On réutilise le moteur de `sport_reel` (Elo, forme, face-à-face, pronostic,
backtest, total de manches). Voir mémoire `racing-api-bet261` (sources de données).
"""
from __future__ import annotations

import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import brotli

import sport_reel as _sr

_H = {
    "User-Agent": "Mozilla/5.0",
    "Origin": "https://www.worldtabletennis.com",
    "Referer": "https://www.worldtabletennis.com/",
}
_CAL = "https://www.worldtabletennis.com/assets/json/{year}_eventcalendar.json"
_RES = ("https://wtt-web-frontdoor-cthahjeqhbh6aqe3.a01.azurefd.net/"
        "websitearchivedresults/{eid}/officialresult/officialresult.json")


def _get(url: str, timeout: float = 25.0):
    """GET JSON WTT, en décompressant le Brotli (le CDN sert `Content-Encoding: br`)."""
    req = urllib.request.Request(url, headers=_H)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        enc = (r.headers.get("Content-Encoding") or "").lower()
    if "br" in enc:
        raw = brotli.decompress(raw)
    return json.loads(raw.decode("utf-8"))


def _iso(s) -> str:
    """Date WTT → ISO UTC. Formats vus : « MM/DD/YYYY HH:MM:SS » et ISO."""
    if not s:
        return ""
    for fmt in ("%m/%d/%Y %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            return datetime.strptime(str(s), fmt).strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            pass
    return str(s)


def evenements(annees) -> list:
    """Tournois des années données (calendrier WTT) : {id, name, end, type}."""
    out = []
    for y in annees:
        try:
            cal = _get(_CAL.format(year=y))
        except Exception:
            continue
        if isinstance(cal, list) and cal and isinstance(cal[0], dict):
            rows = cal[0].get("rows") or []
        elif isinstance(cal, dict):
            rows = cal.get("rows") or next((v for v in cal.values() if isinstance(v, list)), [])
        else:
            rows = []
        for e in rows:
            eid, end = e.get("EventId"), e.get("EndDateTime")
            if eid and end:
                out.append({"id": eid, "name": e.get("EventName"),
                            "end": _iso(end), "type": e.get("EventType")})
    out.sort(key=lambda e: e["end"])
    return out


def parse_results(data) -> list:
    """officialresult.json → matchs de SIMPLE : {home, away, hs, as_ (manches
    gagnées), done, date, n_sets (manches), n_points (total points), periods}."""
    out = []
    for doc in data if isinstance(data, list) else []:
        dc = doc.get("documentCode") or ""
        if "SINGLES" not in dc or "Singles" not in (doc.get("subEventType") or ""):
            continue
        mc = doc.get("match_card")
        if not mc:
            continue
        comps = mc.get("competitiors") or []        # (sic : typo de l'API WTT)
        if len(comps) != 2:
            continue
        na, nb = comps[0].get("competitiorName"), comps[1].get("competitiorName")
        if not na or not nb:
            continue
        ov = str(mc.get("overallScores") or mc.get("resultOverallScores") or "").split("-")
        if len(ov) != 2:
            continue
        try:
            hs, as_ = int(ov[0]), int(ov[1])
        except ValueError:
            continue
        if mc.get("resultStatus") != "OFFICIAL" or hs == as_:
            continue
        # gameScores est REMBOURRÉ à best-of (ex 5 cases), les manches non jouées
        # valant « 0-0 ». Manches réellement jouées = hs + as_ ; on ne somme les
        # points que des manches jouées (on ignore les « 0-0 »).
        n_sets = hs + as_
        pts = 0
        for g in str(mc.get("gameScores") or "").split(","):
            pair = g.split("-")
            if len(pair) != 2:
                continue
            try:
                x, y = int(pair[0]), int(pair[1])
            except ValueError:
                continue
            if x == 0 and y == 0:
                continue
            pts += x + y
        periods = (mc.get("matchConfig") or {}).get("bestOfXGames")
        dt = (mc.get("matchDateTime") or {}).get("startDateUTC") or doc.get("startDateLocal")
        out.append({"home": na, "away": nb, "hs": hs, "as_": as_, "done": True,
                    "date": _iso(dt), "n_sets": n_sets, "n_points": pts or None,
                    "periods": periods})
    return out


def historique(annees=(2025, 2026), max_events: int = 25, timeout: float = 25.0) -> list:
    """Matchs de simple des `max_events` tournois TERMINÉS les plus récents."""
    aujourdhui = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    termines = [e for e in evenements(annees) if e["end"] and e["end"][:10] <= aujourdhui]
    recents = termines[-max_events:]

    def _work(e):
        try:
            return parse_results(_get(_RES.format(eid=e["id"]), timeout))
        except Exception:
            return []

    matches = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        for lot in ex.map(_work, recents):
            matches += lot
    matches.sort(key=lambda g: g.get("date") or "")
    return matches


def joueurs(matches: list) -> list:
    """Liste triée des joueurs présents dans l'historique."""
    s = set()
    for g in matches:
        s.add(g["home"]); s.add(g["away"])
    return sorted(s)


def _median(xs: list):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else None


def _marche_total(passes: list, a: str, b: str, champ: str, ligne: float,
                  sigma: float, n: int = 12) -> dict | None:
    """Marché « total » générique (manches ou points) : moyenne récente de
    `champ` pour chaque joueur → total attendu, plus/moins la `ligne`."""
    import math

    def _moy(j):
        vals = [g[champ] for g in passes
                if (g["home"] == j or g["away"] == j) and g.get(champ)][-n:]
        return sum(vals) / len(vals) if vals else None
    ma, mb = _moy(a), _moy(b)
    if ma is None or mb is None:
        return None
    exp = (ma + mb) / 2.0
    return {"exp": exp, "ligne": ligne,
            "over": 1.0 / (1.0 + math.exp(-(exp - ligne) / sigma))}


def pronostic(matches: list, a: str, b: str) -> dict:
    """Pronostic d'un duel a vs b : forme + face-à-face (via sport_reel) + total
    de MANCHES. Ligne de manches = MÉDIANE de l'historique + 0,5 (robuste). Pas
    de nul, pas d'avantage terrain.

    NB : le total de POINTS n'est PAS exposé — trop pollué par les abandons /
    forfaits fréquents en tournois jeunes (score partiel) pour une ligne fiable."""
    passes = sorted([g for g in matches if g.get("done")], key=lambda g: g.get("date") or "")
    med_s = _median([g["n_sets"] for g in passes if g.get("n_sets")]) or 4
    pr = _sr.pronostic_forme(passes, a, b, nuls=False, home_adv=0)
    pr["sets"] = _marche_total(passes, a, b, "n_sets", med_s + 0.5, 0.8)
    return pr
