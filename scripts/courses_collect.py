"""Collecte des courses virtuelles Bet261 — Étape A du Palier 2 (19/10).

On enregistre chaque course À VENIR (cotes par partant), puis son ORDRE
D'ARRIVÉE une fois courue : re-GET `events/{id}` porte `result` (liste des n°,
du 1er au dernier) et `runners` (noms) APRÈS règlement — voir la mémoire
`racing-api-bet261`. En accumulant, on pourra faire une analyse INDÉPENDANTE des
cotes (biais RNG : un numéro / une piste sur-performe-t-il sa cote ?).

Tout en stdlib + SQLAlchemy, même base /data que le football (table `courses`
dédiée, les tables foot ne sont pas touchées).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import text

import courses_bet261 as cb

_EVENT = "https://hg-event-api-prod.sporty-tech.net/api/events"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_schema(engine) -> None:
    """Crée la table `courses` si absente (idempotent)."""
    with engine.begin() as cx:
        cx.execute(text("""
            CREATE TABLE IF NOT EXISTS courses (
                race_id        INTEGER PRIMARY KEY,
                discipline     INTEGER,
                track          TEXT,
                expected_start TEXT,
                expected_end   TEXT,
                cotes_json     TEXT,
                result_json    TEXT,
                names_json     TEXT,
                captured_at    TEXT,
                settled_at     TEXT
            )"""))


def upsert_upcoming(engine, discipline: int, races: list) -> int:
    """Insère les courses à venir pas encore vues (cotes figées à la capture).
    N'écrase jamais une course déjà en base (donc jamais son résultat)."""
    now = _now()
    n = 0
    with engine.begin() as cx:
        for r in races:
            rid = r.get("id")
            if not rid:
                continue
            if cx.execute(text("SELECT 1 FROM courses WHERE race_id=:i"),
                          {"i": rid}).first():
                continue
            cx.execute(text("""
                INSERT INTO courses (race_id, discipline, track, expected_start,
                                     expected_end, cotes_json, captured_at)
                VALUES (:i, :d, :t, :s, :e, :c, :n)"""),
                       {"i": rid, "d": discipline, "t": r.get("name"),
                        "s": r.get("start"), "e": r.get("end"),
                        "c": json.dumps(r.get("runners") or []), "n": now})
            n += 1
    return n


def fetch_result(event_id: int, timeout: float = 20.0) -> dict:
    """Re-GET `events/{id}` : {state, result:[int]|None, names:{int:str}|None}.
    `result` n'existe qu'une fois la course réglée (`state=="Settled"`)."""
    e = cb._get_json(f"{_EVENT}/{int(event_id)}", timeout)
    res = e.get("result")
    order = None
    if res:
        order = []
        for x in res:
            try:
                order.append(int(x))
            except (TypeError, ValueError):
                pass
    names = {}
    for r in e.get("runners") or []:
        try:
            names[int(r.get("number"))] = r.get("name")
        except (TypeError, ValueError):
            pass
    return {"state": e.get("state"), "result": order or None,
            "names": names or None}


def collect_results(engine, timeout: float = 20.0, limit: int = 60) -> int:
    """Pour les courses dont la fin est passée et sans résultat : va chercher
    l'ordre d'arrivée et l'enregistre. Rend le nombre de courses réglées."""
    now = _now()
    with engine.begin() as cx:
        rows = cx.execute(text("""
            SELECT race_id FROM courses
            WHERE result_json IS NULL AND expected_end IS NOT NULL
              AND expected_end < :now
            ORDER BY expected_end LIMIT :lim"""),
            {"now": now, "lim": limit}).fetchall()
    done = 0
    for (rid,) in rows:
        try:
            r = fetch_result(rid, timeout)
        except Exception:
            continue
        if not r["result"]:
            continue
        with engine.begin() as cx:
            cx.execute(text("""
                UPDATE courses SET result_json=:r, names_json=:n, settled_at=:t
                WHERE race_id=:i"""),
                {"r": json.dumps(r["result"]),
                 "n": json.dumps(r["names"], ensure_ascii=False) if r["names"] else None,
                 "t": now, "i": rid})
        done += 1
    return done


def collect_once(engine, timeout: float = 20.0) -> tuple:
    """Un cycle complet : capte les cotes à venir (2 disciplines) + récupère les
    arrivées des courses finies. Rend (nouvelles_courses, courses_reglees)."""
    ensure_schema(engine)
    new = 0
    for cid in cb.CATEGORIES.values():
        try:
            url = (f"{cb._BASE}?eventCategoryIds={cid}&offset=0&length=21"
                   f"&fetchEventBetTypesMode=1&l=fr")
            raw = cb.parse_events(cb._get_json(url, timeout))
            new += upsert_upcoming(engine, cid, raw)
        except Exception:
            continue
    settled = collect_results(engine, timeout)
    return new, settled
