"""Collecte des courses (Étape A du Palier 2) : schéma, upsert, résultats.

Sans réseau : on monkeypatch l'accès HTTP. Base SQLite temporaire.
"""
import sys
from pathlib import Path

from sqlalchemy import create_engine, text

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "scripts"))
import courses_bet261 as cb  # noqa: E402
import courses_collect as cc  # noqa: E402


def _eng(tmp_path):
    return create_engine(f"sqlite:///{tmp_path/'t.db'}")


def _race(rid, track="PalmRidge", start="2026-10-19T21:40:00Z",
          end="2026-10-19T21:41:03Z", runners=None):
    return {"id": rid, "name": track, "start": start, "end": end,
            "runners": runners or [{"num": 1, "odds": 2.0}, {"num": 2, "odds": 4.0}]}


def test_schema_et_upsert_dedup(tmp_path):
    e = _eng(tmp_path)
    cc.ensure_schema(e)
    assert cc.upsert_upcoming(e, 8006, [_race(1), _race(2)]) == 2
    assert cc.upsert_upcoming(e, 8006, [_race(1), _race(3)]) == 1  # 1 déjà vu
    with e.begin() as cx:
        n = cx.execute(text("SELECT COUNT(*) FROM courses")).scalar()
        cotes = cx.execute(text("SELECT cotes_json FROM courses WHERE race_id=1")).scalar()
    assert n == 3
    assert '"odds": 2.0' in cotes or '"odds":2.0' in cotes


def test_fetch_result_parse(monkeypatch):
    # events/{id} réglé : result (ordre) + runners (noms).
    settled = {"state": "Settled", "result": ["10", "7", "8"],
               "runners": [{"number": 10, "name": "Royal Marriage"},
                           {"number": 7, "name": "Donegal"}]}
    monkeypatch.setattr(cb, "_get_json", lambda u, t=20.0: settled)
    r = cc.fetch_result(123)
    assert r["state"] == "Settled"
    assert r["result"] == [10, 7, 8]            # en int, ordre préservé
    assert r["names"][10] == "Royal Marriage"


def test_fetch_result_course_non_reglee(monkeypatch):
    monkeypatch.setattr(cb, "_get_json", lambda u, t=20.0: {"state": "Undisputed"})
    r = cc.fetch_result(1)
    assert r["result"] is None and r["names"] is None


def test_collect_results_enregistre_l_arrivee(monkeypatch, tmp_path):
    e = _eng(tmp_path)
    cc.ensure_schema(e)
    # une course dont la fin est passée, sans résultat
    cc.upsert_upcoming(e, 8006, [_race(1, end="2020-01-01T00:00:00Z")])
    monkeypatch.setattr(cc, "fetch_result",
                        lambda rid, timeout=20.0: {"state": "Settled",
                                                   "result": [2, 1], "names": {2: "B"}})
    assert cc.collect_results(e) == 1
    with e.begin() as cx:
        res = cx.execute(text("SELECT result_json FROM courses WHERE race_id=1")).scalar()
    assert res == "[2, 1]"
    # 2e passage : plus rien à régler
    assert cc.collect_results(e) == 0


def test_collect_results_ignore_le_futur(monkeypatch, tmp_path):
    e = _eng(tmp_path)
    cc.ensure_schema(e)
    cc.upsert_upcoming(e, 8006, [_race(1, end="2099-01-01T00:00:00Z")])
    appels = []
    monkeypatch.setattr(cc, "fetch_result",
                        lambda rid, timeout=20.0: appels.append(rid) or {"result": None})
    assert cc.collect_results(e) == 0
    assert appels == []                         # fin pas encore passée → pas d'appel
