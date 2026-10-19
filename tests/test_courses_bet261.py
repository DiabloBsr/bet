"""Module courses virtuelles Bet261 (Palier 1) — parsing + dévig, sans réseau.

Les RÉSULTATS ne sont pas exposés par l'API : ce module ne lit que les cotes à
venir et en dérive une proba dévigée. On teste le parsing et la dévig, pas le
HTTP.
"""
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "scripts"))
import courses_bet261 as cb  # noqa: E402


def _ev(name, start, items):
    return {"name": name, "id": 1, "expectedStart": start,
            "eventBetTypes": [{"name": "Vainqueur", "eventBetTypeItems": items}]}


def _it(short, odds, active=True):
    return {"shortName": short, "odds": odds, "active": active, "runnerId": 1}


def test_categories_connues():
    assert cb.CATEGORIES == {"Dashing Derby": 8006, "Platinum Hounds": 8005}


def test_parse_extrait_partants_et_trie_par_heure():
    data = {"0": _ev("Milnerton", "2026-10-19T21:26:00Z",
                      [_it("1", 2.0), _it("2", 4.0)]),
            "1": _ev("Bakersfield", "2026-10-19T21:20:00Z",
                     [_it("1", 3.0), _it("2", 3.0), _it("3", 6.0)])}
    races = cb.parse_events(data)
    assert [r["name"] for r in races] == ["Bakersfield", "Milnerton"]  # trié heure
    assert races[0]["runners"] == [{"num": 1, "odds": 3.0}, {"num": 2, "odds": 3.0},
                                   {"num": 3, "odds": 6.0}]


def test_parse_ignore_partants_sans_cote_ou_inactifs():
    data = {"0": _ev("X", "2026-10-19T21:00:00Z",
                     [_it("1", 2.0), _it("2", 0), _it("3", None),
                      _it("4", 5.0, active=False), {"shortName": "x", "odds": 2.0}])}
    races = cb.parse_events(data)
    assert [r["num"] for r in races[0]["runners"]] == [1]  # seul le 1 est valable


def test_parse_ecarte_course_sans_marche():
    data = {"0": {"name": "Y", "expectedStart": "z", "eventBetTypes": []}}
    assert cb.parse_events(data) == []


def test_devig_normalise_a_un_et_classe_le_favori():
    race = {"name": "Z", "start": "t",
            "runners": [{"num": 1, "odds": 2.0}, {"num": 2, "odds": 4.0},
                        {"num": 3, "odds": 4.0}]}
    c = cb.course_cotes(race)
    ps = [r["p_devig"] for r in c["runners"]]
    assert abs(sum(ps) - 1.0) < 1e-9                 # renormalisé à 1
    assert c["runners"][0]["num"] == 1               # favori = plus courte cote
    assert c["favori"]["num"] == 1
    # 1/2 : 1/4 : 1/4 -> 0.5 : 0.25 : 0.25
    assert abs(c["runners"][0]["p_devig"] - 0.5) < 1e-9


def test_course_cotes_sans_partant():
    c = cb.course_cotes({"name": "V", "start": "t", "runners": []})
    assert c["runners"] == [] and c["favori"] is None
