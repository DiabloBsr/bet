"""Tennis de table WTT : parsing des résultats officiels + pronostic.

Sans réseau : on fournit une structure officialresult.json factice.
"""
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "scripts"))
import wtt_tt as tt  # noqa: E402


def _doc(dc, se, pa, pb, overall, gamescores, bestof=5, dt="01/06/2025 00:30:00"):
    return {"documentCode": dc, "subEventType": se,
            "match_card": {"competitiors": [{"competitiorName": pa}, {"competitiorName": pb}],
                           "overallScores": overall, "gameScores": gamescores,
                           "resultStatus": "OFFICIAL",
                           "matchConfig": {"bestOfXGames": bestof},
                           "matchDateTime": {"startDateUTC": dt}}}


def test_parse_simple_manches_et_points():
    data = [_doc("TTEMSINGLES--U19FNL", "Men Singles", "A", "B", "3-1",
                 "11-3,11-7,6-11,11-8,0-0")]
    g = tt.parse_results(data)[0]
    assert g["home"] == "A" and g["away"] == "B"
    assert g["hs"] == 3 and g["as_"] == 1
    assert g["n_sets"] == 4                       # 3+1 manches jouées (hors 0-0)
    # points = 11+3 +11+7 +6+11 +11+8 = 68
    assert g["n_points"] == 68
    assert g["periods"] == 5 and g["done"] is True


def test_parse_ignore_doubles_et_equipes():
    data = [_doc("TTEMDOUBLES--", "Men Doubles", "A", "B", "3-0", "11-0,11-0,11-0,0-0,0-0"),
            {"documentCode": "TTEMTEAM--", "subEventType": "Men Team", "match_card": {}}]
    assert tt.parse_results(data) == []


def test_parse_ignore_match_sans_vainqueur_ou_non_officiel():
    nul = _doc("TTEWSINGLES--", "Women Singles", "A", "B", "2-2", "11-9,9-11,11-7,7-11,0-0")
    data = [nul]
    assert tt.parse_results(data) == []           # 2-2 : pas de vainqueur -> écarté


def test_date_iso():
    assert tt._iso("01/06/2025 00:30:00") == "2025-01-06T00:30:00Z"
    assert tt._iso("2025-01-05T00:00:00") == "2025-01-05T00:00:00Z"
    assert tt._iso("") == ""


def test_joueurs_tries():
    m = [{"home": "Zoe", "away": "Ana"}, {"home": "Ana", "away": "Bob"}]
    assert tt.joueurs(m) == ["Ana", "Bob", "Zoe"]


def test_pronostic_meilleure_forme_et_marche_manches():
    # A gagne tout, B perd tout -> pronostic A ; marché manches présent.
    m = ([{"home": "A", "away": f"x{i}", "hs": 3, "as_": 0, "done": True,
           "date": f"a{i}", "n_sets": 3, "n_points": 50, "periods": 5} for i in range(6)]
         + [{"home": "B", "away": f"y{i}", "hs": 0, "as_": 3, "done": True,
             "date": f"b{i}", "n_sets": 3, "n_points": 50, "periods": 5} for i in range(6)])
    p = tt.pronostic(m, "A", "B")
    assert p["pick"] == "home" and p["conf"] > 0.5
    assert p["sets"] is not None and "points" not in p   # points retiré
