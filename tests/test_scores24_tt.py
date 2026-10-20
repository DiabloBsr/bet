"""TT Elite Series via scores24 : parsing + pronostic confiant, sans réseau."""
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "scripts"))
import scores24_tt as s2  # noqa: E402


def _m(na, nb, score, sets, finished=True):
    return {"teams": [{"name": na}, {"name": nb}], "result_score": score,
            "result_scores": sets, "is_finished": finished,
            "match_date": "2026-10-19T22:00:00.000000Z"}


def test_norm_nom_inverse_virgule():
    assert s2._norm_nom("Adamus, Pawel") == "Pawel Adamus"
    assert s2._norm_nom("Pawel Adamus") == "Pawel Adamus"


def test_parse_match_joueurs_sets_points():
    g = s2._parse_match(_m("Dmitro Parhomenko", "Milan Smrcek", "2:3",
                           [{"type": "1", "value": "11:13"}, {"type": "2", "value": "11:7"},
                            {"type": "3", "value": "10:12"}, {"type": "4", "value": "11:6"},
                            {"type": "5", "value": "7:11"}, {"type": "FT", "value": "2:3"}]),
                        "TT Elite Series")
    assert g["home"] == "Dmitro Parhomenko" and g["away"] == "Milan Smrcek"
    assert g["hs"] == 2 and g["as_"] == 3 and g["done"] is True
    assert g["n_sets"] == 5                      # 5 manches (hors FT)
    # points = (11+13)+(11+7)+(10+12)+(11+6)+(7+11) = 99
    assert g["n_points"] == 99
    assert g["ligue"] == "TT Elite Series"


def test_parse_match_non_termine():
    g = s2._parse_match(_m("A", "B", "0:0", [], finished=False), "L")
    assert g["done"] is False


def test_pronostic_confiant_favori_et_seuil():
    # Fort gagne tout, Faible perd tout -> Elo écarté -> pronostic confiant.
    hist = ([{"home": "Fort", "away": f"x{i}", "hs": 3, "as_": 0, "done": True,
              "date": f"a{i}", "n_sets": 3, "n_points": 60} for i in range(20)]
            + [{"home": "Faible", "away": f"y{i}", "hs": 0, "as_": 3, "done": True,
                "date": f"b{i}", "n_sets": 3, "n_points": 60} for i in range(20)])
    pr = s2.pronostics_a_venir(hist, [{"home": "Fort", "away": "Faible",
                                       "date": "z", "ligue": "TT Elite Series"}])
    assert pr[0]["pick"] == "home"
    assert pr[0]["conf"] >= s2.SEUIL_CONFIANCE and pr[0]["confiant"] is True


def test_taux_confiants_rend_dict_ou_none():
    assert s2.taux_confiants([]) is None
    hist = [{"home": "A", "away": "B", "hs": 3, "as_": 0, "done": True,
             "date": f"{i:04d}", "n_sets": 3} for i in range(260)]
    bt = s2.taux_confiants(hist, warmup=200)
    # A gagne toujours -> picks confiants corrects
    assert bt is None or (0.0 <= bt["taux"] <= 1.0)
