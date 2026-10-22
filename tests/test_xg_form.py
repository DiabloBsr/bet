"""xG-forme (Understat) embarquée : mappage des noms et mélange au pronostic."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import xg_form as xg  # noqa: E402
import sport_reel as sr  # noqa: E402


def _bundle(tmp_path):
    p = tmp_path / "xg.json"
    p.write_text(json.dumps({"genere_le": "2026-10-22", "xform": {"soccer/eng.1": {
        "Brighton": 0.60, "Tottenham": 0.52, "Manchester City": 0.63,
        "Manchester United": 0.49, "Leeds": 0.51}}}), encoding="utf-8")
    return str(p)


def test_mappe_les_variantes_de_noms(tmp_path):
    p = _bundle(tmp_path)
    out = xg.xform_pour("soccer/eng.1",
                        ["Brighton & Hove Albion", "Tottenham Hotspur", "Leeds United"], p)
    assert out == {"Brighton & Hove Albion": 0.60, "Tottenham Hotspur": 0.52,
                   "Leeds United": 0.51}


def test_pas_de_collision_city_united(tmp_path):
    p = _bundle(tmp_path)
    out = xg.xform_pour("soccer/eng.1", ["Manchester City", "Manchester United"], p)
    assert out["Manchester City"] == 0.63 and out["Manchester United"] == 0.49


def test_ligue_non_couverte_ou_absente(tmp_path):
    p = _bundle(tmp_path)
    assert xg.xform_pour("soccer/uefa.champions", ["Real Madrid"], p) == {}
    assert xg.xform_pour("basketball/nba", ["Lakers"], p) == {}
    assert xg.xform_pour("soccer/eng.1", ["Arsenal"], "/inexistant.json") == {}


def test_norm_retire_suffixes():
    assert xg._norm("Mainz 05") == "mainz"
    assert xg._norm("Parma Calcio 1913") == "parma"


def test_melange_xform_oriente_la_proba():
    passes = ([sr_g("A", f"o{i}", 1, 0, f"a{i}") for i in range(6)]
              + [sr_g("B", f"p{i}", 0, 1, f"b{i}") for i in range(6)])
    pa = sr.pronostic_forme(passes, "A", "B", nuls=True, home_adv=0,
                            xform={"A": 0.75, "B": 0.35})
    pb = sr.pronostic_forme(passes, "A", "B", nuls=True, home_adv=0,
                            xform={"A": 0.35, "B": 0.75})
    assert pa["p"]["home"] > pb["p"]["home"]           # la xG-forme oriente la proba
    assert abs(sum(pa["p"].values()) - 1.0) < 1e-9
    # xform incomplète (une équipe manquante) -> aucun effet
    base = sr.pronostic_forme(passes, "A", "B", nuls=True, home_adv=0)
    sans = sr.pronostic_forme(passes, "A", "B", nuls=True, home_adv=0, xform={"A": 0.9})
    assert sans["p"]["home"] == base["p"]["home"]


def sr_g(h, a, hs, as_, d):
    return {"home": h, "away": a, "hs": hs, "as_": as_, "done": True, "date": d}
