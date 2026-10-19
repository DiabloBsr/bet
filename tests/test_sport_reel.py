"""Prédiction sport réel par Elo — maths Elo + parsing ESPN, sans réseau."""
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "scripts"))
import sport_reel as sr  # noqa: E402


# -------- parsing ESPN --------

def _payload(home, away, hs, as_, completed):
    return {"events": [{"date": "2026-01-18T14:00Z", "competitions": [{
        "status": {"type": {"completed": completed}},
        "competitors": [
            {"homeAway": "home", "team": {"displayName": home}, "score": str(hs)},
            {"homeAway": "away", "team": {"displayName": away}, "score": str(as_)},
        ]}]}]}


def test_parse_extrait_le_match():
    g = sr.parse_games(_payload("Arsenal", "Everton", 2, 0, True))[0]
    assert g["home"] == "Arsenal" and g["away"] == "Everton"
    assert g["hs"] == 2 and g["as_"] == 0 and g["done"] is True


def test_parse_match_a_venir_sans_score():
    data = {"events": [{"date": "2026-02-01T14:00Z", "competitions": [{
        "status": {"type": {"completed": False}},
        "competitors": [
            {"homeAway": "home", "team": {"displayName": "A"}, "score": None},
            {"homeAway": "away", "team": {"displayName": "B"}, "score": None}]}]}]}
    g = sr.parse_games(data)[0]
    assert g["done"] is False and g["hs"] is None


# -------- Elo --------

def test_attendu_symetrique_et_borne():
    assert abs(sr.Elo._attendu(1500, 1500) - 0.5) < 1e-9
    assert sr.Elo._attendu(1900, 1500) > 0.9          # bien plus fort -> proba haute
    assert 0 < sr.Elo._attendu(1500, 1900) < 0.1


def test_proba_somme_a_un():
    e = sr.Elo()
    p = e.proba("A", "B", nuls=True)
    assert abs(p["home"] + p["draw"] + p["away"] - 1.0) < 1e-9
    assert p["draw"] > 0
    p2 = e.proba("A", "B", nuls=False)
    assert p2["draw"] == 0.0 and abs(p2["home"] + p2["away"] - 1.0) < 1e-9


def test_avantage_terrain():
    e = sr.Elo(home_adv=60)
    # deux équipes égales : le domicile est favori grâce à l'avantage terrain.
    p = e.proba("A", "B", nuls=False)
    assert p["home"] > p["away"]


def test_maj_le_vainqueur_monte_le_perdant_descend():
    e = sr.Elo(k=20, home_adv=0)
    e.maj("A", "B", 3, 0)          # A (dom) gagne
    assert e.rating("A") > 1500 and e.rating("B") < 1500
    # somme conservée (k symétrique)
    assert abs((e.rating("A") - 1500) + (e.rating("B") - 1500)) < 1e-9


def test_entrainer_ignore_les_matchs_non_termines():
    games = [
        {"home": "A", "away": "B", "hs": 2, "as_": 0, "done": True},
        {"home": "A", "away": "C", "hs": None, "as_": None, "done": False},  # ignoré
    ]
    e = sr.Elo().entrainer(games)
    assert e.rating("A") > 1500
    assert "C" not in e.r            # le match non joué n'a rien mis à jour


# -------- tennis (1v1, groupings, filtre de tableau) --------

def _tournoi():
    def _m(a, b, a_gagne, done=True):
        return {"id": f"{a}-{b}", "date": "2026-01-18T00:00Z",
                "status": {"type": {"completed": done}},
                "competitors": [{"athlete": {"displayName": a}, "winner": a_gagne},
                                {"athlete": {"displayName": b}, "winner": not a_gagne}]}
    return {"events": [{"date": "2026-01-18T00:00Z", "groupings": [
        {"grouping": {"slug": "mens-singles"}, "competitions": [_m("Sinner", "Alcaraz", True)]},
        {"grouping": {"slug": "womens-singles"}, "competitions": [_m("Swiatek", "Gauff", False)]},
        {"grouping": {"slug": "mens-doubles"}, "competitions": [_m("X", "Y", True)]},
    ]}]}


def test_tennis_parse_simple_messieurs_seulement():
    g = sr.parse_games(_tournoi(), kind="tennis", slug="mens-singles")
    assert len(g) == 1
    assert g[0]["home"] == "Sinner" and g[0]["away"] == "Alcaraz"
    assert g[0]["hs"] == 1 and g[0]["as_"] == 0      # Sinner a gagné


def test_tennis_filtre_exclut_doubles_et_autre_genre():
    hommes = sr.parse_games(_tournoi(), kind="tennis", slug="mens-singles")
    femmes = sr.parse_games(_tournoi(), kind="tennis", slug="womens-singles")
    assert [x["home"] for x in hommes] == ["Sinner"]     # pas de doubles, pas de dames
    assert [x["home"] for x in femmes] == ["Swiatek"]
    # sans slug : tous les SIMPLES (hommes + dames), jamais les doubles
    tous = sr.parse_games(_tournoi(), kind="tennis")
    noms = {x["home"] for x in tous}
    assert noms == {"Sinner", "Swiatek"} and "X" not in noms


def test_predire_attache_proba_et_favori():
    e = sr.Elo()
    e.r = {"Fort": 1800, "Faible": 1400}
    out = sr.predire(e, [{"home": "Fort", "away": "Faible", "date": "d", "done": False}])
    assert out[0]["favori"] == "home"
    assert out[0]["p"]["home"] > out[0]["p"]["away"]
    assert out[0]["rating_home"] == 1800
