"""TT Elite Series (tt-series.com) : parsing joueurs + conversion des matchs.

Aucun réseau : on monkeypatch _get pour renvoyer des réponses fidèles au format
réel de l'API tt/v1 (zawodnicy, mecze-excel)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import tt_elite as te  # noqa: E402

_ZAWODNICY = [
    {"id": "10", "nazwa": "Alpha A", "elo": "1600", "poziom": "1"},
    {"id": "20", "nazwa": "Beta B", "elo": "1400", "poziom": "1"},
    {"id": "30", "nazwa": "Gamma G", "elo": "bizarre", "poziom": "3"},  # elo illisible
]
# Alpha bat Beta ; Gamma bat Alpha.
_MECZE = [
    {"id": "901", "player1_id": "10", "player2_id": "20",
     "match_date": "2026-10-19 00:00:00", "winner_id": "10"},
    {"id": "902", "player1_id": "30", "player2_id": "10",
     "match_date": "2026-10-20 00:00:00", "winner_id": "30"},
    {"id": "903", "player1_id": "20", "player2_id": "999",   # 999 inconnu -> ignoré
     "match_date": "2026-10-20 00:00:00", "winner_id": "20"},
]


def _fake_get(url, timeout=40.0):
    if "zawodnicy" in url:
        return _ZAWODNICY
    if "mecze-excel" in url:
        return _MECZE
    raise AssertionError("url inattendue: " + url)


def test_joueurs_parse_nom_et_elo(monkeypatch):
    monkeypatch.setattr(te, "_get", _fake_get)
    js = te.joueurs()
    assert js["10"]["nom"] == "Alpha A" and js["10"]["elo"] == 1600.0
    assert js["30"]["elo"] is None                      # elo illisible -> None
    assert js["20"]["div"] == "1"


def test_historique_convertit_vainqueur_en_score(monkeypatch):
    monkeypatch.setattr(te, "_get", _fake_get)
    H = te.historique()
    # 903 (adversaire inconnu) est écarté -> 2 matchs exploitables.
    assert len(H) == 2
    m = next(x for x in H if x["id"] == "901")
    assert m["home"] == "Alpha A" and m["away"] == "Beta B"
    assert m["hs"] == 1 and m["as_"] == 0 and m["done"] is True   # Alpha gagne
    assert m["date"] == "2026-10-19"
    g = next(x for x in H if x["id"] == "902")
    assert g["home"] == "Gamma G" and g["hs"] == 1 and g["as_"] == 0  # Gamma gagne


def test_historique_trie_par_date(monkeypatch):
    monkeypatch.setattr(te, "_get", _fake_get)
    H = te.historique()
    assert [x["date"] for x in H] == sorted(x["date"] for x in H)


def test_pronostic_rend_un_pick_et_une_confiance(monkeypatch):
    monkeypatch.setattr(te, "_get", _fake_get)
    H = te.historique()
    p = te.pronostic(H, "Alpha A", "Beta B")
    assert p["pick"] in ("home", "away")
    assert 0.0 <= p["conf"] <= 1.0
    assert p["p"]["draw"] == 0.0                        # pas de nul au tennis de table


def test_joueurs_vide_si_injoignable(monkeypatch):
    def boom(url, timeout=40.0):
        raise OSError("réseau")
    monkeypatch.setattr(te, "_get", boom)
    assert te.joueurs() == {}
    assert te.historique() == []


_JS = {"1": {"nom": "Glanowski Waldemar", "elo": 1500, "div": "1"},
       "2": {"nom": "Kurek Pawel", "elo": 1400, "div": "1"},
       "3": {"nom": "Baron Mariusz", "elo": 1450, "div": "1"}}


def test_trouver_joueur_gere_virgule_et_initiale():
    idx = te._index_noms(_JS)
    assert te.trouver_joueur(idx, "Glanowski, Waldemar") == "Glanowski Waldemar"
    assert te.trouver_joueur(idx, "Kurek, Pawel") == "Kurek Pawel"
    assert te.trouver_joueur(idx, "Kurek, P.") == "Kurek Pawel"       # initiale
    assert te.trouver_joueur(idx, "Inconnu, Xavier") is None


def test_deux_cotes_coupe_la_ligne():
    assert te._deux_cotes("Glanowski, Waldemar - Kurek, Pawel") == (
        "Glanowski, Waldemar", "Kurek, Pawel")
    assert te._deux_cotes("sans separateur") is None


def test_pronostics_liste_reconnait_et_signale(monkeypatch):
    monkeypatch.setattr(te, "_get", _fake_get)
    hist = te.historique()                       # Alpha/Beta/Gamma (cf. _MECZE)
    js = {"10": {"nom": "Alpha A", "elo": 1600, "div": "1"},
          "20": {"nom": "Beta B", "elo": 1400, "div": "1"}}
    # Bruit sans « - » (odds, labels) ignoré ; match avec joueur inconnu signalé.
    bloc = "Alpha, A - Beta, B\n1  2,80  2  1,29\nAlpha, A - Inconnu, Z"
    pronos, inconnus = te.pronostics_liste(hist, js, bloc)
    assert len(pronos) == 1 and {pronos[0]["home"], pronos[0]["away"]} == {"Alpha A", "Beta B"}
    assert inconnus == ["Alpha, A - Inconnu, Z"]     # 1 joueur introuvable → signalé


def test_charger_bundle_prefere_le_fichier(tmp_path, monkeypatch):
    import json
    p = tmp_path / "b.json"
    p.write_text(json.dumps({"genere_le": "2026-10-20",
                             "joueurs": {"1": {"nom": "X Y", "elo": 1500, "div": "1"}},
                             "matchs": [{"id": "1", "date": "2026-10-19", "home": "X Y",
                                         "away": "Z W", "hs": 1, "as_": 0, "done": True}]}),
                encoding="utf-8")
    js, hist, date = te.charger_bundle(str(p))
    assert js and hist and date == "2026-10-20"
    # charger() doit prendre le bundle sans TOUCHER au réseau
    def boom(*a, **k):
        raise AssertionError("ne doit pas appeler le réseau")
    monkeypatch.setattr(te, "joueurs", boom)
    js2, hist2, src = te.charger(str(p))
    assert len(js2) == 1 and "embarqu" in src


def test_charger_bundle_absent_rend_none():
    assert te.charger_bundle("/chemin/inexistant.json") == (None, None, None)


def test_extraire_heure_en_tete():
    assert te._extraire_heure("11:10 Alpha, A - Beta, B") == ("11:10", "Alpha, A - Beta, B")
    assert te._extraire_heure("9h05 X - Y") == ("09:05", "X - Y")
    assert te._extraire_heure("Alpha, A - Beta, B") == (None, "Alpha, A - Beta, B")


def test_pronostics_liste_attache_l_heure(monkeypatch):
    monkeypatch.setattr(te, "_get", _fake_get)
    hist = te.historique()
    js = {"10": {"nom": "Alpha A", "elo": 1600, "div": "1"},
          "20": {"nom": "Beta B", "elo": 1400, "div": "1"}}
    pronos, _ = te.pronostics_liste(hist, js, "11:10 Alpha, A - Beta, B")
    assert pronos and pronos[0]["time"] == "11:10"


def test_extraire_matchs_bloc_bet261_desordonne():
    # Format réel Bet261 : heure APRÈS le match, cotes et libellés à ignorer.
    idx = te._index_noms({"1": {"nom": "Alpha A"}, "2": {"nom": "Beta B"},
                          "3": {"nom": "Gamma C"}, "4": {"nom": "Delta D"}})
    bloc = ("Alpha, A - Beta, B\n"
            "International  -  TT Elite Series   LIVE\n"
            "1   2,80    2   1,29    +7\n"
            "11:10 Aujourd'hui\n"
            "Gamma, C - Delta, D\n"
            "11:20 Aujourd'hui")
    matchs, inconnus = te.extraire_matchs(idx, bloc)
    assert len(matchs) == 2 and inconnus == []       # libellé/cotes ignorés
    assert matchs[0] == {"time": "11:10", "a": "Alpha A", "b": "Beta B"}
    assert matchs[1]["time"] == "11:20"


def test_extraire_matchs_signale_joueur_introuvable():
    idx = te._index_noms({"1": {"nom": "Alpha A"}})
    matchs, inconnus = te.extraire_matchs(idx, "Alpha, A - Inconnu, Z")
    assert matchs == [] and inconnus == ["Alpha, A - Inconnu, Z"]
