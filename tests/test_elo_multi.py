"""Elo multi-saisons openfootball : normalisation des noms, calcul, robustesse.

Aucun réseau : on monkeypatche _gj pour renvoyer des saisons fabriquées."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import elo_multi as em  # noqa: E402
import sport_reel as sr  # noqa: E402


def test_norm_retire_suffixes_et_accents():
    assert em._norm("Arsenal FC") == "arsenal"
    assert em._norm("AFC Bournemouth") == "bournemouth"
    assert em._norm("Atlético de Madrid") == "atletico madrid"
    assert em._norm("1. FSV Mainz 05") == "mainz 05"


def _saison(matchs):
    return {"name": "X", "matches": [
        {"team1": a, "team2": b, "score": {"ft": [hs, as_]}} for a, b, hs, as_ in matchs]}


def test_elo_ligue_calcule_et_classe(monkeypatch):
    # A bat B et C systématiquement -> A doit finir en tête.
    s1 = _saison([("A FC", "B FC", 3, 0), ("A FC", "C FC", 2, 0), ("B FC", "C FC", 1, 1)])
    monkeypatch.setattr(em, "SAISONS", ["2024-25"])
    monkeypatch.setattr(em, "_gj", lambda url, timeout=20.0, essais=2: _saison(
        [("A FC", "B FC", 3, 0), ("A FC", "C FC", 2, 0), ("B FC", "C FC", 1, 1)]))
    r = em.elo_ligue("en.1")
    assert r["a"] > r["b"] and r["a"] > r["c"]          # noms normalisés en clés


def test_elo_ligue_robuste_aux_entrees_malformees(monkeypatch):
    mauvais = {"name": "X", "matches": [
        ["pas", "un", "dict"],                          # élément liste
        {"team1": "A FC", "team2": "B FC", "score": [1, 0]},   # score = liste
        {"team1": "A FC", "team2": "B FC", "score": {"ft": [2, 0]}},  # bon
    ]}
    monkeypatch.setattr(em, "SAISONS", ["2024-25"])
    monkeypatch.setattr(em, "_gj", lambda *a, **k: mauvais)
    r = em.elo_ligue("en.1")                            # ne lève pas
    assert r["a"] > r["b"]                              # seul le match valide compte


def test_ratings_pour_mappe_sur_les_noms_espn(monkeypatch):
    monkeypatch.setattr(em, "SAISONS", ["2024-25"])
    monkeypatch.setattr(em, "_gj", lambda *a, **k: _saison(
        [("Arsenal FC", "Chelsea FC", 2, 0)]))
    out = em.ratings_pour("soccer/eng.1", ["Arsenal", "Chelsea", "Inconnu"])
    assert set(out) == {"Arsenal", "Chelsea"}           # ESPN -> openfootball
    assert out["Arsenal"] > out["Chelsea"]


def test_ratings_pour_ligue_non_couverte():
    assert em.ratings_pour("basketball/nba", ["Lakers"]) == {}


def test_ratings_pour_vide_si_source_hs(monkeypatch):
    monkeypatch.setattr(em, "_gj", lambda *a, **k: None)
    assert em.ratings_pour("soccer/eng.1", ["Arsenal"]) == {}


def test_pronostic_melange_elo_ext_oriente_le_resultat():
    passes = ([sr_g("A", f"o{i}", 1, 0, f"a{i}") for i in range(6)]
              + [sr_g("B", f"p{i}", 0, 1, f"b{i}") for i in range(6)])
    # Même match, mais la 2e référence favorise A dans un cas, B dans l'autre.
    pa = sr.pronostic_forme(passes, "A", "B", nuls=True, home_adv=0,
                            elo_ext={"A": 1900, "B": 1400})
    pb = sr.pronostic_forme(passes, "A", "B", nuls=True, home_adv=0,
                            elo_ext={"A": 1400, "B": 1900})
    assert pa["p"]["home"] > pb["p"]["home"]            # la 2e réf oriente la proba
    assert abs(sum(pa["p"].values()) - 1.0) < 1e-9
    # Absente pour une équipe inconnue -> aucun effet (pas d'erreur).
    sans = sr.pronostic_forme(passes, "A", "B", nuls=True, home_adv=0,
                              elo_ext={"A": 1900})      # B manquant
    base = sr.pronostic_forme(passes, "A", "B", nuls=True, home_adv=0)
    assert sans["p"]["home"] == base["p"]["home"]


def sr_g(h, a, hs, as_, d):
    return {"home": h, "away": a, "hs": hs, "as_": as_, "done": True, "date": d}
