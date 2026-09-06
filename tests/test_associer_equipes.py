"""Rapprochement des noms lus sur une capture avec les vrais noms de la base.

On n'affiche JAMAIS le texte brut de l'OCR : « Fulharn » doit devenir « Fulham ».
Ce rapprochement corrige l'orthographe ET donne l'identité réelle des équipes,
sans quoi le face-à-face à la même cote serait impossible.

Le risque à verrouiller : accepter un rapprochement douteux. Mieux vaut ne rien
identifier que désigner la mauvaise équipe — le face-à-face affiché serait alors
celui d'un autre duo, sans que rien ne l'indique.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import predict_trio as pt  # noqa: E402

LG = "InstantLeague-8035"
EQUIPES = ["Fulham", "C. Palace", "Manchester Red", "Manchester Blue", "Wolverhampton",
           "London Reds", "Spurs", "Sunderland", "Bournemouth", "N. Forest", "Leeds",
           "Everton", "Burnley", "Brighton", "London Blues", "A. Villa", "Liverpool",
           "Brentford", "West Ham", "Newcastle"]


@pytest.fixture
def eng(tmp_path):
    db = tmp_path / "e.db"
    c = sqlite3.connect(db)
    c.executescript("""
        CREATE TABLE events (id INTEGER PRIMARY KEY, competition TEXT, team_a TEXT,
          team_b TEXT, round_info TEXT, expected_start TEXT);
        CREATE TABLE odds_snapshots (id INTEGER PRIMARY KEY, event_id INTEGER,
          odds_home REAL, odds_draw REAL, odds_away REAL, extra_markets TEXT);
        CREATE TABLE results (id INTEGER PRIMARY KEY, event_id INTEGER,
          score_a INTEGER, score_b INTEGER);
    """)
    for i in range(0, len(EQUIPES), 2):
        c.execute("INSERT INTO events (id, competition, team_a, team_b, expected_start) "
                  "VALUES (?,?,?,?,?)", (i + 1, LG, EQUIPES[i], EQUIPES[i + 1],
                                         "2026-01-01 10:00:00"))
    c.commit(); c.close()
    return create_engine(f"sqlite:///{db}")


def test_noms_propres_reconnus_a_lidentique(eng):
    r = pt.associer_equipes(eng, "Fulham / C. Palace", [LG])
    assert (r["home"], r["away"]) == ("Fulham", "C. Palace")
    assert r["score"] == 1.0


@pytest.mark.parametrize("brut,attendu", [
    ("Fulharn / C, Palace", ("Fulham", "C. Palace")),
    ("Manchesler Red / Manchester Blue", ("Manchester Red", "Manchester Blue")),
    ("WOLVERHAMPTON / LONDON REDS", ("Wolverhampton", "London Reds")),
    ("london blues / a. villa", ("London Blues", "A. Villa")),
    ("Liverpool\nBrentford", ("Liverpool", "Brentford")),
])
def test_ocr_abime_est_rattrape(eng, brut, attendu):
    """L'OCR confond rn/m, virgule/point, casse : le nom affiché doit rester juste."""
    r = pt.associer_equipes(eng, brut, [LG])
    assert (r["home"], r["away"]) == attendu, f"{brut!r} mal rapproché"


def test_deux_manchester_ne_sont_pas_confondus(eng):
    """Le piège : deux noms très proches. Ils ne doivent pas s'échanger."""
    r = pt.associer_equipes(eng, "Manchester Blue / Manchester Red", [LG])
    assert r["home"] == "Manchester Blue" and r["away"] == "Manchester Red"


def test_texte_sans_rapport_nidentifie_rien(eng):
    """Mieux vaut ne rien identifier que désigner la mauvaise équipe : le
    face-à-face affiché serait celui d'un autre duo."""
    for brut in ("xyz / qqq", "", "   ", "12:45 / 3G 70%"):
        r = pt.associer_equipes(eng, brut, [LG])
        assert r["home"] is None or r["away"] is None, f"{brut!r} n'aurait pas dû matcher"


def test_une_seule_equipe_lue_ne_donne_pas_de_duo(eng):
    r = pt.associer_equipes(eng, "Fulham", [LG])
    assert r["home"] == "Fulham" and r["away"] is None


def test_seuil_plus_severe_rejette_les_approximations(eng):
    lache = pt.associer_equipes(eng, "Fulharn / C, Palace", [LG], seuil=0.55)
    strict = pt.associer_equipes(eng, "Fulharn / C, Palace", [LG], seuil=0.99)
    assert lache["away"] is not None
    assert strict["away"] is None, "un seuil strict doit refuser une lecture abîmée"


def test_le_score_reflete_la_qualite(eng):
    net = pt.associer_equipes(eng, "Fulham / C. Palace", [LG])["score"]
    abime = pt.associer_equipes(eng, "Fulharn / C, Palace", [LG])["score"]
    assert net > abime, "un texte propre doit avoir un meilleur score qu'un texte abîmé"


def test_base_sans_equipe_ne_plante_pas(tmp_path):
    db = tmp_path / "v.db"
    c = sqlite3.connect(db)
    c.executescript("CREATE TABLE events (id INTEGER PRIMARY KEY, competition TEXT, "
                    "team_a TEXT, team_b TEXT, round_info TEXT, expected_start TEXT);")
    c.commit(); c.close()
    r = pt.associer_equipes(create_engine(f"sqlite:///{db}"), "Fulham / C. Palace", [LG])
    assert r["home"] is None and r["away"] is None


def test_normalisation_ignore_accents_et_ponctuation():
    assert pt._norme("A. Villa") == pt._norme("a villa")
    assert pt._norme("Alavés") == pt._norme("alaves")
    assert pt._norme("  Spurs  ") == "spurs"
