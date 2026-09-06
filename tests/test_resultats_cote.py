"""Onglet « Qu'est-ce qui tombe à cette cote ? » — relevé historique 1X2.

Le risque de cet onglet n'est pas de planter, c'est de FABRIQUER DU SIGNAL :
sur quelques dizaines de rencontres, un écart de 5 à 10 points est le régime
normal du hasard. Sont donc verrouillés ici l'intervalle de Wilson (l'approche
normale donne une largeur nulle quand une issue ne sort jamais, et déclarerait
alors tout significatif) et la correction de Bonferroni pour les 3 issues
testées à chaque requête (sans elle, ~1 requête sur 7 crie au signal).
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


def _base(tmp_path: Path, matchs) -> str:
    """matchs = [(team_a, team_b, oh, od, oa, sa, sb), ...]"""
    tmp_path.mkdir(parents=True, exist_ok=True)
    db = tmp_path / "c.db"
    c = sqlite3.connect(db)
    c.executescript("""
        CREATE TABLE events (id INTEGER PRIMARY KEY, competition TEXT, team_a TEXT,
          team_b TEXT, round_info TEXT, expected_start TEXT);
        CREATE TABLE odds_snapshots (id INTEGER PRIMARY KEY, event_id INTEGER,
          odds_home REAL, odds_draw REAL, odds_away REAL, extra_markets TEXT);
        CREATE TABLE results (id INTEGER PRIMARY KEY, event_id INTEGER,
          score_a INTEGER, score_b INTEGER);
    """)
    for i, (ta, tb, oh, od, oa, sa, sb) in enumerate(matchs, start=1):
        c.execute("INSERT INTO events (id, competition, team_a, team_b, round_info, "
                  "expected_start) VALUES (?,?,?,?,?,?)",
                  (i, LG, ta, tb, "Journee 5", f"2026-01-{(i % 27) + 1:02d} 10:00:00"))
        c.execute("INSERT INTO odds_snapshots (event_id, odds_home, odds_draw, odds_away) "
                  "VALUES (?,?,?,?)", (i, oh, od, oa))
        c.execute("INSERT INTO results (event_id, score_a, score_b) VALUES (?,?,?)", (i, sa, sb))
    c.commit(); c.close()
    return f"sqlite:///{db}"


def test_retrouve_les_rencontres_a_la_cote_exacte(tmp_path):
    eng = create_engine(_base(tmp_path, [
        ("A", "B", 2.05, 3.22, 3.81, 1, 0),
        ("C", "D", 2.05, 3.22, 3.81, 1, 1),
        ("E", "F", 9.00, 3.22, 3.81, 0, 2),      # cote 1 differente : exclue
    ]))
    r = pt.resultats_a_cette_cote(eng, 2.05, 3.22, 3.81, tol=0.05)
    assert r["n"] == 2, "seules les rencontres aux trois cotes doivent sortir"
    assert {m["home"] for m in r["matchs"]} == {"A", "C"}


def test_tolerance_elargit_la_recherche(tmp_path):
    eng = create_engine(_base(tmp_path, [
        ("A", "B", 2.05, 3.22, 3.81, 1, 0),
        ("C", "D", 2.12, 3.22, 3.81, 1, 0),
    ]))
    assert pt.resultats_a_cette_cote(eng, 2.05, 3.22, 3.81, tol=0.02)["n"] == 1
    assert pt.resultats_a_cette_cote(eng, 2.05, 3.22, 3.81, tol=0.10)["n"] == 2


def test_comptage_des_issues_et_des_buts(tmp_path):
    eng = create_engine(_base(tmp_path, [
        ("A", "B", 2.0, 3.0, 4.0, 2, 0),   # 1
        ("A", "B", 2.0, 3.0, 4.0, 1, 1),   # X
        ("A", "B", 2.0, 3.0, 4.0, 0, 3),   # 2
        ("A", "B", 2.0, 3.0, 4.0, 0, 0),   # X, 0-0
    ]))
    R = pt.resultats_a_cette_cote(eng, 2.0, 3.0, 4.0)["resume"]
    par = {i["sel"]: i for i in R["issues"]}
    assert par["1"]["n"] == 1 and par["X"]["n"] == 2 and par["2"]["n"] == 1
    assert R["buts_moyen"] == 1.75         # totaux 2, 2, 3, 0 -> 7 / 4
    assert R["over25"] == 25.0             # un seul match a 3 buts et plus
    assert R["zero"] == 25.0               # un 0-0
    assert R["btts"] == 25.0               # seul le 1-1 voit les deux marquer


def test_limiter_a_deux_equipes(tmp_path):
    eng = create_engine(_base(tmp_path, [
        ("A", "B", 2.0, 3.0, 4.0, 1, 0),
        ("B", "A", 2.0, 3.0, 4.0, 0, 1),   # meme paire, orientation inverse
        ("C", "D", 2.0, 3.0, 4.0, 1, 0),
    ]))
    r = pt.resultats_a_cette_cote(eng, 2.0, 3.0, 4.0, team_a="A", team_b="B")
    assert r["n"] == 2, "les deux orientations de la paire doivent sortir"


def test_cotes_invalides_sont_refusees(tmp_path):
    eng = create_engine(_base(tmp_path, [("A", "B", 2.0, 3.0, 4.0, 1, 0)]))
    for bad in ((1.0, 3.0, 4.0), (2.0, 0.5, 4.0), ("x", 3.0, 4.0), (None, 3.0, 4.0)):
        assert pt.resultats_a_cette_cote(eng, *bad).get("erreur"), f"{bad} doit être refusé"


def test_aucune_rencontre_ne_plante_pas(tmp_path):
    eng = create_engine(_base(tmp_path, [("A", "B", 2.0, 3.0, 4.0, 1, 0)]))
    r = pt.resultats_a_cette_cote(eng, 7.77, 7.77, 7.77)
    assert r["n"] == 0 and r["matchs"] == []


# ---------- le garde-fou principal : ne pas fabriquer de signal ----------

def test_wilson_supporte_une_issue_jamais_sortie(tmp_path):
    """k=0 : l'approximation normale donnerait un intervalle de largeur NULLE et
    declarerait l'ecart significatif. Wilson doit rendre un intervalle large."""
    eng = create_engine(_base(tmp_path, [("A", "B", 2.0, 3.0, 4.0, 1, 0)] * 12))
    R = pt.resultats_a_cette_cote(eng, 2.0, 3.0, 4.0)["resume"]
    deux = next(i for i in R["issues"] if i["sel"] == "2")
    assert deux["n"] == 0 and deux["reel"] == 0.0
    assert deux["ic_haut"] > 15.0, "intervalle trop etroit sur k=0"
    assert deux["ic_bas"] == 0.0


def test_petit_echantillon_ne_declare_rien_de_notable(tmp_path):
    """8 rencontres ne peuvent rien prouver : aucun ecart ne doit etre notable."""
    eng = create_engine(_base(tmp_path, [("A", "B", 2.0, 3.0, 4.0, 1, 0)] * 4
                                        + [("A", "B", 2.0, 3.0, 4.0, 0, 1)] * 4))
    R = pt.resultats_a_cette_cote(eng, 2.0, 3.0, 4.0)["resume"]
    assert R["notables"] == 0, "un echantillon de 8 ne peut pas etre significatif"


def test_bonferroni_elargit_bien_les_intervalles(tmp_path):
    """Sans correction, ~1 requete sur 7 crierait au signal. L'intervalle doit
    donc etre plus large que le 95 % naif (z=1.96)."""
    eng = create_engine(_base(tmp_path, [("A", "B", 2.0, 3.0, 4.0, 1, 0)] * 30
                                        + [("A", "B", 2.0, 3.0, 4.0, 0, 1)] * 30))
    R = pt.resultats_a_cette_cote(eng, 2.0, 3.0, 4.0)["resume"]
    un = next(i for i in R["issues"] if i["sel"] == "1")
    largeur = un["ic_haut"] - un["ic_bas"]
    naif = 2 * 1.96 * ((0.5 * 0.5 / 60) ** 0.5) * 100
    assert largeur > naif, f"intervalle {largeur:.1f} pas plus large que le naif {naif:.1f}"


def test_un_vrai_ecart_massif_reste_detecte(tmp_path):
    """La correction ne doit pas rendre l'outil aveugle : un book faux de 40
    points sur 200 rencontres doit ressortir."""
    eng = create_engine(_base(tmp_path, [("A", "B", 2.0, 3.0, 4.0, 1, 0)] * 200))
    R = pt.resultats_a_cette_cote(eng, 2.0, 3.0, 4.0, n=300)["resume"]
    un = next(i for i in R["issues"] if i["sel"] == "1")
    assert un["reel"] == 100.0 and un["notable"], "un ecart enorme doit rester detecte"


def test_n_pour_5pp_est_rappele(tmp_path):
    eng = create_engine(_base(tmp_path, [("A", "B", 2.0, 3.0, 4.0, 1, 0)] * 5))
    R = pt.resultats_a_cette_cote(eng, 2.0, 3.0, 4.0)["resume"]
    assert R["n_pour_5pp"] >= 300, "l'ordre de grandeur requis doit etre rappele"


# ---------- lecture d'une capture entiere : le seuil doit suivre le nombre de tests ----------

def test_seuil_selargit_avec_le_nombre_de_tests():
    """Analyser une capture de round = 3 issues x ~10 rencontres. Sans elargir
    le seuil, on afficherait plus d'un faux « ecart notable » par capture."""
    z3, z27 = pt._z_bonferroni(3), pt._z_bonferroni(27)
    assert z3 < z27, "27 comparaisons doivent exiger un seuil plus severe que 3"
    assert 2.3 < z3 < 2.5 and 3.0 < z27 < 3.3


def test_seuil_supporte_les_valeurs_degenerees():
    for n in (0, None, 1, -5):
        z = pt._z_bonferroni(n)
        assert 1.9 < z < 3.5, f"n_tests={n} donne un seuil aberrant ({z})"


def test_meme_donnees_moins_de_notables_quand_on_teste_plus(tmp_path):
    """A donnees IDENTIQUES, declarer un ecart notable doit devenir plus dur
    quand on multiplie les rencontres analysees d'un coup."""
    eng = create_engine(_base(tmp_path, [("A", "B", 2.0, 3.0, 4.0, 1, 0)] * 26
                                        + [("A", "B", 2.0, 3.0, 4.0, 0, 1)] * 14))
    peu = pt.resultats_a_cette_cote(eng, 2.0, 3.0, 4.0, n_tests=3)["resume"]["notables"]
    bcp = pt.resultats_a_cette_cote(eng, 2.0, 3.0, 4.0, n_tests=30)["resume"]["notables"]
    assert bcp <= peu, "un seuil plus severe ne peut pas produire PLUS de notables"


def test_n_tests_est_rappele_dans_le_resume(tmp_path):
    eng = create_engine(_base(tmp_path, [("A", "B", 2.0, 3.0, 4.0, 1, 0)] * 5))
    R = pt.resultats_a_cette_cote(eng, 2.0, 3.0, 4.0, n_tests=27)["resume"]
    assert R["n_tests"] == 27


def test_defaut_inchange_pour_une_seule_rencontre(tmp_path):
    """Le cas « je saisis une cote a la main » ne doit pas devenir plus severe."""
    eng = create_engine(_base(tmp_path, [("A", "B", 2.0, 3.0, 4.0, 1, 0)] * 10))
    assert pt.resultats_a_cette_cote(eng, 2.0, 3.0, 4.0)["resume"]["n_tests"] == 3
