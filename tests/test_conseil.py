"""Onglet « Que jouer sur ce match ? » — conseil tous marchés.

Verrouillé ici :
- la recommandation vient de MA proba calibrée, jamais des cotes ;
- chaque marché porte le libellé EXACT de Bet261 (sinon la cote ne se trouve pas) ;
- le modèle de minute utilise la table EMPIRIQUE (l'exponentiel se trompait de 15pp) ;
- la règle « proba × cote » reste ABSENTE : testée, elle affiche un gain apparent
  supérieur à 1 dans 80 % des cas sans améliorer le ROI.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import predict_trio as pt  # noqa: E402

LG = "InstantLeague-8035"
MARCHE = {
    "Total de buts": {"0": 6.12, "1": 3.88, "2": 3.3, "3": 4.33, "4": 8.66,
                      "5": 23.59, "6": 100.0},
    "Multi-Buts": {"Le total de buts est de 0, 1 ou 2": 1.41,
                   "Le total de buts est de 1, 2 ou 3": 1.29,
                   "Le total de buts est de 2, 3 ou 4": 1.57,
                   "Le total de buts est supérieur à 4": 20.38},
    "+/-": {"> 3.5": 6.39, "< 3.5": 1.11},
    "G/NG": {"Oui": 2.39, "Non": 1.56},
    "Double Chance": {"1X": 1.12, "X2": 1.62, "12": 1.25},
    "Pair/Impair": {"Impair": 1.95, "Pair": 1.76},
    "FTTS": {"1": 1.67, "2": 2.81, "Pas de but": 6.12},
    "Minute du premier but": {"1-15": 5.75, "16-30": 3.65, "31-45": 5.52,
                              "46-60": 6.62, "61-75": 10.84, "76-90": 11.84,
                              "Pas de but": 6.12},
    "Score exact": {"0-0": 6.12, "1-1": 6.74, "2-0": 8.08, "2-1": 7.5, "1-0": 5.84},
}


def _base(tmp_path: Path, sa: int = 2, sb: int = 1, marche=MARCHE) -> str:
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
    for i in range(40):
        c.execute("INSERT INTO events (id, competition, team_a, team_b, expected_start) "
                  "VALUES (?,?,?,?,?)", (i + 1, LG, "Alpha", "Beta", "2026-01-01 10:00:00"))
        c.execute("INSERT INTO results (event_id, score_a, score_b) VALUES (?,?,?)",
                  (i + 1, sa, sb))
    fut = (datetime.now(timezone.utc) + timedelta(minutes=40)).strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO events (id, competition, team_a, team_b, round_info, expected_start) "
              "VALUES (?,?,?,?,?,?)", (900, LG, "Alpha", "Beta", "Journee 9", fut))
    c.execute("INSERT INTO odds_snapshots (event_id, odds_home, odds_draw, odds_away, "
              "extra_markets) VALUES (?,?,?,?,?)",
              (900, 1.85, 3.4, 4.2, json.dumps(marche) if marche else None))
    c.commit(); c.close()
    return f"sqlite:///{db}"


# ---------- marches_probas : coherence de chaque marche ----------

def test_chaque_marche_somme_a_un():
    """Sauf Double Chance (2 issues par pari) et les tranches Multi-Buts qui se
    chevauchent volontairement, et le Score exact tronque au top 10."""
    p = pt.marches_probas(1.6, 1.1)
    for m in ("1X2", "G/NG", "Total de buts", "+/-", "Pair/Impair",
              "Minute du premier but", "FTTS"):
        assert abs(sum(x[1] for x in p[m]) - 1.0) < 0.02, f"{m} ne somme pas a 1"
    assert abs(sum(x[1] for x in p["Double Chance"]) - 2.0) < 0.02


def test_libelles_identiques_a_bet261():
    """Un libelle qui derive = cote introuvable = conseil inutilisable."""
    p = pt.marches_probas(1.6, 1.1)
    assert {x[0] for x in p["G/NG"]} == {"Oui", "Non"}
    assert {x[0] for x in p["+/-"]} == {"> 3.5", "< 3.5"}
    assert {x[0] for x in p["Pair/Impair"]} == {"Pair", "Impair"}
    assert {x[0] for x in p["Double Chance"]} == {"1X", "X2", "12"}
    assert "Le total de buts est de 0, 1 ou 2" in {x[0] for x in p["Multi-Buts"]}
    assert "Le total de buts est supérieur à 4" in {x[0] for x in p["Multi-Buts"]}
    assert {x[0] for x in p["FTTS"]} == {"1", "2", "Pas de but"}
    mins = {x[0] for x in p["Minute du premier but"]}
    assert mins == {"1-15", "16-30", "31-45", "46-60", "61-75", "76-90", "Pas de but"}


def test_minute_utilise_la_table_empirique():
    """L'exponentiel donnait 36 % sur « 1-15 » pour 20 % reel : le pic doit etre
    sur « 16-30 », comme mesure sur 207 861 matchs."""
    assert pt._MIN_TABLE, "table empirique des minutes absente"
    d = dict(pt.marches_probas(1.6, 1.1)["Minute du premier but"])
    assert d["16-30"] > d["1-15"], "le pic doit etre sur 16-30, pas sur 1-15"


def test_equipes_offensives_montent_le_total():
    faible = dict(pt.marches_probas(0.4, 0.3)["+/-"])
    fort = dict(pt.marches_probas(2.6, 2.2)["+/-"])
    assert fort["> 3.5"] > faible["> 3.5"]


def test_favori_domine_le_1x2():
    p = dict(pt.marches_probas(2.4, 0.6)["1X2"])
    assert p["1"] > p["2"] and p["1"] > p["X"]


# ---------- calibration ----------

def test_calibration_par_marche_chargee():
    assert pt._MK_CAL, "calibration par marche absente"
    for m in ("1X2", "G/NG", "Double Chance", "Score exact"):
        assert (pt._MK_CAL.get(m) or {}).get("bins"), f"{m} non calibre"


@pytest.mark.parametrize("bad", [None, float("nan"), "x", [], {}])
def test_calib_marche_ne_plante_pas(bad):
    assert pt.calib_marche("1X2", bad) == 0.0


def test_calib_marche_marche_inconnu_reste_neutre():
    assert pt.calib_marche("Marche Inexistant", 0.42) == 0.42


# ---------- conseil de bout en bout ----------

def test_conseil_rend_tous_les_marches_et_une_reco(tmp_path):
    eng = create_engine(_base(tmp_path))
    fx = pt.rencontres(eng)
    assert len(fx) == 1 and fx[0]["home"] == "Alpha"
    r = pt.conseil(eng, fx[0])
    assert not r.get("erreur")
    marches = {l["marche"] for l in r["lignes"]}
    for attendu in ("1X2", "Total de buts", "+/-", "G/NG", "Multi-Buts",
                    "Score exact", "Minute du premier but", "FTTS"):
        assert attendu in marches, f"{attendu} manquant du conseil"
    assert r["sur"] == r["lignes"][0], "la reco doit etre la ligne la plus probable"
    assert r["journee"] == 9
    assert r["attendus"] > 0


def test_conseil_trie_par_proba_decroissante(tmp_path):
    eng = create_engine(_base(tmp_path))
    r = pt.conseil(eng, pt.rencontres(eng)[0])
    ps = [l["p"] for l in r["lignes"]]
    assert ps == sorted(ps, reverse=True)


def test_conseil_rattache_les_vraies_cotes(tmp_path):
    """Chaque selection doit retrouver SA cote dans le flux, 1X2 compris."""
    eng = create_engine(_base(tmp_path))
    r = pt.conseil(eng, pt.rencontres(eng)[0])
    par_m = {l["marche"]: l for l in r["lignes"]}
    x12 = par_m["1X2"]
    assert x12["odds"] == {"1": 1.85, "X": 3.4, "2": 4.2}[x12["sel"]]
    gng = par_m["G/NG"]
    assert gng["odds"] == MARCHE["G/NG"][gng["sel"]]
    mn = par_m["Minute du premier but"]
    assert mn["odds"] == MARCHE["Minute du premier but"][mn["sel"]]


def test_conseil_n_expose_aucun_gain_espere(tmp_path):
    """La regle « proba x cote » a ete ecartee : aucun champ de gain ne doit
    reapparaitre, sinon l'onglet afficherait un mirage 4 fois sur 5."""
    eng = create_engine(_base(tmp_path))
    r = pt.conseil(eng, pt.rencontres(eng)[0])
    assert "moins_mauvais" not in r
    for l in r["lignes"]:
        assert "gain" not in l, "champ de gain espere reapparu"


def test_conseil_sans_cotes_reste_utilisable(tmp_path):
    """Aucun marche dans le flux : les probas restent, les cotes valent None."""
    eng = create_engine(_base(tmp_path, marche=None))
    r = pt.conseil(eng, pt.rencontres(eng)[0])
    assert not r.get("erreur")
    par_m = {l["marche"]: l for l in r["lignes"]}
    assert par_m["G/NG"]["odds"] is None
    assert par_m["1X2"]["odds"] == 1.85, "le 1X2 vient des colonnes, pas d'extra_markets"


def test_conseil_sans_historique_previent(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    db = tmp_path / "vide.db"
    c = sqlite3.connect(db)
    c.executescript("""
        CREATE TABLE events (id INTEGER PRIMARY KEY, competition TEXT, team_a TEXT,
          team_b TEXT, round_info TEXT, expected_start TEXT);
        CREATE TABLE odds_snapshots (id INTEGER PRIMARY KEY, event_id INTEGER,
          odds_home REAL, odds_draw REAL, odds_away REAL, extra_markets TEXT);
        CREATE TABLE results (id INTEGER PRIMARY KEY, event_id INTEGER,
          score_a INTEGER, score_b INTEGER);
    """)
    fut = (datetime.now(timezone.utc) + timedelta(minutes=40)).strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO events (id, competition, team_a, team_b, expected_start) "
              "VALUES (?,?,?,?,?)", (1, LG, "X", "Y", fut))
    c.execute("INSERT INTO odds_snapshots (event_id, odds_home, odds_draw, odds_away) "
              "VALUES (?,?,?,?)", (1, 2.0, 3.0, 4.0))
    c.commit(); c.close()
    eng = create_engine(f"sqlite:///{db}")
    r = pt.conseil(eng, pt.rencontres(eng)[0])
    assert r.get("erreur"), "doit prevenir au lieu de rendre un conseil sans fondement"


def test_rencontres_filtre_ligue_et_heure(tmp_path):
    eng = create_engine(_base(tmp_path))
    assert len(pt.rencontres(eng, leagues=[LG])) == 1
    assert pt.rencontres(eng, leagues=["InstantLeague-8060"]) == []
    assert pt.rencontres(eng, heure="03:07") == []
