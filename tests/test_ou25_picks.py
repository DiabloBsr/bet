"""Onglet « Over / Under 2,5 » : mes deux pronostics les plus sûrs.

Ce qui est verrouillé ici :
- la prédiction vient de la forme des équipes, JAMAIS des cotes ;
- les cotes affichées sont copiées telles quelles depuis le flux Bet261 ;
- l'Under a un pari en UN SEUL CLIC (« Le total de buts est de 0, 1 ou 2 »),
  l'Over n'en a aucun — Bet261 ne cote pas la ligne 2.5 dans ce sens ;
- la « cote équivalente » d'une mise répartie vaut bien 1/somme(1/cote) ;
- les probabilités annoncées restent sous les plafonds MESURÉS : la queue
  extrême sur-promettait de 3 à 8 points sans eux.
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
}


def _futur(minutes: int = 40) -> str:
    return (datetime.now(timezone.utc) + timedelta(minutes=minutes)).strftime("%Y-%m-%d %H:%M:%S")


def _base(tmp_path: Path, marche: dict | None = MARCHE, competition: str = LG) -> str:
    """Deux duos : « Foot vs Ball » tres offensif, « Mur vs Beton » tres ferme."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    db = tmp_path / "ou.db"
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
                  "VALUES (?,?,?,?,?)", (i + 1, competition, "Foot", "Ball", "2026-01-01 10:00:00"))
        c.execute("INSERT INTO results (event_id, score_a, score_b) VALUES (?,?,?)", (i + 1, 3, 2))
        c.execute("INSERT INTO events (id, competition, team_a, team_b, expected_start) "
                  "VALUES (?,?,?,?,?)", (200 + i, competition, "Mur", "Beton", "2026-01-01 10:00:00"))
        c.execute("INSERT INTO results (event_id, score_a, score_b) VALUES (?,?,?)", (200 + i, 0, 0))
    xm = json.dumps(marche) if marche is not None else None
    for eid, a, b in ((900, "Foot", "Ball"), (901, "Mur", "Beton")):
        c.execute("INSERT INTO events (id, competition, team_a, team_b, round_info, expected_start) "
                  "VALUES (?,?,?,?,?,?)", (eid, competition, a, b, "Journee 9", _futur()))
        c.execute("INSERT INTO odds_snapshots (event_id, odds_home, odds_draw, odds_away, "
                  "extra_markets) VALUES (?,?,?,?,?)", (eid, 2.0, 3.0, 4.0, xm))
    c.commit(); c.close()
    return f"sqlite:///{db}"


# ---------- probabilites ----------

def test_ou25_probas_monotone_et_complementaire():
    xs = [i / 100 for i in range(0, 101, 5)]
    ov = [pt.ou25_probas(x)[0] for x in xs]
    un = [pt.ou25_probas(x)[1] for x in xs]
    assert all(ov[i] <= ov[i + 1] + 1e-9 for i in range(len(ov) - 1)), "over non monotone"
    assert all(un[i] >= un[i + 1] - 1e-9 for i in range(len(un) - 1)), "under non decroissant"


def test_ou25_probas_respecte_les_plafonds_mesures():
    """Sans plafond, la queue extreme sur-promettait de 3 a 8 points."""
    for x in (0.0, 0.5, 1.0):
        o, u = pt.ou25_probas(x)
        assert o <= pt._OU25_OVER_MAX + 1e-9
        assert u <= pt._OU25_UNDER_MAX + 1e-9


@pytest.mark.parametrize("bad", [None, float("nan"), "x", [], {}])
def test_ou25_probas_ne_plante_pas(bad):
    assert pt.ou25_probas(bad) == (0.0, 0.0)


# ---------- choix des deux pronostics ----------

def test_le_bon_match_pour_chaque_sens(tmp_path):
    """Le duo offensif doit sortir en Over, le duo ferme en Under."""
    r = pt.ou25_picks(create_engine(_base(tmp_path)))
    assert r["over"]["home"] == "Foot", "l'Over doit designer le duo offensif"
    assert r["under"]["home"] == "Mur", "l'Under doit designer le duo ferme"
    assert r["over"]["p_over"] > 0.5 and r["under"]["p_under"] > 0.5
    assert r["over"]["attendus"] > r["under"]["attendus"]


def test_under_a_un_pari_en_un_clic_pas_l_over(tmp_path):
    """Bet261 cote l'Under 2.5 (Multi-Buts 0/1/2) mais pas l'Over."""
    r = pt.ou25_picks(create_engine(_base(tmp_path)))
    d = r["under"]["direct"]
    assert d is not None and d["odds"] == 1.41
    assert d["sel"] == "Le total de buts est de 0, 1 ou 2"
    assert d["marche"] == "Multi-Buts"
    assert r["over"]["direct"] is None, "aucun pari direct n'existe pour l'Over 2.5"


def test_cotes_copiees_telles_quelles_du_flux(tmp_path):
    r = pt.ou25_picks(create_engine(_base(tmp_path)))
    assert {c["total"]: c["odds"] for c in r["over"]["cellules"]} == {
        "3": 4.33, "4": 8.66, "5": 23.59, "6": 100.0}
    assert {c["total"]: c["odds"] for c in r["under"]["cellules"]} == {
        "0": 6.12, "1": 3.88, "2": 3.3}


def test_cote_equivalente_est_juste(tmp_path):
    """1/somme(1/cote) : le gain reel d'une mise repartie au prorata."""
    r = pt.ou25_picks(create_engine(_base(tmp_path)))
    for sens in ("over", "under"):
        m = r[sens]
        attendu = 1.0 / sum(1.0 / c["odds"] for c in m["cellules"])
        assert abs(m["equivalent"] - attendu) < 0.02, f"{sens} : equivalence rompue"
        for c in m["cellules"]:
            gain = (c["part"] / 100.0) * c["odds"]
            assert abs(gain - attendu) / attendu < 0.02


def test_repartition_somme_a_cent(tmp_path):
    r = pt.ou25_picks(create_engine(_base(tmp_path)))
    for sens in ("over", "under"):
        assert abs(sum(c["part"] for c in r[sens]["cellules"]) - 100.0) < 0.5


def test_voisins_portent_les_libelles_de_lapp(tmp_path):
    r = pt.ou25_picks(create_engine(_base(tmp_path)))
    v_over = {x["sel"]: x["odds"] for x in r["over"]["voisins"]}
    v_under = {x["sel"]: x["odds"] for x in r["under"]["voisins"]}
    assert v_over["> 3.5"] == 6.39
    assert v_under["< 3.5"] == 1.11
    assert v_over["Le total de buts est superieur a 4".replace("superieur a", "supérieur à")] == 20.38


# ---------- independance aux cotes, robustesse ----------

def test_les_cotes_ne_changent_pas_le_choix(tmp_path):
    """Marche completement retourne : memes matchs et memes probas."""
    retourne = dict(MARCHE)
    retourne["Total de buts"] = {"0": 60.0, "1": 40.0, "2": 30.0, "3": 1.5,
                                 "4": 1.4, "5": 1.3, "6": 1.2}
    a = pt.ou25_picks(create_engine(_base(tmp_path / "a")))
    b = pt.ou25_picks(create_engine(_base(tmp_path / "b", retourne)))
    for sens in ("over", "under"):
        assert a[sens]["home"] == b[sens]["home"], f"{sens} : le choix a suivi les cotes !"
        assert a[sens][f"p_{sens}"] == b[sens][f"p_{sens}"], f"{sens} : la proba a suivi !"
    assert a["over"]["equivalent"] != b["over"]["equivalent"], "seules les cotes changent"


def test_predit_meme_sans_aucun_marche(tmp_path):
    """Aucune cote : la prediction reste rendue, sans rien a cliquer."""
    r = pt.ou25_picks(create_engine(_base(tmp_path, marche=None)))
    assert r["over"] and r["under"]
    assert r["over"]["cellules"] == [] and r["over"]["direct"] is None
    assert r["under"]["direct"] is None and r["under"]["equivalent"] is None


def test_filtre_par_ligue(tmp_path):
    eng = create_engine(_base(tmp_path))
    assert pt.ou25_picks(eng, leagues=[LG])["over"] is not None
    assert pt.ou25_picks(eng, leagues=["InstantLeague-8060"]) == {"over": None, "under": None}


def test_base_vide_ne_plante_pas(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    db = tmp_path / "v.db"
    c = sqlite3.connect(db)
    c.executescript("""
        CREATE TABLE events (id INTEGER PRIMARY KEY, competition TEXT, team_a TEXT,
          team_b TEXT, round_info TEXT, expected_start TEXT);
        CREATE TABLE odds_snapshots (id INTEGER PRIMARY KEY, event_id INTEGER,
          odds_home REAL, odds_draw REAL, odds_away REAL, extra_markets TEXT);
        CREATE TABLE results (id INTEGER PRIMARY KEY, event_id INTEGER,
          score_a INTEGER, score_b INTEGER);
    """)
    c.commit(); c.close()
    assert pt.ou25_picks(create_engine(f"sqlite:///{db}")) == {"over": None, "under": None}
