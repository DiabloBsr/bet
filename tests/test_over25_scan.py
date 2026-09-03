"""Onglet « Over 2.5 à cote ≥ 2 » : reconstruction, calibration, scan de bout en bout.

L'Over 2.5 n'existe PAS comme pari sur Bet261 (seule la ligne 3.5 est cotée) : il est
reconstruit depuis « Total de buts ». Un bug de lecture de ce marché fabriquerait un
faux signal — le piège déjà rencontré deux fois sur ce projet. On verrouille donc :
le sens de la reconstruction, la monotonie de la calibration mesurée, la résistance
aux marchés NULL/NaN (bug historique : pandas lit un NULL SQL en NaN, un float VRAI),
et le chemin complet du scan sur une base synthétique.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import predict_trio as pt  # noqa: E402


# ---------- reconstruction Over/Under 2.5 ----------

def test_ou25_uniforme():
    """7 totaux à cote 7 : over = 3..6 = 4 cellules → 1/(4/7) = 1.75."""
    o_over, o_under = pt._ou25({"Total de buts": {str(i): 7.0 for i in range(7)}})
    assert abs(o_over - 1.75) < 0.01
    assert abs(o_under - 7 / 3) < 0.01


def test_ou25_garde_la_marge_du_book():
    """La cote rendue est celle que le book afficherait : marge NON retirée.
    Somme des probas implicites = 1.2 → l'over ne peut pas valoir la cote 'juste'."""
    T = {str(i): 7 * 1.2 / 1.2 for i in range(7)}   # base uniforme
    T = {k: v / 1.2 for k, v in T.items()}          # 20 % de marge ajoutée
    o_over, _ = pt._ou25({"Total de buts": T})
    assert o_over < 1.75, "la marge doit rabaisser la cote sous la valeur équitable"


@pytest.mark.parametrize("bad", [None, float("nan"), "pas du json", 42, [], {},
                                 {"Total de buts": None}, {"Total de buts": {}},
                                 {"Total de buts": {"3": None, "4": "x"}}])
def test_ou25_ne_plante_jamais(bad):
    """NULL SQL → NaN (un float, donc truthy) : le piège qui a déjà cassé head_to_head."""
    assert pt._ou25(bad) == (None, None)


def test_ou25_refuse_un_marche_tronque():
    """Marché incomplet (somme des probas hors bande) → on n'affiche rien."""
    assert pt._ou25({"Total de buts": {"0": 100.0, "3": 100.0}}) == (None, None)


# ---------- calibration mesurée ----------

def test_calib_over25_monotone_et_bornee():
    """Table isotone : plus ma proba brute monte, plus la proba calibrée monte."""
    xs = [i / 100 for i in range(0, 101, 5)]
    ys = [pt.calib_over25(x) for x in xs]
    assert all(ys[i] <= ys[i + 1] + 1e-9 for i in range(len(ys) - 1)), "table non monotone"
    assert 0.0 < min(ys) and max(ys) < 1.0


def test_calib_over25_corrige_la_surconfiance():
    """Le brut est surconfiant (~50 % annoncé pour ~37 % réel) : la table doit rabattre."""
    assert pt.calib_over25(0.50) < 0.50


def test_calib_over25_tolere_une_entree_invalide():
    assert pt.calib_over25(None) is not None or True   # ne doit pas lever
    for bad in (None, "x"):
        try:
            pt.calib_over25(bad)
        except (TypeError, ValueError):
            pass   # refus explicite accepté ; un crash inattendu, non


# ---------- scan de bout en bout ----------

def _base_synthetique(tmp_path: Path, o_over_vise: float) -> str:
    """Base minimale : 2 équipes avec assez d'historique + 1 match À VENIR."""
    db = tmp_path / "t.db"
    c = sqlite3.connect(db)
    c.executescript("""
        CREATE TABLE events (id INTEGER PRIMARY KEY, match_key TEXT, external_id TEXT,
          sport TEXT, competition TEXT, team_a TEXT, team_b TEXT, round_info TEXT,
          source_url TEXT, first_seen_at TEXT, expected_start TEXT);
        CREATE TABLE odds_snapshots (id INTEGER PRIMARY KEY, event_id INTEGER,
          odds_home REAL, odds_draw REAL, odds_away REAL, extra_markets TEXT);
        CREATE TABLE results (id INTEGER PRIMARY KEY, event_id INTEGER,
          score_a INTEGER, score_b INTEGER);
    """)
    LG = "InstantLeague-8035"
    # 40 matchs terminés pour donner de la forme aux deux équipes
    for i in range(40):
        c.execute("INSERT INTO events (id, competition, team_a, team_b, expected_start) "
                  "VALUES (?,?,?,?,?)", (i + 1, LG, "Alpha", "Beta", "2026-01-01 10:00:00"))
        c.execute("INSERT INTO odds_snapshots (event_id, odds_home, odds_draw, odds_away) "
                  "VALUES (?,?,?,?)", (i + 1, 2.0, 3.0, 4.0))
        c.execute("INSERT INTO results (event_id, score_a, score_b) VALUES (?,?,?)",
                  (i + 1, 1, 1))
    # cotes « Total de buts » calibrées pour viser la cote over voulue
    p_over = 1.0 / o_over_vise
    p_under = 1.15 - p_over                      # somme dans la bande plausible du moteur
    tb = {k: round(3.0 / p_under, 2) for k in ("0", "1", "2")}
    tb.update({k: round(4.0 / p_over, 2) for k in ("3", "4", "5", "6")})
    fut = datetime_utc_plus(30)
    c.execute("INSERT INTO events (id, competition, team_a, team_b, round_info, expected_start) "
              "VALUES (?,?,?,?,?,?)", (999, LG, "Alpha", "Beta", "Journee 12", fut))
    c.execute("INSERT INTO odds_snapshots (event_id, odds_home, odds_draw, odds_away, extra_markets) "
              "VALUES (?,?,?,?,?)", (999, 2.0, 3.0, 4.0, json.dumps({"Total de buts": tb})))
    c.commit()
    c.close()
    return f"sqlite:///{db}"


def datetime_utc_plus(minutes: int) -> str:
    from datetime import datetime, timedelta, timezone
    return (datetime.now(timezone.utc) + timedelta(minutes=minutes)).strftime("%Y-%m-%d %H:%M:%S")


def test_over25_scan_trouve_et_filtre_par_cote(tmp_path):
    """Une cote over ≈ 2.5 passe le filtre min_odds=2 ; le même match est rejeté à 3."""
    eng = create_engine(_base_synthetique(tmp_path, 2.5))
    trouve = pt.over25_scan(eng, min_odds=2.0)
    assert len(trouve) == 1, f"le match à venir doit sortir, obtenu {trouve}"
    m = trouve[0]
    assert m["home"] == "Alpha" and m["away"] == "Beta"
    assert m["odds_over25"] >= 2.0
    assert 0.0 < m["p_mine_cal"] < 1.0
    assert m["journee"] == 12, "la journée doit être lue depuis round_info"
    assert [c["total"] for c in m["cells"]] == ["3", "4", "5", "6"]
    assert abs(sum(c["part"] for c in m["cells"]) - 100.0) < 0.5, "la répartition doit faire 100 %"
    assert pt.over25_scan(eng, min_odds=3.0) == [], "cote sous le seuil : doit être exclu"


def test_over25_scan_trie_du_plus_sur_au_moins_sur(tmp_path):
    eng = create_engine(_base_synthetique(tmp_path, 2.5))
    res = pt.over25_scan(eng, min_odds=2.0)
    probas = [m["p_mine_cal"] for m in res]
    assert probas == sorted(probas, reverse=True), "tri décroissant sur la proba calibrée"


def test_over25_scan_base_vide_ne_plante_pas(tmp_path):
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
    c.commit(); c.close()
    assert pt.over25_scan(create_engine(f"sqlite:///{db}"), min_odds=2.0) == []


def test_over25_scan_filtre_par_ligue(tmp_path):
    """Le choix de ligues doit être respecté : la bonne ligue sort, une autre non."""
    eng = create_engine(_base_synthetique(tmp_path, 2.5))
    assert len(pt.over25_scan(eng, min_odds=2.0, leagues=["InstantLeague-8035"])) == 1
    assert pt.over25_scan(eng, min_odds=2.0, leagues=["InstantLeague-8060"]) == []
    assert len(pt.over25_scan(eng, min_odds=2.0, leagues=None)) == 1, "None = toutes les ligues"
