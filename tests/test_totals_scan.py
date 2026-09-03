"""Onglet « Total de buts — mon top 3 » : distribution, calibration, scan.

Le total prédit vient de MA grille Poisson (forme Bet261), pas des cotes. Les
pièges verrouillés ici sont ceux déjà rencontrés sur ce projet : un marché lu de
travers fabrique un faux signal, et un NULL SQL arrive en NaN (un float, donc
truthy) qui fait exploser tout `.get()`. On verrouille aussi la calibration :
sans elle l'analyse annonce ~35 % là où le réel fait ~27 %.
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


def _futur(minutes: int = 30) -> str:
    return (datetime.now(timezone.utc) + timedelta(minutes=minutes)).strftime("%Y-%m-%d %H:%M:%S")


def _base(tmp_path: Path, cotes_totaux: dict | None = None, competition: str = LG) -> str:
    """2 équipes avec de l'historique + 1 match à venir portant « Total de buts »."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    db = tmp_path / "t.db"
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
                  "VALUES (?,?,?,?,?)", (i + 1, competition, "Alpha", "Beta", "2026-01-01 10:00:00"))
        c.execute("INSERT INTO odds_snapshots (event_id, odds_home, odds_draw, odds_away) "
                  "VALUES (?,?,?,?)", (i + 1, 2.0, 3.0, 4.0))
        c.execute("INSERT INTO results (event_id, score_a, score_b) VALUES (?,?,?)", (i + 1, 1, 1))
    tb = cotes_totaux if cotes_totaux is not None else {str(k): 5.0 for k in range(7)}
    c.execute("INSERT INTO events (id, competition, team_a, team_b, round_info, expected_start) "
              "VALUES (?,?,?,?,?,?)", (999, competition, "Alpha", "Beta", "Journee 7", _futur()))
    c.execute("INSERT INTO odds_snapshots (event_id, odds_home, odds_draw, odds_away, extra_markets) "
              "VALUES (?,?,?,?,?)", (999, 2.0, 3.0, 4.0, json.dumps({"Total de buts": tb})))
    c.commit(); c.close()
    return f"sqlite:///{db}"


# ---------- distribution des totaux ----------

def test_predict_own_expose_une_distribution_valide(tmp_path):
    own = pt.predict_own(create_engine(_base(tmp_path)), "Alpha", "Beta", lg=LG)
    assert own is not None
    t = own["totals"]
    assert len(t) == 7, "0..5 exacts + « 6 et plus »"
    assert abs(sum(t) - 1.0) < 1e-3, "la distribution doit sommer à 1"
    assert all(x >= 0 for x in t)


def test_distribution_suit_le_niveau_des_equipes(tmp_path):
    """Deux équipes qui font 1-1 en boucle : le total le plus probable doit être bas."""
    own = pt.predict_own(create_engine(_base(tmp_path)), "Alpha", "Beta", lg=LG)
    assert own["totals"].index(max(own["totals"])) <= 3


# ---------- calibration ----------

def test_calib_totals_monotone_et_bornee():
    ys = [pt.calib_totals(i / 100) for i in range(0, 101, 5)]
    assert all(ys[i] <= ys[i + 1] + 1e-9 for i in range(len(ys) - 1)), "table non monotone"
    assert 0.0 < min(ys) and max(ys) < 1.0


def test_calib_totals_rabat_la_surconfiance():
    """Le brut annonce ~35 % là où le mesuré plafonne vers 27 %."""
    assert pt.calib_totals(0.38) < 0.32


@pytest.mark.parametrize("bad", [None, float("nan"), "x", [], {}])
def test_calib_totals_ne_plante_pas(bad):
    assert pt.calib_totals(bad) == 0.0


# ---------- scan ----------

def test_totals_scan_predit_et_cote(tmp_path):
    res = pt.totals_scan(create_engine(_base(tmp_path)), top=3)
    assert len(res) == 1
    m = res[0]
    assert m["home"] == "Alpha" and m["away"] == "Beta"
    assert 0 <= m["total"] <= 6
    assert m["odds"] == 5.0, "la cote doit venir du marché du total prédit"
    assert 0.0 < m["p_mine_cal"] < 1.0
    assert m["journee"] == 7
    assert len(m["top3"]) == 3, "les 3 totaux les plus probables"
    assert m["attendus"] > 0


def test_totals_scan_respecte_le_choix_des_ligues(tmp_path):
    eng = create_engine(_base(tmp_path))
    assert len(pt.totals_scan(eng, leagues=[LG])) == 1
    assert pt.totals_scan(eng, leagues=["InstantLeague-8060"]) == []
    assert len(pt.totals_scan(eng, leagues=None)) == 1, "None = toutes les ligues"


def test_totals_scan_respecte_la_limite_top(tmp_path):
    assert len(pt.totals_scan(create_engine(_base(tmp_path)), top=1)) <= 1


def test_totals_scan_trie_du_plus_sur_au_moins_sur(tmp_path):
    ps = [m["p_mine_cal"] for m in pt.totals_scan(create_engine(_base(tmp_path)), top=10)]
    assert ps == sorted(ps, reverse=True)


def test_totals_scan_predit_meme_si_le_total_nest_pas_cote(tmp_path):
    """Ma prédiction ne dépend PAS du book : un total non coté sort quand même,
    simplement sans cote. Le marché ne décide pas quels matchs sont prédits."""
    eng = create_engine(_base(tmp_path, cotes_totaux={"5": 9.0, "6": 12.0}))
    res = pt.totals_scan(eng)
    assert len(res) == 1, "le match doit être prédit malgré l'absence de cote"
    assert res[0]["odds"] is None
    assert 0 <= res[0]["total"] <= 6
    assert 0.0 < res[0]["p_mine_cal"] < 1.0


def test_totals_scan_predit_sans_aucun_marche_de_totaux(tmp_path):
    """Aucun marché « Total de buts » du tout : la prédiction reste possible."""
    eng = create_engine(_base(tmp_path, cotes_totaux={}))
    res = pt.totals_scan(eng)
    assert len(res) == 1 and res[0]["odds"] is None


def test_les_cotes_ne_changent_ni_la_prediction_ni_le_classement(tmp_path):
    """Preuve directe d'indépendance : on retourne complètement le marché
    (cotes inversées, favori du book déplacé) — mon total prédit et ma
    probabilité doivent être STRICTEMENT identiques."""
    normal = {str(k): 5.0 for k in range(7)}
    retourne = {"0": 1.2, "1": 1.3, "2": 40.0, "3": 35.0, "4": 30.0, "5": 25.0, "6": 20.0}
    a = pt.totals_scan(create_engine(_base(tmp_path / "a", normal)))
    b = pt.totals_scan(create_engine(_base(tmp_path / "b", retourne)))
    assert len(a) == len(b) == 1
    assert a[0]["total"] == b[0]["total"], "le total prédit a suivi les cotes !"
    assert a[0]["p_mine"] == b[0]["p_mine"], "ma proba a suivi les cotes !"
    assert a[0]["p_mine_cal"] == b[0]["p_mine_cal"]
    assert [t["total"] for t in a[0]["top3"]] == [t["total"] for t in b[0]["top3"]]
    assert a[0]["odds"] != b[0]["odds"], "seule la cote affichée doit différer"


def test_totals_scan_survit_a_un_marche_pourri(tmp_path):
    """extra_markets NULL/NaN/texte : jamais de crash (bug NaN historique)."""
    db = tmp_path / "p.db"
    c = sqlite3.connect(db)
    c.executescript("""
        CREATE TABLE events (id INTEGER PRIMARY KEY, competition TEXT, team_a TEXT,
          team_b TEXT, round_info TEXT, expected_start TEXT);
        CREATE TABLE odds_snapshots (id INTEGER PRIMARY KEY, event_id INTEGER,
          odds_home REAL, odds_draw REAL, odds_away REAL, extra_markets TEXT);
        CREATE TABLE results (id INTEGER PRIMARY KEY, event_id INTEGER,
          score_a INTEGER, score_b INTEGER);
    """)
    for n, xm in enumerate(["pas du json", None, "{}", '{"Total de buts": null}'], start=1):
        c.execute("INSERT INTO events (id, competition, team_a, team_b, expected_start) "
                  "VALUES (?,?,?,?,?)", (n, LG, "A", "B", _futur()))
        c.execute("INSERT INTO odds_snapshots (event_id, odds_home, odds_draw, odds_away, "
                  "extra_markets) VALUES (?,?,?,?,?)", (n, 2.0, 3.0, 4.0, xm))
    c.commit(); c.close()
    assert pt.totals_scan(create_engine(f"sqlite:///{db}")) == []


def test_totals_scan_base_vide_ne_plante_pas(tmp_path):
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
    assert pt.totals_scan(create_engine(f"sqlite:///{db}")) == []
