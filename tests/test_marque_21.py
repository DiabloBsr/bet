"""Annotation ⚽ des pronostics 2-1 / 1-2 pour la CAN (Olivio, 19/10).

Sans condition : dès que le pronostic est 2-1 ou 1-2 en CAN, le score porte ⚽,
sur le score exact principal ET les deux scores « Sinon ». Hors CAN : rien.
"""
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "scripts"))
import predict_trio as pt  # noqa: E402


# -------- _marque_21 : la brique --------

def test_marque_21_sur_2_1_et_1_2_en_can():
    # `derniers` non None (même vide) = CAN -> l'emoji sort, sans condition.
    assert pt._marque_21("2-1", {}) == " ⚽"
    assert pt._marque_21("1-2", {}) == " ⚽"


def test_marque_21_rien_sur_les_autres_scores():
    for sc in ("1-1", "0-0", "2-0", "1-0", "3-1", "2-2"):
        assert pt._marque_21(sc, {}) == ""


def test_marque_21_rien_hors_can():
    # derniers None = ligue hors CAN -> jamais d'emoji.
    assert pt._marque_21("2-1", None) == ""
    assert pt._marque_21("1-2", None) == ""


# -------- intégration dans le tableau --------

def _m(score, sinon=None):
    top3 = [(score, 0.17)] + (sinon or [("0-0", 0.14), ("1-0", 0.10)])
    return {"team_a": "A", "team_b": "B", "cotes": [2.0, 3.2, 3.8],
            "x12": [0.33, 0.34, 0.33], "confidence": 0.25, "over25_pct": 45.0,
            "top1_calibre": (score, 0.17), "consensus_top3": top3}


def test_le_tableau_annote_le_2_1():
    l = pt.tableau_round([_m("2-1")], derniers={})["lignes"][0]
    assert l["Score exact"].startswith("2-1 ·")
    assert l["Score exact"].endswith("⚽"), l["Score exact"]


def test_le_tableau_annote_le_1_2():
    l = pt.tableau_round([_m("1-2")], derniers={})["lignes"][0]
    assert l["Score exact"].endswith("⚽"), l["Score exact"]


def test_pas_de_ballon_hors_can():
    l = pt.tableau_round([_m("2-1")])["lignes"][0]   # derniers=None
    assert "⚽" not in l["Score exact"]


def test_le_ballon_sur_un_score_sinon():
    m = _m("0-0", sinon=[("2-1", 0.14), ("1-0", 0.10)])
    l = pt.tableau_round([m], derniers={})["lignes"][0]
    assert "⚽" not in l["Score exact"]            # tête = 0-0
    assert "2-1 14 % ⚽" in l["Sinon"], l["Sinon"]


def test_le_1_1_garde_son_emoji_pas_le_ballon():
    # Un 1-1 reste annoté par son propre emoji (🔒…), pas par ⚽.
    D = {"A": (0, 0), "B": (0, 0)}
    l = pt.tableau_round([_m("1-1")], derniers=D)["lignes"][0]
    assert l["Score exact"].endswith("🔒")
    assert "⚽" not in l["Score exact"]
