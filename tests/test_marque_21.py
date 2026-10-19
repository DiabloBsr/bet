"""Annotation des pronostics 2-1 / 1-2 pour la CAN (Olivio, 19/10).

Un 2-1/1-2 suit la MÊME condition que le 1-1 (d'après le dernier match de
chaque équipe → 🔒 💀 🔻 🔺) ; si AUCUNE condition ne s'applique, il porte ⚽.
Sur le score exact principal ET les deux scores « Sinon ». Hors CAN : rien.
"""
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "scripts"))
import predict_trio as pt  # noqa: E402

# A/B : les deux ont fait 0-0 -> 🔒 ; Z : aucune condition (a gagné 2-0).
D = {"A": (0, 0), "B": (0, 0), "C": (0, 1), "Z": (2, 0), "Y": (2, 0)}


def _m(score, a="Z", b="Y", sinon=None):
    top3 = [(score, 0.17)] + (sinon or [("0-0", 0.14), ("1-0", 0.10)])
    return {"team_a": a, "team_b": b, "cotes": [2.0, 3.2, 3.8],
            "x12": [0.33, 0.34, 0.33], "confidence": 0.25, "over25_pct": 45.0,
            "top1_calibre": (score, 0.17), "consensus_top3": top3}


# -------- _marque_21 : la brique --------

def test_ballon_quand_aucune_condition():
    # Z et Y ont gagné 2-0 : aucune condition 1-1 -> ⚽.
    assert pt._marque_21("2-1", _m("2-1"), D) == " ⚽"
    assert pt._marque_21("1-2", _m("1-2"), D) == " ⚽"


def test_meme_condition_que_le_1_1():
    # A et B ont fait 0-0 -> 🔒, comme pour un 1-1.
    assert pt._marque_21("2-1", _m("2-1", "A", "B"), D) == " 🔒"
    assert pt._marque_21("1-2", _m("1-2", "A", "B"), D) == " 🔒"


def test_rien_sur_les_autres_scores():
    for sc in ("1-1", "0-0", "2-0", "1-0", "3-1", "2-2"):
        assert pt._marque_21(sc, _m(sc), D) == ""


def test_rien_hors_can():
    assert pt._marque_21("2-1", _m("2-1"), None) == ""
    assert pt._marque_21("1-2", _m("1-2"), None) == ""


# -------- intégration dans le tableau --------

def test_le_tableau_met_le_ballon_sans_condition():
    l = pt.tableau_round([_m("2-1")], derniers=D)["lignes"][0]
    assert l["Score exact"].startswith("2-1 ·")
    assert l["Score exact"].endswith("⚽"), l["Score exact"]


def test_le_tableau_met_la_condition_1_1_sur_un_2_1():
    l = pt.tableau_round([_m("2-1", "A", "B")], derniers=D)["lignes"][0]
    assert l["Score exact"].endswith("🔒"), l["Score exact"]
    assert "⚽" not in l["Score exact"]


def test_pas_d_annotation_hors_can():
    l = pt.tableau_round([_m("2-1")])["lignes"][0]   # derniers=None
    assert "⚽" not in l["Score exact"] and "🔒" not in l["Score exact"]


def test_le_ballon_sur_un_score_sinon():
    m = _m("0-0", sinon=[("1-2", 0.14), ("1-0", 0.10)])
    l = pt.tableau_round([m], derniers=D)["lignes"][0]
    assert "⚽" not in l["Score exact"]            # tête = 0-0
    assert "1-2 14 % ⚽" in l["Sinon"], l["Sinon"]


def test_le_1_1_garde_son_emoji():
    l = pt.tableau_round([_m("1-1", "A", "B")], derniers=D)["lignes"][0]
    assert l["Score exact"].endswith("🔒")
