"""Annotation des pronostics 1-1 pour la CAN (Olivio, 15/10).

Sur un pronostic 1-1, selon le DERNIER match de chaque équipe (contre
n'importe qui, pas leur face-à-face) :
  🧱 les deux ont fait 0-0 ;
  🔻 domicile 0-0 + extérieur battu (🔻🔻 si 0-1) ;
  🔺 extérieur 0-0 + domicile vainqueur (🔺🔺 si 1-0).
Uniquement la CAN, uniquement les lignes 1-1.
"""
import sys
from pathlib import Path

import pandas as pd

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "scripts"))
import predict_trio as pt  # noqa: E402

D = {"A": (0, 0), "B": (0, 0), "C": (0, 1), "E": (0, 2),
     "D": (1, 0), "F": (2, 0), "G": (1, 1)}


# -------- emoji_nul_11 : les trois cas --------

def test_cas1_les_deux_ont_fait_0_0():
    assert pt.emoji_nul_11("A", "B", D) == "🧱"


def test_cas2_domicile_0_0_exterieur_battu():
    assert pt.emoji_nul_11("A", "C", D) == "🔻🔻"   # extérieur perdu 0-1
    assert pt.emoji_nul_11("A", "E", D) == "🔻"     # extérieur perdu 0-2


def test_cas3_exterieur_0_0_domicile_vainqueur():
    assert pt.emoji_nul_11("D", "B", D) == "🔺🔺"   # domicile gagné 1-0
    assert pt.emoji_nul_11("F", "B", D) == "🔺"     # domicile gagné 2-0


def test_aucun_cas_ne_s_applique():
    assert pt.emoji_nul_11("G", "A", D) == ""       # domicile 1-1 (ni 0-0)
    assert pt.emoji_nul_11("A", "D", D) == ""        # domicile 0-0 mais ext. a gagné
    assert pt.emoji_nul_11("A", "ZZZ", D) == ""      # équipe sans historique
    assert pt.emoji_nul_11("A", "B", {}) == ""       # pas de derniers


def test_les_cas_s_excluent():
    # domicile 0-0 et extérieur 0-0 -> cas 1, jamais cas 2.
    assert pt.emoji_nul_11("A", "B", D) == "🧱"


# -------- derniers_resultats --------

def test_dernier_match_par_equipe(monkeypatch):
    # du + récent au + ancien : le 1er vu par équipe est son dernier match.
    rows = [("A", "B", 0, 0), ("C", "A", 2, 1), ("B", "D", 3, 0)]
    monkeypatch.setattr(pt.pd, "read_sql",
                        lambda q, e: pd.DataFrame(rows, columns=["ta", "tb", "sa", "sb"]))
    d = pt.derniers_resultats(None, "X")
    assert d["A"] == (0, 0)     # son dernier : A-B 0-0 (vu de A)
    assert d["B"] == (0, 0)     # idem (vu de B)
    assert d["C"] == (2, 1)     # C-A 2-1 vu de C
    assert d["D"] == (0, 3)     # B-D 3-0 vu de D


# -------- intégration dans le tableau --------

def _m11(a, b, score="1-1"):
    return {"team_a": a, "team_b": b, "cotes": [2.0, 3.2, 3.8],
            "x12": [0.33, 0.34, 0.33], "confidence": 0.25, "over25_pct": 45.0,
            "top1_calibre": (score, 0.17),
            "consensus_top3": [(score, 0.17), ("0-0", 0.14), ("2-1", 0.10)]}


def test_le_tableau_annote_le_1_1():
    lignes = pt.tableau_round([_m11("A", "B")], derniers=D)["lignes"]
    assert lignes[0]["Score exact"].startswith("1-1 ·")
    assert lignes[0]["Score exact"].endswith("🧱"), lignes[0]["Score exact"]


def test_pas_d_annotation_sans_derniers():
    lignes = pt.tableau_round([_m11("A", "B")])["lignes"]
    assert "🧱" not in lignes[0]["Score exact"]


def test_pas_d_annotation_si_le_score_n_est_pas_1_1():
    lignes = pt.tableau_round([_m11("A", "B", score="2-1")], derniers=D)["lignes"]
    assert "🧱" not in lignes[0]["Score exact"] and lignes[0]["Score exact"].startswith("2-1")


# -------- câblage : CAN seule --------

def test_la_can_seule_est_annotee():
    assert pt.CAN_LG in pt.LIGUES_ANNOTE_11
    src = (RACINE / "scripts" / "predict_trio.py").read_text(encoding="utf-8")
    bloc = src[src.index("def predict_round("):src.index("def main():")]
    assert "derniers_resultats(engine, lg) if lg in LIGUES_ANNOTE_11 else None" in bloc
    assert '"derniers": derniers' in bloc


# -------- annotation AUSSI sur les scores "Sinon" (15/10) --------

def test_un_1_1_en_sinon_est_annote():
    """Le pronostic de tête n'est pas 1-1, mais un des scores « Sinon » l'est :
    il doit porter l'emoji (cas Mozambique-Tanzanie, 15/10)."""
    m = {"team_a": "A", "team_b": "B", "cotes": [2.0, 3.2, 3.8],
         "x12": [0.33, 0.34, 0.33], "confidence": 0.25, "over25_pct": 45.0,
         "top1_calibre": ("0-0", 0.20),
         "consensus_top3": [("0-0", 0.20), ("1-1", 0.15), ("1-0", 0.10)]}
    l = pt.tableau_round([m], derniers=D)["lignes"][0]
    assert "🧱" not in l["Score exact"]          # tête = 0-0, pas d'emoji
    assert "1-1 15 % 🧱" in l["Sinon"], l["Sinon"]


def test_le_1_1_de_tete_et_de_sinon_sont_annotes():
    m = {"team_a": "A", "team_b": "B", "cotes": [2.0, 3.2, 3.8],
         "x12": [0.33, 0.34, 0.33], "confidence": 0.25, "over25_pct": 45.0,
         "top1_calibre": ("1-1", 0.17),
         "consensus_top3": [("1-1", 0.17), ("1-1", 0.12), ("0-0", 0.10)]}
    l = pt.tableau_round([m], derniers=D)["lignes"][0]
    assert l["Score exact"].endswith("🧱")
    assert "1-1 12 % 🧱" in l["Sinon"]


def test_sinon_sans_1_1_pas_d_emoji():
    m = {"team_a": "A", "team_b": "B", "cotes": [2.0, 3.2, 3.8],
         "x12": [0.33, 0.34, 0.33], "confidence": 0.25, "over25_pct": 45.0,
         "top1_calibre": ("0-0", 0.20),
         "consensus_top3": [("0-0", 0.20), ("1-0", 0.12), ("0-1", 0.10)]}
    l = pt.tableau_round([m], derniers=D)["lignes"][0]
    assert "🧱" not in l["Sinon"] and "🧱" not in l["Score exact"]


# -------- cas 4 : les deux ont perdu sans marquer 💀 (16/10) --------

def test_cas4_les_deux_ont_perdu_sans_marquer():
    D4 = {"A": (0, 2), "B": (0, 1), "C": (0, 3)}
    assert pt.emoji_nul_11("A", "B", D4) == "💀"   # 0-2 et 0-1
    assert pt.emoji_nul_11("A", "C", D4) == "💀"   # 0-2 et 0-3


def test_cas4_ne_declenche_pas_si_une_equipe_a_marque():
    # B a perdu 1-2 (a marqué) -> pas de double défaite blanche.
    D = {"A": (0, 2), "B": (1, 2)}
    assert pt.emoji_nul_11("A", "B", D) == ""


def test_cas4_ne_declenche_pas_si_une_equipe_n_a_pas_perdu():
    # B a fait 0-0 (nul, pas défaite) -> ni cas 4 ni cas 1 (A pas 0-0).
    D = {"A": (0, 2), "B": (0, 0)}
    assert pt.emoji_nul_11("A", "B", D) == ""
    # B a gagné 0 encaissé mais a marqué (2-0) -> pas perdu.
    D2 = {"A": (0, 2), "B": (2, 0)}
    assert pt.emoji_nul_11("A", "B", D2) == ""


def test_les_quatre_cas_s_excluent():
    # 0-0/0-0 -> 🧱, pas 💀 ; 0-X/0-Y -> 💀, jamais 🧱.
    assert pt.emoji_nul_11("A", "B", {"A": (0, 0), "B": (0, 0)}) == "🧱"
    assert pt.emoji_nul_11("A", "B", {"A": (0, 1), "B": (0, 1)}) == "💀"


def test_le_cas4_annote_le_1_1_au_tableau():
    m = {"team_a": "A", "team_b": "B", "cotes": [2.0, 3.2, 3.8],
         "x12": [0.33, 0.34, 0.33], "confidence": 0.25, "over25_pct": 45.0,
         "top1_calibre": ("1-1", 0.17),
         "consensus_top3": [("1-1", 0.17), ("0-0", 0.14), ("2-1", 0.10)]}
    l = pt.tableau_round([m], derniers={"A": (0, 2), "B": (0, 1)})["lignes"][0]
    assert l["Score exact"].endswith("💀"), l["Score exact"]
