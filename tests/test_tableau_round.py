"""La prédiction du round en tableau, avec l'over/under 2,5 (04/10).

« Affiche le résultat de prédiction de cette ligne sous forme tableau, et
ajoute aussi des pronostics over/under 2,5. »
"""
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "scripts"))
import predict_trio as pt  # noqa: E402


def _m(a, b, x12, conf=0.25, o25=55.0, cotes=(2.0, 3.2, 3.8)):
    return {"match": f"{a} v {b}", "team_a": a, "team_b": b,
            "cotes": list(cotes), "x12": list(x12), "over25_pct": o25,
            "confidence": conf, "top1_calibre": ("1-0", 0.13),
            "consensus_top3": [("1-0", 0.12), ("1-1", 0.11), ("0-1", 0.09)]}


# --------------------------------------------------------------------------
# LE PRONOSTIC 1X2 : LA REGLE DE LA VUE EN LISTES, A L'IDENTIQUE
# --------------------------------------------------------------------------

def test_le_favori_domicile():
    assert pt.issue_round(_m("A", "B", (0.5, 0.3, 0.2))) == ("1", "A", 0.5, 2.0)


def test_le_favori_exterieur():
    assert pt.issue_round(_m("A", "B", (0.2, 0.3, 0.5))) == ("2", "B", 0.5, 3.8)


def test_le_nul_seulement_s_il_est_strictement_devant():
    assert pt.issue_round(_m("A", "B", (0.3, 0.4, 0.3)))[:2] == ("X", "Nul")
    # Egalites : le 1 l'emporte, puis le 2 -- comme avant.
    assert pt.issue_round(_m("A", "B", (0.4, 0.4, 0.2)))[0] == "1"
    assert pt.issue_round(_m("A", "B", (0.2, 0.4, 0.4)))[0] == "2"


# --------------------------------------------------------------------------
# LE TABLEAU
# --------------------------------------------------------------------------

def test_les_colonnes_demandees():
    lignes = pt.tableau_round([_m("A", "B", (0.5, 0.3, 0.2))])["lignes"]
    assert list(lignes[0]) == list(pt.ROUND_COLONNES)
    l = lignes[0]
    assert (l["Match"], l["Pronostic"], l["1"], l["X"], l["2"]) == ("A – B", "A", 50, 30, 20)
    assert l["Cote"] == 2.0


def test_l_over_under_2_5_est_celui_que_le_moteur_calculait():
    l = pt.tableau_round([_m("A", "B", (0.5, 0.3, 0.2), o25=62.4)])["lignes"][0]
    assert (l["Over 2,5"], l["Under 2,5"]) == (62, 38)


def test_sans_over_2_5_les_cases_restent_vides():
    l = pt.tableau_round([_m("A", "B", (0.5, 0.3, 0.2), o25=None)])["lignes"][0]
    assert l["Over 2,5"] is None and l["Under 2,5"] is None


def test_l_ordre_de_la_vue_en_listes():
    """Le Top 3 (score le plus concentre) d'abord, marque 🏆, puis les autres
    du plus sur au moins sur."""
    ms = [_m("P1", "Q1", (0.40, 0.3, 0.3), conf=0.20),
          _m("P2", "Q2", (0.70, 0.2, 0.1), conf=0.21),
          _m("P3", "Q3", (0.45, 0.3, 0.25), conf=0.40),
          _m("P4", "Q4", (0.50, 0.3, 0.2), conf=0.35),
          _m("P5", "Q5", (0.60, 0.2, 0.2), conf=0.30)]
    lignes = pt.tableau_round(ms)["lignes"]
    assert [l["Match"][:2] for l in lignes] == ["P3", "P4", "P5", "P2", "P1"]
    assert [l["Top 3"] for l in lignes] == ["🏆"] * 3 + [""] * 2


def test_mon_pronostic_est_colore():
    t = pt.tableau_round([_m("A", "B", (0.2, 0.3, 0.5), o25=40.0),
                          _m("C", "D", (0.5, 0.3, 0.2), o25=70.0)])
    assert t["reco"][0] == {"2", "Under 2,5"}
    assert t["reco"][1] == {"1", "Over 2,5"}


def test_le_cote_over_under_se_lit_sur_la_valeur_exacte():
    """Arrondies, 50,4 / 49,6 feraient 50 / 50 : pas d'egalite pour autant."""
    t = pt.tableau_round([_m("A", "B", (0.5, 0.3, 0.2), o25=50.4)])
    assert "Over 2,5" in t["reco"][0]
    t = pt.tableau_round([_m("A", "B", (0.5, 0.3, 0.2), o25=50.0)])
    assert t["reco"][0] == {"1"}


def test_les_cases_colorees_sont_des_colonnes():
    t = pt.tableau_round([_m("A", "B", (0.3, 0.4, 0.3), o25=45.0)])
    assert t["reco"][0] <= set(pt.ROUND_COLONNES)


def test_un_round_vide():
    assert pt.tableau_round([]) == {"lignes": [], "reco": []}
    assert pt.tableau_round(None) == {"lignes": [], "reco": []}
