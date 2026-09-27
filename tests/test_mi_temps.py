"""Les deux mi-temps dans l'onglet « Que jouer ? » (demande du 27/09).

MESURE FONDATRICE, sur 208 331 resultats propres : la 1re mi-temps ne depasse
JAMAIS 3 buts -- 0 (31,35 %), 1 (27,87 %), 2 (27,23 %), 3 (13,55 %), pas une
exception. La 2e non plus, a 47 matchs pres (0,02 %). C'est la vraie contrainte
du moteur, et elle explique celle deja trouvee sur le plein-temps : TOTAL_MAX =
6, c'est 3 + 3.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import predict_trio as pt  # noqa: E402

LAMBDAS = [(1.57, 1.16), (0.4, 0.4), (3.5, 0.3), (2.2, 2.4), (0.15, 6.0)]


@pytest.mark.parametrize("la,lb", LAMBDAS)
def test_deux_periodes_rendues(la, lb):
    mt = pt.marches_mi_temps(la, lb)
    assert set(mt) == {"1re mi-temps", "2e mi-temps"}
    for d in mt.values():
        assert [k for k, _ in d["x12"]] == ["1", "X", "2"]
        assert d["scores"] and d["attendus"] > 0


@pytest.mark.parametrize("la,lb", LAMBDAS)
def test_aucun_score_au_dela_du_plafond(la, lb):
    """Un 4-0 de mi-temps n'est jamais sorti en 208 331 matchs : la grille ne
    doit pas lui donner la moindre masse, quelles que soient les intensites."""
    for d in pt.marches_mi_temps(la, lb, top=49).values():
        for sc, pr in d["scores"]:
            a, b = (int(x) for x in sc.split("-"))
            assert a + b <= pt.HALF_MAX, f"{sc} depasse le plafond de mi-temps"
            assert pr >= 0
        # Le score le plus probable, lui, porte toujours de la masse. Les
        # suivants peuvent etre arrondis a 0 % : la calibration est ancree en
        # (0,0), donc une proba brute infime en ressort nulle -- c'est voulu,
        # et ces scores-la ne sont de toute facon jamais affiches (top 3).
        assert d["scores"][0][1] > 0


def test_le_plafond_plein_temps_est_la_somme_des_deux():
    # 6 = 3 + 3. Si l'un des deux bouge sans l'autre, le moteur se contredit.
    assert pt.TOTAL_MAX == 2 * pt.HALF_MAX


# --------------------------------------------------------------------------
# LES DEUX MI-TEMPS SONT DIFFERENTES — et dans le bon sens
# --------------------------------------------------------------------------

def test_la_seconde_porte_plus_de_buts_que_la_premiere():
    # Mesure : 1,2303 but en 1re mi-temps, 1,5031 en 2e. Le rapport tient a
    # PART_MT1 = 0,45, dont la part REELLE mesuree vaut 0,4501.
    mt = pt.marches_mi_temps(1.57, 1.16)
    assert mt["2e mi-temps"]["attendus"] > mt["1re mi-temps"]["attendus"]


def test_les_buts_attendus_collent_a_la_mesure():
    # Nourri des lambdas moyennes reelles, le modele doit retrouver les
    # moyennes reelles par periode, a 0,05 but pres.
    mt = pt.marches_mi_temps(1.571 / pt.LAM_SCALE, 1.163 / pt.LAM_SCALE)
    assert abs(mt["1re mi-temps"]["attendus"] - 1.2303) < 0.05
    assert abs(mt["2e mi-temps"]["attendus"] - 1.5031) < 0.05


def test_les_echelles_ne_valent_pas_un():
    """Plafonner puis renormaliser abaisse la moyenne : sans correction, le
    modele sous-estime les buts de chaque periode. Les deux echelles ont ete
    ajustees sur le TRAIN et VERIFIEES hors echantillon."""
    assert pt.HALF_SCALE["1re mi-temps"] > 1.0
    # La 2e corrige plus : elle porte plus de buts, le plafond y mord davantage.
    assert pt.HALF_SCALE["2e mi-temps"] > pt.HALF_SCALE["1re mi-temps"]


# --------------------------------------------------------------------------
# CALIBRATION
# --------------------------------------------------------------------------

def test_la_table_de_mi_temps_est_chargee():
    # Sans elle, l'app afficherait du brut : 46,9 % annonces pour 42,1 %
    # touches sur le 1X2 de 2e periode.
    assert pt.mi_temps_calibre() is True
    for d in pt.marches_mi_temps(1.5, 1.2).values():
        assert d["calibre"] is True


def test_calib_mi_temps_encaisse_les_entrees_illisibles():
    for mauvais in (None, "x", float("nan"), [], {}):
        assert pt.calib_mi_temps("1re mi-temps 1X2", mauvais) == 0.0


def test_calib_mi_temps_sans_table_rend_le_brut():
    # Fichier absent au deploiement : mieux vaut une valeur non corrigee qu'un
    # zero silencieux. L'interface dit laquelle des deux elle montre.
    sauve = pt._MT_CAL
    try:
        pt._MT_CAL = {}
        assert pt.calib_mi_temps("1re mi-temps 1X2", 0.42) == pytest.approx(0.42)
        assert pt.mi_temps_calibre() is False
    finally:
        pt._MT_CAL = sauve


def test_la_calibration_plafonne_au_dernier_point_mesure():
    # Le defaut deja corrige ailleurs dans ce depot : extrapoler au-dela du
    # dernier point mesure affichait 100 % sur un marche mesure a 26,7 %.
    for cle, tab in pt._MT_CAL.items():
        haut = max(b["real"] for b in tab["bins"])
        assert pt.calib_mi_temps(cle, 0.999) <= haut + 1e-9, cle


# --------------------------------------------------------------------------
# BRANCHEMENT SUR L'ONGLET
# --------------------------------------------------------------------------

def test_les_mi_temps_ne_rejoignent_pas_les_marches_cotes():
    """Elles ne sont pas cotees par Bet261 : les verser dans `lignes` ferait
    recommander « a jouer » un pari qui n'existe pas."""
    src = (Path(__file__).resolve().parents[1] / "scripts" / "predict_trio.py"
           ).read_text(encoding="utf-8")
    bloc = src[src.index("def conseil("):src.index("def _z_bonferroni(")]
    assert '"mi_temps": marches_mi_temps(' in bloc
    assert "lignes.append" in bloc and "mi_temps" not in bloc.split("lignes.append")[1][:400]


def test_l_onglet_affiche_les_deux_periodes():
    src = (Path(__file__).resolve().parents[1] / "scripts" / "dashboard_trio.py"
           ).read_text(encoding="utf-8")
    assert 'res_c.get("mi_temps")' in src
    assert "1re mi-temps" in src and "2e mi-temps" in src


def test_la_calibration_part_au_deploiement():
    # Oubliee dans la liste, l'app en ligne afficherait du brut sans le dire.
    src = (Path(__file__).resolve().parents[1] / "deploy" / "deploy_hf.py"
           ).read_text(encoding="utf-8")
    assert "config/mitemps_calibration.json" in src
