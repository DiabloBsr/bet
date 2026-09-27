"""Le seuil de cote sur l'onglet « Que jouer ? » (demande du 27/09).

« Garder tout mais affiche juste ceux de ton pronostic >= 1,2 ». Le conseil de
tete etant toujours un double chance a petite cote, ce seuil est la seule chose
qui distingue un conseil jouable d'un conseil qui ne paie rien.
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import predict_trio as pt  # noqa: E402


def c(odds, marche="Double Chance", sel="X2"):
    """Un resultat de `conseil()` reduit a ce que le filtre regarde."""
    return {"home": "A", "away": "B", "sur": {"marche": marche, "sel": sel,
                                              "odds": odds, "p": 0.8}}


def test_garde_au_dessus_et_ecarte_en_dessous():
    res = [c(1.29), c(1.19), c(1.20), c(1.00)]
    gardees, trop_bas, sans_cote = pt.filtrer_conseils(res, 1.20)
    assert [g["sur"]["odds"] for g in gardees] == [1.29, 1.20]
    assert (trop_bas, sans_cote) == (2, 0)


def test_le_seuil_est_inclusif():
    # « superieur ou EGAL a 1,2 » : 1.20 passe.
    gardees, trop_bas, _ = pt.filtrer_conseils([c(1.20)], 1.20)
    assert len(gardees) == 1 and trop_bas == 0


def test_inclusif_malgre_le_binaire():
    # 1.20 et 2.30 ne sont pas representables exactement en binaire : sans la
    # tolerance, une cote EGALE au seuil pouvait etre rejetee.
    for v in (1.20, 1.15, 2.30, 1.05, 3.70):
        gardees, trop_bas, _ = pt.filtrer_conseils([c(v)], v)
        assert len(gardees) == 1, f"{v} doit passer son propre seuil"
        assert trop_bas == 0


def test_conseil_non_cote_ecarte_mais_compte():
    # Faute de prix on ne peut pas affirmer qu'il passe le seuil. Mais il doit
    # etre annonce : une rencontre qui disparait en silence est un bug.
    gardees, trop_bas, sans_cote = pt.filtrer_conseils([c(None)], 1.20)
    assert gardees == [] and (trop_bas, sans_cote) == (0, 1)


def test_cote_illisible_traitee_comme_absente():
    # L'historique de ce depot : un champ de marche est parfois une chaine, un
    # entier, ou NaN. Aucun de ces cas ne doit lever.
    for mauvais in ("", "n/a", float("nan"), [], {}):
        gardees, trop_bas, sans_cote = pt.filtrer_conseils([c(mauvais)], 1.20)
        assert gardees == [], f"{mauvais!r} ne doit pas passer"
        assert (trop_bas, sans_cote) == (0, 1), f"{mauvais!r} doit etre compte"


def test_cote_en_chaine_numerique_est_lue():
    gardees, _, sans_cote = pt.filtrer_conseils([c("1.30")], 1.20)
    assert len(gardees) == 1 and sans_cote == 0


def test_une_erreur_reste_visible():
    # Elle n'a pas de cote, mais la masquer ferait croire que la rencontre
    # n'existe pas plutot qu'elle n'a pas pu etre analysee.
    res = [{"erreur": "Pas assez d'historique"}, c(1.00)]
    gardees, trop_bas, sans_cote = pt.filtrer_conseils(res, 1.20)
    assert len(gardees) == 1 and gardees[0].get("erreur")
    assert (trop_bas, sans_cote) == (1, 0)


def test_seuil_a_un_affiche_tout():
    res = [c(1.00), c(1.01), c(2.50)]
    gardees, trop_bas, sans_cote = pt.filtrer_conseils(res, 1.00)
    assert len(gardees) == 3 and (trop_bas, sans_cote) == (0, 0)


def test_rien_ne_passe_sans_lever():
    gardees, trop_bas, _ = pt.filtrer_conseils([c(1.05), c(1.10)], 1.20)
    assert gardees == [] and trop_bas == 2


def test_entrees_degenerees():
    assert pt.filtrer_conseils(None, 1.20) == ([], 0, 0)
    assert pt.filtrer_conseils([], 1.20) == ([], 0, 0)
    # Une entree qui n'est pas un dict est ignoree, pas une cause de plantage.
    assert pt.filtrer_conseils(["bruit", None, 42], 1.20) == ([], 0, 0)


def test_sur_absent_ou_vide():
    # `conseil()` rend `sur: None` quand aucune ligne n'a pu etre calculee.
    for r in ({"sur": None}, {"sur": {}}, {}):
        gardees, trop_bas, sans_cote = pt.filtrer_conseils([r], 1.20)
        assert gardees == [] and (trop_bas, sans_cote) == (0, 1)


def test_comptes_et_gardees_couvrent_toute_l_entree():
    # Aucune rencontre ne doit s'evaporer : gardees + ecartees == entree.
    res = [c(1.29), c(1.19), c(None), {"erreur": "x"}, c(1.21), c(1.00)]
    gardees, trop_bas, sans_cote = pt.filtrer_conseils(res, 1.20)
    assert len(gardees) + trop_bas + sans_cote == len(res)


def test_le_filtre_ne_touche_pas_au_detail():
    # « Garder tout » : la rencontre retenue ressort ENTIERE, ses onze marches
    # inclus. Le filtre trie, il ne rogne pas.
    r = c(1.30)
    r["lignes"] = [{"marche": f"m{i}", "sel": "s", "p": 0.5, "odds": 1.01}
                   for i in range(11)]
    gardees, _, _ = pt.filtrer_conseils([r], 1.20)
    assert gardees[0] is r
    assert len(gardees[0]["lignes"]) == 11
