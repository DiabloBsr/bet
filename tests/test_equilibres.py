"""L'onglet « 1X2 à cote 2 » : les rencontres sans favori court (27/09).

« Crée un onglet où 1X2 égal à cote 2 chacun. »

Pris au mot, c'est impossible : trois cotes EGALES a 2,00 exigent une somme
d'inverses de 1,500, soit 50 % de marge. Mesure sur 155 298 rencontres cotees
en base : le book tourne a 1,060 (min 1,055, max 1,070), soit ~6 %. La
rencontre la plus equilibree possible sous cette marge porte trois cotes
voisines de 2,83 -- et AUCUNE des 155 298 n'a ses trois cotes au-dessus de
3,00.

Le seuil se lit donc « au moins » : a 2,00 il retient 34,3 % des rencontres.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import predict_trio as pt  # noqa: E402

RACINE = Path(__file__).resolve().parents[1]


def m(a, x, b, **kw):
    """Une rencontre reduite a ce que le debusqueur regarde."""
    d = {"home": "Dom", "away": "Ext", "cotes": {"1": a, "X": x, "2": b}}
    d.update(kw)
    return d


def r(a, x, b, **kw):
    d = {"home": "Dom", "away": "Ext", "cotes": {"1": a, "X": x, "2": b}}
    d.update(kw)
    return d


def test_garde_les_rencontres_dont_les_trois_cotes_atteignent_le_seuil():
    g, e, s = pt.filtrer_equilibres([r(2.5, 3.4, 2.8), r(2.0, 3.2, 2.1)], 2.0)
    assert len(g) == 2 and (e, s) == (0, 0)


def test_ecarte_des_qu_UNE_des_trois_est_sous_le_seuil():
    # C'est bien le MINIMUM des trois qui decide : un favori court suffit a
    # disqualifier la rencontre, meme si les deux autres paient tres bien.
    for trio in ((1.4, 4.5, 7.0), (2.5, 1.9, 3.0), (3.0, 3.1, 1.95)):
        g, e, s = pt.filtrer_equilibres([r(*trio)], 2.0)
        assert g == [] and e == 1, trio


def test_le_seuil_est_inclusif():
    g, e, _ = pt.filtrer_equilibres([r(2.0, 2.0, 2.0)], 2.0)
    assert len(g) == 1 and e == 0


def test_inclusif_malgre_le_binaire():
    for v in (2.0, 2.05, 2.20, 2.35, 2.50):
        g, e, _ = pt.filtrer_equilibres([r(v, v, v)], v)
        assert len(g) == 1, v
        assert e == 0


def test_une_cote_manquante_ecarte_mais_compte():
    """On ne peut ni affirmer qu'elle passe le seuil, ni la taire."""
    for trio in ((2.5, None, 2.8), (None, 3.2, 2.9), (2.4, 3.1, None)):
        g, e, s = pt.filtrer_equilibres([r(*trio)], 2.0)
        assert g == [] and (e, s) == (0, 1), trio


def test_cotes_illisibles_traitees_comme_absentes():
    for mauvais in ("", "n/a", 0, -1, float("nan")):
        g, e, s = pt.filtrer_equilibres([r(mauvais, 3.2, 2.9)], 2.0)
        assert g == [] and s == 1, mauvais


def test_une_cote_a_un_est_trop_courte_pas_illisible():
    """⚠️ `_odd_pos` accepte 1,00 : c'est la notion de cote valide deja en
    place dans le depot, et ce filtre s'y range plutot que d'en inventer une
    plus stricte. Une cote a 1,00 est donc ECARTEE — elle est bien sous le
    seuil — et non comptee comme illisible. Le resultat pour l'utilisateur est
    le meme, seul le motif annonce change."""
    g, e, s = pt.filtrer_equilibres([r(1.0, 3.2, 2.9)], 2.0)
    assert g == [] and (e, s) == (1, 0)


def test_une_erreur_reste_visible():
    g, e, s = pt.filtrer_equilibres([{"erreur": "historique insuffisant"},
                                     r(1.4, 4.5, 7.0)], 2.0)
    assert len(g) == 1 and g[0].get("erreur")
    assert e == 1


def test_comptes_couvrent_toute_l_entree():
    entree = [r(2.5, 3.4, 2.8), r(1.4, 4.5, 7.0), r(2.5, None, 2.8),
              {"erreur": "x"}, r(2.1, 3.3, 2.6)]
    g, e, s = pt.filtrer_equilibres(entree, 2.0)
    assert len(g) + e + s == len(entree)


def test_entrees_degenerees():
    assert pt.filtrer_equilibres(None) == ([], 0, 0)
    assert pt.filtrer_equilibres([]) == ([], 0, 0)
    assert pt.filtrer_equilibres(["bruit", None, 42]) == ([], 0, 0)


def test_le_seuil_par_defaut_est_deux():
    assert pt.COTE_EQUILIBRE == 2.0
    g, _, _ = pt.filtrer_equilibres([r(2.5, 3.4, 2.8)])
    assert len(g) == 1


def test_le_filtre_ne_rogne_pas_la_rencontre():
    """Il choisit, il ne modifie pas : la rencontre retenue ressort entiere."""
    m = r(2.5, 3.4, 2.8, pieges=["nul menaçant"], probas={"1": .4},
          seq_a="VVNDV")
    g, _, _ = pt.filtrer_equilibres([m], 2.0)
    assert g[0] is m and g[0]["pieges"] == ["nul menaçant"]


def test_le_filtre_par_seuil_reste_disponible():
    """⚠️ `filtrer_equilibres` n'est plus cable a l'ecran depuis la
    rectification du 27/09 : l'onglet cherche desormais une CIBLE exacte, pas
    un seuil. La fonction est conservee — elle ne coute rien tant que rien ne
    l'appelle, et repondre un jour a « les trois cotes au moins X » ne
    demandera que son affichage."""
    assert callable(pt.filtrer_equilibres) and pt.COTE_EQUILIBRE == 2.0


def test_l_onglet_reutilise_le_moteur_existant():
    """`round_1x2` avait ete conservee apres le retrait de son onglet,
    precisement pour qu'un nouvel ecran n'ait que son affichage a ecrire."""
    src = (RACINE / "scripts" / "dashboard_trio.py").read_text(encoding="utf-8")
    assert "Débusqueur 1X2 à cote 2" in src
    assert "round_1x2(" in src, "l'onglet doit reutiliser le moteur, pas le refaire"
    assert "debusquer_cotes(" in src


# --------------------------------------------------------------------------
# LE DEBUSQUEUR — cible exacte sur les TROIS cotes (rectification du 27/09)
#
# « Ce que je veux, c'est que vous débusquiez le match 1X2 à cote 2 chacun,
#   c'est-à-dire 1 = 2, X = 2 et 2 = 2. »
#
# Mesure : AUCUN des 184 105 releves de cotes en base n'atteint cette cible,
# meme a +/- 0,80. Les plus proches sont autour de 2,80 / 2,85 / 2,82. C'est
# arithmetique : trois cotes a 2,00 font 50 % de marge, le book tourne a 6 %.
# --------------------------------------------------------------------------

def test_la_cible_par_defaut_est_deux_partout():
    assert pt.CIBLE_TROIS_COTES == (2.0, 2.0, 2.0)


def test_ecart_nul_sur_la_cible_exacte():
    assert pt.ecart_aux_cibles({"1": 2.0, "X": 2.0, "2": 2.0}) == 0.0


def test_ecart_somme_les_trois_distances():
    assert pt.ecart_aux_cibles({"1": 2.1, "X": 1.9, "2": 2.2}) == pytest.approx(0.4)


def test_ecart_none_des_qu_une_cote_manque():
    """Sur deux cotes, la distance paraitrait MEILLEURE qu'une rencontre
    complete — la rencontre remonterait en tete du classement a tort."""
    for c in ({"1": 2.0, "X": None, "2": 2.0}, {"1": 2.0, "X": 2.0}, {}):
        assert pt.ecart_aux_cibles(c) is None


def test_trouve_la_cible_dans_la_tolerance():
    r = pt.debusquer_cotes([m(2.0, 2.02, 1.98), m(1.4, 4.5, 7.0)], tol=0.05)
    assert len(r["trouvees"]) == 1
    assert r["trouvees"][0]["ecart"] == pytest.approx(0.04)


def test_la_tolerance_porte_sur_chaque_cote_pas_sur_la_somme():
    """Trois ecarts de 0,04 font 0,12 au total : un seuil GLOBAL de 0,05 les
    refuserait, alors que chacune est dans la tolerance."""
    r = pt.debusquer_cotes([m(2.04, 1.96, 2.04)], tol=0.05)
    assert len(r["trouvees"]) == 1 and r["trouvees"][0]["ecart"] == pytest.approx(0.12)


def test_une_seule_cote_hors_tolerance_disqualifie():
    r = pt.debusquer_cotes([m(2.0, 2.0, 2.2)], tol=0.05)
    assert r["trouvees"] == [] and len(r["proches"]) == 1


def test_les_plus_proches_sont_classes_par_ecart():
    r = pt.debusquer_cotes([m(1.4, 4.5, 7.0), m(2.83, 2.87, 2.78), m(2.5, 3.0, 2.6)],
                           tol=0.01)
    assert [x["ecart"] for x in r["proches"]] == sorted(x["ecart"] for x in r["proches"])
    assert r["proches"][0]["cotes"]["1"] == 2.5


def test_une_trouvee_n_est_pas_aussi_une_proche():
    r = pt.debusquer_cotes([m(2.0, 2.0, 2.0), m(2.83, 2.87, 2.78)], tol=0.05)
    assert len(r["trouvees"]) == 1 and len(r["proches"]) == 1
    assert r["proches"][0]["cotes"]["1"] == 2.83


def test_le_nombre_de_proches_est_plafonne():
    lot = [m(2.0 + i / 10, 2.0, 2.0) for i in range(1, 12)]
    assert len(pt.debusquer_cotes(lot, tol=0.01, proches=5)["proches"]) == 5


def test_sans_cote_comptee_et_jamais_classee():
    r = pt.debusquer_cotes([m(2.0, None, 2.0), m(2.0, 2.0, 2.0)], tol=0.05)
    assert r["sans_cote"] == 1
    assert len(r["trouvees"]) == 1 and r["proches"] == []


def test_les_erreurs_ne_sont_pas_examinees():
    r = pt.debusquer_cotes([{"erreur": "x"}, m(2.0, 2.0, 2.0)], tol=0.05)
    assert r["examinees"] == 1


def test_cible_libre_et_non_figee_a_deux():
    r = pt.debusquer_cotes([m(1.5, 3.5, 6.0)], cibles=(1.5, 3.5, 6.0), tol=0.0)
    assert len(r["trouvees"]) == 1 and r["trouvees"][0]["ecart"] == 0.0


def test_entrees_degenerees():
    for e in (None, [], ["bruit", None, 42]):
        r = pt.debusquer_cotes(e)
        assert r["trouvees"] == [] and r["proches"] == [] and r["examinees"] == 0


def test_la_rencontre_n_est_pas_rognee():
    """L'ecart est AJOUTE, le reste de la rencontre survit intact."""
    src = m(2.0, 2.0, 2.0, pieges=["nul menaçant"], probas={"1": 0.4},
            seq_a="VVNDV", equipe="Dom")
    g = pt.debusquer_cotes([src], tol=0.05)["trouvees"][0]
    assert g["pieges"] == ["nul menaçant"] and g["seq_a"] == "VVNDV"
    assert g["ecart"] == 0.0
    # L'original n'est pas modifie : « ecart » ne doit pas y apparaitre.
    assert "ecart" not in src
