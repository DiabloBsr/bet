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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import predict_trio as pt  # noqa: E402

RACINE = Path(__file__).resolve().parents[1]


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


def test_l_onglet_reutilise_le_moteur_existant():
    """`round_1x2` avait ete conservee apres le retrait de son onglet,
    precisement pour qu'un nouvel ecran n'ait que son affichage a ecrire."""
    src = (RACINE / "scripts" / "dashboard_trio.py").read_text(encoding="utf-8")
    assert "1X2 à cote 2" in src
    assert "round_1x2(" in src, "l'onglet doit reutiliser le moteur, pas le refaire"
    assert "filtrer_equilibres(" in src
