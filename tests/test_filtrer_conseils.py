"""Le tri de l'onglet Que jouer ? — seuil de cote et conseil retenu.

Demande du 27/09 : « afficher les rencontres ou les predictions X2 a une cote
superieure 1,20, et les autres enleve ». Deux criteres, donc : QUEL conseil, et
a QUELLE cote. Le conseil de tete etant structurellement un double chance
(1X / X2 / 12), ces deux criteres ne se recouvrent pas — sur les douze
rencontres mesurees, sept passent le seuil mais deux seulement sont des X2.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import predict_trio as pt  # noqa: E402


def c(odds, marche="Double Chance", sel="X2"):
    """Un resultat de `conseil()` reduit a ce que le tri regarde."""
    return {"home": "A", "away": "B",
            "sur": {"marche": marche, "sel": sel, "odds": odds, "p": 0.8}}


# --------------------------------------------------------------------------
# Le SEUIL DE COTE
# --------------------------------------------------------------------------

def test_garde_au_dessus_et_ecarte_en_dessous():
    t = pt.filtrer_conseils([c(1.29), c(1.19), c(1.20), c(1.00)], 1.20)
    assert [g["sur"]["odds"] for g in t.gardees] == [1.29, 1.20]
    assert (t.trop_bas, t.sans_cote, t.hors_selection) == (2, 0, 0)


def test_le_seuil_est_inclusif():
    # « superieur ou EGAL a 1,2 » : 1.20 passe.
    t = pt.filtrer_conseils([c(1.20)], 1.20)
    assert len(t.gardees) == 1 and t.trop_bas == 0


def test_inclusif_malgre_le_binaire():
    # 1.20 et 2.30 ne sont pas representables exactement en binaire : sans la
    # tolerance, une cote EGALE au seuil pouvait etre rejetee.
    for v in (1.20, 1.15, 2.30, 1.05, 3.70):
        t = pt.filtrer_conseils([c(v)], v)
        assert len(t.gardees) == 1, f"{v} doit passer son propre seuil"
        assert t.trop_bas == 0


def test_seuil_a_un_affiche_tout():
    t = pt.filtrer_conseils([c(1.00), c(1.01), c(2.50)], 1.00)
    assert len(t.gardees) == 3
    assert (t.trop_bas, t.sans_cote, t.hors_selection) == (0, 0, 0)


def test_rien_ne_passe_sans_lever():
    t = pt.filtrer_conseils([c(1.05), c(1.10)], 1.20)
    assert t.gardees == [] and t.trop_bas == 2


# --------------------------------------------------------------------------
# Les COTES ILLISIBLES — un champ de marche est parfois une chaine, un entier,
# ou NaN : tout l'historique de ce depot le rappelle.
# --------------------------------------------------------------------------

def test_conseil_non_cote_ecarte_mais_compte():
    # Faute de prix on ne peut pas affirmer qu'il passe le seuil. Mais il doit
    # etre annonce : une rencontre qui disparait en silence est un bug.
    t = pt.filtrer_conseils([c(None)], 1.20)
    assert t.gardees == [] and (t.trop_bas, t.sans_cote) == (0, 1)


def test_cote_illisible_traitee_comme_absente():
    for mauvais in ("", "n/a", float("nan"), [], {}):
        t = pt.filtrer_conseils([c(mauvais)], 1.20)
        assert t.gardees == [], f"{mauvais!r} ne doit pas passer"
        assert (t.trop_bas, t.sans_cote) == (0, 1), f"{mauvais!r} doit etre compte"


def test_cote_en_chaine_numerique_est_lue():
    t = pt.filtrer_conseils([c("1.30")], 1.20)
    assert len(t.gardees) == 1 and t.sans_cote == 0


def test_sur_absent_ou_vide():
    # `conseil()` rend `sur: None` quand aucune ligne n'a pu etre calculee.
    for r in ({"sur": None}, {"sur": {}}, {}):
        t = pt.filtrer_conseils([r], 1.20)
        assert t.gardees == [] and (t.trop_bas, t.sans_cote) == (0, 1)


def test_entrees_degenerees():
    assert pt.filtrer_conseils(None, 1.20) == ([], 0, 0, 0)
    assert pt.filtrer_conseils([], 1.20) == ([], 0, 0, 0)
    # Une entree qui n'est pas un dict est ignoree, pas une cause de plantage.
    assert pt.filtrer_conseils(["bruit", None, 42], 1.20) == ([], 0, 0, 0)


# --------------------------------------------------------------------------
# Le CONSEIL RETENU
# --------------------------------------------------------------------------

def test_x2_seul_ecarte_les_autres_double_chance():
    res = [c(1.27, sel="X2"), c(1.29, sel="12"), c(1.24, sel="1X"),
           c(1.28, sel="X2")]
    t = pt.filtrer_conseils(res, 1.20, {"X2"})
    assert [g["sur"]["odds"] for g in t.gardees] == [1.27, 1.28]
    assert (t.hors_selection, t.trop_bas) == (2, 0)


def test_selection_vide_accepte_tout():
    res = [c(1.27, sel="X2"), c(1.29, sel="12")]
    for vide in (None, (), [], set()):
        t = pt.filtrer_conseils(res, 1.20, vide)
        assert len(t.gardees) == 2 and t.hors_selection == 0


def test_la_selection_est_verifiee_avant_la_cote():
    # Un « 1X » a 1,05 echoue aux DEUX criteres. Le compter deux fois gonflerait
    # le total et ferait mentir le bandeau : il est hors selection, point.
    t = pt.filtrer_conseils([c(1.05, sel="1X")], 1.20, {"X2"})
    assert (t.hors_selection, t.trop_bas, t.sans_cote) == (1, 0, 0)


def test_trop_bas_ne_parle_que_de_la_selection_voulue():
    # C'est la seule information actionnable : un X2 conseille mais mal paye.
    t = pt.filtrer_conseils([c(1.05, sel="X2"), c(1.05, sel="12")], 1.20, {"X2"})
    assert (t.hors_selection, t.trop_bas) == (1, 1)


def test_plusieurs_selections_acceptees():
    res = [c(1.27, sel="X2"), c(1.29, sel="12"), c(1.24, sel="1X")]
    t = pt.filtrer_conseils(res, 1.20, {"X2", "12"})
    assert len(t.gardees) == 2 and t.hors_selection == 1


def test_un_conseil_hors_double_chance_est_hors_selection():
    # `sur` est la ligne la plus probable des onze marches : rien ne GARANTIT
    # que ce soit un double chance, meme si c'est le cas sur les 12 mesures.
    t = pt.filtrer_conseils([c(3.2, marche="Multi-Buts", sel="1-6")], 1.20, {"X2"})
    assert t.hors_selection == 1 and t.gardees == []


# --------------------------------------------------------------------------
# CE QUI N'EST JAMAIS MASQUE, et l'invariant de comptage
# --------------------------------------------------------------------------

def test_une_erreur_reste_visible():
    # Elle n'a pas de conseil a trier, et la masquer ferait croire que la
    # rencontre n'existe pas plutot qu'elle n'a pas pu etre analysee.
    t = pt.filtrer_conseils([{"erreur": "Pas assez d'historique"}, c(1.00)], 1.20)
    assert len(t.gardees) == 1 and t.gardees[0].get("erreur")
    assert t.trop_bas == 1


def test_erreur_gardee_meme_avec_une_selection():
    t = pt.filtrer_conseils([{"erreur": "x"}, c(1.5, sel="12")], 1.20, {"X2"})
    assert len(t.gardees) == 1 and t.gardees[0].get("erreur")
    assert t.hors_selection == 1


def test_comptes_et_gardees_couvrent_toute_l_entree():
    # Aucune rencontre ne doit s'evaporer.
    res = [c(1.29), c(1.19), c(None), {"erreur": "x"}, c(1.21), c(1.00)]
    t = pt.filtrer_conseils(res, 1.20)
    assert len(t.gardees) + t.hors_selection + t.trop_bas + t.sans_cote == len(res)


def test_comptes_couvrent_toute_l_entree_avec_selection():
    res = [c(1.27, sel="X2"), c(1.05, sel="X2"), c(1.29, sel="12"),
           c(None, sel="X2"), {"erreur": "x"}]
    t = pt.filtrer_conseils(res, 1.20, {"X2"})
    assert len(t.gardees) + t.hors_selection + t.trop_bas + t.sans_cote == len(res)


def test_le_filtre_ne_touche_pas_au_detail():
    # « Garder tout » : la rencontre retenue ressort ENTIERE, ses onze marches
    # inclus. Le tri choisit, il ne rogne pas.
    r = c(1.30)
    r["lignes"] = [{"marche": f"m{i}", "sel": "s", "p": 0.5, "odds": 1.01}
                   for i in range(11)]
    t = pt.filtrer_conseils([r], 1.20, {"X2"})
    assert t.gardees[0] is r
    assert len(t.gardees[0]["lignes"]) == 11


def test_scenario_reel_du_27_09():
    # Les 12 rencontres mesurees en base, conseil et cote reels.
    reel = [("1X", 1.08), ("X2", 1.27), ("12", 1.29), ("1X", 1.19), ("12", 1.21),
            ("X2", 1.28), ("1X", 1.00), ("1X", 1.16), ("12", 1.24), ("X2", 1.05),
            ("12", 1.23), ("12", 1.21)]
    res = [c(o, sel=sel) for sel, o in reel]
    t = pt.filtrer_conseils(res, 1.20, {"X2"})
    assert [g["sur"]["odds"] for g in t.gardees] == [1.27, 1.28]
    assert t.hors_selection == 9      # les 1X et les 12
    assert t.trop_bas == 1            # le X2 a 1,05
    # Sans restriction de selection, on retrouve les sept du seuil seul.
    assert len(pt.filtrer_conseils(res, 1.20).gardees) == 7
