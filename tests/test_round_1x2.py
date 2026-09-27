"""L'onglet « Mon 1X2 du round » : pronostic, pieges, grosses cotes (27/09).

Les seuils sont ceux de la version du 09/09 de cet ecran, rejoues a
l'identique. Deux des quatre pieges d'alors lisaient la confiance du modele
V2/V5 et son accord avec V2 : ce modele ne tourne plus ici, ces deux signaux
ont donc ete REMPLACES par un seul, « aucun favori net », qui ne depend que de
ma propre lecture.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import predict_trio as pt  # noqa: E402

RACINE = Path(__file__).resolve().parents[1]


def sig(p1, pn, p2, oh, od, oa):
    return pt.signaux_1x2(p1, pn, p2, oh, od, oa, "Dom", "Ext")


# --------------------------------------------------------------------------
# FAVORI FRAGILE
# --------------------------------------------------------------------------

def test_favori_court_et_surestime_par_le_book():
    # Book : 1/1,60 devige ~= 60 %. Moi : 52 %. Ecart > 5 points, cote <= 1,70.
    pieges, _ = sig(0.52, 0.32, 0.16, 1.60, 3.9, 6.2)
    assert any("favori fragile" in x for x in pieges)


def test_favori_court_mais_confirme_ne_declenche_pas():
    pieges, _ = sig(0.64, 0.22, 0.14, 1.60, 3.9, 6.2)
    assert not any("favori fragile" in x for x in pieges)


def test_favori_long_ne_declenche_pas_meme_si_je_le_baisse():
    # Au-dela de 1,70 le favori n'est plus « cense » etre solide : c'est la
    # cote courte jouee les yeux fermes qui fait perdre, pas celle-ci.
    pieges, _ = sig(0.30, 0.30, 0.40, 2.50, 3.3, 2.40)
    assert not any("favori fragile" in x for x in pieges)


def test_le_favori_exterieur_est_teste_aussi():
    # L'ancienne version regardait bien les deux camps : un favori a
    # l'exterieur doit declencher comme un favori a domicile.
    pieges, _ = sig(0.16, 0.32, 0.52, 6.2, 3.9, 1.60)
    assert any("favori fragile" in x for x in pieges)


# --------------------------------------------------------------------------
# NUL MENACANT
# --------------------------------------------------------------------------

def test_nul_menacant_sur_favori_court():
    pieges, _ = sig(0.45, 0.33, 0.22, 1.90, 3.4, 4.5)
    assert any("nul mena" in x for x in pieges)


def test_nul_fort_mais_favori_long_ne_declenche_pas():
    pieges, _ = sig(0.35, 0.33, 0.32, 2.60, 3.2, 2.90)
    assert not any("nul mena" in x for x in pieges)


def test_nul_sous_le_seuil_ne_declenche_pas():
    pieges, _ = sig(0.55, 0.29, 0.16, 1.90, 3.4, 4.5)
    assert not any("nul mena" in x for x in pieges)


# --------------------------------------------------------------------------
# AUCUN FAVORI NET — le signal qui REMPLACE les deux anciens
# --------------------------------------------------------------------------

def test_aucun_favori_net():
    pieges, _ = sig(0.35, 0.33, 0.32, 2.60, 3.2, 2.90)
    assert any("aucun favori net" in x for x in pieges)


def test_pas_de_signal_quand_une_issue_se_detache():
    pieges, _ = sig(0.55, 0.28, 0.17, 1.85, 3.5, 5.0)
    assert not any("aucun favori net" in x for x in pieges)


def test_ce_signal_ne_depend_pas_des_cotes():
    """Il lit MA lecture seule : il doit sortir meme sans aucune cote."""
    pieges, _ = sig(0.35, 0.33, 0.32, None, None, None)
    assert pieges == [x for x in pieges if "aucun favori net" in x]
    assert len(pieges) == 1


# --------------------------------------------------------------------------
# SANS COTE, ON N'INVENTE PAS
# --------------------------------------------------------------------------

def test_sans_cote_les_pieges_qui_comparent_au_book_se_taisent():
    pieges, grosses = sig(0.52, 0.32, 0.16, None, None, None)
    assert not any("favori fragile" in x or "nul mena" in x for x in pieges)
    assert grosses == []


def test_cotes_illisibles_traitees_comme_absentes():
    for mauvais in ("", "n/a", 0, -1, 1.0, float("nan")):
        pieges, grosses = sig(0.52, 0.32, 0.16, mauvais, mauvais, mauvais)
        assert not any("favori fragile" in x for x in pieges), mauvais
        assert grosses == [], mauvais


def test_probas_absentes_ne_font_pas_lever():
    pieges, grosses = sig(None, None, None, 1.6, 3.9, 6.2)
    assert isinstance(pieges, list) and isinstance(grosses, list)


# --------------------------------------------------------------------------
# GROSSES COTES — un CONSTAT, jamais une value
# --------------------------------------------------------------------------

def test_grosse_cote_signalee_avec_ma_proba():
    _, grosses = sig(0.52, 0.32, 0.16, 1.60, 3.9, 6.2)
    assert len(grosses) == 1
    assert grosses[0]["equipe"] == "Ext" and grosses[0]["odds"] == 6.2
    assert grosses[0]["p"] == 0.16


def test_le_nul_peut_etre_une_grosse_cote():
    _, grosses = sig(0.60, 0.12, 0.28, 1.45, 5.5, 3.2)
    assert [g["sel"] for g in grosses] == ["X"]
    assert grosses[0]["equipe"] == "Nul"


def test_sous_le_seuil_rien_n_est_signale():
    _, grosses = sig(0.45, 0.30, 0.25, 2.1, 3.4, 4.9)
    assert grosses == []


def test_le_seuil_est_inclusif():
    _, grosses = sig(0.45, 0.30, 0.25, 2.1, 3.4, pt.GROSSE_COTE)
    assert len(grosses) == 1


def test_aucune_regle_proba_fois_cote():
    """Testee DEUX fois dans ce depot, revelee signal INVERSE : elle
    selectionne les matchs ou le modele s'ecarte le plus du book, donc ses
    propres erreurs. Elle ne doit pas revenir par cette porte."""
    # La PREUVE est comportementale, pas textuelle : une issue a 20,00 que je
    # donne a 2 % doit sortir comme FAIT. Son produit vaut 0,02 x 20 = 0,40,
    # tres en dessous de 1 : toute regle de rentabilite l'aurait ecartee.
    _, grosses = sig(0.78, 0.20, 0.02, 1.25, 5.8, 20.0)
    assert [g["sel"] for g in grosses] == ["X", "2"]
    # Et symetriquement, une grosse cote au produit FAVORABLE n'est pas
    # privilegiee : elle sort comme les autres, sans mention ni tri.
    _, g2 = sig(0.10, 0.20, 0.70, 9.0, 5.0, 1.2)
    assert [g["sel"] for g in g2] == ["1", "X"]
    assert all("value" not in str(g).lower() for g in g2)


# --------------------------------------------------------------------------
# BRANCHEMENT
# --------------------------------------------------------------------------

def test_le_pronostic_ne_vient_pas_de_la_cote():
    src = (RACINE / "scripts" / "predict_trio.py").read_text(encoding="utf-8")
    bloc = src[src.index("def round_1x2("):src.index("def fiabilite_marche(")]
    assert "predict_own(" in bloc, "le pronostic doit venir de ma propre analyse"
    assert 'calib_marche("1X2"' in bloc, "et passer par la calibration du marche"


def test_l_onglet_existe_avec_les_controles_demandes():
    src = (RACINE / "scripts" / "dashboard_trio.py").read_text(encoding="utf-8")
    assert "Mon 1X2 du round" in src
    assert "Heure Mada du round" in src
    assert 'selectbox("Ligue", list(LEAGUES), index=0, key="rd_lg")' in src
    assert "round_1x2(" in src


# --------------------------------------------------------------------------
# LA FONCTION ENTIERE, REELLEMENT APPELEE
#
# ⚠️ Ce bloc existe a cause d'un bug reel du 27/09 : `round_1x2` faisait
# `dict(own["x12"])`, or `x12` est une LISTE de trois nombres. Les tests
# etaient verts parce qu'ils n'appelaient jamais la fonction -- ils
# eprouvaient la partie pure et relisaient le source pour le reste. Sans
# rencontre a venir en base, la fonction sort avant la ligne fautive.
#
# On simule donc les deux seules portes vers la base : la liste des rencontres
# et l'analyse d'une rencontre. Tout le reste s'execute pour de vrai.
# --------------------------------------------------------------------------

import types  # noqa: E402

import pytest  # noqa: E402


class _Renc:
    """Une ligne de `_upcoming_df`, telle que `itertuples()` la rend."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Frame(list):
    def itertuples(self):
        return iter(self)


@pytest.fixture
def base_simulee(monkeypatch):
    """Deux rencontres, dont une sans historique exploitable."""
    frame = _Frame([
        _Renc(team_a="Dom", team_b="Ext", local="21:03", rd="Journee 7",
              oh=1.60, od=3.90, oa=6.20),
        _Renc(team_a="Sans", team_b="Histo", local="21:03", rd="Journee 7",
              oh=2.10, od=3.20, oa=3.60),
    ])
    monkeypatch.setattr(pt, "_upcoming_df", lambda *a, **k: frame)

    def faux_own(engine, a, b, lg=None, n=60, journee=None):
        if a == "Sans":
            return None
        return {"x12": [0.52, 0.32, 0.16], "lam_a": 1.6, "lam_b": 1.1,
                "seq_a": "VVNDV", "seq_b": "DDNVD"}
    monkeypatch.setattr(pt, "predict_own", faux_own)
    return frame


def test_la_fonction_entiere_tourne(base_simulee):
    """Le test qui aurait attrape le bug `dict(x12)`."""
    res = pt.round_1x2(object(), "InstantLeague-8035")
    assert len(res) == 2
    m = res[0]
    assert m["sel"] in ("1", "X", "2")
    assert m["equipe"] == "Dom"
    assert 0.0 < m["p"] <= 1.0
    assert m["odds"] == 1.60
    assert m["journee"] == 7
    assert m["attendus"] == 2.7


def test_le_1x2_est_calibre_et_non_brut(base_simulee):
    # Brut : 52 / 32 / 16. La calibration du marche 1X2 doit deplacer ces
    # valeurs -- sinon c'est qu'elle n'est pas appliquee.
    m = pt.round_1x2(object(), "InstantLeague-8035")[0]
    assert m["probas"]["1"] != 0.52


def test_les_signaux_sont_bien_attaches(base_simulee):
    m = pt.round_1x2(object(), "InstantLeague-8035")[0]
    assert any("favori fragile" in x for x in m["pieges"])
    assert [g["equipe"] for g in m["grosses_cotes"]] == ["Ext"]


def test_une_rencontre_sans_historique_est_dite_pas_cachee(base_simulee):
    """La masquer ferait croire que la rencontre n'existe pas, plutot
    qu'elle n'a pas pu etre analysee."""
    res = pt.round_1x2(object(), "InstantLeague-8035")
    assert res[1]["erreur"] and res[1]["home"] == "Sans"


def test_x12_de_mauvaise_forme_n_explose_pas(monkeypatch, base_simulee):
    """Le bug d'origine, pris a la racine : une forme inattendue doit donner
    une ligne d'erreur lisible, jamais une trace en pleine page."""
    for mauvais in ([], [0.5], {"1": 0.5}, None, "abc"):
        monkeypatch.setattr(pt, "predict_own",
                            lambda *a, **k: {"x12": mauvais, "lam_a": 1.0,
                                             "lam_b": 1.0})
        res = pt.round_1x2(object(), "InstantLeague-8035")
        assert all(m.get("erreur") for m in res), mauvais


def test_le_plafond_de_rencontres_est_respecte(monkeypatch):
    frame = _Frame([_Renc(team_a=f"A{i}", team_b=f"B{i}", local="21:03",
                          rd="J1", oh=2.0, od=3.2, oa=3.8) for i in range(50)])
    monkeypatch.setattr(pt, "_upcoming_df", lambda *a, **k: frame)
    monkeypatch.setattr(pt, "predict_own",
                        lambda *a, **k: {"x12": [0.5, 0.3, 0.2],
                                         "lam_a": 1.0, "lam_b": 1.0})
    assert len(pt.round_1x2(object(), "lg", limite=12)) == 12


def test_aucune_rencontre_rend_une_liste_vide(monkeypatch):
    monkeypatch.setattr(pt, "_upcoming_df", lambda *a, **k: _Frame())
    assert pt.round_1x2(object(), "lg") == []
