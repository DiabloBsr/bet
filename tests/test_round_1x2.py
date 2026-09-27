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


def m(a, x, b, **kw):
    """Une rencontre reduite a ce que le debusqueur regarde."""
    d = {"home": "Dom", "away": "Ext",
         "cotes": {"1": a, "X": x, "2": b}}
    d.update(kw)
    return d

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
    """L'analyse d'une rencontre vit dans `_analyse_1x2` depuis le 27/09 :
    `round_1x2` et le débusqueur des 5 rounds la partagent, deux écritures du
    même pronostic divergeraient au premier réglage de l'une."""
    src = (RACINE / "scripts" / "predict_trio.py").read_text(encoding="utf-8")
    bloc = src[src.index("def _analyse_1x2("):src.index("def round_1x2(")]
    assert "predict_own(" in bloc, "le pronostic doit venir de ma propre analyse"
    assert 'calib_marche("1X2"' in bloc, "et passer par la calibration du marche"
    # Et les deux appelants passent bien par elle, sans refaire le calcul.
    aval = src[src.index("def round_1x2("):src.index("def fiabilite_marche(")]
    assert aval.count("_analyse_1x2(") == 2
    assert "predict_own(" not in aval, "le pronostic est recopie quelque part"


def test_le_moteur_reste_disponible_sans_son_onglet():
    """⚠️ L'onglet « Mon 1X2 du round » a été retiré le 27/09, la section
    de prédiction du round restaurée couvrant le même besoin.

    Ce test affirmait sa présence à l'écran. Il vérifie désormais que le
    MOTEUR reste entier : `round_1x2` et `signaux_1x2` ne coûtent rien tant
    que rien ne les appelle, et remettre un écran ne demandera alors que son
    affichage. C'est le même parti que pour les quatre onglets coupés la
    veille.
    """
    assert callable(pt.round_1x2) and callable(pt.signaux_1x2)
    # Les seuils aussi : ce sont eux qui seraient perdus en premier.
    for nom in ("PIEGE_COTE_FAVORI", "PIEGE_ECART", "PIEGE_NUL",
                "PIEGE_NUL_COTE", "PIEGE_SANS_FAVORI", "GROSSE_COTE"):
        assert isinstance(getattr(pt, nom), (int, float)), nom
    src = (RACINE / "scripts" / "dashboard_trio.py").read_text(encoding="utf-8")
    assert "Mon 1X2 du round" not in src, "l'onglet est censé avoir été retiré"


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


# --------------------------------------------------------------------------
# BALAYAGE MULTI-LIGUES (27/09) — « debusque dans toutes les ligues »
# --------------------------------------------------------------------------

def _renc(**kw):
    d = dict(team_a="Dom", team_b="Ext", local="21:03", rd="Journee 7",
             oh=2.83, od=2.79, oa=2.87, c="InstantLeague-8035")
    d.update(kw)
    return _Renc(**d)


@pytest.fixture
def deux_ligues(monkeypatch):
    """Deux rencontres, dans DEUX competitions differentes."""
    frame = _Frame([
        _renc(team_a="Benin", team_b="Mozambique", c="InstantLeague-8060"),
        _renc(team_a="Tondela", team_b="Moreirense", c="InstantLeague-8044"),
    ])
    monkeypatch.setattr(pt, "_upcoming_df", lambda *a, **k: frame)
    vues = []

    def faux_own(engine, a, b, lg=None, n=60, journee=None):
        vues.append(lg)
        return {"x12": [0.34, 0.33, 0.33], "lam_a": 1.3, "lam_b": 1.3}
    monkeypatch.setattr(pt, "predict_own", faux_own)
    return vues


def test_chaque_rencontre_est_analysee_dans_SA_ligue(deux_ligues):
    """⚠️ LE PIEGE DU MULTI-LIGUES. La fonction passait la ligue DEMANDEE a
    `predict_own`. Sur un balayage, la forme des equipes aurait ete cherchee
    dans la mauvaise competition : `predict_own` rend None, et toutes les
    rencontres seraient sorties en « historique insuffisant » — un ecran vide
    sans message d'erreur."""
    pt.round_1x2(object(), ["InstantLeague-8060", "InstantLeague-8044"])
    assert deux_ligues == ["InstantLeague-8060", "InstantLeague-8044"]


def test_la_ligue_voyage_avec_la_rencontre(deux_ligues):
    # Sur neuf ligues, l'heure seule ne situe plus rien : l'ecran doit
    # pouvoir nommer la competition.
    res = pt.round_1x2(object(), ["InstantLeague-8060", "InstantLeague-8044"])
    assert [m["ligue"] for m in res] == ["InstantLeague-8060", "InstantLeague-8044"]


def _espion(monkeypatch):
    """Enregistre les ligues reçues par `_upcoming_df` et rend un cadre vide.

    ⚠️ Pas de `setdefault(...) or cadre` : `setdefault` rend la valeur
    enregistrée, qui est véridique, donc le `or` court-circuite et la fonction
    renvoie la LISTE au lieu du cadre. C'est ce qui a fait tomber ce test à
    l'écriture.
    """
    recu = {}

    def faux(engine, lgs, *a, **k):
        recu["lgs"] = lgs
        return _Frame()
    monkeypatch.setattr(pt, "_upcoming_df", faux)
    return recu


def test_une_chaine_reste_une_seule_ligue(monkeypatch):
    """`list("InstantLeague-8035")` en ferait une liste de caracteres, et le
    filtre ne retiendrait plus aucune rencontre."""
    recu = _espion(monkeypatch)
    pt.round_1x2(object(), "InstantLeague-8035")
    assert recu["lgs"] == ["InstantLeague-8035"]


def test_liste_vide_balaie_tout(monkeypatch):
    recu = _espion(monkeypatch)
    pt.round_1x2(object(), [])
    assert recu["lgs"] is None, "None = aucune restriction de ligue"


def test_les_ligues_vides_sont_ignorees(monkeypatch):
    recu = _espion(monkeypatch)
    pt.round_1x2(object(), ["InstantLeague-8060", None, "", "InstantLeague-8044"])
    assert recu["lgs"] == ["InstantLeague-8060", "InstantLeague-8044"]


def test_le_debusqueur_trouve_les_exemples_du_cabinet():
    """Les trois releves cites : cible 2,83, tolerance 0,10."""
    exemples = [m(2.87, 2.79, 2.82), m(2.81, 2.84, 2.83), m(2.83, 2.87, 2.78)]
    r = pt.debusquer_cotes(exemples, cibles=(2.83, 2.83, 2.83), tol=0.10)
    assert len(r["trouvees"]) == 3, "les trois exemples doivent sortir"


def test_la_cible_a_deux_ne_les_trouve_pas():
    """Meme lot, cible 2,00 : aucun. C'est ce qui justifie d'avoir cale la
    valeur par defaut sur 2,83."""
    exemples = [m(2.87, 2.79, 2.82), m(2.81, 2.84, 2.83), m(2.83, 2.87, 2.78)]
    r = pt.debusquer_cotes(exemples, cibles=(2.0, 2.0, 2.0), tol=0.10)
    assert r["trouvees"] == [] and len(r["proches"]) == 3


# --------------------------------------------------------------------------
# DEBUSQUEUR DES 5 PROCHAINS ROUNDS (27/09)
#
# « Je n'ai pas besoin d'indicateur d'heure, mais je veux que tu debusques
#   toutes les cotes dans les 5 rounds a venir du 1X2 a une cote 2, peu
#   importe l'apres-virgule. »
#
# « Cote 2 peu importe l'apres-virgule » = partie entiere 2, soit 2,00 a 2,99.
# Mesure : 1 060 releves sur 184 105 (0,58 %), sur 8 ligues.
# --------------------------------------------------------------------------

def test_partie_entiere():
    assert pt._partie_entiere(2.87) == 2
    assert pt._partie_entiere(2.00) == 2
    assert pt._partie_entiere(2.99) == 2
    assert pt._partie_entiere(3.00) == 3
    assert pt._partie_entiere(1.99) == 1
    for mauvais in (None, "", "n/a", 0, -1, float("nan")):
        assert pt._partie_entiere(mauvais) is None, mauvais


@pytest.fixture
def cinq_rounds(monkeypatch):
    """Trois heures dans une ligue, deux dans une autre ; cotes variees."""
    def rc(ta, h, c, oh, od, oa):
        return _Renc(team_a=ta, team_b=ta + "b", local="21:03", rd="Journee 7",
                     oh=oh, od=od, oa=oa, c=c, expected_start=h)
    frame = _Frame([
        rc("A1", "h1", "LG1", 2.87, 2.79, 2.82),   # retenue
        rc("A2", "h1", "LG1", 1.40, 4.50, 7.00),   # ecartee
        rc("A3", "h2", "LG1", 2.05, 2.95, 2.50),   # retenue
        rc("A4", "h3", "LG1", 3.00, 2.50, 2.50),   # ecartee : une cote a 3,00
        rc("B1", "h9", "LG2", 2.10, 2.20, 2.30),   # retenue
    ])
    monkeypatch.setattr(pt, "_upcoming_df", lambda *a, **k: frame)
    monkeypatch.setattr(pt, "predict_own",
                        lambda *a, **k: {"x12": [0.34, 0.33, 0.33],
                                         "lam_a": 1.3, "lam_b": 1.3})
    return frame


def test_ne_garde_que_les_trois_cotes_commencant_par_deux(cinq_rounds):
    r = pt.debusquer_rounds(object(), n_rounds=5, entier=2)
    assert [m["home"] for m in r["trouvees"]] == ["A1", "A3", "B1"]
    assert r["examinees"] == 5 and r["ligues"] == 2


def test_une_cote_a_trois_pile_disqualifie(cinq_rounds):
    """2,99 passe, 3,00 non : « peu importe l'apres-virgule » porte bien sur
    la PARTIE ENTIERE, et la borne haute est exclusive."""
    r = pt.debusquer_rounds(object(), n_rounds=5, entier=2)
    assert "A4" not in [m["home"] for m in r["trouvees"]]


def test_le_nombre_de_rounds_se_compte_PAR_LIGUE(cinq_rounds):
    """Sans cela, une ligue rapide mangerait la place des autres."""
    r = pt.debusquer_rounds(object(), n_rounds=1, entier=2)
    # LG1 ne garde que « h1 », LG2 garde « h9 » : A3 disparait, B1 reste.
    assert [m["home"] for m in r["trouvees"]] == ["A1", "B1"]


def test_la_partie_entiere_est_reglable(cinq_rounds):
    r = pt.debusquer_rounds(object(), n_rounds=5, entier=1)
    assert r["trouvees"] == []


def test_aucune_rencontre_a_venir(monkeypatch):
    monkeypatch.setattr(pt, "_upcoming_df", lambda *a, **k: _Frame())
    r = pt.debusquer_rounds(object())
    assert r == {"trouvees": [], "examinees": 0, "rounds": 0, "ligues": 0}


def test_l_analyse_ne_tourne_que_sur_les_survivantes(cinq_rounds, monkeypatch):
    """⚠️ C'EST LA RAISON D'ETRE DU TRI EN DEUX TEMPS. Cinq rounds sur neuf
    ligues font ~450 rencontres ; les analyser toutes demanderait ~900
    requetes de forme pour n'en retenir que 0,58 %."""
    appels = []
    monkeypatch.setattr(pt, "predict_own",
                        lambda e, a, b, **k: (appels.append(a) or
                                              {"x12": [0.34, 0.33, 0.33],
                                               "lam_a": 1.3, "lam_b": 1.3}))
    pt.debusquer_rounds(object(), n_rounds=5, entier=2)
    assert appels == ["A1", "A3", "B1"], "l'analyse a tourne sur des ecartees"


def test_le_plafond_de_rencontres_est_respecte(monkeypatch):
    frame = _Frame([_Renc(team_a=f"T{i}", team_b="X", local="21:03", rd="J1",
                          oh=2.1, od=2.2, oa=2.3, c="LG1",
                          expected_start="h1") for i in range(40)])
    monkeypatch.setattr(pt, "_upcoming_df", lambda *a, **k: frame)
    monkeypatch.setattr(pt, "predict_own",
                        lambda *a, **k: {"x12": [0.34, 0.33, 0.33],
                                         "lam_a": 1.3, "lam_b": 1.3})
    assert len(pt.debusquer_rounds(object(), limite=12)["trouvees"]) == 12


def test_l_onglet_ne_demande_plus_l_heure():
    src = (RACINE / "scripts" / "dashboard_trio.py").read_text(encoding="utf-8")
    bloc = src[src.index("Débusqueur 1X2 à cote 2"):src.index("HISTORIQUE & FACE")]
    assert "Heure Mada" not in bloc, "l'indicateur d'heure devait disparaître"
    assert "debusquer_rounds(" in bloc
    assert "Rounds à venir" in bloc
