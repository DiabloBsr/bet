"""Le tableau a l'heure choisie (demande d'Olivio du 04/10).

« Cree-moi un onglet ou tu mets dans un tableau les predictions des matchs a
l'heure que je choisis, avec leur pourcentage de chance, de celui qui gagne
ou fait match nul, et aussi leur over/under. »

La base locale est figee au 5 juillet : aucune rencontre a venir. Tout passe
donc par une base simulee (`_upcoming_df` et `predict_own` remplaces).
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "scripts"))
import predict_trio as pt  # noqa: E402

ANG, ESP = "InstantLeague-8035", "InstantLeague-8037"

# home, away, ligue, heure locale
RENCONTRES = [
    # Ligues ENTREMELEES a la meme minute, comme en base : le tableau doit
    # les regrouper.
    ("Burnley", "Brentford", ANG, "17:12"),
    ("Elche", "Girona", ESP, "17:12"),
    ("Arsenal", "Chelsea", ANG, "17:12"),
    ("Leeds", "Fulham", ANG, "17:14"),
]
# Ce que « ma » forme dirait de chaque equipe a domicile : x12 brut, lambdas.
FORMES = {
    "Burnley": ([0.52, 0.26, 0.22], 1.7, 0.9, 0.55),
    "Arsenal": ([0.30, 0.28, 0.42], 1.1, 1.4, 0.48),
    "Elche": ([0.36, 0.33, 0.31], 1.0, 0.9, 0.38),
    "Leeds": ([0.45, 0.27, 0.28], 1.5, 1.2, 0.60),
}


@pytest.fixture
def base(monkeypatch):
    appels = []

    def faux_upcoming(engine, leagues=None, minutes=120, start_local=None,
                      end_local=None):
        appels.append({"leagues": leagues, "minutes": minutes,
                       "start": start_local, "end": end_local})
        lignes = [{"c": lg, "team_a": a, "team_b": b, "expected_start": h,
                   "rd": "Journée 13", "oh": 2.1, "od": 3.2, "oa": 3.4,
                   "xm": "{}", "local": h}
                  for a, b, lg, h in RENCONTRES]
        df = pd.DataFrame(lignes)
        if start_local and end_local:
            df = df[(df.local >= start_local) & (df.local <= end_local)]
        if leagues:
            df = df[df.c.isin(leagues)]
        return df.sort_values("local", kind="stable")

    def faux_own(engine, team_a, team_b, lg=None, journee=None, n=60):
        if team_a not in FORMES:
            return None
        x12, la, lb, po = FORMES[team_a]
        return {"x12": x12, "lam_a": la, "lam_b": lb, "p_over25": po,
                "seq_a": "VVNDV", "seq_b": "DNDDV"}

    monkeypatch.setattr(pt, "_upcoming_df", faux_upcoming)
    monkeypatch.setattr(pt, "predict_own", faux_own)
    return appels


# --------------------------------------------------------------------------
# L'HEURE
# --------------------------------------------------------------------------

def test_l_heure_choisie_borne_la_recherche_a_cette_minute(base):
    res = pt.tableau_heure(None, heure="17:12")
    assert base[0]["start"] == base[0]["end"] == "17:12"
    assert res["heure"] == "17:12"
    assert [l["home"] for l in res["lignes"]] == ["Burnley", "Arsenal", "Elche"]


def test_une_heure_sans_zero_initial_est_completee(base):
    pt.tableau_heure(None, heure="9:03")
    assert base[0]["start"] == "09:03"


def test_sans_heure_on_prend_la_prochaine(base):
    """Vide = la prochaine heure de coup d'envoi, et SEULEMENT elle."""
    res = pt.tableau_heure(None)
    assert res["heure"] == "17:12"
    assert {l["home"] for l in res["lignes"]} == {"Burnley", "Arsenal", "Elche"}


def test_les_ligues_filtrent(base):
    res = pt.tableau_heure(None, leagues=[ESP], heure="17:12")
    assert [l["home"] for l in res["lignes"]] == ["Elche"]
    assert res["lignes"][0]["tag"] == "ESP"


def test_aucune_rencontre_rend_un_tableau_vide(base):
    res = pt.tableau_heure(None, heure="03:00")
    assert res == {"heure": "03:00", "lignes": [], "total": 0}


def test_le_plafond_est_annonce(base):
    res = pt.tableau_heure(None, heure="17:12", limite=2)
    assert len(res["lignes"]) == 2 and res["total"] == 3


# --------------------------------------------------------------------------
# LES CHIFFRES : AUCUN NEUF
# --------------------------------------------------------------------------

def test_le_1x2_est_celui_du_debusqueur(base):
    """Meme match, memes pourcentages dans les deux ecrans."""
    l = pt.tableau_heure(None, heure="17:12")["lignes"][0]
    attendu = {k: round(pt.calib_marche("1X2", v), 4)
               for k, v in zip("1X2", FORMES["Burnley"][0])}
    assert l["probas"] == attendu
    assert l["sel"] == "1" and l["equipe"] == "Burnley"


def test_le_1x2_du_tableau_fait_100(base):
    """⚠️ Calibrees issue par issue, les trois chances descendaient jusqu'a
    80 % au total sur le round du 05/07 a 10:47 (8 / 13 / 67 %)."""
    for l in pt.tableau_heure(None, heure="17:12")["lignes"]:
        assert abs(sum(l["probas_100"].values()) - 1.0) < 1e-3, l["probas_100"]


def test_le_pronostic_garde_sa_chance_mesuree(base):
    """L'issue de tete ne bouge pas : c'est la seule valeur mesuree, et la
    meme que dans le debusqueur."""
    for l in pt.tableau_heure(None, heure="17:12")["lignes"]:
        assert l["probas_100"][l["sel"]] == l["probas"][l["sel"]]


def test_les_deux_autres_se_partagent_le_reste_au_prorata():
    b = [0.75, 0.15, 0.10]
    p = pt._1x2_somme_100(b)
    assert p["1"] == round(pt.calib_marche("1X2", 0.75), 4)
    assert abs(p["X"] / p["2"] - 0.15 / 0.10) < 1e-2
    assert p["1"] < 0.75, "la calibration rabote bien le favori"


def test_un_partage_impossible_ne_leve_pas():
    assert pt._1x2_somme_100(None) == {}
    assert pt._1x2_somme_100([0.5, 0.5]) == {}
    p = pt._1x2_somme_100([1.0, 0.0, 0.0])
    assert abs(sum(p.values()) - 1.0) < 1e-3


def test_les_lignes_sont_groupees_par_ligue(base):
    lignes = pt.tableau_heure(None, heure="17:12")["lignes"]
    tags = [l["tag"] for l in lignes]
    assert tags == sorted(tags)


def test_un_pronostic_exterieur_nomme_l_equipe_exterieure(base):
    l = pt.tableau_heure(None, heure="17:12")["lignes"][1]   # Arsenal 0.42 en 2
    assert l["sel"] == "2" and l["equipe"] == "Chelsea"


def test_l_over_under_2_5_est_la_fonction_calibree(base):
    l = pt.tableau_heure(None, heure="17:12")["lignes"][0]
    o, u = pt.ou25_probas(FORMES["Burnley"][3])
    assert (l["over25"], l["under25"]) == (round(o, 4), round(u, 4))


def test_l_over_under_3_5_est_la_ligne_calibree_de_bet261(base):
    l = pt.tableau_heure(None, heure="17:12")["lignes"][0]
    brut = dict(pt.marches_probas(1.7, 0.9)["+/-"])
    assert l["over35"] == round(pt.calib_marche("+/-", brut["> 3.5"]), 4)
    assert l["under35"] == round(pt.calib_marche("+/-", brut["< 3.5"]), 4)
    # Peu de buts attendus (2,6) : l'under 3,5 doit dominer.
    assert l["under35"] > l["over35"]


def test_les_pourcentages_restent_des_probabilites(base):
    for l in pt.tableau_heure(None, heure="17:12")["lignes"]:
        for k in ("over25", "under25", "over35", "under35"):
            assert 0.0 <= l[k] <= 1.0, (k, l[k])
        assert all(0.0 <= v <= 1.0 for v in l["probas"].values())


def test_un_match_sans_historique_reste_dans_le_tableau(base, monkeypatch):
    vrai = pt.predict_own
    monkeypatch.setattr(pt, "predict_own",
                        lambda e, a, b, **k: None if a == "Arsenal" else vrai(e, a, b, **k))
    lignes = pt.tableau_heure(None, heure="17:12")["lignes"]
    assert len(lignes) == 3
    assert lignes[1].get("erreur")


# --------------------------------------------------------------------------
# L'AFFICHAGE
# --------------------------------------------------------------------------

def test_l_affichage_a_les_colonnes_demandees(base):
    rows = pt.tableau_affichage(pt.tableau_heure(None, heure="17:12")["lignes"])
    assert list(rows[0]) == list(pt.TABLEAU_COLONNES)
    r = rows[0]
    assert r["Match"] == "Burnley – Brentford" and r["Pronostic"] == "Burnley"
    for k in pt.TABLEAU_COLONNES[3:]:
        assert isinstance(r[k], int) and 0 <= r[k] <= 100, (k, r[k])
    for r in rows:   # a l'arrondi pres
        assert 99 <= r["1"] + r["X"] + r["2"] <= 101, r


def test_un_nul_s_ecrit_nul(base, monkeypatch):
    monkeypatch.setitem(FORMES, "Elche", ([0.30, 0.40, 0.30], 0.8, 0.8, 0.30))
    rows = pt.tableau_affichage(pt.tableau_heure(None, heure="17:12")["lignes"])
    assert rows[2]["Pronostic"] == "Nul"


def test_une_ligne_en_erreur_garde_ses_cases_vides():
    rows = pt.tableau_affichage([{"home": "A", "away": "B", "tag": "ANG",
                                  "erreur": "historique insuffisant"}])
    assert rows[0]["Pronostic"] == "— historique insuffisant"
    assert all(rows[0][k] is None for k in pt.TABLEAU_COLONNES[3:])


# --------------------------------------------------------------------------
# LA COLORATION DU PRONOSTIC (04/10)
# --------------------------------------------------------------------------

def test_le_pronostic_de_chaque_ligne_est_colore(base):
    lignes = pt.tableau_heure(None, heure="17:12")["lignes"]
    reco = pt.tableau_reco(lignes)
    assert len(reco) == len(lignes)
    burnley = reco[0]                      # 1 a 52 %, 2,6 buts attendus
    assert burnley & {"1", "X", "2"} == {"1"}
    assert len(burnley & {"Over 2,5", "Under 2,5"}) == 1
    assert burnley & {"Over 3,5", "Under 3,5"} == {"Under 3,5"}


def test_la_case_coloree_est_le_pronostic_du_debusqueur(base):
    """Pas de nouveau calcul : la couleur suit `sel`, le pronostic affiche."""
    lignes = pt.tableau_heure(None, heure="17:12")["lignes"]
    for l, cases in zip(lignes, pt.tableau_reco(lignes)):
        assert l["sel"] in cases


def test_le_cote_colore_est_le_plus_probable():
    l = {"sel": "2", "over25": 0.61, "under25": 0.39,
         "over35": 0.30, "under35": 0.70}
    assert pt.tableau_reco([l]) == [{"2", "Over 2,5", "Under 3,5"}]


def test_une_egalite_ne_colore_rien():
    l = {"sel": "X", "over25": 0.5, "under25": 0.5,
         "over35": 0.3, "under35": 0.7}
    assert pt.tableau_reco([l]) == [{"X", "Under 3,5"}]


def test_une_ligne_en_erreur_n_est_pas_coloree():
    assert pt.tableau_reco([{"erreur": "historique insuffisant",
                             "sel": "1"}]) == [set()]
    assert pt.tableau_reco(None) == []


def test_les_noms_colores_sont_des_colonnes_du_tableau(base):
    lignes = pt.tableau_heure(None, heure="17:12")["lignes"]
    for cases in pt.tableau_reco(lignes):
        assert cases <= set(pt.TABLEAU_COLONNES)
