"""La fiabilite mesuree de chaque marche, a cote de chaque pronostic (27/09).

« Integre ces 11 marches cotes par Bet261 — ceux qu'on peut jouer — dans
Que jouer, selon ton propre pronostic. »

Les 11 marches etaient deja listes par rencontre. Ce qui manquait, c'est le
second nombre : a quel point CHAQUE marche est previsible. L'ecart est enorme
— Double Chance 78,1 %, Score exact 11,8 % — et sans lui, « Score exact -> 2-1,
14 % » se lit comme « 1X2 -> 1, 14 % », alors que l'un sort une fois sur huit
et l'autre une fois sur deux.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import predict_trio as pt  # noqa: E402

RACINE = Path(__file__).resolve().parents[1]

# Les 11 marches cotes, tels que `marches_probas` les rend.
COTES = list(pt.marches_probas(1.5, 1.2).keys())


def test_les_onze_marches_cotes_sont_bien_onze():
    assert len(COTES) == 11


def test_chacun_des_onze_porte_une_fiabilite_mesuree():
    """Un marche sans mesure afficherait un pronostic sans garde-fou."""
    for m in COTES:
        f = pt.fiabilite_marche(m)
        assert f is not None, f"{m} n'a pas de fiabilite mesuree"
        assert 0.0 < f["reel"] < 1.0, f"{m} : taux aberrant {f}"
        assert f["n"] > 1000, f"{m} : mesure sur trop peu de matchs"


def test_l_ecart_entre_marches_est_bien_la():
    # C'est tout l'interet de l'information : ces deux-la ne se lisent pas
    # de la meme facon, et rien ne le disait a l'ecran.
    dc = pt.fiabilite_marche("Double Chance")["reel"]
    se = pt.fiabilite_marche("Score exact")["reel"]
    assert dc > 0.7 and se < 0.2
    assert dc > se * 3


def test_les_mi_temps_aussi():
    # Meme fonction, autre table : elle doit servir les deux.
    for cle in ("1re mi-temps 1X2", "2e mi-temps Score exact"):
        assert pt.fiabilite_marche(cle) is not None


def test_marche_inconnu_rend_none_sans_lever():
    for mauvais in ("Inconnu", "", None, 42):
        assert pt.fiabilite_marche(mauvais) is None


def test_la_fiabilite_ne_recalcule_rien():
    """Elle lit la MEME table que la calibration des probabilites : deux
    sources se contrediraient au premier reajustement de l'une."""
    src = (RACINE / "scripts" / "predict_trio.py").read_text(encoding="utf-8")
    bloc = src[src.index("def fiabilite_marche("):src.index("def rencontres(")]
    assert "_MK_CAL" in bloc and "_MT_CAL" in bloc
    assert "global_reel" in bloc


def test_l_onglet_affiche_la_fiabilite_sur_chaque_ligne():
    src = (RACINE / "scripts" / "dashboard_trio.py").read_text(encoding="utf-8")
    bloc = src[src.index('for l in res_c["lignes"]'):]
    assert "fiabilite_marche" in bloc[:900]
    assert "ce marché touche".encode().decode("unicode_escape") in bloc[:900]         or "touche" in bloc[:900]


def test_aucune_variable_ecrasee_dans_la_boucle():
    """`quoi` servait au bandeau AVANT la boucle, et la boucle le reecrivait.
    Sans consequence aujourd'hui, mais c'est exactement la forme du bug qui
    fait mentir un message des qu'on deplace une ligne."""
    src = (RACINE / "scripts" / "dashboard_trio.py").read_text(encoding="utf-8")
    boucle = src[src.index("for res_c in gardees:"):]
    assert "quoi = {" not in boucle, "`quoi` est de nouveau ecrase dans la boucle"
