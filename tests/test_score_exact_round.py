"""Le score exact sur TOUTES les rencontres du round (27/09).

« Dans la prédiction du round restaurée, affiche aussi les scores exacts de
tous les matchs. »

Le score était calculé pour chaque rencontre mais rendu sur le seul Top 3. Ce
fichier vérifie qu'il l'est partout, et surtout qu'il l'est PAR LE MÊME CODE :
deux écritures du même format divergent à la première retouche de l'une.

04/10 : les deux listes (Top 3, puis « les autres matchs ») sont devenues UN
tableau, à la demande d'Olivio. L'invariant ne change pas, il change de
place : le format du score n'existe plus que dans `tableau_round`.
"""
import re
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
DASH = RACINE / "scripts" / "dashboard_trio.py"
sys.path.insert(0, str(RACINE / "scripts"))
import predict_trio as pt  # noqa: E402


def _sans_commentaires(p: Path) -> str:
    """Ces vérifications lisent la SOURCE : sans ce nettoyage elles mesureraient
    la prose, et les commentaires qui EXPLIQUENT le changement contiennent par
    construction les chaînes cherchées."""
    return re.sub(r"(^|[^:])#.*$", r"",
                  p.read_text(encoding="utf-8"), flags=re.M)


SRC = _sans_commentaires(DASH)


def _match(a, b, conf, scores, x12=(0.5, 0.3, 0.2)):
    return {"match": f"{a} v {b}", "team_a": a, "team_b": b,
            "cotes": [2.0, 3.2, 3.8], "x12": list(x12), "over25_pct": 55.0,
            "confidence": conf, "top1_calibre": (scores[0][0], 0.13),
            "consensus_top3": scores}


ROUND = [_match(f"A{i}", f"B{i}", 0.20 + i / 100,
                [("1-0", 0.12), ("1-1", 0.11), ("2-1", 0.09)])
         for i in range(6)]


def test_le_score_n_est_ecrit_qu_a_un_seul_endroit():
    """C'est l'invariant qui compte : le format n'existe qu'une fois. Le
    tableau de bord ne l'écrit plus lui-même, il appelle `tableau_round`."""
    assert "def _score(" not in SRC and "def _autres_scores(" not in SRC
    assert "· score **" not in SRC, "le format du score est écrit dans l'écran"
    assert SRC.count("tableau_round(") == 1


def test_chaque_rencontre_a_son_score_exact():
    lignes = pt.tableau_round(ROUND)["lignes"]
    assert len(lignes) == len(ROUND)
    for l in lignes:
        assert l["Score exact"] == "1-0 · 13 %", l


def test_les_alternatives_accompagnent_le_score():
    """Sur un marché touché à 11,8 %, donner UN score sans ses suivants laisse
    croire à une précision qu'il n'a pas."""
    for l in pt.tableau_round(ROUND)["lignes"]:
        assert l["Sinon"] == "1-1 11 % · 2-1 9 %", l


def test_aucun_score_exact_ne_revient_dans_les_marches_bannis():
    """`SCORES_EXACTS` bannit ces MARCHÉS d'un autre bandeau. Le score du round
    est une prédiction, pas un marché à jouer : les deux ne doivent pas se
    confondre, et l'interdiction doit rester en place."""
    assert "SCORES_EXACTS = {" in SRC
    assert "Score exact" in SRC.split("SCORES_EXACTS = {")[1][:120]
