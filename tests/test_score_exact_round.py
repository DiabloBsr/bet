"""Le score exact sur TOUTES les rencontres du round (27/09).

« Dans la prédiction du round restaurée, affiche aussi les scores exacts de
tous les matchs. »

Le score était calculé pour chaque rencontre mais rendu sur le seul Top 3. Ce
fichier vérifie qu'il l'est partout, et surtout qu'il l'est PAR LE MÊME CODE :
deux écritures du même format divergent à la première retouche de l'une.
"""
import ast
import re
from pathlib import Path

DASH = Path(__file__).resolve().parents[1] / "scripts" / "dashboard_trio.py"


def _sans_commentaires(p: Path) -> str:
    """Ces vérifications lisent la SOURCE : sans ce nettoyage elles mesureraient
    la prose, et les commentaires qui EXPLIQUENT le changement contiennent par
    construction les chaînes cherchées."""
    return re.sub(r"(^|[^:])#.*$", r"",
                  DASH.read_text(encoding="utf-8"), flags=re.M)


SRC = _sans_commentaires(DASH)

# La section du round SEULE. « sinon : » et les alternatives existent aussi
# dans l'onglet « Que jouer ? », pour un tout autre contenu : compter sur le
# fichier entier ferait echouer ce test au premier ajout ailleurs.
ROUND = SRC[SRC.index("        def _score(r):"):SRC.index("        pieges = []")]


def test_le_score_est_rendu_par_une_seule_fonction():
    """C'est l'invariant qui compte : le format n'existe qu'une fois."""
    assert SRC.count("def _score(") == 1
    assert SRC.count("def _autres_scores(") == 1
    # Aucune autre construction du libelle « score **X-Y** » dans le fichier.
    assert SRC.count("· score **") == 1, "le format du score est écrit ailleurs"


def test_les_deux_listes_appellent_le_meme_rendu():
    """Compte les APPELS par l'arbre syntaxique : une recherche de chaîne
    comptait aussi la ligne de définition, et aurait compté un appel commente."""
    arbre = ast.parse(DASH.read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(arbre)
              if isinstance(n, ast.FunctionDef) and n.name == "main")
    appels = [n.func.id for n in ast.walk(fn)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
              and n.func.id in ("_score", "_autres_scores")]
    # Le Top 3 et « les autres matchs » : un score chacun.
    assert appels.count("_score") == 2
    # Deux par liste : le test de présence, puis le rendu.
    assert appels.count("_autres_scores") == 4


def test_la_liste_des_autres_matchs_annonce_le_score():
    assert "Les autres matchs — avec score exact" in SRC


def test_les_alternatives_accompagnent_le_score():
    """Sur un marché touché à 11,8 %, donner UN score sans ses suivants laisse
    croire à une précision qu'il n'a pas."""
    assert ROUND.count("sinon : ") == 2


def test_les_fonctions_sont_definies_avant_leur_usage():
    """En Streamlit l'ordre du code est l'ordre d'exécution : une fonction
    définie plus bas que son appel lèverait dès le premier rendu."""
    arbre = ast.parse(DASH.read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(arbre)
              if isinstance(n, ast.FunctionDef) and n.name == "main")
    pos = {n.name: n.lineno for n in ast.walk(fn)
           if isinstance(n, ast.FunctionDef) and n.name in ("_score", "_autres_scores")}
    assert set(pos) == {"_score", "_autres_scores"}
    appels = [n.lineno for n in ast.walk(fn)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
              and n.func.id in pos]
    assert appels, "aucun appel trouvé"
    assert min(appels) > max(pos.values())


def test_aucun_score_exact_ne_revient_dans_les_marches_bannis():
    """`SCORES_EXACTS` bannit ces MARCHÉS d'un autre bandeau. Le score du round
    est une prédiction, pas un marché à jouer : les deux ne doivent pas se
    confondre, et l'interdiction doit rester en place."""
    import sys
    sys.path.insert(0, str(DASH.parent))
    assert "SCORES_EXACTS = {" in SRC
    assert "Score exact" in SRC.split("SCORES_EXACTS = {")[1][:120]
