"""Tout module importe par l'app doit PARTIR au deploiement.

⚠️ PIEGE PAYE LE 29/09. `scripts/forme_pastilles.py` a ete cree puis importe
par `dashboard_trio`, sans etre ajoute a `DEFAULT_FILES`. Le depot etait vert,
les 400 tests passaient, et l'app en ligne serait morte a l'import — un ecran
blanc, sans rien pour comprendre.

Ce fichier ferme la porte : il lit les imports REELS des scripts deployes et
verifie que chacun de leurs voisins locaux figure dans la liste.
"""
import ast
import re
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
SCRIPTS = RACINE / "scripts"
DEPLOY = RACINE / "deploy" / "deploy_hf.py"


def _liste_deployee() -> set:
    src = DEPLOY.read_text(encoding="utf-8")
    d = src.index("DEFAULT_FILES = [")
    return set(re.findall(r'"([^"]+)"', src[d:src.index("]", d)]))


def _modules_locaux() -> set:
    """Les modules qu'un script peut importer par leur nom nu."""
    return {p.stem for p in SCRIPTS.glob("*.py")}


def _noms_importes(arbre) -> set:
    """`import x`, `from x import y` et `from scripts.x import y` -> {"x"}."""
    noms = set()
    for n in ast.walk(arbre):
        if isinstance(n, ast.Import):
            modules = [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            modules = [n.module]
        else:
            continue
        for m in modules:
            parts = m.split(".")
            noms.add(parts[1] if parts[0] == "scripts" and len(parts) > 1
                     else parts[0])
    return noms


def _manquants(deployes: set) -> list:
    locaux = _modules_locaux()
    manquants = []
    for chemin in sorted(deployes):
        if not chemin.startswith("scripts/") or not chemin.endswith(".py"):
            continue
        f = RACINE / chemin
        if not f.exists():
            manquants.append(f"{chemin} : liste mais absent du depot")
            continue
        for nom in _noms_importes(ast.parse(f.read_text(encoding="utf-8"))):
            if nom in locaux and f"scripts/{nom}.py" not in deployes:
                manquants.append(
                    f"{chemin} importe `{nom}` : ajoute scripts/{nom}.py "
                    f"a DEFAULT_FILES, sinon l'app en ligne meurt a l'import")
    return sorted(set(manquants))


def test_chaque_import_local_des_scripts_deployes_est_deploye():
    manquants = _manquants(_liste_deployee())
    assert not manquants, "\n".join(manquants)


def test_le_garde_fou_detecte_bien_un_oubli():
    """Sans ce second test, un analyseur trop permissif donnerait un vert
    trompeur — c'est deja arrive sur un autre garde-fou de ce depot.
    On rejoue l'oubli du 29/09 : la liste privee de forme_pastilles."""
    faux = _liste_deployee() - {"scripts/forme_pastilles.py"}
    assert any("`forme_pastilles`" in m for m in _manquants(faux)), \
        "l'analyseur ne voit plus l'import de forme_pastilles"


def test_le_garde_fou_lit_aussi_la_forme_scripts_point():
    """`from scripts.x import y` doit compter comme un import de x."""
    arbre = ast.parse("from scripts.ui_theme import hero\nimport scripts.a.b")
    assert _noms_importes(arbre) == {"ui_theme", "a"}
