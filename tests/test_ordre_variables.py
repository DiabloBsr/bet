"""Aucune variable ne doit être lue avant d'être écrite dans le rendu Streamlit.

BUG RÉEL (remonté par capture) : le bloc de traitement de l'image appelait
`r_tol` et `r_lgs` alors que ces widgets étaient déclarés PLUS BAS. En Streamlit
l'ordre du code est l'ordre d'exécution : dès le premier dépôt de fichier, la
page tombait en `UnboundLocalError`. Aucun test ne pouvait le voir — ils
n'exécutent jamais le rendu.

Ce fichier contient donc DEUX tests : celui qui contrôle le tableau de bord, et
celui qui vérifie que l'analyseur détecte bien un cas connu-mauvais. Sans ce
second test, un analyseur trop permissif afficherait un vert trompeur — c'est
exactement ce qui s'est produit à la première écriture de ce garde-fou.
"""
from __future__ import annotations

import ast
import builtins
from pathlib import Path

DASH = Path(__file__).resolve().parents[1] / "scripts" / "dashboard_trio.py"

MAUVAIS = '''
from outil import depose, calcul, widget_nombre, widget_liste, choisir, afficher
D = {}

def main():
    img = depose()
    if img is not None:
        lots = [D[k] for k in r_lgs]
        calcul(tol=r_tol)
    r_tol = widget_nombre()
    r_lgs = widget_liste()
'''

# ⚠️ Ajoute le 27/09. `fautes()` ne voyait QUE les noms affectes plus bas
# (`l < e`). Un nom jamais affecte du tout passait au travers -- et c'est
# exactement ce qui est arrive : en supprimant l'onglet de prediction, `_dfi`
# a disparu alors que l'onglet historique le lisait encore. La suite est restee
# verte sur un NameError certain.
JAMAIS_AFFECTE = '''
from outil import depose, calcul, widget_nombre, widget_liste, choisir, afficher
D = {}

def main():
    lg = choisir(index=_dfi)
    afficher(lg)
'''

BON = '''
from outil import depose, calcul, widget_nombre, widget_liste, choisir, afficher
D = {}

def main():
    img = depose()
    r_tol = widget_nombre()
    r_lgs = widget_liste()
    if img is not None:
        lots = [D[k] for k in r_lgs]
        calcul(tol=r_tol)
'''


def _noms_hors_flux(fn: ast.FunctionDef) -> set:
    """Noms qui n'obéissent pas au flux linéaire : paramètres, cibles de boucle
    et de compréhension, alias d'import, variables d'exception, globals, et tout
    ce qui vit dans une fonction imbriquée (exécutée plus tard)."""
    hors = {a.arg for a in fn.args.args}
    for n in ast.walk(fn):
        if isinstance(n, (ast.For, ast.AsyncFor)):
            hors |= {x.id for x in ast.walk(n.target) if isinstance(x, ast.Name)}
        elif isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            for g in n.generators:
                hors |= {x.id for x in ast.walk(g.target) if isinstance(x, ast.Name)}
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            hors |= {(a.asname or a.name).split(".")[0] for a in n.names}
        elif isinstance(n, ast.ExceptHandler) and n.name:
            hors.add(n.name)
        elif isinstance(n, ast.Global):
            hors |= set(n.names)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n is not fn:
            hors.add(n.name)
            hors |= {x.id for x in ast.walk(n) if isinstance(x, ast.Name)}
        elif isinstance(n, ast.Lambda):
            hors |= {x.id for x in ast.walk(n) if isinstance(x, ast.Name)}
    return hors


def fautes(src: str, fonction: str = "main") -> list:
    """Noms lus AVANT leur première affectation, dans le flux linéaire."""
    tree = ast.parse(src)
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == fonction), None)
    if fn is None:
        return []
    hors = _noms_hors_flux(fn)
    # Uniquement le niveau MODULE : parcourir tout l'arbre y ferait entrer les
    # affectations internes a main(), qui sont precisement celles a controler.
    # C'est cette erreur qui rendait le garde-fou aveugle a sa premiere ecriture.
    # Les builtins ne sont jamais « affectes » : sans eux, `len`, `next` ou
    # `Exception` ressortiraient comme des NameError des qu'on a commence a
    # signaler les noms jamais affectes (27/09).
    connus = set(dir(builtins))
    for x in tree.body:
        if isinstance(x, ast.Assign):
            connus |= {n.id for n in x.targets if isinstance(n, ast.Name)}
        elif isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            connus.add(x.name)
        elif isinstance(x, (ast.Import, ast.ImportFrom)):
            connus |= {(a.asname or a.name).split(".")[0] for a in x.names}
        elif isinstance(x, (ast.If, ast.Try)):        # imports/constantes gardes
            for y in ast.walk(x):
                if isinstance(y, (ast.Import, ast.ImportFrom)):
                    connus |= {(a.asname or a.name).split(".")[0] for a in y.names}
                elif isinstance(y, ast.Assign):
                    connus |= {n.id for n in y.targets if isinstance(n, ast.Name)}
    ecrit, lu = {}, {}
    for n in ast.walk(fn):
        if not isinstance(n, ast.Name):
            continue
        cible = ecrit if isinstance(n.ctx, ast.Store) else lu
        if isinstance(n.ctx, (ast.Store, ast.Load)):
            cible[n.id] = min(cible.get(n.id, 10 ** 9), n.lineno)
    out = []
    for nom, l in lu.items():
        if nom in hors or nom in connus:
            continue
        e = ecrit.get(nom)
        if e is None:
            # Jamais affecte : NameError garanti des que la ligne s'execute.
            # Aussi grave que l'affectation trop tardive, et plus silencieux.
            out.append(f"{nom} : lu ligne {l}, jamais affecté")
        elif l < e:
            out.append(f"{nom} : lu ligne {l}, affecté seulement ligne {e}")
    return sorted(out)


# ---------- l'analyseur doit d'abord prouver qu'il detecte ----------

def test_analyseur_detecte_le_cas_connu_mauvais():
    """Sans ce test, un analyseur trop permissif donnerait un vert trompeur."""
    f = fautes(MAUVAIS)
    noms = " ".join(f)
    assert "r_lgs" in noms and "r_tol" in noms, f"cas fautif non détecté : {f}"


def test_analyseur_detecte_le_nom_jamais_affecte():
    """Un nom jamais affecte est un NameError certain, et plus discret qu'une
    affectation trop tardive : rien dans le fichier ne le rappelle."""
    f = fautes(JAMAIS_AFFECTE)
    assert any("_dfi" in x and "jamais" in x for x in f),         f"nom jamais affecte non detecte : {f}"


def test_analyseur_ne_crie_pas_sur_du_code_correct():
    assert fautes(BON) == [], f"faux positif sur du code correct : {fautes(BON)}"


# ---------- le contrôle qui compte ----------

def test_dashboard_sans_lecture_avant_affectation():
    f = fautes(DASH.read_text(encoding="utf-8"))
    assert not f, ("variable(s) lue(s) avant affectation dans main() — la page "
                   "tombera en UnboundLocalError à l'exécution :\n  " + "\n  ".join(f))
