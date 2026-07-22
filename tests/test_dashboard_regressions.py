"""Garde-fous de non-régression sur le dashboard (statique, sans lancer Streamlit).

Ces tests reproduisent des bugs RÉELLEMENT survenus en production.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

DASH = Path(__file__).resolve().parents[1] / "scripts" / "dashboard_trio.py"


def _tree():
    return ast.parse(DASH.read_text(encoding="utf-8"))


def test_pandas_is_imported_at_module_level():
    """BUG RÉEL : `pd` importé seulement en local dans main() -> UnboundLocalError
    quand une nouvelle ligne l'utilisait avant l'import. Il DOIT être module-level."""
    tree = _tree()
    top = [n for n in tree.body if isinstance(n, ast.Import)]
    names = {a.asname or a.name for imp in top for a in imp.names}
    assert "pd" in names, "pandas doit être importé au niveau module (alias pd)"


def test_main_has_no_local_pandas_import():
    """Un seul import local de pandas dans main() re-crée le piège UnboundLocalError
    (Python marque alors `pd` comme local pour TOUTE la fonction)."""
    tree = _tree()
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    locals_ = [
        a.asname or a.name
        for n in ast.walk(main)
        if isinstance(n, ast.Import)
        for a in n.names
    ]
    assert "pd" not in locals_, (
        "import pandas local dans main() -> rend `pd` local à toute la fonction "
        "et casse tout usage placé avant l'import")


def test_dashboard_compiles():
    compile(DASH.read_text(encoding="utf-8"), str(DASH), "exec")


def _is_expander(node) -> bool:
    return isinstance(node, ast.With) and any(
        isinstance(it.context_expr, ast.Call)
        and isinstance(it.context_expr.func, ast.Attribute)
        and it.context_expr.func.attr == "expander"
        for it in node.items
    )


def test_no_nested_expanders():
    """Streamlit interdit st.expander() imbriqué (lève StreamlitAPIException).

    Vérification par AST (l'indentation seule donne des faux positifs : un expander
    dans un `if`/`for` est profondément indenté sans être imbriqué)."""
    def walk(node, depth):
        d = depth + 1 if _is_expander(node) else depth
        assert d <= 1, f"expander imbriqué ligne {getattr(node, 'lineno', '?')}"
        for child in ast.iter_child_nodes(node):
            walk(child, d)

    walk(_tree(), 0)


def test_widget_keys_are_unique():
    """Deux widgets Streamlit avec la même key -> DuplicateWidgetID à l'exécution."""
    src = DASH.read_text(encoding="utf-8")
    keys = re.findall(r'key\s*=\s*"([^"]+)"', src)
    dupes = {k for k in keys if keys.count(k) > 1}
    assert not dupes, f"clés de widget dupliquées : {sorted(dupes)}"
