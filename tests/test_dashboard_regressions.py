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


# ---- accès base : le scraper écrit en parallèle, un verrou est NORMAL ----

# Ces deux-là ont déjà leur propre try/except avec un message adapté.
SPINNERS_AUTORISES = ("Calcul de la calibration", "Fit V5+V2")


def test_db_calls_are_guarded():
    """BUG RÉEL : un `st.spinner` nu autour d'un accès base affiche une trace Python
    en pleine page quand SQLite est verrouillé (crash-loop Hugging Face). Tout accès
    doit passer par `_db()`, qui explique et arrête proprement le rendu."""
    src = DASH.read_text(encoding="utf-8")
    nus = [ln.strip() for ln in src.splitlines()
           if "with st.spinner(" in ln and not any(a in ln for a in SPINNERS_AUTORISES)]
    assert not nus, f"accès base non gardé (utiliser _db) : {nus}"
    assert "def _db(" in src, "le garde-fou _db() a disparu"
    assert src.count("with _db(") >= 12, "des accès base ne passent plus par _db()"


def test_db_guard_stops_the_page():
    """La garde doit ARRÊTER le rendu : sinon le code suivant plante sur une
    variable jamais affectée (NameError), ce qui masque la vraie cause."""
    tree = _tree()
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_db")
    handlers = [h for h in ast.walk(fn) if isinstance(h, ast.ExceptHandler)]
    assert handlers, "_db doit intercepter les erreurs base"
    calls = {getattr(c.func, "attr", "") for h in handlers for c in ast.walk(h)
             if isinstance(c, ast.Call)}
    assert "stop" in calls, "_db doit appeler st.stop() après avoir expliqué"


def test_history_block_degrades_without_stopping():
    """Les 3 onglets d'historique sont indépendants : l'un peut échouer sur un
    verrou sans priver l'utilisateur des deux autres (donc pas de st.stop() ici)."""
    src = DASH.read_text(encoding="utf-8")
    tree = _tree()
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_hist_block")
    seg = ast.get_source_segment(src, fn) or ""
    assert "_safe(" in seg, "les lectures base de _hist_block doivent passer par _safe"
    assert "head_to_head(engine" not in seg, "appel base non gardé dans _hist_block"


# ---- app cloud (streamlit_app.py) : meme piege de calibration ----

CLOUD = Path(__file__).resolve().parents[1] / "scripts" / ".." / "streamlit_app.py"


def test_cloud_app_passes_the_league_to_calibration():
    """BUG RÉEL : l'app cloud appelait _apply_calib/_over25_calib SANS ligue, donc
    appliquait la table anglaise aux 9 ligues. Mêmes cotes -> Over 2.5 48.4% (ANG)
    vs 44.8% (CAN) : 3.6pp d'écart qui étaient auparavant confondus."""
    src = CLOUD.read_text(encoding="utf-8", errors="replace")
    assert "_apply_calib(dict(cons), lg)" in src, "_apply_calib appelé sans ligue"
    assert "_over25_calib(oh, od, oa, lg)" in src, "_over25_calib appelé sans ligue"
    assert 'lg = f"InstantLeague-{ev[\'lid\']}"' in src, "la ligue n'est plus dérivée du match"


def test_cloud_app_loads_per_league_tables():
    """Le bootstrap doit remplir _CALIB_BY_LG : écrire dans _CALIB seul ne calibre
    plus rien depuis le passage aux tables par ligue (perte silencieuse)."""
    src = CLOUD.read_text(encoding="utf-8", errors="replace")
    assert "_CALIB_BY_LG" in src, "le bootstrap n'alimente pas les tables par ligue"


def test_embedded_calibration_has_every_league():
    """config/score_calibration.json est la copie embarquée (data/ n'est pas versionné) :
    si elle se désynchronise, l'app en ligne calibre avec une table périmée."""
    import json
    p = Path(__file__).resolve().parents[1] / "config" / "score_calibration.json"
    if not p.exists():
        return
    c = json.loads(p.read_text(encoding="utf-8"))
    assert len(c.get("per_league", {})) >= 9, "tables par ligue manquantes dans config/"
