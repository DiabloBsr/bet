"""Fraîcheur des résultats et keep-alive du Space (04/10/2026).

Le face-à-face gelait car la collecte en ligne s'arrêtait quand le Space HF
s'endormait (gcTimeout 48 h). Deux garde-fous : un ping GitHub Actions qui
garde le Space éveillé, et un indicateur de fraîcheur dans l'app.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "scripts"))
import predict_trio as pt  # noqa: E402


@pytest.fixture
def base(monkeypatch):
    rows = {"m": ["2026-07-05 10:48:00"]}

    def faux(q, engine):
        return pd.DataFrame(rows)
    monkeypatch.setattr(pt.pd, "read_sql", faux)
    return rows


def test_date_du_dernier_resultat_et_age(base):
    txt, age = pt.derniere_date_resultat(None)
    assert txt.startswith("05/07/2026")
    assert age > 2, "une base de juillet lue en octobre doit être signalée comme vieille"


def test_une_base_vide_ne_leve_pas(base):
    base["m"] = [None]
    assert pt.derniere_date_resultat(None) is None


def test_une_lecture_impossible_ne_leve_pas(monkeypatch):
    monkeypatch.setattr(pt.pd, "read_sql",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError()))
    assert pt.derniere_date_resultat(None) is None


def test_le_dashboard_affiche_la_fraicheur():
    src = (RACINE / "scripts" / "dashboard_trio.py").read_text(encoding="utf-8")
    assert "derniere_date_resultat(engH)" in src
    assert "collecte en ligne est" in src   # le message d'alerte


def test_le_workflow_keep_alive_existe_et_ping_assez_souvent():
    import yaml
    wf = RACINE / ".github" / "workflows" / "keep-space-awake.yml"
    assert wf.exists()
    d = yaml.safe_load(wf.read_text(encoding="utf-8"))
    on = d.get("on", d.get(True))   # PyYAML lit 'on' comme True
    cron = on["schedule"][0]["cron"]
    # toutes les 12 h : l'intervalle doit rester < 48 h (gcTimeout du Space)
    assert cron == "0 */12 * * *"
    corps = wf.read_text(encoding="utf-8")
    assert "oliviobsr-vfoot-trio.hf.space" in corps.lower()
