"""Le trio V2/V5 s'entraîne sur CHAQUE ligue, pas seulement l'anglaise (11/10).

« Résoudre le problème de marché sur la ligue Allemagne pour que le moteur
s'entraîne aussi dans cette ligue. »

Avant, `fit()` et `load_hist()` étaient figés sur l'anglaise (LG=8035) ; les
autres ligues tournaient en « marché pur ». Désormais c'est paramétré par
ligue, chaque ligue ayant son historique coté et sa calibration.
"""
import sys
from pathlib import Path

import pandas as pd

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "scripts"))
import predict_trio as pt  # noqa: E402

ALL = "InstantLeague-8043"   # Allemagne


def test_load_hist_cible_la_ligue_demandee(monkeypatch):
    vues = {}

    def faux_read_sql(q, engine):
        vues["q"] = q
        return pd.DataFrame()
    monkeypatch.setattr(pt.pd, "read_sql", faux_read_sql)
    pt.load_hist(None, ALL)
    assert f"e.competition='{ALL}'" in vues["q"]


def test_load_hist_defaut_reste_l_anglaise(monkeypatch):
    vues = {}
    monkeypatch.setattr(pt.pd, "read_sql",
                        lambda q, e: vues.update(q=q) or pd.DataFrame())
    pt.load_hist(None)
    assert f"e.competition='{pt.LG}'" in vues["q"]
    assert pt.LG == "InstantLeague-8035"


def test_fit_entraine_sur_la_ligue_et_passe_la_competition(monkeypatch):
    """fit(engine, lg) charge l'historique de lg ET dit à V5 quelle ligue
    (pour les classements), au lieu de l'anglaise en dur."""
    recu = {}
    monkeypatch.setattr(pt, "load_hist",
                        lambda e, lg: recu.update(hist_lg=lg) or pd.DataFrame({"x": [1, 2]}))
    monkeypatch.setattr(pt, "fit_model_v5",
                        lambda h, **k: recu.update(v5_kwargs=k) or "m5")
    monkeypatch.setattr(pt, "fit_model_v2", lambda h: "v2")
    m5, v2, n = pt.fit(None, ALL)
    assert (m5, v2, n) == ("m5", "v2", 2)
    assert recu["hist_lg"] == ALL
    assert recu["v5_kwargs"].get("competition") == ALL
    assert recu["v5_kwargs"].get("form_alpha") == 0.0


def test_fit_defaut_anglaise(monkeypatch):
    recu = {}
    monkeypatch.setattr(pt, "load_hist",
                        lambda e, lg: recu.update(lg=lg) or pd.DataFrame())
    monkeypatch.setattr(pt, "fit_model_v5", lambda h, **k: None)
    monkeypatch.setattr(pt, "fit_model_v2", lambda h: None)
    pt.fit(None)
    assert recu["lg"] == pt.LG


def test_chaque_ligue_de_l_app_a_sa_calibration():
    """Sans table par ligue, un modèle entraîné sur l'Allemagne aurait des
    probas dé-calibrées. Les 9 ligues de l'app doivent être couvertes."""
    import json
    LEAGUES = {
        "InstantLeague-8035", "InstantLeague-8065", "InstantLeague-8056",
        "InstantLeague-8060", "InstantLeague-8036", "InstantLeague-8037",
        "InstantLeague-8044", "InstantLeague-8043", "InstantLeague-8042",
    }
    cal = json.loads((RACINE / "config" / "score_calibration.json")
                     .read_text(encoding="utf-8"))["per_league"]
    assert LEAGUES <= set(cal), LEAGUES - set(cal)


def test_le_dashboard_entraine_par_ligue():
    src = (RACINE / "scripts" / "dashboard_trio.py").read_text(encoding="utf-8")
    # _fit prend la ligue, le cache est par ligue, l'appel passe lg.
    assert "def _fit(lg=" in src
    assert "cached_fit(lg)" in src
    assert "pt.fit(eng, lg)" in src
    # l'ancien message "mode marché pur ... entraînés sur l'anglaise" a disparu.
    assert "mode MARCHÉ pur" not in src
    assert "V2/V5 sont\n                   \"entraînés sur l'anglaise" not in src
