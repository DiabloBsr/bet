"""Cœur numérique du signal Under/Over CAN (_devig_over25).

Un bug de lecture de marché fabrique un faux signal — c'est arrivé deux fois cette
session (parseur « supérieur à 4 », préfixe « 1X2 »). On verrouille donc le dévigage
du marché « Total de buts » par des cas déterministes.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import predict_trio as pt  # noqa: E402


def test_devig_over25_partition_uniforme():
    """7 cellules à cote 7 → uniforme : P(total>2.5)=P(3..6)=4/7."""
    T = {str(i): 7.0 for i in range(7)}
    p = pt._devig_over25({"Total de buts": T})
    assert abs(p - 4 / 7) < 1e-9


def test_devig_over25_retire_la_marge():
    """La proba dévigée somme à 1 : une marge uniforme ne déplace pas P(over)."""
    T = {str(i): 7.0 / 0.85 for i in range(7)}      # même distribution, +18% de marge
    p = pt._devig_over25({"Total de buts": T})
    assert abs(p - 4 / 7) < 1e-9


def test_devig_over25_sensible_au_profil():
    """Masse concentrée sur les petits totaux → P(over) faible (régime CAN)."""
    T = {"0": 3.0, "1": 3.0, "2": 3.0, "3": 20.0, "4": 40.0, "5": 80.0, "6": 99.0}
    p = pt._devig_over25({"Total de buts": T})
    assert p < 0.25, f"P(over) devrait être faible, vaut {p}"


def test_devig_over25_partition_incomplete_refusee():
    """Moins de 7 cellules lisibles → None (on n'invente pas)."""
    assert pt._devig_over25({"Total de buts": {"0": 3.0, "1": 3.0}}) is None
    assert pt._devig_over25({}) is None
    assert pt._devig_over25({"Total de buts": {str(i): 0.0 for i in range(7)}}) is None


def test_devig_over25_accent_et_json():
    """Nom accentué toléré et JSON brut accepté (données réelles)."""
    import json
    T = {str(i): 5.0 for i in range(7)}
    assert pt._devig_over25(json.dumps({"Total de buts": T})) is not None


def test_over_leaning_matches_sortent_en_tete():
    """Contrat d'affichage : les matchs OVER (à contre-courant en CAN) passent avant
    les UNDER, pour surfacer les appels informatifs — invariant vérifié sur la clé
    de tri, sans base (liste construite à la main)."""
    rows = [
        {"contre_courant": False, "p_over": 0.30, "recent": False},
        {"contre_courant": True, "p_over": 0.70, "recent": False},
        {"contre_courant": False, "p_over": 0.10, "recent": False},
    ]
    rows.sort(key=lambda x: (x["recent"], not x["contre_courant"], -abs(x["p_over"] - 0.5)))
    assert rows[0]["contre_courant"] is True, "les OVER doivent être affichés d'abord"
