"""Table de cas figés pour la lecture des libellés + le garde-fou anti-faux-edge.

Régression réelle : `"Le total de buts est supérieur à 4"` lu comme « exactement 4 »
a fabriqué un ROI de +35% (gain payé à la cote d'un événement bien plus rare).
Chaque libellé vu en base a désormais son cas ici.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from market_ranges import (MAX_GOALS, SuspectResultError, check_roi,  # noqa: E402
                           parse_goal_range)

# libellés réellement présents en base (relevés sur 604 384 offres Multi-Buts)
REELS = [
    ("Le total de buts est supérieur à 4", {5, 6, 7, 8, 9, 10, 11, 12}),
    ("Le total de buts est de 0, 1 ou 2", {0, 1, 2}),
    ("Le total de buts est de 1, 2 ou 3", {1, 2, 3}),
    ("Le total de buts est de 2, 3 ou 4", {2, 3, 4}),
    ("> 3.5", {4, 5, 6, 7, 8, 9, 10, 11, 12}),      # marché "+/-"
    ("< 3.5", {0, 1, 2, 3}),
]


@pytest.mark.parametrize("label,attendu", REELS)
def test_libelles_reels(label, attendu):
    assert parse_goal_range(label) == frozenset(attendu)


def test_superieur_est_strict():
    """LE bug historique : 'supérieur à 4' ne doit JAMAIS contenir 4."""
    r = parse_goal_range("Le total de buts est supérieur à 4")
    assert 4 not in r and 5 in r


def test_accents_indifferents():
    assert parse_goal_range("supérieur à 3") == parse_goal_range("superieur a 3")


def test_seuil_decimal_nest_pas_lu_comme_deux_entiers():
    """2e piège de la même famille : '> 3.5' ne doit pas donner {3, 5}."""
    r = parse_goal_range("> 3.5")
    assert r == frozenset(range(4, MAX_GOALS + 1))
    assert 3 not in r
    assert parse_goal_range("< 3.5") == frozenset({0, 1, 2, 3})


def test_over_under_sont_complementaires():
    """Sur un même seuil, Over et Under doivent partitionner exactement les totaux."""
    for seuil in ("0.5", "1.5", "2.5", "3.5", "4.5"):
        over = parse_goal_range(f"> {seuil}")
        under = parse_goal_range(f"< {seuil}")
        assert over & under == frozenset(), f"chevauchement sur {seuil}"
        assert over | under == frozenset(range(MAX_GOALS + 1)), f"trou sur {seuil}"


def test_strict_vs_inclusif():
    assert parse_goal_range("> 4") == frozenset(range(5, MAX_GOALS + 1))
    assert parse_goal_range(">= 4") == frozenset(range(4, MAX_GOALS + 1))
    assert parse_goal_range("< 4") == frozenset({0, 1, 2, 3})
    assert parse_goal_range("<= 4") == frozenset({0, 1, 2, 3, 4})


@pytest.mark.parametrize("label,attendu", [
    ("3+", set(range(3, MAX_GOALS + 1))),
    ("au moins 2 buts", set(range(2, MAX_GOALS + 1))),
    ("4 ou plus", set(range(4, MAX_GOALS + 1))),
    ("moins de 3", {0, 1, 2}),
    ("inférieur à 2", {0, 1}),
    ("au plus 2", {0, 1, 2}),
    ("entre 2 et 5", {2, 3, 4, 5}),
])
def test_autres_formulations(label, attendu):
    assert parse_goal_range(label) == frozenset(attendu)


def test_illisible_renvoie_none():
    """Aucun repli 'par défaut' : une offre illisible doit être écartée, pas devinée."""
    assert parse_goal_range("Pair / Impair") is None
    assert parse_goal_range("") is None


def test_les_plages_ne_sont_jamais_vides_a_tort():
    for label, attendu in REELS:
        assert parse_goal_range(label), f"{label} ne doit pas donner un ensemble vide"


# ---- garde-fou : un ROI positif significatif doit ARRÊTER le backtest ----

def test_roi_negatif_passe():
    pnl = np.full(5000, -0.09)
    assert "SUSPECT" not in check_roi(pnl, "marge normale")


def test_roi_positif_significatif_leve():
    """Le faux +35% doit lever avant d'atteindre l'utilisateur."""
    with pytest.raises(SuspectResultError):
        check_roi(np.full(5000, 0.35), "faux edge")


def test_roi_positif_mais_bruit_ne_leve_pas():
    """Un ROI légèrement positif noyé dans le bruit n'est pas un bug : on le laisse passer."""
    rng = np.random.default_rng(0)
    pnl = rng.normal(0.001, 3.0, 400)          # SE ~0.15 -> 3 sigmas couvrent 0
    assert "SUSPECT" not in check_roi(pnl, "bruit")


def test_mode_non_bloquant_retourne_le_message():
    msg = check_roi(np.full(5000, 0.35), "faux edge", raise_on_suspect=False)
    assert msg.startswith("SUSPECT")
