"""Unit tests for the trap detector (pure logic, no I/O, no DB).

Encode les vérités MESURÉES du projet (voir THEORIES_TESTED.md) : aucun marché n'est
+EV, combiner multiplie la marge, un panier de simples la moyenne. Ces tests garantissent
qu'une future modif ne fera pas dire au détecteur qu'un pari est rentable.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import trap_detector as td


def test_every_market_is_negative_ev():
    """Invariant du projet : aucun marché n'a un ROI positif."""
    for key in td.MARKETS:
        assert td.evaluate_single(key).roi < 0, f"{key} ne doit jamais être +EV"


def test_thinnest_margin_markets_are_the_least_bad():
    """1X2 / O-U 3.5 (marge 5.7%) doivent être moins mauvais que Score exact et HT/FT."""
    best = td.evaluate_single("1x2").roi
    assert best > td.evaluate_single("score_exact").roi
    assert best > td.evaluate_single("htft").roi
    assert best > td.evaluate_single("total_buts").roi


def test_combo_multiplies_the_margin():
    """Combiner N legs -EV doit donner un ROI STRICTEMENT pire que 1 leg."""
    one = td.evaluate_single("1x2").roi
    three = td.evaluate_combo([("1x2", 2.0)] * 3).roi
    five = td.evaluate_combo([("1x2", 2.0)] * 5).roi
    assert three < one, "3 legs doit être pire qu'un simple"
    assert five < three, "5 legs doit être pire que 3"


def test_basket_averages_and_beats_the_combo():
    """Un panier de simples moyenne l'EV : meilleur qu'un combiné de mêmes legs."""
    legs = [("1x2", 2.0)] * 3
    assert td.evaluate_basket(legs).roi > td.evaluate_combo(legs).roi


def test_longshot_odds_flag_a_warning():
    """Une grosse cote doit déclencher un avertissement explicite."""
    v = td.evaluate_single("1x2", odds=td.LONGSHOT_ODDS + 1)
    assert any("grosse cote" in r.lower() for r in v.reasons)


def test_cheaper_alternative_points_to_a_known_market():
    """Toute alternative conseillée doit exister dans le catalogue."""
    for key, (alt, _why) in td.CHEAPER.items():
        assert alt in td.MARKETS
        assert td.MARKETS[alt][2] > td.MARKETS[key][2], (
            f"l'alternative {alt} doit être moins mauvaise que {key}")


def test_expected_loss_scales_with_stake():
    v = td.evaluate_single("1x2", odds=2.0, stake=10_000)
    assert v.expected_loss is not None
    assert abs(v.expected_loss - (-v.roi * 10_000)) < 1e-9


def test_exact_score_uses_measured_roi_when_available():
    """Si la table mesurée est présente, un score connu doit l'utiliser (et rester -EV)."""
    if not td.SCORE_ROI:
        return  # table absente en CI : rien à vérifier
    known = next(iter(td.SCORE_ROI))
    v = td.evaluate_exact_score(known)
    assert v.roi == td.SCORE_ROI[known]["roi"]
    assert v.roi < 0


def test_unknown_market_does_not_crash():
    v = td.evaluate_single("marche_inexistant")
    assert v.severity == "⚪"


# ---- calibration consciente de la ligue (bug mesuré : CAN 7.8pp -> 3.5pp) ----

def test_calibration_table_is_league_aware():
    """Chaque ligue doit utiliser SA table, jamais celle d'une autre.

    Les constantes du simulateur sont ajustées sur l'anglaise ; réutiliser sa table
    ailleurs dé-calibre. Hold-out CAN (n=8000, matchs antérieurs à la fenêtre
    d'ajustement) : sans calibration écart max/total 5.0pp, avec la table anglaise
    0.9pp, avec sa propre table 0.8pp (et max/score 2.7 -> 0.9 -> 0.6pp)."""
    import predict_trio as pt
    if not pt._CALIB_BY_LG:
        return  # tables absentes : rien à vérifier
    dist = {"1-0": 0.4, "0-0": 0.3, "2-1": 0.3}
    CAN = "InstantLeague-8060"
    if CAN in pt._CALIB_BY_LG and pt.LG in pt._CALIB_BY_LG:
        assert pt._apply_calib(dist, CAN) != pt._apply_calib(dist, pt.LG), (
            "deux ligues ne doivent pas partager la même correction")
    # une ligue sans table mesurée n'est PAS corrigée au hasard
    assert pt._apply_calib(dist, "InstantLeague-INEXISTANTE") == dist
    assert pt._calib_for("InstantLeague-INEXISTANTE") is None


def test_every_league_has_its_own_calibration():
    """Les 9 ligues suivies doivent chacune avoir une table (sinon elles ne sont pas corrigées)."""
    import predict_trio as pt
    if not pt._CALIB_BY_LG:
        return
    assert len(pt._CALIB_BY_LG) >= 9, f"seulement {len(pt._CALIB_BY_LG)} tables"
    for lg, mat in pt._CALIB_BY_LG.items():
        assert mat.shape == (7, 7), f"{lg} : matrice {mat.shape}"
        assert (mat > 0).all(), f"{lg} : correction nulle ou négative"
