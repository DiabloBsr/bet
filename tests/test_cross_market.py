"""Invariants des prédicats de marché (cross_market_check).

Un prédicat faux fabrique un faux edge — c'est très exactement ce qui s'est produit
avec « supérieur à 4 » lu comme « exactement 4 » (+35% de ROI fantôme). On ne teste
donc pas des cas isolés mais des INVARIANTS DE PARTITION : sur les 28 issues
possibles (tous les scores de total <= 6), les libellés d'un marché doivent couvrir
chaque issue le bon nombre de fois. Un décalage d'une seule case casse le compte.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import cross_market_check as cm  # noqa: E402

SCORES = cm.SCORES


def _couverture(fabrique, labels):
    """Nombre de libellés qui font gagner chaque issue."""
    preds = []
    for l in labels:
        p = fabrique(l)
        assert p is not None, f"libellé non reconnu : {l!r}"
        preds.append(p)
    return [sum(1 for p in preds if p(h, a)) for h, a in SCORES]


def test_les_28_issues_sont_bien_la():
    """Le RNG plafonne à 6 buts : 28 scores possibles, ni plus ni moins."""
    assert len(SCORES) == 28
    assert all(h + a <= 6 for h, a in SCORES)


@pytest.mark.parametrize("nom,fabrique,labels", [
    ("1X2", cm._x12, ["1", "X", "2"]),
    ("G/NG", cm._gng, ["Oui", "Non"]),
    ("Pair/Impair", cm._parite, ["Pair", "Impair"]),
    ("Total de buts", cm._cell_total, [str(i) for i in range(7)]),
    ("+/-", cm._total_global, ["> 3.5", "< 3.5"]),
    ("Total dom.", lambda l: cm._total_equipe(l, True), ["> 3.5", "< 3.5"]),
    ("Total ext.", lambda l: cm._total_equipe(l, False), ["> 3.5", "< 3.5"]),
    ("1X2 & Total", cm._x12_total,
     ["1 / > 3.5", "X / > 3.5", "2 / > 3.5", "1 / < 3.5", "X / < 3.5", "2 / < 3.5"]),
    ("1X2 & G/NG", cm._x12_gng,
     ["1 gagne et les deux équipes marquent", "1 gagne et seulement  1  marque",
      "X et les deux équipes marquent", "X et aucun but",
      "2 gagne et les deux équipes marquent", "2 gagne et seulement 2 marque"]),
])
def test_partition_exacte(nom, fabrique, labels):
    """Chaque issue doit être couverte par EXACTEMENT un libellé (somme des probas = 1)."""
    cov = _couverture(fabrique, labels)
    mauvais = [(s, c) for s, c in zip(SCORES, cov) if c != 1]
    assert not mauvais, f"{nom} ne partitionne pas : {mauvais[:5]}"


def test_double_chance_couvre_chaque_issue_deux_fois():
    """1X, 12, X2 se chevauchent par construction : chaque score en gagne exactement 2."""
    cov = _couverture(cm._dc, ["1X", "12", "X2"])
    assert set(cov) == {2}, f"couverture inattendue : {sorted(set(cov))}"


def test_multibuts_est_chevauchant_donc_non_partitionnant():
    """Garde-fou de lecture : Multi-Buts n'est PAS une partition (plages qui se
    recouvrent). C'est pourquoi on ne peut pas le déviger seul — il faut une
    référence externe (« Score exact »)."""
    cov = _couverture(cm._total_global, [
        "Le total de buts est de 0, 1 ou 2", "Le total de buts est de 1, 2 ou 3",
        "Le total de buts est de 2, 3 ou 4", "Le total de buts est supérieur à 4"])
    assert max(cov) > 1, "Multi-Buts devrait se chevaucher"


def test_score_exact_reference_est_devigee():
    """La référence doit sommer à 1 et retirer la marge."""
    faux = {f"{h}-{a}": 30.0 for h, a in SCORES}      # 28 issues a cote 30 -> marge enorme
    ref = cm._reference({"Score exact": faux})
    assert ref is not None
    loi, sature = ref
    assert abs(sum(loi.values()) - 1.0) < 1e-9
    assert all(abs(v - 1 / 28) < 1e-9 for v in loi.values())
    assert sature == set()


def test_reference_incomplete_est_refusee():
    """Partition incomplète = non dévigeable : on écarte au lieu de deviner."""
    assert cm._reference({"Score exact": {"0-0": 5.0}}) is None
    assert cm._reference({}) is None


def test_reference_signale_les_cases_saturees():
    """Les cases plafonnées doivent être NOMMÉES, pas juste comptées : c'est leur
    position qui permet d'écarter les offres dont la zone gagnante s'appuie dessus
    (la masse fantôme de ~13.8% en CAN tombait pile sur « > 3.5 »)."""
    faux = {f"{h}-{a}": (100.0 if h + a > 3 else 5.0) for h, a in SCORES}
    loi, sature = cm._reference({"Score exact": faux})
    assert sature == {(h, a) for h, a in SCORES if h + a > 3}
    assert all(h + a > 3 for h, a in sature)


def test_le_nom_de_marche_est_exact_pas_un_prefixe():
    """BUG RÉEL : le préfixe « 1X2 » attrapait le marché « 1X2 & Total », et l'on
    comparait alors un marché aux cotes d'un autre."""
    extra = {"1X2 & Total": {"1 / > 3.5": 5.0}, "Double Chance": {"1X": 1.3}}
    assert cm._market(extra, "1X2") is None
    assert cm._market(extra, "1X2 & Total") == {"1 / > 3.5": 5.0}
    assert cm._market(extra, "Double Chance") == {"1X": 1.3}


def test_accents_du_nom_de_marche():
    """« Total equipe extérieur » doit être trouvé malgré l'accent."""
    assert cm._market({"Total equipe extérieur": {"> 3.5": 9.0}},
                      "Total equipe extérieur") == {"> 3.5": 9.0}


def test_libelle_inconnu_est_ignore_pas_devine():
    """Aucun repli : un libellé inconnu doit renvoyer None (l'offre sera ignorée)."""
    assert cm._x12("3") is None
    assert cm._dc("2X") is None
    assert cm._gng("Peut-être") is None
    assert cm._x12_total("1 sans plage") is None
