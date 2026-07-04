"""Shadow-RNG Ensemble System.

Laboratoire de simulation + détection de divergence du RNG d'un bookmaker
de football virtuel. Pas un simple prédicteur : un cadre de SURVEILLANCE.

Briques :
  A - DistributionProfiler  : extrait la "signature" du RNG (biais réel vs théorique,
      régime, mémoire) sur fenêtres glissantes.                      [profiler.py]
  B - ShadowRNGSimulator    : 4 simulateurs Monte-Carlo branchables. [simulators.py]
  C - EnsembleVoter         : vote pondéré + filtre de divergence.   [ensemble.py]

Référence théorique (validée) : 1X2 -> devig -> lambda -> grille Poisson pure.
BASELINE = cette théorique. Tout le reste se mesure CONTRE elle.
"""
from .config import DEFAULT_CONFIG  # noqa: F401
from .profiler import DistributionProfiler  # noqa: F401
from .simulators import (  # noqa: F401
    ShadowRNGSimulator, BaseSimulator,
    BaselineSimulator, TrendSimulator, MemorySimulator, RegimeSimulator,
    build_transition_matrix,
)
from .ensemble import EnsembleVoter  # noqa: F401

__all__ = [
    "DEFAULT_CONFIG", "DistributionProfiler", "ShadowRNGSimulator", "BaseSimulator",
    "BaselineSimulator", "TrendSimulator", "MemorySimulator", "RegimeSimulator",
    "build_transition_matrix", "EnsembleVoter",
]
