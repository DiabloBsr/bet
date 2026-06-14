"""Modele mi-temps (HT) calibre sur la structure du moteur (microscope live, etude),
pour predire des marches BETTABLES pre-match : Mi-tps 1X2, score HT, HT/FT.

Calibration (scripts/_ht_calib.py sur 8035) :
  - split H1/H2 = 45,2% / 54,8% -> lam_HT = F * lam_FT par cote.
  - F_home=0,479, F_away=0,474.
  - Dixon-Coles HT rho=-0,15 : la grille Poisson independante SOUS-estime les nuls HT
    (empirique 41,2% vs Poisson 37,9%) -> rho corrige (modele 41,1%).
  - HT/FT : HT result x 2nde-MT result (lam_2H = (1-F)*lam_FT), independance entre MT.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import poisson

from .predictor_v2 import _dc_tau

F_HOME = 0.479
F_AWAY = 0.474
RHO_HT = -0.15          # Dixon-Coles HT (fitte sur le taux de nul HT)
RHO_2H = -0.066         # 2nde MT : meme correlation que le FT global
MAXG = 7


def _dc_grid(lh: float, la: float, rho: float, max_goals: int = MAXG) -> np.ndarray:
    g = np.zeros((max_goals, max_goals))
    for h in range(max_goals):
        for a in range(max_goals):
            g[h, a] = poisson.pmf(h, lh) * poisson.pmf(a, la) * _dc_tau(h, a, lh, la, rho)
    g = np.clip(g, 0, None)
    s = g.sum()
    if s > 0:
        g /= s
    return g


def ht_grid(lam_h: float, lam_a: float, max_goals: int = MAXG) -> np.ndarray:
    """Grille du score a la mi-temps."""
    return _dc_grid(lam_h * F_HOME, lam_a * F_AWAY, RHO_HT, max_goals)


def second_half_grid(lam_h: float, lam_a: float, max_goals: int = MAXG) -> np.ndarray:
    """Grille des buts de la 2nde mi-temps uniquement."""
    return _dc_grid(lam_h * (1 - F_HOME), lam_a * (1 - F_AWAY), RHO_2H, max_goals)


def _grid_1x2(g: np.ndarray) -> tuple[float, float, float]:
    n = g.shape[0]
    p1 = float(sum(g[h, a] for h in range(n) for a in range(n) if h > a))
    pX = float(sum(g[h, a] for h in range(n) for a in range(n) if h == a))
    return p1, pX, 1.0 - p1 - pX


def ht_predictions(lam_h: float, lam_a: float, top_k: int = 3) -> dict:
    """Mi-tps 1X2, score HT top-k, et HT/FT (9 issues) joints."""
    gh = ht_grid(lam_h, lam_a)
    g2 = second_half_grid(lam_h, lam_a)
    p1, pX, p2 = _grid_1x2(gh)

    # score HT top-k
    n = gh.shape[0]
    flat = gh.flatten()
    idx = np.argsort(flat)[-top_k:][::-1]
    top_ht = [(f"{i // n}-{i % n}", round(float(flat[i]), 4)) for i in idx]

    # HT/FT joint : FT = HT + 2nde MT (independance entre mi-temps)
    htft: dict[str, float] = {}
    for hh in range(n):
        for ha in range(n):
            ph = gh[hh, ha]
            if ph < 1e-6:
                continue
            ht_o = "1" if hh > ha else ("X" if hh == ha else "2")
            for sh in range(n):
                for sa in range(n):
                    p = ph * g2[sh, sa]
                    if p < 1e-7:
                        continue
                    fh, fa = hh + sh, ha + sa
                    ft_o = "1" if fh > fa else ("X" if fh == fa else "2")
                    k = f"{ht_o}/{ft_o}"
                    htft[k] = htft.get(k, 0.0) + p
    htft_sorted = sorted(htft.items(), key=lambda kv: -kv[1])

    return {
        "ht_1x2": {"1": round(p1, 4), "X": round(pX, 4), "2": round(p2, 4)},
        "ht_pick": max((("1", p1), ("X", pX), ("2", p2)), key=lambda kv: kv[1])[0],
        "ht_top_scores": top_ht,
        "htft": {k: round(v, 4) for k, v in htft_sorted},
        "htft_top": htft_sorted[:top_k],
    }
