"""Inversion de marche : recuperer la grille latente (lam_h, lam_a) du moteur a
partir du vecteur de marche complet d'un event, puis lire la distribution
Total-de-buts / Score-exact que le moteur a "en tete".

Fondement (ENGINE_MODEL.md SS3, verifie a 0,000000) : le PRICING est une grille
Poisson(lam_h) x Poisson(lam_a) INDEPENDANTE pure. Donc les cotes 1X2 SEULES
fixent deja exactement (lam_h, lam_a). Les autres marches (Double Chance, Total
de buts, Score exact, G/NG, totals par equipe) sont des fonctions deterministes
de la meme grille : ils CONFIRMENT. Leur valeur ajoutee :
  1. detecter une INCOHERENCE entre marches (residu > 0 = mispricing / value-jitter,
     edges #7-#8) -> seul vrai signal +EV ;
  2. quand Score exact est present et peu cappe, il epingle directement la grille.

Le moteur de SIMULATION (ENGINE_MODEL.md SS4) devie du pricing : +0,12 but,
2-1/1-2 sous-cotes (+17/+20 %), correlation Dixon-Coles rho~-0,066, cap 6 buts.
On applique ces deviations CONNUES (Pass B) pour pousser la prediction de score
exact vers son plafond (~13-14 %).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import poisson

from .predictor_v2 import (
    devig,
    grid_modal_score,
    grid_to_1x2,
    grid_top_k_scores,
    poisson_score_grid,
)

MAXG = 8          # support interne de la grille (h, a in 0..7)
CAP = 6           # total de buts cappe a "6+" (= la borne du marche Total de buts)
CAPPED_COTE = 99.99   # cote >= 99.99 = cellule cappee, interdite (ROI -70 %)

_ARANGE = np.arange(MAXG)
# index des totaux pour bincount (h+a par cellule, cappe a CAP)
_TOTAL_IDX = np.minimum(np.add.outer(_ARANGE, _ARANGE), CAP).ravel()


def _fast_grid(lh: float, la: float, rho: float = 0.0) -> np.ndarray:
    """Grille Poisson(lh)xPoisson(la) [+ Dixon-Coles sur 4 cellules]. Vectorise
    (produit externe de pmf) -> ~30x plus rapide que les boucles Python."""
    ph = poisson.pmf(_ARANGE, lh)
    pa = poisson.pmf(_ARANGE, la)
    g = np.outer(ph, pa)
    if rho:
        g[0, 0] *= 1.0 - lh * la * rho
        g[0, 1] *= 1.0 + lh * rho
        g[1, 0] *= 1.0 + la * rho
        g[1, 1] *= 1.0 - rho
        np.clip(g, 0, None, out=g)
    s = g.sum()
    if s > 0:
        g /= s
    return g


def _g1x2(g: np.ndarray) -> tuple[float, float, float]:
    """(p_home, p_draw, p_away) depuis la grille (g[h,a], home win = h>a)."""
    p_d = float(np.trace(g))
    p_h = float(np.tril(g, -1).sum())
    p_a = float(np.triu(g, 1).sum())
    return p_h, p_d, p_a

# --- Constantes des deviations simulateur (ENGINE_MODEL.md SS4) ---------------
MU_BOOST_H = 1.700 / 1.635   # ~1.0398  (buts home reels vs prices)
MU_BOOST_A = 1.254 / 1.196   # ~1.0485  (buts away reels vs prices)
RHO_SIM = -0.066             # correlation Dixon-Coles fittee du simulateur

# Calibration "cells" : boosts multiplicatifs explicites par cellule
SIM_CELL_BOOST = {
    "engine": {"2-1": 1.17, "1-2": 1.20, "2-2": 1.31, "3-3": 0.51,
               "4-2": 0.80, "2-4": 0.80, "0-2": 0.80, "2-0": 0.80, "5-1": 0.80},
    "alt":    {"2-1": 1.19, "1-2": 1.22, "2-2": 1.31, "3-3": 0.51,
               "4-2": 0.80, "2-4": 0.80, "0-2": 0.80, "2-0": 0.80, "5-1": 0.80},
}

# Poids WLS par marche (prior ; affines par le backtest). DC down-weighte car
# combinaison lineaire exacte du 1X2 (eviter le double-comptage).
DEFAULT_W = {
    "score_exact": 3.0,
    "1x2": 2.0,
    "total": 1.5,
    "ou35": 1.0,
    "gng": 1.0,
    "dc": 0.5,
    "team_home": 0.75,
    "team_away": 0.75,
}

# Bornes physiques (ENGINE_MODEL.md SS3, elargies 10 %)
LAM_BOUNDS = ((0.35, 3.6), (0.40, 3.0))


# ---------------------------------------------------------------------------
# Parsing extra_markets (robuste au mojibake : 'Total equipe ext\x82rieur', ...)
# ---------------------------------------------------------------------------

def parse_extra_markets(raw) -> dict:
    """Renvoie le dict extra_markets brut (str JSON / dict / autre -> {})."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            em = json.loads(raw)
        except Exception:
            return {}
        return em if isinstance(em, dict) else {}
    return {}


def _get_market(em: dict, *, exact: str | None = None, prefix: str | None = None) -> dict | None:
    """Recupere un sous-marche par nom exact ou par prefixe (mojibake-safe)."""
    if not isinstance(em, dict):
        return None
    if exact is not None and isinstance(em.get(exact), dict):
        return em[exact]
    if prefix is not None:
        for k, v in em.items():
            if isinstance(v, dict) and str(k).startswith(prefix):
                return v
    return None


def _to_float(v) -> float | None:
    try:
        f = float(v)
        return f if f > 1.0 else None
    except (TypeError, ValueError):
        return None


def score_exact_odds(em: dict) -> dict[str, float]:
    """{'2-1': cote, ...} avec cles normalisees ; exclut les cellules cappees."""
    cs = _get_market(em, exact="Score exact")
    out: dict[str, float] = {}
    if not isinstance(cs, dict):
        return out
    for k, v in cs.items():
        kk = str(k).strip().replace(":", "-").replace(" ", "")
        f = _to_float(v)
        if f is not None and f < CAPPED_COTE:
            out[kk] = f
    return out


def total_buts_odds(em: dict) -> dict[str, float]:
    """{'0':cote, ..., '6':cote} (6 = 6+)."""
    tb = _get_market(em, exact="Total de buts")
    out: dict[str, float] = {}
    if isinstance(tb, dict):
        for k, v in tb.items():
            f = _to_float(v)
            if f is not None and f < CAPPED_COTE:
                out[str(k).strip()] = f
    return out


# ---------------------------------------------------------------------------
# Devig par marche
# ---------------------------------------------------------------------------

def devig_market(selections: dict, target_sum: float = 1.0) -> dict[str, float]:
    """Dévig proportionnel (Sigma 1/cote), renormalise a target_sum.
    target_sum=2.0 pour Double Chance (chaque issue comptee 2x).
    Exclut les cotes cappees / invalides (<=1.0 = placeholder mort).
    Renvoie {} si < 2 selections valides (1 issue seule -> prob 1.0 absurde).
    """
    inv = {}
    for k, v in selections.items():
        f = _to_float(v)
        if f is not None and f < CAPPED_COTE:
            inv[k] = 1.0 / f
    if len(inv) < 2:
        return {}
    s = sum(inv.values())
    if s <= 0:
        return {}
    return {k: (x / s) * target_sum for k, x in inv.items()}


# ---------------------------------------------------------------------------
# Projection de la grille sur chaque marche (model side)
# ---------------------------------------------------------------------------

def total_distribution(grid: np.ndarray, cap: int = CAP) -> np.ndarray:
    """Loi du total de buts 0..cap (cap = 'cap+'). Vectorise (bincount)."""
    n = grid.shape[0]
    if n == MAXG and cap == CAP:
        return np.bincount(_TOTAL_IDX, weights=grid.ravel(), minlength=cap + 1)
    idx = np.minimum(np.add.outer(np.arange(n), np.arange(n)), cap).ravel()
    return np.bincount(idx, weights=grid.ravel(), minlength=cap + 1)


def _grid_btts_oui(grid: np.ndarray) -> float:
    n = grid.shape[0]
    return float(grid[1:n, 1:n].sum())


def model_market_probs(lam_h: float, lam_a: float, rho: float = 0.0,
                       max_goals: int = MAXG) -> dict:
    """Toutes les probas modele depuis la grille (Pass A si rho=0)."""
    grid = _fast_grid(lam_h, lam_a, rho)
    p1, pX, p2 = _g1x2(grid)
    tot = total_distribution(grid)
    over35 = float(tot[4:].sum())
    n = grid.shape[0]
    home_over35 = float(grid[4:n, :].sum())
    away_over35 = float(grid[:, 4:n].sum())
    btts = _grid_btts_oui(grid)
    return {
        "grid": grid,
        "1x2": {"1": p1, "X": pX, "2": p2},
        "dc": {"1X": p1 + pX, "X2": pX + p2, "12": p1 + p2},
        "total": {str(k): float(tot[k]) for k in range(CAP + 1)},
        "ou35": {">3.5": over35, "<3.5": 1.0 - over35},
        "gng": {"Oui": btts, "Non": 1.0 - btts},
        "team_home": {">3.5": home_over35, "<3.5": 1.0 - home_over35},
        "team_away": {">3.5": away_over35, "<3.5": 1.0 - away_over35},
    }


# ---------------------------------------------------------------------------
# Inversion 1X2 exacte (chemin principal : pricing = Poisson pur)
# ---------------------------------------------------------------------------

def exact_invert_1x2(oh: float, od: float, oa: float) -> tuple[float, float]:
    """Recupere (lam_h, lam_a) tels que la grille Poisson reproduit le 1X2 devigge.
    Unique car le pricing est Poisson independant pur.
    """
    q1, qX, q2 = devig(oh, od, oa)

    def resid(theta):
        lh, la = np.exp(theta)
        g = _fast_grid(lh, la, 0.0)
        p1, pX, p2 = _g1x2(g)
        return [p1 - q1, p2 - q2]

    # init heuristique : plus le favori est court, plus l'ecart lam_h-lam_a est grand
    lh0 = 1.3 + 0.9 * (q1 - q2)
    la0 = 1.3 - 0.9 * (q1 - q2)
    lh0 = min(max(lh0, 0.4), 3.4)
    la0 = min(max(la0, 0.4), 2.9)
    sol = least_squares(resid, np.log([lh0, la0]),
                        bounds=(np.log([0.30, 0.30]), np.log([3.8, 3.2])))
    lh, la = np.exp(sol.x)
    return float(lh), float(la)


# ---------------------------------------------------------------------------
# Inversion full-vecteur (WLS) + residu d'incoherence
# ---------------------------------------------------------------------------

@dataclass
class InversionResult:
    lam_h: float
    lam_a: float
    grid: np.ndarray          # grille pricing Pass A (Poisson pur)
    residual: float           # RMS pondere des ecarts marches non-1X2 (0 = coherent)
    per_market_gap: dict = field(default_factory=dict)
    fit_quality: str = "ok"   # "ok" | "low" | "1x2-only" | "inconsistent"
    n_markets: int = 1


def _observed_targets(em: dict, oh: float, od: float, oa: float) -> dict:
    """Probas devigees observees, par marche present."""
    obs = {"1x2": devig_market({"1": oh, "X": od, "2": oa})}
    dc = _get_market(em, exact="Double Chance")
    if isinstance(dc, dict):
        d = devig_market({k: dc.get(k) for k in ("1X", "X2", "12") if dc.get(k)}, target_sum=2.0)
        if d:
            obs["dc"] = d
    tb = total_buts_odds(em)
    if len(tb) >= 4:
        obs["total"] = devig_market(tb)
    gng = _get_market(em, exact="G/NG")
    if isinstance(gng, dict) and gng.get("Oui") and gng.get("Non"):
        obs["gng"] = devig_market({"Oui": gng["Oui"], "Non": gng["Non"]})
    ou = _get_market(em, exact="+/-")
    if isinstance(ou, dict):
        sel = {k: v for k, v in ou.items() if str(k).replace(" ", "") in (">3.5", "<3.5")}
        if len(sel) == 2:
            obs["ou35"] = devig_market({">3.5": sel.get("> 3.5", sel.get(">3.5")),
                                        "<3.5": sel.get("< 3.5", sel.get("<3.5"))})
    th = _get_market(em, prefix="Total equipe domicile")
    if isinstance(th, dict):
        sel = {k.replace(" ", ""): v for k, v in th.items()}
        if sel.get(">3.5") and sel.get("<3.5"):
            obs["team_home"] = devig_market({">3.5": sel[">3.5"], "<3.5": sel["<3.5"]})
    ta = _get_market(em, prefix="Total equipe ext")
    if isinstance(ta, dict):
        sel = {k.replace(" ", ""): v for k, v in ta.items()}
        if sel.get(">3.5") and sel.get("<3.5"):
            obs["team_away"] = devig_market({">3.5": sel[">3.5"], "<3.5": sel["<3.5"]})
    return obs


def score_exact_gap(grid: np.ndarray, score_obs: dict[str, float]) -> float | None:
    """Ecart RMS conditionnel entre la grille (1X2) et le marche Score exact
    sur les cellules pricees. None si trop peu de cellules. Signal d'incoherence
    le plus direct (le Score exact epingle la grille)."""
    if len(score_obs) < 8:
        return None
    qs = devig_market(score_obs)
    if not qs:
        return None
    cells, denom = [], 0.0
    for k in qs:
        try:
            h, a = map(int, k.split("-"))
        except ValueError:
            continue
        if h < grid.shape[0] and a < grid.shape[0]:
            cells.append((k, h, a))
            denom += grid[h, a]
    if denom <= 0 or not cells:
        return None
    return float(np.sqrt(np.mean([(grid[h, a] / denom - qs[k]) ** 2 for k, h, a in cells])))


def invert_markets(oh: float, od: float, oa: float, extra_markets=None,
                   weights: dict | None = None) -> InversionResult:
    """Inversion. lam vient du 1X2 (EXACT, le pricing est Poisson pur). Les autres
    marches ne servent PAS a corriger lam mais a mesurer le RESIDU d'incoherence
    (gap marche-vs-grille1X2) = signal de mispricing / value-jitter (edges #7-#8).
    """
    w = weights or DEFAULT_W
    em = parse_extra_markets(extra_markets)
    lam_h, lam_a = exact_invert_1x2(oh, od, oa)
    grid = _fast_grid(lam_h, lam_a, 0.0)
    m = model_market_probs(lam_h, lam_a)

    obs = _observed_targets(em, oh, od, oa)
    score_obs = score_exact_odds(em)

    # residu d'incoherence : RMS pondere des ecarts sur les marches NON-1X2
    gaps, sq, wt = {}, 0.0, 0.0
    for mk, q in obs.items():
        if mk == "1x2" or not q:
            continue
        rms = float(np.sqrt(np.mean([(m[mk][s] - q[s]) ** 2 for s in q])))
        gaps[mk] = round(rms, 4)
        sq += w.get(mk, 1.0) * rms ** 2
        wt += w.get(mk, 1.0)
    se_gap = score_exact_gap(grid, score_obs)
    if se_gap is not None:
        gaps["score_exact"] = round(se_gap, 4)
        sq += w.get("score_exact", 3.0) * se_gap ** 2
        wt += w.get("score_exact", 3.0)

    residual = float(np.sqrt(sq / wt)) if wt > 0 else 0.0
    n_markets = 1 + len([k for k in gaps])
    if n_markets <= 1:
        fq = "1x2-only"
    elif residual > 0.06:
        fq = "inconsistent"   # marches en desaccord = mispricing potentiel
    else:
        fq = "ok"

    return InversionResult(lam_h=lam_h, lam_a=lam_a, grid=grid, residual=round(residual, 4),
                           per_market_gap=gaps, fit_quality=fq, n_markets=n_markets)


# ---------------------------------------------------------------------------
# Pass B : appliquer les deviations connues du simulateur
# ---------------------------------------------------------------------------

def apply_sim_deviations(lam_h: float, lam_a: float, mode: str = "dc",
                         max_goals: int = MAXG) -> np.ndarray:
    """Grille du RESULTAT realise (vs grille de pricing).
    mode='dc'    : rescale mu + Dixon-Coles rho (principciel, produit 2-1/1-2 eleves).
    mode='cells' : rescale mu + boosts multiplicatifs explicites par cellule.
    Les deux sont equivalents en intention -> le backtest tranche.
    """
    lh, la = lam_h * MU_BOOST_H, lam_a * MU_BOOST_A
    if mode == "cells":
        g = _fast_grid(lh, la, 0.0)
        for k, fct in SIM_CELL_BOOST["engine"].items():
            h, a = map(int, k.split("-"))
            if h < g.shape[0] and a < g.shape[0]:
                g[h, a] *= fct
        s = g.sum()
        if s > 0:
            g /= s
        return g
    # mode 'dc' (defaut)
    return _fast_grid(lh, la, RHO_SIM)


# ---------------------------------------------------------------------------
# Grille -> predictions
# ---------------------------------------------------------------------------

def grid_predictions(grid: np.ndarray, top_k: int = 3) -> dict:
    """Total le + probable, top-k scores, BTTS, score modal."""
    tot = total_distribution(grid)
    most_likely_total = int(np.argmax(tot))
    return {
        "most_likely_total": most_likely_total,
        "total_dist": {str(k): round(float(tot[k]), 4) for k in range(CAP + 1)},
        "top_scores": [(s, round(p, 4)) for s, p in grid_top_k_scores(grid, top_k)],
        "modal_score": grid_modal_score(grid),
        "btts_oui": round(_grid_btts_oui(grid), 4),
    }
