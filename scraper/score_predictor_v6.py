"""Score Predictor v6 — NOUVEAU module, ne touche ni V5 ni V2.

Issu du backtest (_score_v6.py) : sur le score exact, le MEILLEUR prédicteur
hors-échantillon est l'échelle de cotes score OFFERTE devigée (Top1 11.7% /
Top3 31.1%, au plafond empirique). Le book price la vraie distribution déviée
du RNG (variance réduite + biais BTTS) ; notre grille sim (10.8%/29.4%) ne fait
que l'approcher, et la mélanger n'améliore pas. v6 = book-primary + sim en repli.

Usage:
    from scraper.score_predictor_v6 import predict_score_v6
    res = predict_score_v6(oh, od, oa, extra_markets, top_n=3)
    # res = {"top": [(score, prob), ...], "source": "book"|"sim"|"blend"}
"""
from __future__ import annotations
from .market_inversion import (
    invert_markets, apply_sim_deviations, parse_extra_markets, score_exact_odds,
)

# repli : si moins de N cellules book exploitables, on bascule sur la grille sim
_MIN_BOOK_CELLS = 6
# poids du léger lissage sim sur le book (backtest: 0 = pur book optimal ; on
# garde un epsilon pour départager les cellules book absentes, pas pour "corriger")
_SIM_SMOOTH = 0.0


def _book_dist(em) -> dict:
    se = score_exact_odds(em) if em is not None else {}
    d = {s: 1.0 / c for s, c in se.items() if c and c > 1 and c < 99.99}
    tot = sum(d.values())
    return {k: v / tot for k, v in d.items()} if tot > 0 else {}


def _sim_dist(oh, od, oa, em) -> dict:
    inv = invert_markets(oh, od, oa, em)
    g = apply_sim_deviations(inv.lam_h, inv.lam_a, "cells")
    d = {}
    for h in range(g.shape[0]):
        for a in range(g.shape[1]):
            v = float(g[h, a])
            if v > 0:
                d[f"{h}-{a}"] = v
    tot = sum(d.values())
    return {k: v / tot for k, v in d.items()} if tot > 0 else {}


def predict_score_v6(oh: float, od: float, oa: float, extra_markets=None, top_n: int = 3) -> dict:
    """Top-N scores exacts. Primaire = cotes offertes devigées ; repli = grille sim."""
    em = parse_extra_markets(extra_markets) if extra_markets is not None else {}
    book = _book_dist(em)
    sim = _sim_dist(oh, od, oa, extra_markets)
    if len(book) >= _MIN_BOOK_CELLS:
        if _SIM_SMOOTH > 0 and sim:
            keys = set(book) | set(sim)
            dist = {k: (1 - _SIM_SMOOTH) * book.get(k, 0) + _SIM_SMOOTH * sim.get(k, 0) for k in keys}
            source = "blend"
        else:
            dist, source = book, "book"
    else:
        dist, source = sim, "sim"
    top = sorted(dist.items(), key=lambda x: -x[1])[:top_n]
    return {"top": [(s, round(p, 4)) for s, p in top], "source": source,
            "modal": top[0][0] if top else None}


if __name__ == "__main__":
    # démo
    for oh, od, oa in [(1.20, 6.8, 12.5), (2.4, 3.4, 2.9), (5.1, 4.0, 1.6)]:
        r = predict_score_v6(oh, od, oa, None, top_n=3)
        print(f"{oh}/{od}/{oa} -> {r['modal']} | top {r['top']} (src={r['source']})")
