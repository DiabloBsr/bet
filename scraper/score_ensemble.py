"""Ensemble de score exact V5 + V2 + V6 — NOUVEAU module, ne touche aucun existant.

Backtest OOS (_score_ensemble.py, n=1500) : l'ensemble V5+V2+V6 donne le meilleur
Top1 (12.7%) et un bon Top3 (32.7%), plus ROBUSTE que chaque modèle seul. Il ne
dépasse pas le plafond empirique (~13% Top1 / ~33% Top3) — aucun ne le peut, le
RNG est fort et le book a le vrai modèle — mais c'est l'estimation la plus stable
du score le plus probable.

Cette fonction est PURE : elle prend les 3 distributions déjà calculées par
l'appelant (V5 top5_scores_enriched, V2 predict, V6 predict_score_v6) et les
combine. Poids par défaut égaux (le backtest ne distingue pas mieux dans le bruit).

Usage:
    from scraper.score_ensemble import ensemble_top_scores
    res = ensemble_top_scores(d5, d2, d6, top_n=3)   # d* = dict {score: prob}
"""
from __future__ import annotations


def _norm(d):
    if not d:
        return {}
    t = sum(d.values())
    return {k: v / t for k, v in d.items() if v > 0} if t > 0 else {}


def ensemble_top_scores(d_v5: dict | None, d_v2: dict | None, d_v6: dict | None,
                        weights=(1.0, 1.0, 1.0), top_n: int = 3) -> dict:
    """Combine les distributions de score V5/V2/V6 (moyenne pondérée, normalisées).
    Ignore proprement les composantes absentes. Retourne top-N + le score modal."""
    parts = [(_norm(d_v5), weights[0]), (_norm(d_v2), weights[1]), (_norm(d_v6), weights[2])]
    parts = [(d, w) for d, w in parts if d]
    if not parts:
        return {"top": [], "modal": None, "n_models": 0}
    wsum = sum(w for _, w in parts)
    keys = set().union(*[set(d) for d, _ in parts])
    blend = {k: sum(d.get(k, 0) * w for d, w in parts) / wsum for k in keys}
    top = sorted(blend.items(), key=lambda x: -x[1])[:top_n]
    return {"top": [(s, round(p, 4)) for s, p in top],
            "modal": top[0][0] if top else None,
            "n_models": len(parts)}


def ensemble_from_raw(top5_v5=None, v2_top=None, v6_res=None, top_n: int = 3) -> dict:
    """Convenience : accepte les formats bruts des 3 moteurs.
    top5_v5  : liste [(score, prob), ...] (predict_match_v5 -> top5_scores_enriched)
    v2_top   : liste [(score, prob, breakdown), ...] (ScorePredictorV2.predict)
    v6_res   : dict de predict_score_v6 (clé 'top' = [(score, prob), ...])"""
    d5 = {s: p for s, p in (top5_v5 or [])}
    d2 = {t[0]: t[1] for t in (v2_top or [])}
    d6 = dict((v6_res or {}).get("top", []))
    return ensemble_top_scores(d5, d2, d6, top_n=top_n)
