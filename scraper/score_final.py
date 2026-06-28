"""Moteur FINAL (v2 'fin') — prédicteur de score exact data-driven, corrigé pour
la diversité. Principe : table 2D (force favori × λtot) -> liste des scores
empiriquement fréquents de la cellule ; on tranche entre eux avec le λ CONTINU
(grille simulateur). Résultat : plus varié (11 scores vs 7) ET plus précis
(11.79% vs 11.66% OOS) que le simple modal. Capte les biais du RNG sans
sur-émettre 1-1/2-1.
"""
from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path

from .market_inversion import exact_invert_1x2, apply_sim_deviations

FAV_EDGES = [1.0, 1.3, 1.5, 1.8, 2.2, 1e9]
TOT_EDGES = [0.0, 2.1, 2.5, 2.9, 3.3, 1e9]
TOPK = 4  # nb de candidats gardés par cellule
DEFAULT_TABLE = Path(__file__).resolve().parents[1] / "exports" / "score_final_table.json"


def _band(v: float, edges: list[float]) -> int:
    for i in range(len(edges) - 1):
        if edges[i] <= v < edges[i + 1]:
            return i
    return len(edges) - 2


def fit_final_table(rows, topk: int = TOPK):
    """rows : iterable de (oh,od,oa,sa,sb). Renvoie (table, global_candidats).
    table : {"fi-ti": ["fg-dg", ...]} = top-k scores orientés-favori de la cellule."""
    cells = defaultdict(Counter)
    allc = Counter()
    for oh, od, oa, sa, sb in rows:
        if not (oh > 1 and od > 1 and oa > 1):
            continue
        lh, la = exact_invert_1x2(oh, od, oa)
        fav_home = oh < oa
        favc = oh if fav_home else oa
        fg = int(sa if fav_home else sb)
        dg = int(sb if fav_home else sa)
        fs = f"{fg}-{dg}"
        key = (_band(favc, FAV_EDGES), _band(lh + la, TOT_EDGES))
        cells[key][fs] += 1
        allc[fs] += 1
    table = {f"{fi}-{ti}": [s for s, _ in c.most_common(topk)] for (fi, ti), c in cells.items()}
    glob = [s for s, _ in allc.most_common(topk)] or ["1-1"]
    return table, glob


def save_table(table: dict, glob, path=DEFAULT_TABLE):
    Path(path).write_text(json.dumps({
        "table": table, "global": glob,
        "fav_edges": FAV_EDGES, "tot_edges": TOT_EDGES,
    }, indent=1), encoding="utf-8")


def load_table(path=DEFAULT_TABLE):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    glob = d.get("global", ["1-1"])
    if isinstance(glob, str):   # compat ancienne table (modal unique)
        glob = [glob]
    return d["table"], glob


def _grid_prob(g, fs: str, fav_home: bool) -> float:
    a, b = map(int, fs.split("-"))
    h, aw = (a, b) if fav_home else (b, a)
    return float(g[h, aw]) if h < g.shape[0] and aw < g.shape[0] else 0.0


BUCKETS_PATH = Path(__file__).resolve().parents[1] / "exports" / "score_buckets.json"


def load_buckets(path=BUCKETS_PATH):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return d["buckets"], d.get("global", {"1-1": 1.0})


def ensemble_top3(oh: float, od: float, oa: float, buckets: dict, glob: dict, top_n: int = 3) -> dict:
    """Meilleur moteur Top-3 (31.3% OOS) : moyenne 50/50 de la grille simulateur
    (orientée favori, normalisée) et de la distribution empirique du bucket.
    Renvoie {top (home-away), top_fav, lam_tot}."""
    lh, la = exact_invert_1x2(oh, od, oa)
    lt = lh + la
    fav_home = oh < oa
    favc = oh if fav_home else oa
    g = apply_sim_deviations(lh, la, "cells")
    if not fav_home:
        g = g.T
    s = g.sum()
    grid = {f"{i}-{j}": float(g[i, j]) / s for i in range(g.shape[0]) for j in range(g.shape[0])} if s > 0 else {}
    key = f"{_band(favc, FAV_EDGES)}-{_band(lt, TOT_EDGES)}"
    emp = buckets.get(key, glob)
    agg = {}
    for sc in set(grid) | set(emp):
        agg[sc] = 0.5 * grid.get(sc, 0.0) + 0.5 * emp.get(sc, 0.0)
    ranked = sorted(agg, key=lambda sc: -agg[sc])
    def orient(fs):
        a, b = fs.split("-"); return f"{a}-{b}" if fav_home else f"{b}-{a}"
    top = [(orient(fs), round(agg[fs] * 100, 1)) for fs in ranked[:top_n]]
    return {"top": top, "top_fav": ranked[:top_n], "lam_tot": round(lt, 2)}


def predict_final(oh: float, od: float, oa: float, table: dict, glob=("1-1",), top_n: int = 1) -> dict:
    """Renvoie {score, top (liste orientée home-away), fav_oriented, lam_tot, cell}.
    Tranche les candidats de la cellule via la grille λ continue (plus de variété)."""
    lh, la = exact_invert_1x2(oh, od, oa)
    lt = lh + la
    fav_home = oh < oa
    favc = oh if fav_home else oa
    fi, ti = _band(favc, FAV_EDGES), _band(lt, TOT_EDGES)
    cands = table.get(f"{fi}-{ti}") or (list(glob) if not isinstance(glob, str) else [glob])
    g = apply_sim_deviations(lh, la, "cells")
    ranked = sorted(cands, key=lambda fs: _grid_prob(g, fs, fav_home), reverse=True)
    def orient(fs):
        a, b = fs.split("-"); return f"{a}-{b}" if fav_home else f"{b}-{a}"
    top = [orient(fs) for fs in ranked[:max(top_n, 1)]]
    return {"score": top[0], "top": top, "fav_oriented": ranked[0],
            "lam_tot": round(lt, 2), "cell": (fi, ti)}
