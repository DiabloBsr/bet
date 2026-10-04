"""Prédicteur TRIO — V2 + V5 + arbitre MARCHÉ (poids égaux), importable + CLI.

Trois votes indépendants, moyennés à poids égaux (le marché tranche les
désaccords V2/V5 sans favoritisme) :
  • V2  : team-strength Poisson+DC + blend Score-exact
  • V5  : team-strength + HT/FT
  • MARCHÉ : cotes Score-exact offertes devigées (score_predictor_v6 core) — arbitre neutre

N'utilise QUE des moteurs honnêtes (au plafond). PAS V6/V7/V8/V10 (faux edges réfutés OOS).
CLI : ./.venv/Scripts/python.exe scripts/predict_trio.py [HH:MM]  (heure Mada)
"""
from __future__ import annotations
import sys, json, re
from datetime import datetime, timedelta, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from typing import NamedTuple
from sqlalchemy import create_engine
from scraper.config import load_settings
from scraper.predictor_v2 import (fit_model_v2, predict_match_v2, blended_score_grid,
                                  grid_top_k_scores, market_score_grid)
from scraper.predictor_v5 import fit_model_v5, predict_match_v5
from scraper.market_inversion import exact_invert_1x2, apply_sim_deviations

MADA = timezone(timedelta(hours=3))
K_GRID = 9   # taille de la grille Poisson (partagee predict_own / marches_probas)
# PLAFOND STRUCTUREL du moteur Bet261 : sur 208 762 resultats, le total n'a
# JAMAIS depasse 6 buts. Un 6-0 existe, un 4-3 non. Sans ce plafond, la grille
# de Poisson disperse ~2.2 % de probabilite sur des scores impossibles, et cette
# masse manque la ou elle devrait etre -- les totaux de 3 a 5, precisement ceux
# que le modele sous-estimait (+2.8, +3.2 et +2.3 points d'ecart mesures).
TOTAL_MAX = 6

# Facteur d'echelle des lambdas, AJUSTE sur la 1re moitie chronologique par
# maximum de vraisemblance du score exact observe, verifie sur la seconde.
# Sans lui, le modele annoncait 2.59 buts par match pour 2.72 reels : trop de
# masse sur les totaux de 1 et 2, pas assez sur 3 a 5 -- exactement les scores
# (3-1, 2-2, 3-2, 4-1...) qui ne sortaient jamais en tete. La log-vraisemblance
# est plate entre 1.05 et 1.08 ; on retient la valeur qui aligne la moyenne
# predite sur la moyenne observee de TRAIN.
LAM_SCALE = 1.06
LG = "InstantLeague-8035"

_CALIB_BY_LG: dict = {}      # ligue -> matrice 7x7
_CALIB = None                # table de la ligue de référence (compat ascendante)
_CALIB_REF = LG
try:
    _cp = Path(__file__).resolve().parents[1] / "data" / "vfoot_ml" / "score_calibration.json"
    if _cp.exists():
        _raw = json.loads(_cp.read_text(encoding="utf-8"))
        _CALIB_REF = _raw.get("reference_league", LG)
        _CALIB_BY_LG = {k: np.asarray(v, float)
                        for k, v in (_raw.get("per_league") or {}).items()}
        if not _CALIB_BY_LG and _raw.get("correction"):     # ancien format mono-ligue
            _CALIB_BY_LG = {_CALIB_REF: np.asarray(_raw["correction"], float)}
        _CALIB = _CALIB_BY_LG.get(_CALIB_REF)
except Exception:
    _CALIB_BY_LG, _CALIB = {}, None


def _calib_for(lg: str = None):
    """Table de correction PROPRE à cette ligue, ou None si elle n'en a pas.

    Une table par ligue est indispensable : les constantes du simulateur
    (MU_BOOST, RHO_SIM, SIM_CELL_BOOST) sont ajustées sur l'anglaise, et les
    réutiliser telles quelles ailleurs dé-calibre — mesuré sur CAN (λ=1.49 vs
    2.83), l'écart max passait de 3.5pp à 8.0pp. Une ligue sans table mesurée
    n'est PAS corrigée (mieux vaut non corrigé que corrigé avec la mauvaise)."""
    return _CALIB_BY_LG.get(lg if lg is not None else LG)


def _apply_calib(d: dict, lg: str = None) -> dict:
    """Applique la table de calibration 7x7 de la ligue à {score: p}, renormalise."""
    cal = _calib_for(lg)
    if cal is None or not d:
        return d
    out = {}
    for sc, p in d.items():
        try:
            h, a = map(int, sc.split("-"))
            f = float(cal[h][a]) if (0 <= h < 7 and 0 <= a < 7) else 1.0
        except Exception:
            f = 1.0
        out[sc] = p * f
    tt = sum(out.values()) or 1.0
    return {k: v / tt for k, v in out.items()}


def load_hist(engine):
    return pd.read_sql(f"""SELECT e.team_a,e.team_b,o.odds_home,o.odds_draw,o.odds_away,
        r.score_a,r.score_b,r.ht_score_a,r.ht_score_b FROM events e
        JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
        JOIN results r ON r.event_id=e.id
        WHERE r.ht_score_a IS NOT NULL AND e.competition='{LG}'""", engine)


def fit(engine):
    """Fit V5 + V2. Retourne (m5, v2model, n)."""
    hist = load_hist(engine)
    m5 = fit_model_v5(hist, ht_history=hist.copy(), engine=engine, form_alpha=0.0)
    v2 = fit_model_v2(hist)
    return m5, v2, len(hist)


def _sem(extra_markets):
    if isinstance(extra_markets, str):
        try: extra_markets = json.loads(extra_markets)
        except Exception: return None
    return extra_markets.get("Score exact") if isinstance(extra_markets, dict) else None


def _over25_calib(oh, od, oa, lg: str = None):
    try:
        lh, la = exact_invert_1x2(oh, od, oa)
        g = np.asarray(apply_sim_deviations(lh, la, "cells"), float)[:7, :7]; g /= g.sum()
        cal = _calib_for(lg)
        if cal is not None:
            g = g * cal; g /= g.sum()
        # plafond dur du RNG : total <= 6 buts (0/58083 dépassement) -> cases
        # impossibles zérotées, probas renormalisées (justesse exacte des totaux)
        for h in range(7):
            for a in range(7):
                if h + a > 6:
                    g[h, a] = 0.0
        g = g / (g.sum() or 1.0)
        return round(100 * float(sum(g[h, a] for h in range(7) for a in range(7) if h + a > 2.5)), 1)
    except Exception:
        return None


# ================= TABLEAU COMPLET DES MARCHÉS =================
# Probabilités = cotes offertes DÉVIGÉES par marché. Prouvé (17 campagnes) :
# le book est calibré <2pp partout et cohérent (grille unique) -> ces probas
# sont les meilleures estimations disponibles, marché par marché.
_CANON = ["Mi-tps 1X2", "Mi-tps DC", "Mi-tps CS", "Double Chance", "Score exact", "+/-",
          "HT/FT", "Total de buts", "G/NG", "Les deux équipes marquent / 1ère mi temps",
          "1X2 & Total", "1X2 & G/NG", "Pair/Impair", "Minute du premier but", "FTTS",
          "Multi-Buts", "2ème mi-tps - CS"]


def _canon(k: str) -> str:
    kn = k.replace("\x82", "é").replace("\xe9", "é")
    if kn.startswith(("Total equipe", "Total équipe")):
        return "Total equipe domicile" if "dom" in kn else "Total equipe extérieur"
    if kn.startswith(("G/NG equipe", "G/NG équipe")):
        return "G/NG equipe domicile" if "dom" in kn else "G/NG equipe extérieur"
    for c in _CANON:
        if kn[:10] == c[:10]:
            return c
    return kn


def _devig(valid: dict) -> dict:
    """{sel: proba} — dévig si partition complète cotée, sinon 1/cote brute."""
    tinv = sum(1/o for o in valid.values())
    if tinv >= 0.95:
        return {s: (1/o)/tinv for s, o in valid.items()}
    return {s: 1/o for s, o in valid.items()}


def market_board(extra_markets, oh, od, oa) -> dict:
    """TOUS les marchés d'un match -> {marché: [(sélection, proba, cote), ...]} trié par proba."""
    if isinstance(extra_markets, str):
        try: extra_markets = json.loads(extra_markets)
        except Exception: extra_markets = {}
    mk = {"1X2": {"1": float(oh), "X": float(od), "2": float(oa)}}
    for k, v in (extra_markets or {}).items():
        if isinstance(v, dict):
            mk[_canon(k)] = v
    valid_of = lambda sels: {s: float(o) for s, o in sels.items()
                             if isinstance(o, (int, float)) and 1 < o < 99.99}
    p1x2 = _devig(valid_of(mk["1X2"]))
    pht = _devig(valid_of(mk["Mi-tps 1X2"])) if "Mi-tps 1X2" in mk else {}
    ptot = _devig(valid_of(mk["Total de buts"])) if "Total de buts" in mk else {}
    board = {}
    for mkt, sels in mk.items():
        valid = valid_of(sels)
        if not valid:
            continue
        if mkt == "Double Chance":              # dérivé du 1X2 (sélections chevauchantes)
            pr = {"1X": p1x2.get("1", 0)+p1x2.get("X", 0), "12": p1x2.get("1", 0)+p1x2.get("2", 0),
                  "X2": p1x2.get("X", 0)+p1x2.get("2", 0)}
        elif mkt == "Mi-tps DC" and pht:
            pr = {"1X": pht.get("1", 0)+pht.get("X", 0), "12": pht.get("1", 0)+pht.get("2", 0),
                  "X2": pht.get("X", 0)+pht.get("2", 0)}
        elif mkt == "Multi-Buts" and ptot:      # ranges chevauchants, dérivés du total
            g = lambda *ks: sum(ptot.get(str(k), 0) for k in ks)
            pr = {}
            for s in valid:
                if "0, 1 ou 2" in s: pr[s] = g(0, 1, 2)
                elif "1, 2 ou 3" in s: pr[s] = g(1, 2, 3)
                elif "2, 3 ou 4" in s: pr[s] = g(2, 3, 4)
                else: pr[s] = g(5, 6)
        else:
            pr = _devig(valid)
        board[mkt] = sorted(((s, round(pr.get(s, 0), 4), o) for s, o in valid.items()),
                            key=lambda r: -r[1])
    return board


# marchés "sûrs" pour le cadran de précision (du plus fin au plus large)
CONF_MARKETS = ["1X2", "Mi-tps 1X2", "+/-", "G/NG", "Double Chance", "Mi-tps DC",
                "Total de buts", "Multi-Buts"]


def pick_for_confidence(board: dict, target: float):
    """Meilleur pari (cote la plus haute) dont la proba >= target, tous marchés sûrs
    confondus. Rend (marché, sélection, proba, cote) ou None si aucun n'atteint target."""
    cands = [(mkt, s, p, o) for mkt in CONF_MARKETS
             for (s, p, o) in (board.get(mkt) or []) if p >= target]
    if not cands:
        return None
    return max(cands, key=lambda r: r[3])          # cote max qui tient la confiance


def top_confidence_pick(board: dict):
    """Le pari le PLUS PROBABLE du match (tous marchés sûrs) -> (marché, sél, proba, cote)."""
    cands = [(mkt, s, p, o) for mkt in CONF_MARKETS for (s, p, o) in (board.get(mkt) or [])]
    return max(cands, key=lambda r: r[2]) if cands else None


def upcoming_window(engine, start_local: str, end_local: str, target: float = 0.75,
                    min_odds: float = 1.08, horizon_min: int = 240) -> list:
    """Matchs à venir des 9 LIGUES dont l'heure Mada (HH:MM) tombe dans [start,end].
    Pour chaque match : le pari qui PAIE LE MIEUX tout en restant >= target de confiance
    (et cote >= min_odds pour écarter les 1.01 sans valeur). Trié par proba décroissante.
    Retour : dict par match {match, tag, local, board, best=(marché,sél,proba,cote)}."""
    now = datetime.now(timezone.utc)
    up = pd.read_sql(f"""SELECT e.competition c, e.team_a, e.team_b, e.expected_start,
        o.odds_home oh, o.odds_draw od, o.odds_away oa, o.extra_markets xm, e.id ev FROM events e
        JOIN odds_snapshots o ON o.id=(SELECT MAX(id) FROM odds_snapshots WHERE event_id=e.id)
        LEFT JOIN results r ON r.event_id=e.id
        WHERE r.id IS NULL AND e.expected_start IS NOT NULL
          AND e.competition LIKE 'InstantLeague-%'""", engine)
    if not len(up):
        return []
    up["es"] = pd.to_datetime(up.expected_start, utc=True)
    up = up[(up.es > now - pd.Timedelta(minutes=3)) & (up.es < now + pd.Timedelta(minutes=horizon_min))]
    up["local"] = up.es.dt.tz_convert(MADA).dt.strftime("%H:%M")
    up = up[(up.local >= start_local) & (up.local <= end_local)]
    up = up.sort_values(["es", "ev"]).drop_duplicates(["c", "team_a", "team_b", "expected_start"])
    out = []
    for r in up.itertuples():
        if float(r.oh) <= 1 or float(r.oa) <= 1:
            continue
        board = market_board(r.xm, r.oh, r.od, r.oa)
        # meilleur pari (cote max) avec proba >= target ET cote >= min_odds
        cands = [(mkt, s, p, o) for mkt in CONF_MARKETS
                 for (s, p, o) in (board.get(mkt) or []) if p >= target and o >= min_odds]
        if not cands:
            continue
        best = max(cands, key=lambda x: x[3])       # meilleur payout qui tient la confiance
        out.append({"match": f"{r.team_a} v {r.team_b}", "tag": LEAGUE_TAGS.get(r.c, r.c[-4:]),
                    "local": r.local, "board": board, "best": best})
    out.sort(key=lambda m: -m["best"][2])           # meilleure proba d'abord
    return out


# marchés scannés en mode "cote cible" (large : on filtre par cote, pas par type)
ODDS_SCAN_MARKETS = ["1X2", "Double Chance", "+/-", "Total de buts", "Multi-Buts", "G/NG",
                     "Mi-tps 1X2", "Mi-tps DC", "HT/FT", "1X2 & Total", "1X2 & G/NG",
                     "Total equipe domicile", "Total equipe extérieur"]


def team_strength(engine, lg: str = LG, leagues: list | None = None) -> dict:
    """Profil de force par équipe (historique) : buts marqués/encaissés + % victoire
    domicile/extérieur. Sert de contexte 'équipe forte ou pas'. `leagues` = liste (union)."""
    d = pd.read_sql(f"""SELECT e.team_a, e.team_b, r.score_a, r.score_b FROM events e
        JOIN results r ON r.event_id=e.id
        WHERE r.score_a IS NOT NULL AND {_league_where(lg, leagues)}""", engine)
    prof = {}
    for r in d.itertuples():
        h = prof.setdefault(r.team_a, {"gf": 0, "ga": 0, "n": 0, "w": 0, "wh": 0, "nh": 0})
        a = prof.setdefault(r.team_b, {"gf": 0, "ga": 0, "n": 0, "w": 0, "wh": 0, "nh": 0})
        h["gf"] += r.score_a; h["ga"] += r.score_b; h["n"] += 1; h["nh"] += 1
        a["gf"] += r.score_b; a["ga"] += r.score_a; a["n"] += 1
        h["w"] += int(r.score_a > r.score_b); h["wh"] += int(r.score_a > r.score_b)
        a["w"] += int(r.score_b > r.score_a)
    out = {}
    for t, v in prof.items():
        if v["n"] >= 20:
            out[t] = {"gf": v["gf"]/v["n"], "ga": v["ga"]/v["n"], "winrate": v["w"]/v["n"]}
    return out


def _league_where(lg: str, leagues: list | None) -> str:
    """Clause WHERE de filtrage ligue : liste (IN) si fournie, sinon la ligue unique `lg`."""
    if leagues:
        vals = ",".join("'" + str(x).replace("'", "''") + "'" for x in leagues)
        return f"e.competition IN ({vals})"
    return f"e.competition='{lg}'"


def nodraw_streaks(engine, lg: str = LG, leagues: list | None = None) -> dict:
    """Par équipe : nb de matchs depuis son dernier nul (sécheresse). CONTEXTE
    seulement — ne prédit RIEN (le 'dû' est prouvé faux). `leagues` = liste (union)."""
    d = pd.read_sql(f"""SELECT e.team_a, e.team_b, r.score_a, r.score_b, e.expected_start
        FROM events e JOIN results r ON r.event_id=e.id
        WHERE r.score_a IS NOT NULL AND {_league_where(lg, leagues)} ORDER BY e.expected_start""", engine)
    since = {}
    for r in d.itertuples():
        draw = r.score_a == r.score_b
        for t in (r.team_a, r.team_b):
            since[t] = 0 if draw else since.get(t, 0) + 1
    return since


def find_targets(engine, team: str | None = None, side: str = "any",
                 lo: float = 2.0, hi: float = 3.5, window_min: int = 300,
                 leagues: list | None = None, draw_ctx: dict | None = None,
                 start_local: str | None = None, end_local: str | None = None) -> list:
    """Matchs à venir (9 ligues) où l'équipe visée (ou toute équipe) joue au côté demandé
    avec une cote de victoire dans [lo,hi]. Rend match, équipe, cote, PROBA de victoire
    (implicite dévigée = honnête), adversaire. Trié par proba décroissante (le + probable
    dans la fourchette de cote = ton 'gros coup probable').
    Si start_local/end_local (HH:MM Mada) sont donnés, ne garde que les matchs dont
    l'heure Mada tombe dans [start,end] (sinon : fenêtre glissante depuis maintenant)."""
    now = datetime.now(timezone.utc)
    up = pd.read_sql(f"""SELECT e.competition c, e.team_a, e.team_b, e.expected_start,
        o.odds_home oh, o.odds_draw od, o.odds_away oa FROM events e
        JOIN odds_snapshots o ON o.id=(SELECT MAX(id) FROM odds_snapshots WHERE event_id=e.id)
        LEFT JOIN results r ON r.event_id=e.id
        WHERE r.id IS NULL AND e.expected_start IS NOT NULL
          AND e.competition LIKE 'InstantLeague-%'""", engine)
    if not len(up):
        return []
    up["es"] = pd.to_datetime(up.expected_start, utc=True)
    interval = bool(start_local and end_local)
    horizon = 1440 if interval else window_min       # intervalle -> cherche sur 24 h de matchs publiés
    up = up[(up.es > now - pd.Timedelta(minutes=3)) & (up.es < now + pd.Timedelta(minutes=horizon))]
    if interval:
        up = up.copy()
        up["local"] = up.es.dt.tz_convert(MADA).dt.strftime("%H:%M")
        up = up[(up.local >= start_local) & (up.local <= end_local)]
    if leagues:
        up = up[up.c.isin(leagues)]
    up = up.sort_values("es").drop_duplicates(["c", "team_a", "team_b", "expected_start"])
    tl = (team or "").lower().strip()
    out = []
    for r in up.itertuples():
        oh, od, oa = float(r.oh), float(r.od), float(r.oa)
        if oh <= 1 or oa <= 1:
            continue
        inv = 1/oh + 1/od + 1/oa
        cands = []
        if side in ("any", "home"):
            cands.append((r.team_a, "domicile", oh, (1/oh)/inv, r.team_b))
        if side in ("any", "away"):
            cands.append((r.team_b, "extérieur", oa, (1/oa)/inv, r.team_a))
        if side in ("any", "draw", "nul"):
            dry = ""
            if draw_ctx is not None:
                da, db = draw_ctx.get(r.team_a, 0), draw_ctx.get(r.team_b, 0)
                dry = f"sécheresse nuls: {r.team_a[:12]} {da}, {r.team_b[:12]} {db}"
            cands.append((f"Nul ({r.team_a} v {r.team_b})", "nul", od, (1/od)/inv,
                          f"{r.team_a} v {r.team_b}", dry))
        for cand in cands:
            tm, sd, o, p, opp = cand[:5]
            extra = cand[5] if len(cand) > 5 else ""
            if lo <= o <= hi and (not tl or tl in tm.lower() or (sd == "nul" and tl in opp.lower())):
                out.append({"comp": r.c, "tag": LEAGUE_TAGS.get(r.c, r.c[-4:]),
                            "local": r.es.tz_convert(MADA).strftime("%H:%M"),
                            "team": tm, "side": sd, "opp": opp, "odds": o, "winprob": p,
                            "ctx": extra})
    out.sort(key=lambda x: -x["winprob"])
    return out


CAN_LG = "InstantLeague-8060"


def _can_pick_outsiders(up, lo, hi, p_min, recent):
    """Construit la liste des outsiders à partir d'un DataFrame de matchs CAN."""
    out = []
    for r in up.itertuples():
        oh, od, oa = float(r.oh), float(r.od), float(r.oa)
        if oh <= 1 or oa <= 1 or od <= 1:
            continue
        if oh >= oa:
            side, team, opp, o_out = "domicile", r.team_a, r.team_b, oh
        else:
            side, team, opp, o_out = "extérieur", r.team_b, r.team_a, oa
        if not (lo <= o_out <= hi):
            continue
        inv = 1/oh + 1/od + 1/oa
        p = (1/o_out) / inv                 # proba dévigée (honnête, ~vraie sur CAN calibrée)
        if p < p_min:
            continue
        out.append({"match": f"{team} vs {opp}", "local": r.es.tz_convert(MADA).strftime("%H:%M"),
                    "team": team, "side": side, "opp": opp, "odds": o_out, "p": p, "recent": recent})
    out.sort(key=lambda x: (x["recent"], -x["p"]))
    return out


def can_outsiders(engine, lo: float = 5.0, hi: float = 15.0, minutes: int = 120,
                  p_min: float = 0.0, start_local: str | None = None,
                  end_local: str | None = None) -> list:
    """Matchs CAN (8060) : l'OUTSIDER (côté à cote la plus haute) filtré sur [lo,hi] et
    proba dévigée >= p_min, trié par CHANCE RÉELLE décroissante. D'abord les matchs À VENIR ;
    si aucun n'est capté (scraper en ligne throttlé), REPLI sur les derniers matchs CAN réels
    (flag recent=True) pour rester utile. Rappel : outsider = pari le MOINS MAUVAIS de CAN
    (ROI ~-2.4% vs favori -6%), mais -EV — aucun edge confirmé."""
    now = datetime.now(timezone.utc)
    up = pd.read_sql(f"""SELECT e.team_a, e.team_b, e.expected_start,
        o.odds_home oh, o.odds_draw od, o.odds_away oa FROM events e
        JOIN odds_snapshots o ON o.id=(SELECT MAX(id) FROM odds_snapshots WHERE event_id=e.id)
        LEFT JOIN results r ON r.event_id=e.id
        WHERE r.id IS NULL AND e.expected_start IS NOT NULL
          AND e.competition='{CAN_LG}'""", engine)
    interval = bool(start_local and end_local)
    if len(up):
        up["es"] = pd.to_datetime(up.expected_start, utc=True)
        horizon = 1440 if interval else minutes
        up = up[(up.es > now - pd.Timedelta(minutes=3)) & (up.es < now + pd.Timedelta(minutes=horizon))]
        if interval:
            up = up.copy()
            up["local"] = up.es.dt.tz_convert(MADA).dt.strftime("%H:%M")
            up = up[(up.local >= start_local) & (up.local <= end_local)]
        up = up.sort_values("es").drop_duplicates(["team_a", "team_b", "expected_start"])
        rows = _can_pick_outsiders(up, lo, hi, p_min, recent=False)
        if rows or interval:
            return rows
    # REPLI : aucun match à venir -> derniers matchs CAN réels (exemples)
    rec = pd.read_sql(f"""SELECT e.team_a, e.team_b, e.expected_start,
        o.odds_home oh, o.odds_draw od, o.odds_away oa FROM events e
        JOIN odds_snapshots o ON o.id=(SELECT MAX(id) FROM odds_snapshots WHERE event_id=e.id)
        WHERE e.competition='{CAN_LG}' AND e.expected_start IS NOT NULL
        ORDER BY e.expected_start DESC LIMIT 300""", engine)
    if not len(rec):
        return []
    rec["es"] = pd.to_datetime(rec.expected_start, utc=True)
    rec = rec.drop_duplicates(["team_a", "team_b", "expected_start"])
    return _can_pick_outsiders(rec, lo, hi, p_min, recent=True)


def _devig_over25(xm) -> float | None:
    """P(total > 2.5) DÉVIGÉE depuis le marché « Total de buts » (cellules 0..6).
    C'est le pricing Under/Over le plus DIRECT du book (pas une inversion)."""
    try:
        mk = json.loads(xm) if isinstance(xm, str) else (xm or {})
    except Exception:
        return None
    T = None
    for k, v in (mk or {}).items():
        if str(k).replace("é", "e").startswith("Total de buts"):
            T = v; break
    if not isinstance(T, dict):
        return None
    inv = {i: 1.0 / T[str(i)] for i in range(7)
           if isinstance(T.get(str(i)), (int, float)) and T[str(i)] > 1}
    if len(inv) != 7:
        return None
    s = sum(inv.values())
    return sum(v for i, v in inv.items() if i > 2) / s if s > 0 else None


def can_over_under_signal(engine, minutes: int = 120, start_local: str | None = None,
                          end_local: str | None = None, n_recent: int = 300) -> list:
    """Signal Under/Over 2.5 pour les matchs CAN — INDICATEUR D'AFFICHAGE, PAS une reco.

    Mesuré (rejeu 8000 matchs, marché « Total de buts » dévigé) : la direction
    Under/Over en CAN tombe juste 76.9% du temps, soit +3.6pp au-dessus de « toujours
    under » (IC ±0.9, net) — le SEUL domaine où le book porte une info que la règle
    bête rate. On l'expose donc, en surfaçant les matchs qui penchent OVER (à
    contre-courant du taux de base ~74% under) : ce sont les appels informatifs.

    RIEN À PARIER : la cote intègre déjà ce taux (Under 2.5 CAN se paie ~1.25 -> 0.96 < 1).
    'p_over' = marché « Total de buts » dévigé (le pricing O/U DIRECT du book) ; repli
    sur l'inversion 1X2 si ce marché manque. D'abord les matchs à venir, sinon récents.
    """
    now = datetime.now(timezone.utc)
    interval = bool(start_local and end_local)

    def _rows(df, recent):
        out = []
        for r in df.itertuples():
            oh, od, oa = float(r.oh), float(r.od), float(r.oa)
            if oh <= 1 or od <= 1 or oa <= 1:
                continue
            p_over = _devig_over25(r.xm)                   # pricing O/U direct du book
            source = "marché"
            if p_over is None:                             # repli : inversion 1X2 (modèle)
                pv = _over25_calib(oh, od, oa, CAN_LG)
                if pv is None:
                    continue
                p_over = pv / 100.0; source = "inversion"
            lean = "OVER" if p_over >= 0.5 else "UNDER"
            conf = abs(p_over - 0.5)                       # distance à l'indécision
            niveau = ("forte" if conf >= 0.25 else "moyenne" if conf >= 0.12 else "faible")
            out.append({
                "match": f"{r.team_a} vs {r.team_b}",
                "local": r.es.tz_convert(MADA).strftime("%H:%M"),
                "p_over": round(p_over, 4), "p_under": round(1 - p_over, 4),
                "lean": lean, "confiance": niveau, "source": source,
                "cote_juste": round(1.0 / max(p_over if lean == "OVER" else 1 - p_over, 1e-6), 2),
                "contre_courant": lean == "OVER",          # OVER = à contre-courant en CAN
                "recent": recent,
            })
        # à contre-courant d'abord (les appels informatifs), puis par confiance
        out.sort(key=lambda x: (x["recent"], not x["contre_courant"], -abs(x["p_over"] - 0.5)))
        return out

    up = pd.read_sql(f"""SELECT e.team_a, e.team_b, e.expected_start,
        o.odds_home oh, o.odds_draw od, o.odds_away oa, o.extra_markets xm FROM events e
        JOIN odds_snapshots o ON o.id=(SELECT MAX(id) FROM odds_snapshots WHERE event_id=e.id)
        LEFT JOIN results r ON r.event_id=e.id
        WHERE r.id IS NULL AND e.expected_start IS NOT NULL
          AND e.competition='{CAN_LG}'""", engine)
    if len(up):
        up["es"] = pd.to_datetime(up.expected_start, utc=True)
        horizon = 1440 if interval else minutes
        up = up[(up.es > now - pd.Timedelta(minutes=3)) & (up.es < now + pd.Timedelta(minutes=horizon))]
        if interval:
            up = up.copy()
            up["local"] = up.es.dt.tz_convert(MADA).dt.strftime("%H:%M")
            up = up[(up.local >= start_local) & (up.local <= end_local)]
        up = up.sort_values("es").drop_duplicates(["team_a", "team_b", "expected_start"])
        rows = _rows(up, recent=False)
        if rows or interval:
            return rows
    rec = pd.read_sql(f"""SELECT e.team_a, e.team_b, e.expected_start,
        o.odds_home oh, o.odds_draw od, o.odds_away oa, o.extra_markets xm FROM events e
        JOIN odds_snapshots o ON o.id=(SELECT MAX(id) FROM odds_snapshots WHERE event_id=e.id)
        WHERE e.competition='{CAN_LG}' AND e.expected_start IS NOT NULL
        ORDER BY e.expected_start DESC LIMIT {int(n_recent)}""", engine)
    if not len(rec):
        return []
    rec["es"] = pd.to_datetime(rec.expected_start, utc=True)
    rec = rec.drop_duplicates(["team_a", "team_b", "expected_start"])
    return _rows(rec, recent=True)


def can_team_profiles(engine, min_n: int = 200) -> list:
    """Profil favori/outsider de chaque équipe CAN (historique BDD) : taux de victoire,
    cote moyenne, % de matchs en favori, buts marqués/encaissés. Classé du + fort (favori
    habituel) au + faible (outsider habituel). Contexte : à cote égale, la proba est la même
    (marché calibré) — sert à repérer les outsiders 'les moins risqués' (base plus solide)."""
    d = pd.read_sql(f"""SELECT e.team_a, e.team_b, o.odds_home oh, o.odds_away oa,
        r.score_a sa, r.score_b sb FROM events e
        JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
        JOIN results r ON r.event_id=e.id
        WHERE e.competition='{CAN_LG}' AND r.score_a IS NOT NULL""", engine)
    prof = {}
    for r in d.itertuples():
        oh, oa = r.oh, r.oa
        if not (oh and oa and 1 < oh < 99.99 and 1 < oa < 99.99):
            continue
        for team, mo, gf, ga, is_fav in [(r.team_a, oh, r.sa, r.sb, oh < oa),
                                         (r.team_b, oa, r.sb, r.sa, oa < oh)]:
            t = prof.setdefault(team, {"n": 0, "w": 0, "fav": 0, "osum": 0.0, "gf": 0, "ga": 0})
            t["n"] += 1; t["w"] += int(gf > ga); t["fav"] += int(is_fav)
            t["osum"] += mo; t["gf"] += gf; t["ga"] += ga
    out = []
    for team, t in prof.items():
        if t["n"] < min_n:
            continue
        out.append({"team": team, "n": t["n"], "winrate": t["w"]/t["n"],
                    "avg_odds": t["osum"]/t["n"], "fav_pct": t["fav"]/t["n"],
                    "gf": t["gf"]/t["n"], "ga": t["ga"]/t["n"]})
    out.sort(key=lambda x: -x["winrate"])
    return out


def low_total_scan(engine, minutes: int = 120, leagues: list | None = None,
                   start_local: str | None = None, end_local: str | None = None) -> list:
    """Détecteur 0/1 but : matchs à venir (9 ligues) triés par P(≤1 but) dévigée décroissante.
    Pour chaque : proba de 0 but, proba de ≤1 but, cotes offertes « 0 » et « 1 ».
    D'abord les matchs À VENIR ; repli sur les derniers matchs réels si rien de capté.
    ⚠️ Info seulement — parier ces petits totaux est OVERPRICÉ (ROI mesuré ~-10%, marché
    Total de buts = 10.7% de marge ; en CAN 0 but = -10.6%). Sert à repérer les matchs
    défensifs, pas à gagner."""
    now = datetime.now(timezone.utc)
    base = f"""SELECT e.competition c, e.team_a, e.team_b, e.expected_start,
        o.odds_home oh, o.odds_draw od, o.odds_away oa, o.extra_markets xm FROM events e
        JOIN odds_snapshots o ON o.id=(SELECT MAX(id) FROM odds_snapshots WHERE event_id=e.id)"""

    def _pick(df, recent):
        res = []
        for r in df.itertuples():
            oh, od, oa = float(r.oh), float(r.od), float(r.oa)
            if oh <= 1 or oa <= 1 or od <= 1:
                continue
            board = market_board(r.xm, oh, od, oa)
            tot = board.get("Total de buts", [])
            if not tot:
                continue
            pm = {sel: (p, o) for sel, p, o in tot}
            if "0" not in pm and "1" not in pm:
                continue
            p0, o0 = pm.get("0", (0.0, None))
            p1, o1 = pm.get("1", (0.0, None))
            if leagues and r.c not in leagues:
                continue
            res.append({"match": f"{r.team_a} vs {r.team_b}", "tag": LEAGUE_TAGS.get(r.c, r.c[-4:]),
                        "local": r.es.tz_convert(MADA).strftime("%H:%M"), "p0": p0, "p_le1": p0 + p1,
                        "o0": o0, "o1": o1, "recent": recent})
        res.sort(key=lambda x: (x["recent"], -x["p_le1"]))
        return res

    up = pd.read_sql(base + """ LEFT JOIN results r ON r.event_id=e.id
        WHERE r.id IS NULL AND e.expected_start IS NOT NULL
          AND e.competition LIKE 'InstantLeague-%'""", engine)
    interval = bool(start_local and end_local)
    if len(up):
        up["es"] = pd.to_datetime(up.expected_start, utc=True)
        horizon = 1440 if interval else minutes
        up = up[(up.es > now - pd.Timedelta(minutes=3)) & (up.es < now + pd.Timedelta(minutes=horizon))]
        if interval:
            up = up.copy()
            up["local"] = up.es.dt.tz_convert(MADA).dt.strftime("%H:%M")
            up = up[(up.local >= start_local) & (up.local <= end_local)]
        up = up.sort_values("es").drop_duplicates(["c", "team_a", "team_b", "expected_start"])
        rows = _pick(up, recent=False)
        if rows or interval:
            return rows
    rec = pd.read_sql(base + f""" WHERE e.expected_start IS NOT NULL
        {("AND e.competition IN (" + ",".join("'"+x+"'" for x in leagues) + ")") if leagues else "AND e.competition LIKE 'InstantLeague-%'"}
        ORDER BY e.expected_start DESC LIMIT 400""", engine)
    if not len(rec):
        return []
    rec["es"] = pd.to_datetime(rec.expected_start, utc=True)
    rec = rec.drop_duplicates(["c", "team_a", "team_b", "expected_start"])
    return _pick(rec, recent=True)


def _lg_clause(leagues, col="e.competition"):
    if leagues:
        vals = ",".join("'" + str(x).replace("'", "''") + "'" for x in leagues)
        return f"AND {col} IN ({vals})"
    return f"AND {col} LIKE 'InstantLeague-%'"


def league_teams(engine, league: str) -> list:
    """Liste des équipes d'une ligue (pour les menus déroulants)."""
    lg = str(league).replace("'", "''")
    d = pd.read_sql(f"""SELECT team_a t FROM events WHERE competition='{lg}'
        UNION SELECT team_b t FROM events WHERE competition='{lg}'""", engine)
    return sorted(x for x in d.t.dropna().tolist() if x)


def match_history(engine, team: str, n: int = 5, leagues: list | None = None) -> list:
    """Les n derniers matchs JOUÉS d'une équipe, du + récent au + ancien. Rend date (Mada),
    ligue, adversaire, côté (dom/ext), score, résultat (V/N/D), cote 1X2 de l'équipe, total buts."""
    t = str(team).replace("'", "''")
    d = pd.read_sql(f"""SELECT e.competition c, e.team_a, e.team_b, e.expected_start,
        o.odds_home oh, o.odds_away oa, r.score_a sa, r.score_b sb FROM events e
        JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
        JOIN results r ON r.event_id=e.id
        WHERE r.score_a IS NOT NULL AND (e.team_a='{t}' OR e.team_b='{t}') {_lg_clause(leagues)}
        ORDER BY e.expected_start DESC LIMIT {int(n)}""", engine)
    out = []
    for r in d.itertuples():
        home = (r.team_a == team)
        gf, ga = (r.sa, r.sb) if home else (r.sb, r.sa)
        es = pd.to_datetime(r.expected_start, utc=True).tz_convert(MADA)
        out.append({"date": es.strftime("%d/%m %H:%M"), "tag": LEAGUE_TAGS.get(r.c, r.c[-4:]),
                    "opp": r.team_b if home else r.team_a, "side": "dom" if home else "ext",
                    "gf": int(gf), "ga": int(ga), "res": "V" if gf > ga else ("N" if gf == ga else "D"),
                    "odds": float(r.oh if home else r.oa), "tot": int(gf + ga)})
    return out


def _odd_min1(x):
    return round(float(x), 2) if isinstance(x, (int, float)) and x and x > 1 else None

def _mk(xm):
    """extra_markets -> dict, ou None. Robuste : la colonne est NULL en base pour
    les vieux snapshots -> pandas NaN (un float, donc truthy !) -> None."""
    if isinstance(xm, str):
        try:
            mk = json.loads(xm)
        except Exception:
            return None
    elif isinstance(xm, dict):
        mk = xm
    else:
        return None                        # NaN / None / autre : pas de marché
    return mk if isinstance(mk, dict) else None

def _odd_pos(x):
    return round(float(x), 2) if isinstance(x, (int, float)) and x and x > 0 else None

def _ou35(mk):
    """Cotes INITIALES Over/Under 3.5 (marché « +/- »). On garde même une valeur ≤1
    (under quasi-certain des matchs défensifs) pour TOUJOURS afficher les deux cotes."""
    pm = mk.get("+/-") if mk else None
    if not isinstance(pm, dict):
        return None, None
    return _odd_pos(pm.get("> 3.5")), _odd_pos(pm.get("< 3.5"))

def _dc(mk):
    """Double chance tel que coté par le book : 1X / X2 / 12."""
    dc = mk.get("Double Chance") if mk else None
    if not isinstance(dc, dict):
        return None, None, None
    return _odd_pos(dc.get("1X")), _odd_pos(dc.get("X2")), _odd_pos(dc.get("12"))

def _ou25(mk):
    """Over/Under 2.5 RECONSTITUÉ. Bet261 ne cote que la ligne 3.5 sur « +/- » ;
    le 2.5 n'existe nulle part dans le flux. On le rebâtit depuis « Total de buts »,
    qui cote chaque total exact : under = 0/1/2, over = 3 et plus (le « 6 » est un
    6+, donc le marché est complet). On somme les probas implicites SANS les
    normaliser — la marge du book reste dedans, si bien que la cote obtenue est
    celle qu'il afficherait, et non une cote « juste » gonflée par le dévig.
    Renvoie (None, None) si la somme des probas sort d'une bande plausible :
    marché tronqué ou cotes verrouillées, mieux vaut ne rien afficher.
    """
    if not isinstance(mk, dict):        # NaN (NULL SQL lu par pandas = float truthy),
        return None, None               # str, int... : jamais de .get sur autre chose
    tb = mk.get("Total de buts")
    if not isinstance(tb, dict):
        return None, None
    p_under = p_over = 0.0
    for k, o in tb.items():
        try:
            n = int(str(k).strip().rstrip("+"))
            cote = float(o)
        except (TypeError, ValueError):
            continue
        if cote <= 0:
            continue
        if n <= 2:
            p_under += 1.0 / cote
        else:
            p_over += 1.0 / cote
    if p_under <= 0 or p_over <= 0:
        return None, None
    if not (1.0 <= p_under + p_over <= 1.6):   # marge normale ~12 % ; hors bande = marché cassé
        return None, None
    return round(1.0 / p_over, 2), round(1.0 / p_under, 2)

def _goals(gj):
    """Minutes des buts, séparées équipe domicile / extérieur (ordre croissant)."""
    if isinstance(gj, str):
        try:
            arr = json.loads(gj)
        except Exception:
            return [], []
    elif isinstance(gj, list):
        arr = gj
    else:
        return [], []
    hm, am = [], []
    for g in (arr if isinstance(arr, list) else []):
        if not isinstance(g, dict):
            continue
        mn = g.get("minute")
        if not isinstance(mn, (int, float)):
            continue
        (hm if g.get("team") == "Home" else am).append(int(mn))
    return sorted(hm), sorted(am)

def _total_lines(mk, tot):
    """Toutes les lignes cotées du marché « Total de buts », dans l'ordre, chacune
    marquée si le total RÉELLEMENT sorti retombe dessus.

    La dernière ligne est un « et plus » : le 6 vaut 6+, c'est ce qui rend la somme
    des probas implicites cohérente (~1.12, la marge habituelle du book). Un match à
    7 buts y retombe donc, et s'affiche « 6+ » et non « 7 » — il n'a jamais existé
    de ligne exacte à 7.
    """
    tb = mk.get("Total de buts") if mk else None
    if not isinstance(tb, dict):
        return []
    lignes = {}
    for k, o in tb.items():
        try:
            n = int(str(k).strip().rstrip("+"))
        except (TypeError, ValueError):
            continue
        cote = _odd_pos(o)
        if cote is not None:
            lignes[n] = cote
    if not lignes:
        return []
    plafond = max(lignes)
    sorti = plafond if tot >= plafond else tot
    return [{"label": f"{n}+" if n == plafond else str(n),
             "odd": lignes[n], "hit": n == sorti} for n in sorted(lignes)]


def _match_rows(d) -> list:
    """DataFrame de rencontres terminées -> lignes prêtes pour l'UI. Partagé par
    head_to_head et recent_matches : une seule définition des cotes affichées."""
    out = []
    for r in d.itertuples():
        es = pd.to_datetime(r.expected_start, utc=True).tz_convert(MADA)
        mk = _mk(r.xm)
        ov, un = _ou35(mk)
        ov25, un25 = _ou25(mk)
        dc1x, dcx2, dc12 = _dc(mk)
        tot_reel = int(r.sa + r.sb)
        totals = _total_lines(mk, tot_reel)
        hm, am = _goals(r.gj)
        out.append({"date": es.strftime("%d/%m %H:%M"),
                    # ⚠️ La date AFFICHEE est « 05/07 21:03 » : sans annee, elle ne
                    # peut pas servir a ordonner ni a couper un historique. La date
                    # brute part donc avec la ligne, pour que l'appelant puisse
                    # demander « ce qui precede CE match » sans re-interroger la base.
                    "es": r.expected_start,
                    "home": r.team_a, "away": r.team_b,
                    "comp": r.c, "tag": LEAGUE_TAGS.get(r.c, str(r.c)[-4:]),
                    "journee": str(r.rd) if r.rd not in (None, "") else None,
                    "sa": int(r.sa), "sb": int(r.sb), "tot": int(r.sa + r.sb),
                    "oh": _odd_min1(r.oh), "od": _odd_min1(r.od), "oa": _odd_min1(r.oa),
                    "o_over35": ov, "o_under35": un,
                    "o_over25": ov25, "o_under25": un25,
                    "dc_1x": dc1x, "dc_x2": dcx2, "dc_12": dc12,
                    "totals": totals,
                    "home_min": hm, "away_min": am})
    return out


def head_to_head(engine, team_a: str, team_b: str, leagues: list | None = None, n: int = 30) -> list:
    """Tous les face-à-face directs entre 2 équipes (les deux orientations), du + récent au + ancien.
    Inclut les cotes 1X2 (1er snapshot) offertes CE match-là : oh/od/oa (None si absentes)."""
    a = str(team_a).replace("'", "''"); b = str(team_b).replace("'", "''")
    d = pd.read_sql(f"""SELECT e.competition c, e.team_a, e.team_b, e.expected_start, e.round_info rd,
        o.odds_home oh, o.odds_draw od, o.odds_away oa,
        (SELECT extra_markets FROM odds_snapshots WHERE event_id=e.id
         AND extra_markets IS NOT NULL ORDER BY id LIMIT 1) xm,
        r.score_a sa, r.score_b sb, r.goals_json gj
        FROM events e JOIN results r ON r.event_id=e.id
        LEFT JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
        WHERE r.score_a IS NOT NULL {_lg_clause(leagues)}
        AND ((e.team_a='{a}' AND e.team_b='{b}') OR (e.team_a='{b}' AND e.team_b='{a}'))
        ORDER BY e.expected_start DESC LIMIT {int(n)}""", engine)

    return _match_rows(d)


def formes_avant_h2h(engine, team_a: str, team_b: str, leagues: list | None = None,
                     h2h: list | None = None, n: int = 5) -> dict:
    """Pour CHAQUE face-a-face, la forme des deux equipes JUSTE AVANT ce match.

    Rend {es_du_match: {"a": "VNDVV", "b": "DDNVD"}}, du plus recent au plus
    ancien dans chaque chaine -- meme sens de lecture que le reste de l'app.

    ── POURQUOI « AVANT », ET PAS LA FORME ACTUELLE ─────────────────────────────

    Regarder un face-a-face de la saison passee avec la forme d'aujourd'hui ne
    dit rien : on saurait comment les equipes vont, pas comment elles allaient
    en y entrant. La coupure se fait donc a la date du match lui-meme, et ce
    match-la est EXCLU de sa propre forme.

    ── DEUX REQUETES, PAS DEUX PAR RENCONTRE ────────────────────────────────────

    L'historique complet de chaque equipe est charge UNE fois, puis decoupe en
    memoire pour chacun des face-a-face. Une requete par rencontre ferait
    soixante allers-retours sur une base que le collecteur ecrit en parallele.
    """
    lignes = list(h2h or [])
    if not lignes:
        return {}

    def _hist(team):
        t = str(team).replace("'", "''")
        d = pd.read_sql(f"""SELECT e.expected_start es, e.team_a ta,
            r.score_a sa, r.score_b sb
            FROM events e JOIN results r ON r.event_id=e.id
            WHERE r.score_a IS NOT NULL {_lg_clause(leagues)}
              AND (e.team_a='{t}' OR e.team_b='{t}')
            ORDER BY e.expected_start DESC""", engine)
        # (date, resultat) du point de vue de CETTE equipe.
        out = []
        for r in d.itertuples():
            mine, opp = (r.sa, r.sb) if r.ta == team else (r.sb, r.sa)
            out.append((r.es, "V" if mine > opp else ("N" if mine == opp else "D")))
        return out

    ha, hb = _hist(team_a), _hist(team_b)

    def _avant(hist, borne):
        # STRICTEMENT avant : un match ne fait pas partie de sa propre forme.
        return "".join(res for es, res in hist if es < borne)[:int(n)]

    formes = {}
    for m in lignes:
        borne = m.get("es")
        if borne is None:
            continue
        formes[borne] = {"a": _avant(ha, borne), "b": _avant(hb, borne)}
    return formes


def recent_matches(engine, leagues: list | None = None, n: int = 30) -> list:
    """Les n dernières rencontres TERMINÉES des 9 ligues, de la + récente à la + ancienne.

    Contrairement à head_to_head, on ne part pas d'une paire d'équipes : c'est le flux
    récent brut. Même forme de sortie (1X2, O/U 2.5 reconstitué, double chance), donc
    le même rendu peut servir les deux vues.
    """
    d = pd.read_sql(f"""SELECT e.competition c, e.team_a, e.team_b, e.expected_start, e.round_info rd,
        o.odds_home oh, o.odds_draw od, o.odds_away oa,
        (SELECT extra_markets FROM odds_snapshots WHERE event_id=e.id
         AND extra_markets IS NOT NULL ORDER BY id LIMIT 1) xm,
        r.score_a sa, r.score_b sb, r.goals_json gj
        FROM events e JOIN results r ON r.event_id=e.id
        LEFT JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
        WHERE r.score_a IS NOT NULL {_lg_clause(leagues)}
        ORDER BY e.expected_start DESC LIMIT {int(n)}""", engine)

    return _match_rows(d)


def _upcoming_df(engine, leagues=None, minutes=120, start_local=None, end_local=None):
    now = datetime.now(timezone.utc)
    up = pd.read_sql("""SELECT e.competition c, e.team_a, e.team_b, e.expected_start,
        e.round_info rd,
        o.odds_home oh, o.odds_draw od, o.odds_away oa, o.extra_markets xm FROM events e
        JOIN odds_snapshots o ON o.id=(SELECT MAX(id) FROM odds_snapshots WHERE event_id=e.id)
        LEFT JOIN results r ON r.event_id=e.id
        WHERE r.id IS NULL AND e.expected_start IS NOT NULL
          AND e.competition LIKE 'InstantLeague-%'""", engine)
    if not len(up):
        return up
    up["es"] = pd.to_datetime(up.expected_start, utc=True)
    interval = bool(start_local and end_local)
    horizon = 1440 if interval else minutes
    up = up[(up.es > now - pd.Timedelta(minutes=3)) & (up.es < now + pd.Timedelta(minutes=horizon))].copy()
    up["local"] = up.es.dt.tz_convert(MADA).dt.strftime("%H:%M")
    if interval:
        up = up[(up.local >= start_local) & (up.local <= end_local)]
    if leagues:
        up = up[up.c.isin(leagues)]
    return up.sort_values("es").drop_duplicates(["c", "team_a", "team_b", "expected_start"])


BIG_ODDS_MARKETS = ["1X2", "Double Chance", "Total de buts", "+/-", "G/NG", "Multi-Buts", "Score exact"]


def big_odds_fixtures(engine, leagues=None, min_odds=5.0, max_odds=50.0, markets=None,
                      minutes=120, start_local=None, end_local=None, top=60,
                      with_context=False, ctx_n=5) -> list:
    """Débusqueur : matchs à venir dont une sélection (marchés choisis) est à GROSSE COTE
    (min_odds..max_odds). Trié par proba dévigée décroissante (le + probable des gros paris d'abord).
    Chaque ligne porte les 2 équipes. with_context=True attache la FORME récente de chaque équipe
    + le résumé face-à-face (pour une vision globale directement à côté de la cote)."""
    up = _upcoming_df(engine, leagues, minutes, start_local, end_local)
    if not len(up):
        return []
    mkts = markets or BIG_ODDS_MARKETS
    out = []
    for r in up.itertuples():
        if float(r.oh) <= 1 or float(r.oa) <= 1:
            continue
        board = market_board(r.xm, r.oh, r.od, r.oa)
        for mk in mkts:
            for sel, p, o in board.get(mk, []):
                if min_odds <= o <= max_odds:
                    out.append({"tag": LEAGUE_TAGS.get(r.c, r.c[-4:]), "local": r.local,
                                "home": r.team_a, "away": r.team_b, "comp": r.c,
                                "market": mk, "sel": sel, "odds": float(o), "p": float(p)})
    out.sort(key=lambda x: -x["p"])
    out = out[:top]
    if with_context:
        lgs = leagues if leagues else None
        cache = {}
        for m in out:
            for side in ("home", "away"):
                t = m[side]
                if t not in cache:
                    cache[t] = match_history(engine, t, ctx_n, lgs)
                m[side + "_hist"] = cache[t]
            h2h = head_to_head(engine, m["home"], m["away"], lgs, n=20)
            m["h2h_n"] = len(h2h)
            m["h2h_zeros"] = sum(1 for x in h2h if x["tot"] == 0)
            m["h2h_avg"] = round(sum(x["tot"] for x in h2h) / len(h2h), 1) if h2h else 0.0
            m["h2h_recent"] = h2h[:5]
    return out


def under35_scan(engine, target: float = 1.68, tol: float = 0.03, leagues=None,
                 minutes: int = 180, start_local=None, end_local=None,
                 n_recent: int = 400) -> list:
    """Matchs À VENIR (9 ligues) dont la cote UNDER 3.5 (marché « +/- » : « < 3.5 »)
    vaut `target` ± `tol` (tol=0.03 => ~exactement la cible), TRIÉS par round (heure de
    coup d'envoi croissante). Créneau horaire optionnel (start_local/end_local, HH:MM Mada).
    D'abord les matchs à venir ; repli sur les derniers matchs réels si aucun capté."""
    def _pick(df, recent):
        out = []
        for r in df.itertuples():
            xm = r.xm
            if isinstance(xm, str):
                try:
                    mk = json.loads(xm)
                except Exception:
                    continue
            elif isinstance(xm, dict):
                mk = xm
            else:
                continue                        # NaN (extra_markets NULL) / None
            pm = mk.get("+/-") if isinstance(mk, dict) else None
            if not isinstance(pm, dict):
                continue
            un, ov = pm.get("< 3.5"), pm.get("> 3.5")
            if not (isinstance(un, (int, float)) and un > 1):
                continue          # under coté ≤1 = pas un vrai pari (matchs très défensifs)
            if abs(float(un) - target) <= tol:
                out.append({
                    "tag": LEAGUE_TAGS.get(r.c, str(r.c)[-4:]), "local": r.local, "es": r.es,
                    "home": r.team_a, "away": r.team_b, "under35": round(float(un), 2),
                    "over35": round(float(ov), 2) if isinstance(ov, (int, float)) and ov > 1 else None,
                    "recent": recent})
        out.sort(key=lambda x: x["es"])        # ordre du round (heure croissante)
        return out

    interval = bool(start_local and end_local)
    up = _upcoming_df(engine, leagues, minutes, start_local, end_local)
    if len(up):
        rows = _pick(up, recent=False)
        if rows or interval:
            return rows
    if interval:
        return []                              # créneau fixé : pas de repli sur le passé
    # repli : derniers matchs réels
    ph = _lg_clause(leagues)
    rec = pd.read_sql(f"""SELECT e.competition c, e.team_a, e.team_b, e.expected_start,
        o.extra_markets xm FROM events e
        JOIN odds_snapshots o ON o.id=(SELECT MAX(id) FROM odds_snapshots WHERE event_id=e.id)
        WHERE e.expected_start IS NOT NULL {ph}
        ORDER BY e.expected_start DESC LIMIT {int(n_recent)}""", engine)
    if not len(rec):
        return []
    rec["es"] = pd.to_datetime(rec.expected_start, utc=True)
    rec["local"] = rec.es.dt.tz_convert(MADA).dt.strftime("%H:%M")
    rec = rec.drop_duplicates(["c", "team_a", "team_b", "expected_start"])
    return _pick(rec, recent=True)


def combo_by_target(engine, target_odds: float, n_legs: int = 3, leagues=None,
                    start_local=None, end_local=None, top: int = 6, p_min: float = 0.35) -> list:
    """Constructeur : combiné de n_legs matchs (à venir, ligue/créneau choisis) dont la cote
    produit >= target_odds, du PLUS PROBABLE au moins probable (via build_combos)."""
    up = _upcoming_df(engine, leagues, 120, start_local, end_local)
    if not len(up):
        return []
    matches = []
    for r in up.itertuples():
        if float(r.oh) <= 1 or float(r.oa) <= 1:
            continue
        matches.append({"match": f"[{LEAGUE_TAGS.get(r.c, r.c[-4:])} {r.local}] {r.team_a} vs {r.team_b}",
                        "board": market_board(r.xm, r.oh, r.od, r.oa)})
    return build_combos(matches, target_odds=target_odds, max_legs=n_legs, min_legs=n_legs,
                        top=top, p_min=p_min)


def _can_bet_pool(engine, bet: str, lo: float, hi: float) -> list:
    """Pool historique CAN de (gagné 0/1, cote offerte) pour un type de pari + bande de cote."""
    import json as _json
    d = pd.read_sql("""SELECT o.odds_home oh, o.odds_draw od, o.odds_away oa,
        o.extra_markets xm, r.score_a sa, r.score_b sb FROM events e
        JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
        JOIN results r ON r.event_id=e.id
        WHERE e.competition='InstantLeague-8060' AND r.score_a IS NOT NULL""", engine)

    def gm(x, p):
        for k, v in (x or {}).items():
            if k.replace("\x82", "e").replace("\xe9", "e").startswith(p):
                return v
        return None
    pool = []
    for r in d.itertuples():
        oh, oa = r.oh, r.oa
        if not (oh and oa and oh > 1 and oa > 1):
            continue
        tot = r.sa + r.sb
        odds = win = None
        if bet == "outsider":
            odds = max(oh, oa); win = int((r.sb > r.sa) if oh < oa else (r.sa > r.sb))
        elif bet == "favori":
            odds = min(oh, oa); win = int((r.sa > r.sb) if oh < oa else (r.sb > r.sa))
        elif bet in ("zero", "under35"):
            try:
                mk = _json.loads(r.xm) if isinstance(r.xm, str) else (r.xm or {})
            except Exception:
                continue
            if bet == "zero":
                tb = gm(mk, "Total de buts")
                odds = tb.get("0") if isinstance(tb, dict) else None; win = int(tot == 0)
            else:
                pm = gm(mk, "+/-")
                odds = pm.get("< 3.5") if isinstance(pm, dict) else None; win = int(tot <= 3)
        if odds and 1 < odds < 99.99 and lo <= odds <= hi:
            pool.append((win, float(odds)))
    return pool


def can_simulate(engine, bet="outsider", lo=6.0, hi=10.0, stake=1000.0, n_bets=100,
                 bankroll=50000.0, stop_loss=0.5, take_profit=1.0, n_sims=3000) -> dict:
    """Simulateur Monte-Carlo : rejoue n_bets paris (tirés au hasard dans le pool historique
    CAN réel) sur n_sims sessions, mise plate, avec stop-loss / take-profit. Rend la
    distribution des résultats (pas une prédiction — un miroir honnête de la variance)."""
    import random
    pool = _can_bet_pool(engine, bet, lo, hi)
    if len(pool) < 100:
        return {"error": "pool trop petit", "n_pool": len(pool)}
    wr = sum(w for w, _ in pool) / len(pool)
    roi = sum(w * o - 1 for w, o in pool) / len(pool)
    random.seed(20260720)
    lo_bk, hi_bk = bankroll * (1 - stop_loss), bankroll * (1 + take_profit)
    finals, profit, ruin, curves = [], 0, 0, []
    npool = len(pool)
    for s in range(n_sims):
        bk = bankroll; curve = [bk]
        for _ in range(n_bets):
            if bk < stake or bk <= lo_bk or bk >= hi_bk:
                break
            w, o = pool[random.randrange(npool)]
            bk += (o - 1) * stake if w else -stake
            curve.append(bk)
        finals.append(bk)
        profit += int(bk > bankroll)
        ruin += int(bk <= lo_bk)
        if s < 40:
            curves.append(curve)
    finals.sort()
    n = len(finals)
    return {
        "n_pool": npool, "win_rate": wr, "roi": roi,
        "pct_profit": 100 * profit / n, "pct_ruin": 100 * ruin / n,
        "median": finals[n // 2], "mean": sum(finals) / n,
        "p10": finals[int(0.10 * n)], "p90": finals[int(0.90 * n)],
        "best": finals[-1], "worst": finals[0], "start": bankroll,
        "curves": curves, "n_bets": n_bets, "stake": stake,
    }


def goal_totalizer(engine, minutes: int = 30, leagues: list | None = None) -> list:
    """Pour chaque match à venir : distribution des TOTAUX de buts + top scores exacts
    (probas dévigées = calibrées, cotes offertes). Honnête : marchés −EV, aucun edge.
    Retour : liste de dict {match, tag, local, totals=[(sel,p,o)], scores=[(sel,p,o)]}."""
    now = datetime.now(timezone.utc)
    up = pd.read_sql(f"""SELECT e.competition c, e.team_a, e.team_b, e.expected_start,
        o.odds_home oh, o.odds_draw od, o.odds_away oa, o.extra_markets xm FROM events e
        JOIN odds_snapshots o ON o.id=(SELECT MAX(id) FROM odds_snapshots WHERE event_id=e.id)
        LEFT JOIN results r ON r.event_id=e.id
        WHERE r.id IS NULL AND e.expected_start IS NOT NULL
          AND e.competition LIKE 'InstantLeague-%'""", engine)
    if not len(up):
        return []
    up["es"] = pd.to_datetime(up.expected_start, utc=True)
    up = up[(up.es > now - pd.Timedelta(minutes=3)) & (up.es < now + pd.Timedelta(minutes=minutes))]
    if leagues:
        up = up[up.c.isin(leagues)]
    up = up.sort_values("es").drop_duplicates(["c", "team_a", "team_b", "expected_start"])
    out = []
    for r in up.itertuples():
        if float(r.oh) <= 1 or float(r.oa) <= 1:
            continue
        board = market_board(r.xm, r.oh, r.od, r.oa)
        totals = board.get("Total de buts", [])
        scores = board.get("Score exact", [])
        if not totals and not scores:
            continue
        out.append({"match": f"{r.team_a} vs {r.team_b}",
                    "tag": LEAGUE_TAGS.get(r.c, r.c[-4:]),
                    "local": r.es.tz_convert(MADA).strftime("%H:%M"),
                    "totals": totals, "scores": scores[:10]})
    return out


def odds_window(engine, start_local: str, end_local: str, target_odds: float,
                tol: float = 0.12, horizon_min: int = 240, leagues: list | None = None,
                markets: list | None = None) -> list:
    """Sur le créneau [start,end] et les 9 LIGUES : tous les paris dont la cote est
    proche de target_odds (±tol), classés par PROBABILITÉ décroissante.
    Retour : liste de dict {match, tag, local, market, sel, p, o}."""
    now = datetime.now(timezone.utc)
    lo_c, hi_c = target_odds * (1 - tol), target_odds * (1 + tol)
    up = pd.read_sql(f"""SELECT e.competition c, e.team_a, e.team_b, e.expected_start,
        o.odds_home oh, o.odds_draw od, o.odds_away oa, o.extra_markets xm, e.id ev FROM events e
        JOIN odds_snapshots o ON o.id=(SELECT MAX(id) FROM odds_snapshots WHERE event_id=e.id)
        LEFT JOIN results r ON r.event_id=e.id
        WHERE r.id IS NULL AND e.expected_start IS NOT NULL
          AND e.competition LIKE 'InstantLeague-%'""", engine)
    if not len(up):
        return []
    up["es"] = pd.to_datetime(up.expected_start, utc=True)
    up = up[(up.es > now - pd.Timedelta(minutes=3)) & (up.es < now + pd.Timedelta(minutes=horizon_min))]
    if leagues:
        up = up[up.c.isin(leagues)]
    up["local"] = up.es.dt.tz_convert(MADA).dt.strftime("%H:%M")
    up = up[(up.local >= start_local) & (up.local <= end_local)]
    up = up.sort_values(["es", "ev"]).drop_duplicates(["c", "team_a", "team_b", "expected_start"])
    scan_mkts = markets if markets else ODDS_SCAN_MARKETS
    out = []
    for r in up.itertuples():
        if float(r.oh) <= 1 or float(r.oa) <= 1:
            continue
        board = market_board(r.xm, r.oh, r.od, r.oa)
        tag = LEAGUE_TAGS.get(r.c, r.c[-4:])
        for mkt in scan_mkts:
            for (s, p, o) in (board.get(mkt) or []):
                if lo_c <= o <= hi_c:
                    out.append({"match": f"{r.team_a} v {r.team_b}", "tag": tag, "local": r.local,
                                "market": mkt, "sel": s, "p": p, "o": o})
    out.sort(key=lambda x: -x["p"])                 # plus probable à cette cote d'abord
    return out


# marchés autorisés pour le combiné conseillé : marges fines (~5.7-7.3%) + les
# CONJONCTIFS natifs (1X2&Total ~8.7%, 1X2&G/NG ~10.3%) — moins chers que
# d'empiler 2 jambes du même match (2x6% composés = ~12%) pour monter la cote.
COMBO_MARKETS = {"1X2", "Double Chance", "+/-", "G/NG", "1X2 & Total", "1X2 & G/NG"}
# spécialisation TOTALS (préférence utilisateur). Backtest 3312 rounds : chaque jambe
# coûte ~6-10% -> autoriser les combos à 1 jambe (un Over 3.5 seul à cote ~2.8-3.3 bat
# un triple under à même cote de ~13 points de ROI).
TOTALS_MARKETS = {"+/-", "Total de buts", "Multi-Buts"}


def build_combos(matches: list, target_odds: float = 3.0, max_legs: int = 3, top: int = 3,
                 markets: set | None = None, min_legs: int = 2, p_min: float = 0.45) -> list:
    """Combiné conseillé : parmi les combis de min_legs..max_legs jambes (1 par match),
    retourne les PLUS PROBABLES dont la cote produit >= target_odds.
    Politique max-gain/min-risque : à cote cible fixée, maximiser P(réussite).
    Indépendance inter-matchs prouvée -> P(combo) = produit des probas."""
    from itertools import combinations, product as iproduct
    mkts = markets if markets is not None else COMBO_MARKETS
    legs_by_match = []
    for m in matches:
        legs = [(m["match"], mkt, s, p, o)
                for mkt, rows in (m.get("board") or {}).items() if mkt in mkts
                for s, p, o in rows if p >= p_min and o >= 1.10]
        legs.sort(key=lambda l: -l[3])
        legs_by_match.append(legs[:5])
    # anti-explosion : ne garde que les 12 matchs aux meilleures jambes
    idxs = sorted((i for i, l in enumerate(legs_by_match) if l),
                  key=lambda i: -legs_by_match[i][0][3])[:12]
    out = []
    for r in range(max(1, min_legs), max_legs + 1):
        for mix in combinations(idxs, r):
            for choice in iproduct(*[legs_by_match[i] for i in mix]):
                oprod = pprod = 1.0
                for _, _, _, p, o in choice:
                    oprod *= o; pprod *= p
                if oprod >= target_odds:
                    out.append({"legs": choice, "odds": round(oprod, 2),
                                "p": round(pprod, 4), "ev": round(pprod*oprod - 1, 4)})
    out.sort(key=lambda c: (-c["p"], c["odds"]))
    seen, dedup = set(), []
    for c in out:                                # 1 combi max par ensemble de matchs
        key = frozenset(l[0] for l in c["legs"])
        if key not in seen:
            seen.add(key); dedup.append(c)
        if len(dedup) >= top:
            break
    return dedup


LEAGUE_TAGS = {"InstantLeague-8035": "ANG", "InstantLeague-8065": "CDM", "InstantLeague-8056": "UCL",
               "InstantLeague-8060": "CAN", "InstantLeague-8036": "ITA", "InstantLeague-8037": "ESP",
               "InstantLeague-8042": "FRA", "InstantLeague-8043": "ALL", "InstantLeague-8044": "POR"}

# ligues du panneau SPÉCIAL (mesuré) : ALL surbut (BTTS 61%, O2.5 66%, 0-0 4%),
# POR bascule défensive (O2.5 47%), CDM terrain neutre (dom 38% ≈ ext 38%).
SPECIAL3 = {"InstantLeague-8065": "CDM", "InstantLeague-8043": "ALL", "InstantLeague-8044": "POR"}


def _devig_btts(xm) -> float | None:
    """P(les deux marquent) DÉVIGÉE depuis le marché « G/NG » (Oui/Non). Direct."""
    try:
        mk = json.loads(xm) if isinstance(xm, str) else (xm or {})
    except Exception:
        return None
    g = None
    for k, v in (mk or {}).items():
        if str(k).replace("é", "e").startswith("G/NG"):
            g = v; break
    if not isinstance(g, dict):
        return None
    oui = no = None
    for k, o in g.items():
        kk = str(k).replace("é", "e").strip().lower()
        if not isinstance(o, (int, float)) or o <= 1:
            continue
        if kk.startswith("oui"):
            oui = o
        elif kk.startswith("non"):
            no = o
    if not (oui and no):
        return None
    s = 1 / oui + 1 / no
    return (1 / oui) / s if s > 0 else None




# Calibration MESUREE de p_over25 (analyse propre) -> taux reel, population cote>=2.
# Apprise sur la 1re moitie chronologique, verifiee sur la 2e (ecarts < 1pp) :
# le brut est surconfiant (~50% annonce pour ~37% reel), la table corrige ce biais.
_O25_CAL = []
try:
    _p25 = Path(__file__).resolve().parents[1] / "config" / "over25_calibration.json"
    if _p25.exists():
        _b = json.loads(_p25.read_text(encoding="utf-8")).get("bins") or []
        _O25_CAL = sorted(((b["lo"] + b["hi"]) / 2.0, float(b["real"])) for b in _b)
except Exception:
    _O25_CAL = []


def calib_over25(p_raw: float) -> float:
    """p_over25 brute -> proba CALIBREE (taux reel mesure). Identite si table absente."""
    if not isinstance(p_raw, (int, float)) or p_raw != p_raw:   # non numerique ou NaN
        return 0.0
    if not _O25_CAL:
        return float(p_raw)
    return _interp_calib([x for x, _ in _O25_CAL], [y for _, y in _O25_CAL], p_raw)



# Calibration MESUREE du TOTAL DE BUTS predit (analyse propre) -> taux reel.
# Apprise sur la 1re moitie chronologique, table isotone, verifiee sur la 2e :
# le brut annonce ~33-38 % la ou le reel fait ~27-29 %. Elle rabat ce biais.
_TOT_CAL = []
try:
    _ptot = Path(__file__).resolve().parents[1] / "config" / "totals_calibration.json"
    if _ptot.exists():
        _bt = json.loads(_ptot.read_text(encoding="utf-8")).get("bins") or []
        _TOT_CAL = sorted(((b["lo"] + b["hi"]) / 2.0, float(b["real"])) for b in _bt)
except Exception:
    _TOT_CAL = []


def calib_totals(p_raw) -> float:
    """Proba brute du total predit -> proba CALIBREE (taux reel mesure)."""
    if not isinstance(p_raw, (int, float)) or p_raw != p_raw:
        return 0.0
    if not _TOT_CAL:
        return float(p_raw)
    return _interp_calib([x for x, _ in _TOT_CAL], [y for _, y in _TOT_CAL], p_raw)


def totals_scan(engine, leagues=None, minutes: int = 180, start_local=None,
                end_local=None, top: int = 3) -> list:
    """TOTAL DE BUTS predit par l'analyse propre (Poisson sur la forme Bet261,
    cotes NON utilisees), pour les matchs a venir des ligues choisies.

    Renvoie les `top` matchs ou je suis le PLUS SUR de mon total, classes par
    proba calibree decroissante. Mesure (split chrono, TEST jamais vu) : le haut
    du classement touche ~28 % contre ~23.5 % pour l'ensemble — le tri apporte
    vraiment, mais le marche « Total de buts » porte ~10.7 % de marge : aucun
    de ces paris n'est gagnant sur la duree.
    """
    up = _upcoming_df(engine, leagues, minutes, start_local, end_local)
    out = []
    if not len(up):
        return out
    for r in up.itertuples():
        # Les cotes ne servent JAMAIS a predire ni a classer : elles sont lues
        # uniquement pour AFFICHER ce que le book paierait sur MON total. Un
        # marche absent, illisible ou incomplet ne fait donc plus disparaitre
        # le match -- il est predit quand meme, simplement sans cote.
        xm = r.xm
        mk = None
        if isinstance(xm, str):
            try:
                mk = json.loads(xm)
            except Exception:
                mk = None
        elif isinstance(xm, dict):
            mk = xm
        tb = mk.get("Total de buts") if isinstance(mk, dict) else None
        if not isinstance(tb, dict):
            tb = None
        jn = None
        _d = re.findall(r"\d+", str(getattr(r, "rd", "") or ""))
        if _d:
            jn = int(_d[0])
        try:
            own = predict_own(engine, r.team_a, r.team_b, lg=r.c, journee=jn)
        except Exception:
            own = None
        if not own or not own.get("totals"):
            continue
        dist = own["totals"]
        k = max(range(len(dist)), key=lambda i: dist[i])
        cote = _odd_pos(tb.get(str(k))) if tb else None
        top3 = sorted(range(len(dist)), key=lambda i: -dist[i])[:3]
        out.append({
            "tag": LEAGUE_TAGS.get(r.c, str(r.c)[-4:]), "local": r.local, "es": r.es,
            "home": r.team_a, "away": r.team_b, "journee": jn,
            "total": k, "label": f"{k}+" if k >= 6 else str(k),
            "odds": round(float(cote), 2) if cote else None,
            "p_mine": round(float(dist[k]), 3),
            "p_mine_cal": round(calib_totals(dist[k]), 3),
            "top3": [{"total": f"{i}+" if i >= 6 else str(i),
                      "p": round(float(dist[i]), 3),
                      "odds": (round(float(_odd_pos(tb.get(str(i)))), 2)
                               if tb and _odd_pos(tb.get(str(i))) else None)} for i in top3],
            "attendus": round(own["lam_a"] + own["lam_b"], 2),
            "lam_a": own["lam_a"], "lam_b": own["lam_b"],
            "seq_a": own.get("seq_a", ""), "seq_b": own.get("seq_b", "")})
    out.sort(key=lambda x: -x["p_mine_cal"])   # le plus SUR d'abord
    return out[:top]



# ---------------------------------------------------------------------------
# OVER / UNDER 2.5 : mes deux pronostics les plus surs.
# Calibration apprise sur la population COTEE (celle que le dashboard voit),
# 1re moitie chronologique, table isotone, verifiee sur la 2e (ecarts < 2.5pp).
# PLAFONDS : sur la queue extreme -- celle qu'on affiche justement en pronostic --
# le taux REELLEMENT observe plafonne. Mesure sur TEST jamais vu :
#   Over  : annonce 66-68 %  -> touche 75-76 %  (je sous-promets, OK)
#   Under : annonce 79-83 %  -> touche 75-77 %  sur le sous-ensemble ou le pari
#           existe vraiment (Multi-Buts « 0,1,2 ») : la je SUR-promets.
# On borne donc chaque sens a ce qui a ete constate, plutot que de laisser la
# table extrapoler au-dela de ce qui est prouve.
_OU25_OVER_MAX, _OU25_UNDER_MAX = 0.76, 0.78
_OU25_CAL = []
try:
    _pou = Path(__file__).resolve().parents[1] / "config" / "ou25_calibration.json"
    if _pou.exists():
        _bo = json.loads(_pou.read_text(encoding="utf-8")).get("bins") or []
        _OU25_CAL = sorted(((b["lo"] + b["hi"]) / 2.0, float(b["real"])) for b in _bo)
except Exception:
    _OU25_CAL = []



# ---------------------------------------------------------------------------
# TOUS LES MARCHES depuis MES deux lambdas (aucune cote en entree).
# Une seule implementation, partagee par le backtest ET le dashboard : c'est ce
# qui garantit que le taux mesure decrit bien ce qui est affiche.
MINUTE_BUCKETS = (("1-15", 0, 15), ("16-30", 15, 30), ("31-45", 30, 45),
                  ("46-60", 45, 60), ("61-75", 60, 75), ("76-90", 75, 90))

# Minute du 1er but : le modele exponentiel (intensite constante) est FAUX sur ce
# moteur -- mesure sur 207 861 matchs : il annoncait 36 % sur « 1-15 » quand le
# reel fait 20 %, et ratait le vrai pic (« 16-30 », 30 %). Le jeu a une periode de
# chauffe. On utilise donc la distribution EMPIRIQUE par bande de buts attendus,
# apprise sur la 1re moitie chronologique et stable a 1.5pp sur la seconde.
_MIN_TABLE = []
try:
    _pmt = Path(__file__).resolve().parents[1] / "config" / "minute_table.json"
    if _pmt.exists():
        _MIN_TABLE = json.loads(_pmt.read_text(encoding="utf-8")).get("bandes") or []
except Exception:
    _MIN_TABLE = []


def _minute_dist(lam: float):
    """Distribution empirique du 1er but pour ce niveau de buts attendus.
    Renvoie None si la table est absente (repli sur l'exponentiel)."""
    for b in _MIN_TABLE:
        if float(b["lo"]) <= lam < float(b["hi"]):
            return b
    return _MIN_TABLE[-1] if _MIN_TABLE else None


# HT/FT : part des buts marquee en 1re mi-temps, MESUREE sur 208 425 matchs.
# Remarquablement stable : 45.04 % sur la 1re moitie chronologique contre 44.98 %
# sur la seconde, et identique a domicile (45.08 %) et a l'exterieur (44.91 %) --
# aucun ajustement par camp n'est justifie. Variation par ligue : 43.5 a 46.1 %.
PART_MT1 = 0.45
KH = 7                      # buts par equipe et par mi-temps (0..6) : largement suffisant
HTFT_LABELS = ("1/1", "1/X", "1/2", "X/1", "X/X", "X/2", "2/1", "2/X", "2/2")


def _masques_htft():
    """Masques (9, KH*KH, KH*KH) : pour chaque issue HT/FT, les combinaisons
    (buts 1re MT) x (buts 2e MT) qui la realisent. Calcules une seule fois."""
    i, j = np.meshgrid(np.arange(KH), np.arange(KH), indexing="ij")
    k, l = i.copy(), j.copy()
    ht = np.sign(i - j).ravel()                       # issue a la mi-temps
    a1, b1 = i.ravel(), j.ravel()
    a2, b2 = k.ravel(), l.ravel()
    ft = np.sign((a1[:, None] + a2[None, :]) - (b1[:, None] + b2[None, :]))
    # Le plafond porte sur le total du MATCH ENTIER, pas sur une mi-temps :
    # une combinaison 1re + 2e mi-temps qui depasse 6 buts ne peut pas sortir.
    total = (a1[:, None] + a2[None, :]) + (b1[:, None] + b2[None, :])
    possible = total <= TOTAL_MAX
    code = {1: "1", 0: "X", -1: "2"}
    M = np.zeros((9, KH * KH, KH * KH))
    for g, lib in enumerate(HTFT_LABELS):
        h, f = lib.split("/")
        M[g] = ((np.array([code[x] for x in ht])[:, None] == h) &
                (np.vectorize(code.get)(ft) == f) & possible).astype(float)
    return M


_HTFT_M = _masques_htft()


def _grille(la: float, lb: float, k: int = None):
    """Grille des scores exacts, PLAFONNEE au total maximum du moteur puis
    renormalisee. La masse des totaux impossibles est ainsi redistribuee au
    prorata sur les scores qui peuvent reellement sortir."""
    k = k or K_GRID
    la, lb = float(la) * LAM_SCALE, float(lb) * LAM_SCALE
    g = np.outer(_poisson(la, k), _poisson(lb, k))
    idx = np.add.outer(np.arange(k), np.arange(k))
    g = np.where(idx <= TOTAL_MAX, g, 0.0)
    tot = g.sum()
    return g / tot if tot > 0 else g


def _poisson(lam: float, k: int):
    from math import exp, factorial
    return np.exp(-lam) * np.array([lam ** n / factorial(n) for n in range(k)])



def _htft(la: float, lb: float):
    """Les 9 issues mi-temps/fin de match.

    Chaque mi-temps est un Poisson independant, l'intensite etant repartie selon
    PART_MT1 (mesure). On croise ensuite la grille de la 1re mi-temps avec celle
    de la 2e : l'issue a la pause vient de la premiere, l'issue finale de la
    somme des deux -- c'est ce croisement qui distingue HT/FT d'un simple 1X2.
    """
    la, lb = la * LAM_SCALE, lb * LAM_SCALE      # meme echelle que _grille
    p1a = _poisson(max(la * PART_MT1, 1e-9), KH)
    p1b = _poisson(max(lb * PART_MT1, 1e-9), KH)
    p2a = _poisson(max(la * (1.0 - PART_MT1), 1e-9), KH)
    p2b = _poisson(max(lb * (1.0 - PART_MT1), 1e-9), KH)
    g1 = np.outer(p1a, p1b).ravel(); g1 /= g1.sum()
    g2 = np.outer(p2a, p2b).ravel(); g2 /= g2.sum()
    v = np.einsum("a,gab,b->g", g1, _HTFT_M, g2)
    # Le plafond retire de la masse : on renormalise pour rester une loi de proba.
    tot = float(v.sum())
    return [float(x / tot) for x in v] if tot > 0 else [float(x) for x in v]


# PLAFOND PAR MI-TEMPS — mesure du 27/09 sur 208 331 resultats propres.
#
# La 1re mi-temps ne depasse JAMAIS 3 buts : 0 (31,35 %), 1 (27,87 %),
# 2 (27,23 %), 3 (13,55 %). Pas une seule exception sur 208 331 matchs. La 2e
# non plus, a 47 matchs pres (0,02 %) -- assez rare pour etre traite comme du
# bruit, trop rare pour justifier d'ouvrir la grille.
#
# C'est la VRAIE contrainte du moteur, et elle explique celle qu'on avait
# trouvee sur le plein-temps : TOTAL_MAX = 6, c'est 3 + 3. Plafonner par
# mi-temps est donc plus strict que plafonner le total -- un 4-0 a la pause
# suivi d'un 0-2 ferait bien 6 au total, mais ne peut pas se produire.
HALF_MAX = 3

# ECHELLE DE LAMBDA PAR MI-TEMPS — ajustee par maximum de vraisemblance sur le
# TRAIN (1re moitie chronologique, 103 714 matchs), verifiee sur le TEST.
#
# Pourquoi elle ne vaut pas 1 : plafonner a HALF_MAX puis renormaliser DEPLACE
# la masse des totaux impossibles vers les petits scores, ce qui abaisse la
# moyenne. Il faut donc entrer une intensite plus forte pour que la grille
# plafonnee retrouve les 1,2303 buts reels de la 1re mi-temps et les 1,5031 de
# la seconde. La seconde corrige plus (1,19 contre 1,09) parce qu'elle porte
# plus de buts : le plafond y mord davantage.
#
# Optimums PLATS (1,04-1,14 et 1,14-1,24), et gain confirme HORS echantillon :
# log-vraisemblance TEST -1,9629 -> -1,9595 et -2,1284 -> -2,1135. Une echelle
# qui n'ameliorerait que le TRAIN serait du surapprentissage, pas un reglage.
HALF_SCALE = {"1re mi-temps": 1.09, "2e mi-temps": 1.19}


def _grille_mt(la: float, lb: float, part: float, echelle: float = 1.0):
    """Grille des scores exacts d'UNE mi-temps, plafonnee a HALF_MAX.

    Meme mecanique que `_grille` : echelle LAM_SCALE, Poisson independants,
    plafond, puis renormalisation -- la masse des totaux impossibles est
    redistribuee au prorata sur les scores qui peuvent reellement sortir.

    `part` est la fraction d'intensite de la mi-temps : PART_MT1 pour la
    premiere, son complement pour la seconde. La meme repartition que `_htft`,
    et elle est juste : la part REELLE mesuree vaut 0,4501 pour un PART_MT1 de
    0,45.
    """
    ech = float(echelle)
    la = max(float(la) * LAM_SCALE * part * ech, 1e-9)
    lb = max(float(lb) * LAM_SCALE * part * ech, 1e-9)
    g = np.outer(_poisson(la, KH), _poisson(lb, KH))
    idx = np.add.outer(np.arange(KH), np.arange(KH))
    g = np.where(idx <= HALF_MAX, g, 0.0)
    tot = g.sum()
    return g / tot if tot > 0 else g


def marches_mi_temps(lam_a: float, lam_b: float, top: int = 4) -> dict:
    """1X2 et score exact, pour CHACUNE des deux mi-temps.

    Rendu : {"1re mi-temps": {"x12": [...], "scores": [...]}, "2e mi-temps": ...}

    Le 1X2 d'une mi-temps n'est PAS celui du match : il porte sur les buts
    marques DANS cette periode seulement. Un « X » en 2e mi-temps veut dire
    « aucune des deux equipes ne prend l'avantage sur la periode », pas
    « match nul ».

    Aucune de ces probabilites n'est calibree : la calibration isotone du depot
    a ete etablie sur les onze marches plein-temps cotes par le book, et aucun
    marche de mi-temps n'y figure. Ce sont donc des sorties de modele brutes,
    et l'interface doit le dire.
    """
    out = {}
    for nom, part in (("1re mi-temps", PART_MT1), ("2e mi-temps", 1.0 - PART_MT1)):
        g = _grille_mt(lam_a, lam_b, part, HALF_SCALE.get(nom, 1.0))
        ph = float(np.tril(g, -1).sum())     # domicile devant sur la periode
        pn = float(np.trace(g))              # aucune des deux ne prend l'avantage
        pv = float(np.triu(g, 1).sum())      # exterieur devant sur la periode
        scores = sorted(((f"{i}-{j}", float(g[i, j]))
                         for i in range(KH) for j in range(KH) if g[i, j] > 0),
                        key=lambda kv: -kv[1])[:int(top)]
        # CALIBREES, comme tous les autres marches du tableau de bord. Mesure
        # sur 103 715 matchs de TEST jamais vus : le 1X2 de 2e mi-temps annoncait
        # 46,9 % pour 42,1 % touches ; apres correction, 42,2 % pour 42,1 %.
        out[nom] = {
            "x12": [(k, round(calib_mi_temps(f"{nom} 1X2", v), 4))
                    for k, v in (("1", ph), ("X", pn), ("2", pv))],
            "scores": [(sc, round(calib_mi_temps(f"{nom} Score exact", pr), 4))
                       for sc, pr in scores],
            "calibre": mi_temps_calibre(),
            "attendus": round(float(np.sum(
                np.add.outer(np.arange(KH), np.arange(KH)) * g)), 2),
        }
    return out


def marches_probas(lam_a: float, lam_b: float) -> dict:
    """Probabilites de CHAQUE marche Bet261, derivees de mes buts attendus.

    Grille Poisson tronquee a K_GRID puis renormalisee (identique a predict_own).
    Les minutes viennent du meme modele lu comme un processus de Poisson
    d'intensite constante sur 90 minutes : P(1er but apres t) = exp(-lam*t/90).
    """
    from math import exp
    la, lb = float(lam_a), float(lam_b)
    g = _grille(la, lb)
    idx = np.add.outer(np.arange(K_GRID), np.arange(K_GRID))
    tot = np.array([g[idx == k].sum() if k < 6 else g[idx >= 6].sum() for k in range(7)])
    ph = float(np.tril(g, -1).sum()); pn = float(np.trace(g)); pv = float(np.triu(g, 1).sum())
    p_h0 = float(g[0, :].sum()); p_a0 = float(g[:, 0].sum()); p_00 = float(g[0, 0])
    btts = 1.0 - p_h0 - p_a0 + p_00
    lam = max(la + lb, 1e-9)
    bande = _minute_dist(lam)
    if bande:
        dist_min = [(lib, float(bande["dist"].get(lib, 0.0)))
                    for lib, _, _ in MINUTE_BUCKETS]
        p_rien = float(bande["dist"].get("Pas de but", exp(-lam)))
    else:                                   # table absente : repli exponentiel
        dist_min = [(lib, exp(-lam * t0 / 90.0) - exp(-lam * t1 / 90.0))
                    for lib, t0, t1 in MINUTE_BUCKETS]
        p_rien = exp(-lam)
    out = {
        "1X2": [("1", ph), ("X", pn), ("2", pv)],
        "Double Chance": [("1X", ph + pn), ("X2", pn + pv), ("12", ph + pv)],
        "G/NG": [("Oui", btts), ("Non", 1.0 - btts)],
        "Total de buts": [(str(k), float(tot[k])) for k in range(7)],
        "+/-": [("> 3.5", float(tot[4:].sum())), ("< 3.5", float(tot[:4].sum()))],
        "Multi-Buts": [("Le total de buts est de 0, 1 ou 2", float(tot[0:3].sum())),
                       ("Le total de buts est de 1, 2 ou 3", float(tot[1:4].sum())),
                       ("Le total de buts est de 2, 3 ou 4", float(tot[2:5].sum())),
                       ("Le total de buts est supérieur à 4", float(tot[5:].sum()))],
        "Pair/Impair": [("Pair", float(tot[0::2].sum())), ("Impair", float(tot[1::2].sum()))],
        "Score exact": sorted(((f"{i}-{j}", float(g[i, j]))
                               for i in range(7) for j in range(7)),
                              key=lambda kv: -kv[1])[:10],
        "Minute du premier but": dist_min + [("Pas de but", p_rien)],
        "FTTS": [("1", (la / lam) * (1.0 - p_rien)), ("2", (lb / lam) * (1.0 - p_rien)),
                 ("Pas de but", p_rien)],
        "HT/FT": list(zip(HTFT_LABELS, _htft(la, lb))),
    }
    return {m: [(sel, round(float(pr), 4)) for sel, pr in v] for m, v in out.items()}



# Calibration par marche (isotone, TRAIN chrono, population cotee). Sans elle,
# comparer les marches entre eux serait biaise : le 1X2 brut sur-promet de 8pp
# quand le G/NG est juste -- le classement designerait le mauvais pari.
_MK_CAL = {}
try:
    _pmc = Path(__file__).resolve().parents[1] / "config" / "marches_calibration.json"
    if _pmc.exists():
        _MK_CAL = json.loads(_pmc.read_text(encoding="utf-8")).get("marches") or {}
except Exception:
    _MK_CAL = {}

# Calibration des marches de MI-TEMPS. Fichier SEPARE, et pas une entree de plus
# dans `marches_calibration.json` : celui-ci porte la mention « population cotee »,
# or aucun marche de mi-temps n'est cote par le book. Les deux tables n'ont donc
# pas la meme population de reference, et les melanger rendrait cette note fausse.
_MT_CAL = {}
try:
    _pmt = Path(__file__).resolve().parents[1] / "config" / "mitemps_calibration.json"
    if _pmt.exists():
        _MT_CAL = json.loads(_pmt.read_text(encoding="utf-8")).get("marches") or {}
except Exception:
    _MT_CAL = {}


def calib_mi_temps(cle: str, p_raw) -> float:
    """Proba brute d'un marche de mi-temps -> proba calibree.

    Meme mecanique que `calib_marche`, table differente. Sans table, la valeur
    brute ressort telle quelle : mieux vaut une sortie non corrigee qu'un
    silence, et l'interface dit laquelle des deux elle affiche.
    """
    if not isinstance(p_raw, (int, float)) or p_raw != p_raw:
        return 0.0
    b = (_MT_CAL.get(cle) or {}).get("bins") or []
    if not b:
        return float(p_raw)
    xs = [(x["lo"] + x["hi"]) / 2.0 for x in b]
    ys = [float(x["real"]) for x in b]
    return _interp_calib(xs, ys, p_raw)


def mi_temps_calibre() -> bool:
    """La table de mi-temps est-elle chargee ? L'interface doit pouvoir le dire."""
    return bool(_MT_CAL)



def _interp_calib(xs, ys, p_raw: float) -> float:
    """Interpolation d'une table de calibration, ANCREE en (0,0) et (1,1).

    Traitement ASYMETRIQUE des deux bouts, et c'est deliberе :

    - EN BAS, ancrage en (0,0). Sans lui, np.interp extrapole a plat : une table
      HT/FT apprise sur des probas de 20 a 75 % rendait 21.3 % pour une entree
      de 2 %. Les alternatives affichees sous la recommandation etaient gonflees
      et pouvaient s'ordonner a l'envers. Une proba nulle doit rester nulle.

    - EN HAUT, PLAFOND au dernier taux MESURE, pas d'ancre en (1,1). La table ne
      sait rien au-dela de ce qu'elle a observe : sur « Total de buts » le
      meilleur taux jamais constate est 26.7 %, et une ancre en (1,1) faisait
      annoncer 99.9 %. Un « + / - » a 100 % est apparu ainsi en test.

    L'asymetrie tient a la DIRECTION de l'erreur : extrapoler a plat gonfle en
    bas (dangereux) et rabote en haut (prudent). On ne promet jamais mieux que
    ce qui a ete observe.
    """
    if not xs:
        return float(p_raw)
    xs2 = [0.0] + list(xs)
    ys2 = [0.0] + list(ys)
    return float(min(np.interp(float(p_raw), xs2, ys2), ys2[-1]))


def calib_marche(marche: str, p_raw) -> float:
    """Proba brute d'un marche -> proba CALIBREE (taux reel mesure)."""
    if not isinstance(p_raw, (int, float)) or p_raw != p_raw:
        return 0.0
    b = (_MK_CAL.get(marche) or {}).get("bins") or []
    if not b:
        return float(p_raw)
    xs = [(x["lo"] + x["hi"]) / 2.0 for x in b]
    ys = [float(x["real"]) for x in b]
    return _interp_calib(xs, ys, p_raw)


# Seuils des signaux du round. Ils viennent de la version du 09/09 de cet
# ecran, conservee telle quelle : les rejouer a l'identique evite de faire
# passer pour une amelioration ce qui ne serait qu'un reglage different.
PIEGE_COTE_FAVORI = 1.7      # en dessous, le favori est cense etre solide
PIEGE_ECART = 0.05           # ecart a la proba du book qui rend ce favori suspect
PIEGE_NUL = 0.30             # part de nul qui menace un favori
PIEGE_NUL_COTE = 2.2
PIEGE_SANS_FAVORI = 0.40     # sous ce seuil, aucune issue ne se detache
GROSSE_COTE = 5.0


# Seuil par defaut des rencontres « sans favori court » : les TROIS issues
# payees au moins autant.
#
# ── POURQUOI 2,00, ET POURQUOI PAS « EGAL A 2,00 » ───────────────────────────
#
# Mesure du 27/09 sur les 155 298 rencontres cotees en base : la somme des
# inverses des trois cotes vaut 1,060 en moyenne (min 1,055, max 1,070), soit
# une marge de book d'environ 6 %. Trois cotes EGALES a 2,00 exigeraient une
# somme de 1,500 -- une marge de 50 %. Cela n'existe pas, et pas seulement
# ici : aucun operateur ne cote ainsi.
#
# Le match le plus equilibre possible sous cette marge porte trois cotes
# voisines de 2,83. C'est ce que montre la mesure : 62 rencontres seulement
# ont leurs trois cotes au-dessus de 2,80, et AUCUNE au-dessus de 3,00.
#
# Le seuil se lit donc « au moins », pas « egal ». A 2,00 il retient 34,3 % des
# rencontres ; a 2,20, 20,9 % ; a 2,50, 4,8 %.
COTE_EQUILIBRE = 2.0


def filtrer_equilibres(rencontres, cote_min: float = COTE_EQUILIBRE) -> tuple:
    """Ne garde que les rencontres dont les TROIS cotes 1X2 atteignent le seuil.

    Rend `(gardees, ecartees, sans_cote)`. Les deux comptes sont rendus pour
    etre AFFICHES : une rencontre qui disparait sans etre annoncee est un bug
    d'interface.

    Une rencontre a qui il manque une des trois cotes est ECARTEE mais COMPTEE
    -- on ne peut ni affirmer qu'elle passe le seuil, ni la taire.
    """
    seuil = float(cote_min)
    gardees, ecartees, sans_cote = [], 0, 0
    for r in rencontres or []:
        if not isinstance(r, dict):
            continue
        if r.get("erreur"):
            gardees.append(r)
            continue
        c = r.get("cotes") or {}
        trois = [_odd_pos(c.get(k)) for k in ("1", "X", "2")]
        if not all(trois):
            sans_cote += 1
        elif min(trois) + 1e-9 < seuil:
            ecartees += 1
        else:
            gardees.append(r)
    return gardees, ecartees, sans_cote


CIBLE_TROIS_COTES = (2.0, 2.0, 2.0)


def ecart_aux_cibles(cotes, cibles=CIBLE_TROIS_COTES):
    """Ecart total entre les trois cotes d'une rencontre et une cible.

    Rend `None` des qu'une des trois manque : sans elle, la distance serait
    calculee sur deux cotes et paraitrait meilleure qu'une rencontre complete.
    """
    c = cotes or {}
    trois = [_odd_pos(c.get(k)) for k in ("1", "X", "2")]
    if not all(trois):
        return None
    return float(sum(abs(o - float(t)) for o, t in zip(trois, cibles)))


def debusquer_cotes(rencontres, cibles=CIBLE_TROIS_COTES, tol: float = 0.05,
                    proches: int = 5) -> dict:
    """Cherche les rencontres dont les TROIS cotes valent la cible, a `tol` pres.

    Rend {"trouvees": [...], "proches": [...], "sans_cote": n, "examinees": n}.

    ── CE QUE LA MESURE DIT DE LA CIBLE (2,00 / 2,00 / 2,00) ────────────────────

    Elle n'existe pas, et ce n'est pas une question de chance. Sur les 184 105
    releves de cotes en base : AUCUN n'a ses trois cotes a 2,00, meme avec une
    tolerance de +/- 0,80. Les plus proches tournent autour de 2,80 / 2,85 /
    2,82, soit un ecart total de 2,48.

    La raison est arithmetique : trois cotes a 2,00 donnent une somme
    d'inverses de 1,500, c'est-a-dire 50 % de marge pour l'operateur. Le book
    mesure 1,060 ici, soit ~6 %. Sous cette marge, la rencontre la plus
    equilibree possible porte trois cotes voisines de 2,83 -- et c'est
    exactement ce qu'on observe.

    ── POURQUOI « PROCHES » EXISTE ──────────────────────────────────────────────

    Un ecran qui ne rend jamais rien n'apprend rien. Quand la cible n'est
    atteinte par personne, les rencontres les plus proches sont rendues, avec
    leur ecart, pour que la reponse soit lisible plutot qu'absente.
    """
    tol = float(tol)
    trouvees, tous, sans_cote, vus = [], [], 0, 0
    for r in rencontres or []:
        if not isinstance(r, dict) or r.get("erreur"):
            continue
        vus += 1
        c = r.get("cotes") or {}
        trois = [_odd_pos(c.get(k)) for k in ("1", "X", "2")]
        if not all(trois):
            sans_cote += 1
            continue
        d = ecart_aux_cibles(c, cibles)
        enrichie = dict(r, ecart=round(d, 2))
        tous.append(enrichie)
        # La tolerance porte sur CHAQUE cote, pas sur la somme : trois ecarts
        # de 0,04 feraient 0,12 au total et passeraient a tort un seuil global.
        if all(abs(o - float(t)) <= tol + 1e-9 for o, t in zip(trois, cibles)):
            trouvees.append(enrichie)
    tous.sort(key=lambda x: x["ecart"])
    noms = {id(x) for x in trouvees}
    return {"trouvees": trouvees,
            "proches": [x for x in tous if id(x) not in noms][:int(proches)],
            "sans_cote": sans_cote, "examinees": vus}


def signaux_1x2(p1, pn, p2, oh, od, oa, home="1", away="2") -> tuple:
    """Pieges et grosses cotes d'une rencontre, a partir des seules probas et cotes.

    Extraite de `round_1x2` pour etre eprouvee sans base : c'est ici que vivent
    les seuils, donc c'est ici que se logeraient les erreurs. Rend
    `(pieges, grosses_cotes)`.

    Sans cote 1 ou 2, les deux pieges qui comparent au book sont IMPOSSIBLES a
    evaluer -- on ne les invente pas. Seul « aucun favori net », qui ne depend
    que de moi, reste rendu.
    """
    p1, pn, p2 = (float(x or 0.0) for x in (p1, pn, p2))
    oh, od, oa = (_odd_pos(x) for x in (oh, od, oa))
    p_sel = max(p1, pn, p2)
    raisons = []
    if oh and oa:
        inv = (1.0 / oh) + (1.0 / oa) + (1.0 / od if od else 0.0)
        if inv > 0:
            if oh <= oa:
                o_fav, p_fav, pm_fav = oh, p1, (1.0 / oh) / inv
            else:
                o_fav, p_fav, pm_fav = oa, p2, (1.0 / oa) / inv
            if o_fav <= PIEGE_COTE_FAVORI and p_fav < pm_fav - PIEGE_ECART:
                raisons.append(f"favori fragile (book {pm_fav*100:.0f}%, "
                               f"moi {p_fav*100:.0f}%)")
            if pn >= PIEGE_NUL and o_fav <= PIEGE_NUL_COTE:
                raisons.append(f"nul menaçant ({pn*100:.0f}%)")
    if p_sel < PIEGE_SANS_FAVORI:
        raisons.append(f"aucun favori net (au mieux {p_sel*100:.0f}%)")

    noms = {"1": home, "X": "Nul", "2": away}
    probas = {"1": p1, "X": pn, "2": p2}
    grosses = [{"sel": k, "equipe": noms[k], "odds": round(float(c), 2),
                "p": round(probas[k], 4)}
               for k, c in (("1", oh), ("X", od), ("2", oa))
               if c and float(c) >= GROSSE_COTE]
    return raisons, grosses


def _analyse_1x2(engine, r, lg_defaut=None) -> dict:
    """Mon 1X2 sur UNE rencontre, avec ses pieges et ses grosses cotes.

    Extraite de `round_1x2` pour etre partagee avec le debusqueur : deux
    ecritures du meme pronostic divergeraient au premier reglage de l'une.
    """
    jn = None
    _d = re.findall(r"\d+", str(getattr(r, "rd", "") or ""))
    if _d:
        jn = int(_d[0])
    # ⚠️ La ligue de CETTE rencontre, pas celle demandee : sur un balayage
    # multi-ligues, passer la ligue demandee ferait chercher la forme des
    # equipes dans la mauvaise competition -- `predict_own` rendrait None, et
    # toutes les rencontres sortiraient en « historique insuffisant ».
    lg_r = getattr(r, "c", None) or lg_defaut
    own = predict_own(engine, r.team_a, r.team_b, lg=lg_r, journee=jn)
    if not own:
        return {"home": r.team_a, "away": r.team_b, "local": r.local,
                "ligue": lg_r, "erreur": "historique insuffisant"}
    # ⚠️ `x12` est une LISTE de trois nombres, pas des paires. La TAILLE ne
    # suffit pas a la valider : une chaine de trois caracteres la passerait et
    # ressortirait en trois probas nulles, donc en faux « aucun favori net ».
    bruts = list(own.get("x12") or [])
    if len(bruts) != 3 or not all(
            isinstance(v, (int, float)) and v == v for v in bruts):
        return {"home": r.team_a, "away": r.team_b, "local": r.local,
                "ligue": lg_r, "erreur": "analyse 1X2 indisponible"}
    p1, pn, p2 = (calib_marche("1X2", v) for v in bruts)
    sel, p_sel = max((("1", p1), ("X", pn), ("2", p2)), key=lambda kv: kv[1])
    oh, od, oa = _odd_pos(r.oh), _odd_pos(r.od), _odd_pos(r.oa)
    cotes = {"1": oh, "X": od, "2": oa}
    raisons, grosses = signaux_1x2(p1, pn, p2, oh, od, oa, r.team_a, r.team_b)
    return {
        "home": r.team_a, "away": r.team_b, "local": r.local, "journee": jn,
        "ligue": lg_r, "tag": getattr(r, "tag", None),
        "sel": sel, "p": round(p_sel, 4),
        "equipe": {"1": r.team_a, "2": r.team_b}.get(sel, "Nul"),
        "odds": round(float(cotes[sel]), 2) if cotes.get(sel) else None,
        "probas": {"1": round(p1, 4), "X": round(pn, 4), "2": round(p2, 4)},
        "cotes": {k: (round(float(v), 2) if v else None) for k, v in cotes.items()},
        "pieges": raisons, "grosses_cotes": grosses,
        "seq_a": own.get("seq_a", ""), "seq_b": own.get("seq_b", ""),
        "attendus": round(own["lam_a"] + own["lam_b"], 2),
        # Pour l'over/under du tableau a l'heure (04/10) : sans eux, il
        # faudrait rejouer `predict_own`, soit deux requetes de plus par match.
        "lam_a": own["lam_a"], "lam_b": own["lam_b"],
        "p_over25": own.get("p_over25"), "x12": [float(v) for v in bruts],
    }


def round_1x2(engine, lg, heure=None, limite: int = 30) -> list:
    """Mon 1X2 sur tout un round, avec les pieges et les grosses cotes.

    Le pronostic vient de MA SEULE analyse (`predict_own` : forme Bet261), puis
    de la calibration du marche 1X2. Les cotes ne servent qu'a deux choses :
    chiffrer le gain, et reperer les DESACCORDS avec le book -- jamais a
    choisir l'issue. Le detail de chaque rencontre vit dans `_analyse_1x2`.

    `lg` accepte UNE ligue ou plusieurs. Une chaine reste une chaine :
    `list("Instant...")` en ferait une liste de caracteres, et le filtre ne
    retiendrait plus rien.
    """
    ligues = [lg] if isinstance(lg, str) else [x for x in (lg or []) if x]
    up = _upcoming_df(engine, ligues or None, 1440,
                      *((str(heure).strip().zfill(5),) * 2 if heure else ()))
    if not len(up):
        return []
    out = []
    for r in up.itertuples():
        out.append(_analyse_1x2(engine, r, ligues[0] if ligues else None))
        if len(out) >= int(limite):
            break
    return out


def tableau_heure(engine, leagues=None, heure=None, limite: int = 40) -> dict:
    """Le tableau d'UNE heure de coup d'envoi (demande du 04/10) : pour chaque
    rencontre, mes chances de 1, de nul, de 2, et d'over/under 2,5 et 3,5.

    Rend {"heure": "HH:MM" | None, "lignes": [...], "total": n}. Heure vide =
    la prochaine heure de coup d'envoi des ligues choisies.

    ── AUCUN CHIFFRE NEUF, ET C'EST VOULU ───────────────────────────────────────

    - 1X2 : `_analyse_1x2`, la meme analyse que le debusqueur. Une rencontre
      affiche donc les memes pourcentages dans les deux ecrans.
    - O/U 2,5 : `ou25_probas`, calibree sur 29 835 matchs et bornee au mesure.
    - O/U 3,5, la ligne que Bet261 cote (« +/- ») : `calib_marche`, comme le
      detail des marches de « Que jouer ? ».

    ⚠️ Les trois chances du 1X2 font 100 % ici, et c'est le seul ecart avec le
    debusqueur : voir `_1x2_somme_100`. Le pronostic et sa chance, eux, sont
    les memes dans les deux ecrans.
    """
    lgs = [leagues] if isinstance(leagues, str) else [x for x in (leagues or []) if x]
    h = str(heure).strip().zfill(5) if heure else None
    if h:
        up = _upcoming_df(engine, lgs or None, 1440, h, h)
    else:
        up = _upcoming_df(engine, lgs or None, 240)
        if len(up):
            h = str(up["local"].iloc[0])      # trie par coup d'envoi
            up = up[up["local"] == h]
    if not len(up):
        return {"heure": h, "lignes": [], "total": 0}
    lignes = []
    for r in up.itertuples():
        if len(lignes) >= int(limite):
            break
        a = _analyse_1x2(engine, r, getattr(r, "c", None))
        a["tag"] = LEAGUE_TAGS.get(getattr(r, "c", None), str(getattr(r, "c", ""))[-4:])
        if not a.get("erreur"):
            a["probas_100"] = _1x2_somme_100(a.get("x12"))
            o25, u25 = ou25_probas(a.get("p_over25"))
            pm = dict(marches_probas(a["lam_a"], a["lam_b"])["+/-"])
            a["over25"], a["under25"] = round(o25, 4), round(u25, 4)
            a["over35"] = round(calib_marche("+/-", pm["> 3.5"]), 4)
            a["under35"] = round(calib_marche("+/-", pm["< 3.5"]), 4)
        lignes.append(a)
    # Groupees par ligue : a une meme minute, plusieurs ligues s'entremelent
    # au gre des secondes du coup d'envoi (mesure : 57 rencontres, 5 ligues).
    lignes.sort(key=lambda a: a.get("tag") or "")
    return {"heure": h, "lignes": lignes, "total": len(up)}


def _1x2_somme_100(bruts) -> dict:
    """1, X, 2 qui font 100 %, sans toucher a la seule valeur MESUREE.

    ── POURQUOI PAS LA CALIBRATION ISSUE PAR ISSUE DU DEBUSQUEUR ────────────────

    La table du 1X2 a ete apprise sur l'issue la PLUS PROBABLE de chaque match
    (ses paliers commencent a 33 %). Elle rabote le favori -- le modele brut
    sur-promet -- mais ne rend ces points a personne : sur le round du 05/07 a
    10:47, la somme descendait jusqu'a 80 %, et un tableau qui affiche 8 / 13 /
    67 % laisse le lecteur chercher les 12 % manquants.

    Ici : l'issue de tete garde sa chance calibree (taux reel mesure), et les
    deux autres se partagent le reste au prorata de mon analyse brute. Leur
    partage n'est pas mesure -- il ne l'etait pas davantage dans l'autre ecran,
    ou la table etait extrapolee sous son premier palier.
    """
    if not bruts or len(bruts) != 3:
        return {}
    b = [float(v) for v in bruts]
    i = max(range(3), key=lambda k: b[k])
    tete = calib_marche("1X2", b[i])
    autres = sum(b) - b[i]
    out = [((1.0 - tete) * v / autres) if autres > 0 else (1.0 - tete) / 2
           for v in b]
    out[i] = tete
    return {k: round(v, 4) for k, v in zip("1X2", out)}


# Colonnes du tableau, dans l'ordre d'affichage. Les pourcentages sont des
# entiers de 0 a 100 : l'ecran n'a plus qu'a ajouter « % ».
TABLEAU_COLONNES = ("Ligue", "Match", "Pronostic", "1", "X", "2",
                    "Over 2,5", "Under 2,5", "Over 3,5", "Under 3,5")


def tableau_reco(lignes) -> list:
    """Les cases a colorer dans le tableau, ligne par ligne (Olivio, 04/10).

    Pour chaque rencontre : la colonne de MON pronostic 1X2 (`sel`, le meme
    que le debusqueur, pas un nouveau calcul), et le cote le plus probable de
    chaque over/under. Une egalite ne colore rien : il n'y a pas de pronostic
    a montrer. Une ligne en erreur non plus.

    Rend une liste d'ensembles de noms de colonnes, dans l'ordre des lignes --
    celui de `tableau_affichage`.
    """
    out = []
    for a in lignes or []:
        cases = set()
        if not a.get("erreur"):
            if a.get("sel") in ("1", "X", "2"):
                cases.add(a["sel"])
            for o, u, co, cu in (("over25", "under25", "Over 2,5", "Under 2,5"),
                                 ("over35", "under35", "Over 3,5", "Under 3,5")):
                vo, vu = a.get(o), a.get(u)
                if isinstance(vo, (int, float)) and isinstance(vu, (int, float)):
                    if vo > vu:
                        cases.add(co)
                    elif vu > vo:
                        cases.add(cu)
        out.append(cases)
    return out


def tableau_affichage(lignes) -> list:
    """Les lignes de `tableau_heure`, pretes pour un tableau. Sans Streamlit.

    Une rencontre sans historique suffisant reste dans le tableau, cases
    vides et raison ecrite : la faire disparaitre ferait croire qu'elle
    n'existe pas a cette heure.
    """
    def _pc(v):
        return int(round(float(v) * 100)) if isinstance(v, (int, float)) and v == v else None
    out = []
    for a in lignes or []:
        ligne = {"Ligue": a.get("tag") or "",
                 "Match": f"{a.get('home', '?')} – {a.get('away', '?')}"}
        if a.get("erreur"):
            ligne["Pronostic"] = f"— {a['erreur']}"
            ligne.update({k: None for k in TABLEAU_COLONNES[3:]})
        else:
            pr = a.get("probas_100") or a.get("probas") or {}
            ligne["Pronostic"] = a.get("equipe") or ""
            ligne.update({"1": _pc(pr.get("1")), "X": _pc(pr.get("X")),
                          "2": _pc(pr.get("2")),
                          "Over 2,5": _pc(a.get("over25")),
                          "Under 2,5": _pc(a.get("under25")),
                          "Over 3,5": _pc(a.get("over35")),
                          "Under 3,5": _pc(a.get("under35"))})
        out.append({k: ligne.get(k) for k in TABLEAU_COLONNES})
    return out


def issue_round(m) -> tuple:
    """Le pronostic 1X2 d'une rencontre de la prediction du round.

    Rend (sel, issue, proba, cote) : sel en 1 / X / 2, issue = nom de
    l'equipe ou « Nul ». Meme regle que la vue en listes qu'elle remplace :
    le 1 l'emporte sur une egalite, puis le 2, le nul seulement s'il est
    strictement devant. Une seule ecriture, partagee par le tableau et le
    pari suggere.
    """
    oh, od, oa = m["cotes"]
    ph, pd_, pa = m["x12"]
    if ph >= pd_ and ph >= pa:
        return "1", m.get("team_a") or "1", ph, oh
    if pa >= pd_:
        return "2", m.get("team_b") or "2", pa, oa
    return "X", "Nul", pd_, od


# Colonnes du tableau de la prediction du round (04/10), dans l'ordre.
ROUND_COLONNES = ("Top 3", "Match", "Pronostic", "1", "X", "2", "Cote",
                  "Score exact", "Sinon", "Over 2,5", "Under 2,5", "Moteurs O/U")


def libelle_jambe(match: str, marche: str, sel: str) -> str:
    """« Leeds – London Reds → 1X2 : London Reds » : une jambe lisible.

    En 1X2 et en double chance, 1 et 2 deviennent le nom des equipes : sous
    un tableau de vingt rencontres, « 2 » seul oblige a remonter la ligne.
    """
    a, _, b = str(match).partition(" v ")
    noms = {"1": a or "1", "2": b or "2", "X": "Nul"}
    if marche == "1X2":
        lisible = noms.get(sel, sel)
    elif marche == "Double Chance" and len(sel) == 2:
        lisible = f"{sel} ({noms.get(sel[0], sel[0])} ou {noms.get(sel[1], sel[1])})"
    else:
        lisible = sel
    return f"{a} – {b} → {marche} : {lisible}" if b else f"{match} → {marche} : {lisible}"


def combine_round(matches) -> dict | None:
    """LE combine de 3 matchs que proposent les moteurs pour ce round (04/10).

    « Je veux un combine de 3 matchs propose par les moteurs. »

    Rien de neuf : `build_combos`, le constructeur de combines de l'app, avec
    les marches et le seuil de jambe de la famille « surs » du tracker
    (`COMBO_MARKETS`, 45 % au moins par jambe) -- mais EXACTEMENT 3 matchs,
    une jambe chacun, et le seul plus probable dont la cote atteint 3. Sans ce
    plancher, le plus probable serait trois jambes a 1,05 : une cote de 1,15.

    Rend {"jambes": [{"libelle", "p", "o"}...], "cote", "p"} ou None si le
    round n'a pas trois matchs aux jambes assez sures. Chance d'une jambe =
    cote du marche devigee (l'arbitre du trio) ; celle du combine est leur
    produit, les matchs etant independants.
    """
    combos = build_combos(list(matches or []), 3.0, 3, top=1,
                          markets=COMBO_MARKETS, min_legs=3, p_min=0.45)
    if not combos:
        return None
    c = combos[0]
    return {"jambes": [{"libelle": libelle_jambe(mn, mkt, s), "p": p, "o": o}
                       for mn, mkt, s, p, o in c["legs"]],
            "cote": c["odds"], "p": c["p"]}


def _accord_lisible(mo) -> str:
    """« 3/3 Over », « 2/3 Under » : combien de moteurs portent le sens retenu."""
    if not mo or not mo.get("sens"):
        return ""
    return f"{mo['accord']} {mo['sens']}"


def tableau_round(matches) -> dict:
    """La prediction du round en TABLEAU, avec l'over/under 2,5 (04/10).

    « Affiche le resultat de prediction de cette ligne sous forme tableau, et
    ajoute aussi des pronostics over/under 2,5. »

    Rend {"lignes": [...], "reco": [...]} : les lignes dans l'ordre de la vue
    qu'il remplace -- le Top 3 du round d'abord (score exact le plus
    concentre, marque 🏆), puis les autres du plus sur au moins sur -- et,
    pour chacune, les cases de MON pronostic a colorer.

    1X2, score exact et confiance sortent du trio deja affiche. L'over 2,5
    est la recommandation DES MOTEURS (`ou25_moteurs` : V2, V5 et le marche
    a poids egaux, 04/10), et « Moteurs O/U » dit combien la portent.
    L'under est son complement.
    """
    brut = []
    for m in matches or []:
        sel, issue, pi, ci = issue_round(m)
        ph, pd_, pa = m["x12"]
        t1 = m.get("top1_calibre") or (m.get("consensus_top3") or [(None, 0)])[0]
        sinon = " · ".join(f"{sc} {pr * 100:.0f} %"
                           for sc, pr in (m.get("consensus_top3") or [])[1:3] if sc)
        # Over 2,5 = ce que RECOMMANDENT les moteurs (V2, V5, marche a poids
        # egaux, 04/10) ; repli sur le marche seul pour un resultat ancien.
        mo = m.get("ou25_moteurs") or {}
        if isinstance(mo.get("over"), (int, float)):
            o25 = 100.0 * mo["over"]
        else:
            o25 = m.get("over25_pct")
            o25 = float(o25) if isinstance(o25, (int, float)) and o25 == o25 else None
        brut.append({
            "_pi": pi, "_conf": m.get("confidence") or 0, "_sel": sel, "_o25": o25,
            "Match": f"{m.get('team_a', '?')} – {m.get('team_b', '?')}",
            "Pronostic": issue,
            "1": int(round(ph * 100)), "X": int(round(pd_ * 100)),
            "2": int(round(pa * 100)),
            "Cote": round(float(ci), 2) if ci else None,
            "Score exact": (f"{t1[0]} · {t1[1] * 100:.0f} %"
                            if t1 and t1[0] else ""),
            "Sinon": sinon,
            "Over 2,5": int(round(o25)) if o25 is not None else None,
            "Under 2,5": int(round(100 - o25)) if o25 is not None else None,
            "Moteurs O/U": _accord_lisible(mo),
        })
    top3 = sorted(brut, key=lambda r: -r["_conf"])[:3]
    ids = {id(r) for r in top3}
    ordre = top3 + sorted((r for r in brut if id(r) not in ids),
                          key=lambda r: -r["_pi"])
    lignes, reco = [], []
    for r in ordre:
        r["Top 3"] = "🏆" if id(r) in ids else ""
        lignes.append({k: r[k] for k in ROUND_COLONNES})
        cases = {r["_sel"]}
        # Sur la valeur EXACTE : arrondies, 50,4 / 49,6 feraient 50 / 50.
        if r["_o25"] is not None and r["_o25"] != 50:
            cases.add("Over 2,5" if r["_o25"] > 50 else "Under 2,5")
        reco.append(cases)
    return {"lignes": lignes, "reco": reco}


def _partie_entiere(o) -> int | None:
    """Partie entiere d'une cote lisible : 2,87 -> 2. None si illisible."""
    v = _odd_pos(o)
    return int(v) if v else None


def debusquer_rounds(engine, ligues=None, n_rounds: int = 5, entier: int = 2,
                     limite: int = 60) -> dict:
    """Les rencontres des `n_rounds` prochains rounds dont les TROIS cotes 1X2
    ont `entier` pour partie entiere -- 2 signifie « 2,00 a 2,99 ».

    Rend {"trouvees": [...], "examinees": n, "rounds": n, "ligues": n}.

    ── POURQUOI LE FILTRE SUR LES COTES PASSE EN PREMIER ────────────────────────

    Cinq rounds sur neuf ligues, c'est de l'ordre de 450 rencontres. Analyser
    chacune demande deux requetes de forme, soit ~900 requetes -- pour ne
    retenir que 0,58 % d'entre elles (mesure : 1 060 releves sur 184 105).

    Le tri sur les cotes ne coute RIEN : il lit trois nombres deja charges. On
    le fait donc avant, et `predict_own` ne tourne que sur les survivantes.
    L'ecran passe de dizaines de secondes a une reponse immediate.

    ── CE QU'EST UN ROUND ICI ───────────────────────────────────────────────────

    Toutes les rencontres d'une ligue partageant la meme heure de coup d'envoi
    (9 a 18 selon la ligue). « Les 5 prochains rounds » se lit donc PAR LIGUE :
    les cinq prochaines heures de chacune, et non les cinq prochaines heures
    toutes ligues confondues -- sans quoi une ligue rapide mangerait la place
    des autres.
    """
    lgs = [ligues] if isinstance(ligues, str) else [x for x in (ligues or []) if x]
    up = _upcoming_df(engine, lgs or None, 1440)
    if not len(up):
        return {"trouvees": [], "examinees": 0, "rounds": 0, "ligues": 0}

    # Les n premieres heures de chaque ligue.
    gardees, par_ligue = [], {}
    for r in up.itertuples():
        c = getattr(r, "c", None)
        heures = par_ligue.setdefault(c, [])
        h = getattr(r, "expected_start", None)
        if h not in heures:
            if len(heures) >= int(n_rounds):
                continue
            heures.append(h)
        gardees.append(r)

    ent = int(entier)
    retenues = [r for r in gardees
                if all(_partie_entiere(o) == ent for o in (r.oh, r.od, r.oa))]
    out = []
    for r in retenues[:int(limite)]:
        out.append(_analyse_1x2(engine, r, getattr(r, "c", None)))
    return {"trouvees": out, "examinees": len(gardees),
            "rounds": int(n_rounds),
            "ligues": len({getattr(r, "c", None) for r in gardees})}


def fiabilite_marche(marche: str) -> dict | None:
    """Ce que ce marche TOUCHE reellement, mesure sur l'historique.

    Rend {"reel": 0.781, "n": 59670} ou None si le marche n'a pas ete mesure.

    ── POURQUOI CETTE INFORMATION EST A COTE DE CHAQUE PRONOSTIC (27/09) ────────

    La probabilite affichee repond a « quelles chances pour CE match ». Elle ne
    dit rien de « a quel point ce marche-la est previsible ». Or l'ecart est
    enorme d'un marche a l'autre : la Double Chance sort juste dans 78,1 % des
    cas, le Score exact dans 11,8 %. Lire « Score exact -> 2-1, 14 % » sans
    savoir que ce marche n'est touche qu'une fois sur huit donne une confiance
    que le chiffre seul ne justifie pas.

    Les deux nombres viennent de la MEME table de calibration qui corrige deja
    les probabilites : rien n'est recalcule ici, et rien ne peut donc diverger.
    """
    v = _MK_CAL.get(marche) or _MT_CAL.get(marche)
    if not isinstance(v, dict):
        return None
    r, n = v.get("global_reel"), v.get("n")
    if not isinstance(r, (int, float)) or r != r:
        return None
    return {"reel": float(r), "n": int(n or 0)}


def rencontres(engine, leagues=None, minutes: int = 240, heure=None) -> list:
    """Rencontres a venir, pour le selecteur de l'onglet conseil."""
    if heure:
        h = str(heure).strip().zfill(5)
        up = _upcoming_df(engine, leagues, minutes, h, h)
    else:
        up = _upcoming_df(engine, leagues, minutes)
    out = []
    for r in getattr(up, "itertuples", list)():
        tag = LEAGUE_TAGS.get(r.c, str(r.c)[-4:])
        out.append({"label": f"[{tag} {r.local}] {r.team_a} vs {r.team_b}",
                    "c": r.c, "tag": tag, "local": r.local,
                    "home": r.team_a, "away": r.team_b,
                    "oh": r.oh, "od": r.od, "oa": r.oa, "xm": r.xm,
                    "rd": getattr(r, "rd", None)})
    return out


def conseil(engine, renc: dict) -> dict:
    """Analyse TOUS les marches d'une rencontre et dit quoi jouer.

    Les probabilites viennent de MA seule analyse (forme Bet261), calibrees par
    marche sur l'historique. Les cotes ne servent qu'a chiffrer le gain et a
    departager a probabilite egale -- jamais a choisir la prediction.
    """
    xm = renc.get("xm")
    mk = None
    if isinstance(xm, str):
        try:
            mk = json.loads(xm)
        except Exception:
            mk = None
    elif isinstance(xm, dict):
        mk = xm
    if not isinstance(mk, dict):
        mk = {}
    jn = None
    _d = re.findall(r"\d+", str(renc.get("rd") or ""))
    if _d:
        jn = int(_d[0])
    own = predict_own(engine, renc["home"], renc["away"], lg=renc["c"], journee=jn)
    if not own:
        return {"erreur": "Pas assez d'historique en base pour ces deux équipes."}
    probas = marches_probas(own["lam_a"], own["lam_b"])

    def _cote(marche, sel):
        if marche == "1X2":
            return _odd_pos({"1": renc.get("oh"), "X": renc.get("od"),
                             "2": renc.get("oa")}.get(sel))
        d = mk.get(marche)
        return _odd_pos(d.get(sel)) if isinstance(d, dict) else None

    lignes = []
    for marche, sels in probas.items():
        sel, p_raw = max(sels, key=lambda x: x[1])
        p = calib_marche(marche, p_raw)
        o = _cote(marche, sel)
        lignes.append({
            "marche": marche, "sel": sel, "p": round(p, 3), "p_brute": round(p_raw, 3),
            "odds": round(float(o), 2) if o else None,
            "fiable": bool((_MK_CAL.get(marche) or {}).get("bins")),
            "top3": [{"sel": x[0], "p": round(calib_marche(marche, x[1]), 3),
                      "odds": (round(float(_cote(marche, x[0])), 2)
                               if _cote(marche, x[0]) else None)}
                     for x in sorted(sels, key=lambda x: -x[1])[:3]]})
    lignes.sort(key=lambda x: -x["p"])
    return {
        "home": renc["home"], "away": renc["away"], "tag": renc["tag"],
        "local": renc["local"], "journee": jn,
        "lam_a": own["lam_a"], "lam_b": own["lam_b"],
        "attendus": round(own["lam_a"] + own["lam_b"], 2),
        "seq_a": own.get("seq_a", ""), "seq_b": own.get("seq_b", ""),
        "season_a": own.get("season_a"), "season_b": own.get("season_b"),
        "lignes": lignes,
        # Les deux mi-temps, demandees le 27/09. Elles ne rejoignent PAS
        # `lignes` : ce ne sont pas des marches cotes par le book, elles n'ont
        # donc ni cote ni place dans le classement par probabilite qui designe
        # le conseil. Les melanger ferait recommander « a jouer » un pari qui
        # n'existe pas sur Bet261.
        "mi_temps": marches_mi_temps(own["lam_a"], own["lam_b"]),
        # LA recommandation : le pari le plus probable, toutes lignes confondues.
        # Mesure sur 29 835 matchs de TEST : annonce 79.7 % -> touche 80.1 %.
        #
        # La regle concurrente « proba x cote » (le pari le moins -EV) a ete
        # TESTEE puis ECARTEE : meme ROI (-7.10 % contre -7.16 %, indiscernables),
        # mais elle annonce 44.9 % pour 35.9 % reel et afficherait un gain
        # apparent > 1.00 dans 80 % des cas. Maximiser proba x cote revient a
        # selectionner les marches ou MON modele s'ecarte le plus du book en ma
        # faveur -- c'est-a-dire mes propres erreurs. On ne l'expose pas.
        "sur": lignes[0] if lignes else None,
    }



class TriConseils(NamedTuple):
    """Resultat d'un tri : ce qui reste, et pourquoi le reste est parti.

    Les trois compteurs existent pour etre AFFICHES. Une rencontre qui
    disparait sans etre annoncee est un bug d'interface, pas un filtre.
    """
    gardees: list
    hors_selection: int
    trop_bas: int
    sans_cote: int


def filtrer_conseils(resultats, cote_min: float, selections=None) -> TriConseils:
    """Ne garde que les conseils voulus, au-dessus de `cote_min`.

    `selections` : les libelles de conseil acceptes -- {"X2"} pour ne voir que
    les X2, None ou vide pour tous les accepter.

    ── POURQUOI CE TRI EXISTE (27/09) ───────────────────────────────────────────

    « Afficher les rencontres ou les predictions X2 a une cote superieure 1,20
    et les autres enleve. »

    Le conseil de tete est `lignes[0]`, la ligne la plus probable des onze
    marches ; c'est donc structurellement un double chance (1X / X2 / 12) --
    douze rencontres sur douze a la mesure. Deux criteres, donc, et non un :
    QUEL double chance, et a QUELLE cote. Mesure sur ces douze rencontres :
    sept passent le seuil de 1,20, mais deux seulement sont des X2.

    ── L'ORDRE DES DEUX CRITERES N'EST PAS INDIFFERENT ──────────────────────────

    La selection est verifiee EN PREMIER. Une rencontre conseillee « 1X » a
    1,05 echoue aux deux criteres ; la compter deux fois gonflerait le total et
    ferait mentir le bandeau. Elle est donc comptee « hors selection », et
    `trop_bas` ne parle que des conseils VOULUS mais trop peu payes -- ce qui
    est la seule information actionnable des deux.

    ── CE QUI N'EST JAMAIS MASQUE ───────────────────────────────────────────────

    Une erreur d'analyse est GARDEE : elle n'a pas de conseil a trier, et la
    masquer ferait croire que la rencontre n'existe pas plutot qu'elle n'a pas
    pu etre analysee.

    Un conseil sans cote lisible est ecarte mais COMPTE : faute de prix on ne
    peut ni affirmer qu'il passe le seuil, ni le taire.
    """
    seuil = float(cote_min)
    voulues = {str(x) for x in (selections or ())}
    gardees, hors, trop_bas, sans_cote = [], 0, 0, 0
    for r in resultats or []:
        if not isinstance(r, dict):
            continue
        if r.get("erreur"):
            gardees.append(r)
            continue
        sur = r.get("sur") or {}
        if voulues and str(sur.get("sel")) not in voulues:
            hors += 1
            continue
        o = sur.get("odds")
        try:
            o = float(o)
        except (TypeError, ValueError):
            o = None
        if o is None or o != o:          # absente, ou NaN
            sans_cote += 1
        elif o + 1e-9 < seuil:
            trop_bas += 1
        else:
            gardees.append(r)
    return TriConseils(gardees, hors, trop_bas, sans_cote)


def _z_bonferroni(n_tests: int) -> float:
    """Seuil normal a 95 % corrige pour `n_tests` comparaisons simultanees.

    Analyser une capture de round entier, c'est tester 3 issues x ~10 rencontres
    d'un coup : sans elargir le seuil, on afficherait plus d'un faux « ecart
    notable » par capture. NormalDist vient de la bibliotheque standard, donc
    pas de dependance ajoutee.
    """
    from statistics import NormalDist
    alpha = 0.05 / max(int(n_tests or 1), 1)
    return float(NormalDist().inv_cdf(1.0 - alpha / 2.0))



_ACCENTS = str.maketrans("àâäáãçéèêëíìîïñóòôöõúùûüýÿ", "aaaaaceeeeiiiinooooouuuuyy")


def _norme(t: str) -> str:
    """Forme comparable : minuscules, sans accent ni ponctuation, espaces reduits."""
    t = str(t or "").lower().translate(_ACCENTS)
    return " ".join("".join(c if c.isalnum() else " " for c in t).split())


def associer_equipes(engine, texte: str, leagues=None, seuil: float = 0.55) -> dict:
    """Rapproche un texte OCR des VRAIS noms d'equipes de la base.

    L'OCR ecrit « Fulharn » ou « Manchesler Red » : on n'affiche donc jamais ce
    qu'il rend, on affiche le nom de la base le plus proche. Cela corrige
    l'orthographe ET donne l'identite reelle des equipes, ce qui permet ensuite
    d'aller chercher leur face-a-face.

    Retour {home, away, score, ligue} ; les noms valent None si rien ne colle.
    """
    from difflib import SequenceMatcher
    sep = chr(10)
    lignes = [l for l in str(texte or "").replace("/", sep).splitlines() if l.strip()]
    if not lignes:
        return {"home": None, "away": None, "score": 0.0, "ligue": None}
    comps = list(leagues) if leagues else list(LEAGUE_TAGS)
    connues = []
    for c in comps:
        for nom in league_teams(engine, c):
            connues.append((nom, c, _norme(nom)))
    if not connues:
        return {"home": None, "away": None, "score": 0.0, "ligue": None}

    def _meilleur(txt):
        n = _norme(txt)
        if not n:
            return None, None, 0.0
        best, bestc, bests = None, None, 0.0
        for nom, comp, ref in connues:
            r = SequenceMatcher(None, n, ref).ratio()
            if r > bests:
                best, bestc, bests = nom, comp, r
        return best, bestc, bests

    trouves = [_meilleur(l) for l in lignes[:3]]
    trouves = [t for t in trouves if t[0] and t[2] >= seuil]
    if len(trouves) < 2:
        seul = trouves[0] if trouves else (None, None, 0.0)
        return {"home": seul[0], "away": None, "score": round(seul[2], 2),
                "ligue": seul[1]}
    (ha, ca, sa_), (ab, cb, sb_) = trouves[0], trouves[1]
    return {"home": ha, "away": ab, "score": round(min(sa_, sb_), 2),
            "ligue": ca if ca == cb else None}


def resultats_a_cette_cote(engine, oh: float, od: float, oa: float, tol: float = 0.05,
                           leagues=None, team_a: str | None = None,
                           team_b: str | None = None, n: int = 200,
                           n_tests: int = 3) -> dict:
    """Ce qui est REELLEMENT tombe, historiquement, a cette cote 1X2.

    On cherche les rencontres TERMINEES dont la cote d'ouverture (1er snapshot)
    vaut 1/X/2 a `tol` pres, puis on montre les resultats et ce qu'ils donnent.
    Deux equipes peuvent etre precisees pour ne garder que leurs face-a-face.

    Rien n'est predit ici : c'est un releve. La comparaison « taux reel vs
    probabilite implicite de la cote » dit seulement si le book etait juste sur
    ce prix -- jusqu'ici il l'a toujours ete, a la marge pres.
    """
    try:
        oh, od, oa, tol = float(oh), float(od), float(oa), abs(float(tol))
    except (TypeError, ValueError):
        return {"erreur": "Cotes invalides."}
    if min(oh, od, oa) <= 1.0:
        return {"erreur": "Une cote doit être supérieure à 1."}
    where = [f"o.odds_home BETWEEN {oh - tol} AND {oh + tol}",
             f"o.odds_draw BETWEEN {od - tol} AND {od + tol}",
             f"o.odds_away BETWEEN {oa - tol} AND {oa + tol}"]
    if team_a and team_b:
        a = str(team_a).replace("'", "''"); b = str(team_b).replace("'", "''")
        where.append(f"((e.team_a='{a}' AND e.team_b='{b}') OR "
                     f"(e.team_a='{b}' AND e.team_b='{a}'))")
    d = pd.read_sql(f"""SELECT e.competition c, e.team_a, e.team_b, e.expected_start,
        e.round_info rd, o.odds_home oh, o.odds_draw od, o.odds_away oa,
        r.score_a sa, r.score_b sb
        FROM events e JOIN results r ON r.event_id=e.id
        JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
        WHERE r.score_a IS NOT NULL {_lg_clause(leagues)} AND {' AND '.join(where)}
        ORDER BY e.expected_start DESC LIMIT {int(n)}""", engine)
    if not len(d):
        return {"matchs": [], "n": 0}
    d["tot"] = d.sa.astype(int) + d.sb.astype(int)
    d["issue"] = np.where(d.sa > d.sb, "1", np.where(d.sa == d.sb, "X", "2"))
    matchs = [{
        "date": pd.to_datetime(r.expected_start, utc=True).tz_convert(MADA).strftime("%d/%m %H:%M"),
        "tag": LEAGUE_TAGS.get(r.c, str(r.c)[-4:]),
        "journee": (int(re.findall(r"\d+", str(r.rd))[0])
                    if r.rd and re.findall(r"\d+", str(r.rd)) else None),
        "home": r.team_a, "away": r.team_b, "sa": int(r.sa), "sb": int(r.sb),
        "tot": int(r.tot), "issue": r.issue,
        "cotes": [round(float(r.oh), 2), round(float(r.od), 2), round(float(r.oa), 2)],
    } for r in d.itertuples()]
    nb = len(d)
    inv = 1 / oh + 1 / od + 1 / oa
    resume = {"n": nb, "buts_moyen": round(float(d.tot.mean()), 2),
              "over25": round(100.0 * float((d.tot >= 3).mean()), 1),
              "over35": round(100.0 * float((d.tot >= 4).mean()), 1),
              "btts": round(100.0 * float(((d.sa > 0) & (d.sb > 0)).mean()), 1),
              "zero": round(100.0 * float((d.tot == 0).mean()), 1),
              "marge": round(100.0 * (inv - 1.0), 1), "issues": []}
    # Un releve sur quelques dizaines de matchs produit MECANIQUEMENT des ecarts
    # de plusieurs points. Sans intervalle de confiance, cet onglet ferait voir
    # un signal a chaque requete. On tranche donc explicitement bruit / notable.
    # z corrige de Bonferroni : les TROIS issues sont testees a chaque requete.
    # A 95 % non corrige, une requete sur sept afficherait un « ecart notable »
    # par pur hasard (verifie : 2 sur 15 tests) -- exactement le piege de
    # comparaison multiple qui a deja fabrique 3 faux positifs sur 76 500
    # cellules dans ce projet.
    Z_BONF = _z_bonferroni(n_tests)

    def _wilson(k, n, z=Z_BONF):
        """Intervalle de Wilson a 95 %. L'approximation normale donne une largeur
        NULLE quand k vaut 0 ou n -- elle declarerait alors n'importe quel ecart
        significatif sur un petit echantillon, ce qui est le cas d'usage courant
        ici (une paire precise + une cote precise = quelques dizaines de matchs)."""
        if n <= 0:
            return 0.0, 1.0
        pr = k / n
        den = 1.0 + z * z / n
        centre = (pr + z * z / (2 * n)) / den
        demi = (z / den) * ((pr * (1 - pr) / n + z * z / (4 * n * n)) ** 0.5)
        return max(0.0, centre - demi), min(1.0, centre + demi)

    for lib, cote in (("1", oh), ("X", od), ("2", oa)):
        k = int((d.issue == lib).sum())
        pr = k / nb
        lo_ic, hi_ic = _wilson(k, nb)
        dev = 1.0 / cote / inv
        resume["issues"].append({
            "sel": lib, "cote": round(cote, 2), "n": k,
            "reel": round(100.0 * pr, 1),
            "ic_bas": round(100.0 * lo_ic, 1), "ic_haut": round(100.0 * hi_ic, 1),
            "implicite": round(100.0 / cote, 1),
            "devigue": round(100.0 * dev, 1),
            "ecart": round(100.0 * (pr - dev), 1),
            # « notable » = le prix devigue tombe HORS de l'intervalle a 95 %.
            "notable": bool(dev < lo_ic or dev > hi_ic),
        })
    resume["notables"] = sum(1 for i in resume["issues"] if i["notable"])
    resume["n_tests"] = int(n_tests or 3)
    # n minimal pour qu'un ecart de 5 points soit seulement detectable
    resume["n_pour_5pp"] = int(round((1.96 ** 2) * 0.25 / (0.05 ** 2)))
    sc = d.groupby([d.sa, d.sb]).size().sort_values(ascending=False)
    resume["scores"] = [{"score": f"{int(a)}-{int(b)}", "n": int(k),
                         "pct": round(100.0 * int(k) / nb, 1)}
                        for (a, b), k in list(sc.items())[:6]]
    return {"matchs": matchs, "n": nb, "resume": resume,
            "cible": [round(oh, 2), round(od, 2), round(oa, 2)], "tol": tol}


def ou25_probas(p_raw_over):
    """P(over 2.5) brute -> (over calibre, under calibre), bornes au mesure."""
    if not isinstance(p_raw_over, (int, float)) or p_raw_over != p_raw_over:
        return 0.0, 0.0
    o = float(p_raw_over)
    if _OU25_CAL:
        o = _interp_calib([x for x, _ in _OU25_CAL], [y for _, y in _OU25_CAL], o)
    return min(o, _OU25_OVER_MAX), min(1.0 - o, _OU25_UNDER_MAX)


def _paris_reels(mk, sens):
    """Ce qui est REELLEMENT cliquable dans Bet261 pour ce sens, libelles exacts.
    Renvoie (direct, cellules, voisins) : `direct` est un pari en un seul clic
    (il n'existe QUE pour l'under), `cellules` est la mise a repartir."""
    tb = mk.get("Total de buts") if isinstance(mk, dict) else None
    mb = mk.get("Multi-Buts") if isinstance(mk, dict) else None
    pm = mk.get("+/-") if isinstance(mk, dict) else None
    direct, cellules, voisins = None, [], []
    if sens == "under":
        if isinstance(mb, dict):
            for lib, o in mb.items():
                if "0, 1 ou 2" in str(lib) and _odd_pos(o):
                    direct = {"marche": "Multi-Buts", "sel": str(lib),
                              "odds": round(float(_odd_pos(o)), 2)}
        totaux = ("0", "1", "2")
    else:
        totaux = ("3", "4", "5", "6")
    if isinstance(tb, dict):
        for k in totaux:
            o = _odd_pos(tb.get(k))
            if o:
                cellules.append({"total": k, "odds": round(float(o), 2)})
    inv = sum(1.0 / c["odds"] for c in cellules)
    for c in cellules:
        c["part"] = round(100.0 * (1.0 / c["odds"]) / inv, 1) if inv else None
    equiv = round(1.0 / inv, 2) if inv else None
    if isinstance(pm, dict):
        lib = "> 3.5" if sens == "over" else "< 3.5"
        o = _odd_pos(pm.get(lib))
        if o:
            voisins.append({"marche": "+/-", "sel": lib, "odds": round(float(o), 2)})
    if isinstance(mb, dict):
        for lib in ("Le total de buts est de 2, 3 ou 4",
                    "Le total de buts est supérieur à 4" if sens == "over"
                    else "Le total de buts est de 1, 2 ou 3"):
            o = _odd_pos(mb.get(lib))
            if o:
                voisins.append({"marche": "Multi-Buts", "sel": lib,
                                "odds": round(float(o), 2)})
    return direct, cellules, equiv, voisins


def _toutes_cotes(mk, oh=None, od=None, oa=None):
    """Toutes les cotes lisibles du match — sert a RETROUVER un match a partir
    d'une cote vue dans l'application (1X2, totaux, Multi-Buts, +/-)."""
    out = []
    for v in (oh, od, oa):
        o = _odd_pos(v)
        if o:
            out.append(round(float(o), 2))
    for nom in ("Total de buts", "Multi-Buts", "+/-", "Double Chance", "G/NG"):
        d = mk.get(nom) if isinstance(mk, dict) else None
        if isinstance(d, dict):
            for v in d.values():
                o = _odd_pos(v)
                if o:
                    out.append(round(float(o), 2))
    return out


def ou25_picks(engine, leagues=None, minutes: int = 240, start_local=None,
               end_local=None, heure=None, cote=None, tol: float = 0.05) -> dict:
    """Mes DEUX pronostics les plus surs sur la ligne 2.5 : l'over et l'under.

    La prediction vient de la SEULE forme des equipes dans le virtuel Bet261
    (Poisson attaque/defense) ; les cotes ne servent qu'a montrer quoi cliquer.
    Renvoie {"over": {...} | None, "under": {...} | None}.
    """
    # Heure ciblee : on borne la fenetre a cette minute exacte (bornes inclusives),
    # ce qui isole le round demande sans que l'utilisateur ait a saisir un creneau.
    if heure:
        start_local = end_local = str(heure).strip().zfill(5)
    up = _upcoming_df(engine, leagues, minutes, start_local, end_local)
    cands = []
    if not len(up):
        return {"over": None, "under": None}
    for r in up.itertuples():
        xm = r.xm
        mk = None
        if isinstance(xm, str):
            try:
                mk = json.loads(xm)
            except Exception:
                mk = None
        elif isinstance(xm, dict):
            mk = xm
        if not isinstance(mk, dict):
            mk = {}
        # Cote ciblee : on ne garde que les matchs ou cette cote existe vraiment,
        # tous marches confondus. C'est ainsi qu'on retrouve LE match vu dans l'app.
        if cote:
            try:
                cible = float(cote)
            except (TypeError, ValueError):
                cible = None
            if cible:
                vues = _toutes_cotes(mk, getattr(r, "oh", None), getattr(r, "od", None),
                                     getattr(r, "oa", None))
                if not any(abs(o - cible) <= tol for o in vues):
                    continue
        jn = None
        _d = re.findall(r"\d+", str(getattr(r, "rd", "") or ""))
        if _d:
            jn = int(_d[0])
        try:
            own = predict_own(engine, r.team_a, r.team_b, lg=r.c, journee=jn)
        except Exception:
            own = None
        if not own or own.get("p_over25") is None:
            continue
        p_o, p_u = ou25_probas(own["p_over25"])
        cands.append({
            "tag": LEAGUE_TAGS.get(r.c, str(r.c)[-4:]), "local": r.local, "es": r.es,
            "home": r.team_a, "away": r.team_b, "journee": jn, "mk": mk,
            "p_over": round(p_o, 3), "p_under": round(p_u, 3),
            "attendus": round(own["lam_a"] + own["lam_b"], 2),
            "lam_a": own["lam_a"], "lam_b": own["lam_b"],
            "seq_a": own.get("seq_a", ""), "seq_b": own.get("seq_b", "")})
    if not cands:
        return {"over": None, "under": None}

    def _monte(c, sens):
        d, cel, eq, vois = _paris_reels(c.pop("mk") if "mk" in c else {}, sens)
        c.update({"sens": sens, "direct": d, "cellules": cel,
                  "equivalent": eq, "voisins": vois})
        return c
    best_o = max(cands, key=lambda c: c["p_over"])
    best_u = max(cands, key=lambda c: c["p_under"])
    out_o = _monte(dict(best_o), "over")
    out_u = _monte(dict(best_u), "under")
    return {"over": out_o, "under": out_u}


def over25_scan(engine, min_odds: float = 2.0, leagues=None, minutes: int = 180,
                start_local=None, end_local=None, top: int = 40) -> list:
    """Matchs À VENIR dont l'OVER 2.5 se paie >= `min_odds`, classés par MA proba
    (analyse Poisson de la forme Bet261 : cotes NON utilisées) — le plus sûr d'abord.

    Bet261 ne cote pas la ligne 2.5 (seul le 3.5 existe sur « +/- ») : elle est
    RECONSTITUÉE depuis « Total de buts » (cellules 3,4,5,6 = plus de 2.5 buts),
    marge du book incluse — c'est la cote qu'il afficherait. Le pari s'exécute
    en misant ces 4 cellules au prorata de 1/cote : la répartition est donnée
    dans `cells` (clé `part`, en % de la mise).
    """
    up = _upcoming_df(engine, leagues, minutes, start_local, end_local)
    out = []
    if not len(up):
        return out
    for r in up.itertuples():
        xm = r.xm
        if isinstance(xm, str):
            try:
                mk = json.loads(xm)
            except Exception:
                continue
        elif isinstance(xm, dict):
            mk = xm
        else:
            continue                          # NaN (extra_markets NULL) / None
        if not isinstance(mk, dict):
            continue
        o_over, o_under = _ou25(mk)
        if not (isinstance(o_over, (int, float)) and float(o_over) >= float(min_odds)):
            continue
        jn = None
        _d = re.findall(r"\d+", str(getattr(r, "rd", "") or ""))
        if _d:
            jn = int(_d[0])
        try:
            own = predict_own(engine, r.team_a, r.team_b, lg=r.c, journee=jn)
        except Exception:
            own = None
        if not own or own.get("p_over25") is None:
            continue
        tb = mk.get("Total de buts")
        cells = []
        if isinstance(tb, dict):
            for k in ("3", "4", "5", "6"):
                o = _odd_pos(tb.get(k))
                if o:
                    cells.append({"total": k, "odds": round(float(o), 2)})
        # Paris REELLEMENT cliquables dans Bet261, avec le libelle exact de l'app
        # et MA proba brute pour chacun. Sans ca, l'onglet n'affichait qu'une cote
        # synthetique introuvable dans l'application.
        dist = own.get("totals") or []
        def _pdist(lo, hi):
            return round(sum(dist[k] for k in range(lo, min(hi, 6) + 1)), 3) if dist else None
        reels = []
        mb = mk.get("Multi-Buts") if isinstance(mk, dict) else None
        if isinstance(mb, dict):
            for lib, pr in (("Le total de buts est de 2, 3 ou 4", _pdist(2, 4)),
                            ("Le total de buts est de 1, 2 ou 3", _pdist(1, 3)),
                            ("Le total de buts est supérieur à 4", _pdist(5, 6))):
                o_mb = _odd_pos(mb.get(lib))
                if o_mb:
                    reels.append({"marche": "Multi-Buts", "sel": lib,
                                  "odds": round(float(o_mb), 2), "p": pr})
        pm_ = mk.get("+/-") if isinstance(mk, dict) else None
        if isinstance(pm_, dict):
            for lib, pr in (("> 3.5", _pdist(4, 6)), ("< 3.5", _pdist(0, 3))):
                o_pm = _odd_pos(pm_.get(lib))
                if o_pm:
                    reels.append({"marche": "+/-", "sel": lib,
                                  "odds": round(float(o_pm), 2), "p": pr})
        tot_inv = sum(1.0 / c["odds"] for c in cells)
        for c in cells:
            c["part"] = round(100.0 * (1.0 / c["odds"]) / tot_inv, 1) if tot_inv else None
            c["p"] = round(dist[int(c["total"])], 3) if dist else None
        p = float(own["p_over25"])
        out.append({
            "tag": LEAGUE_TAGS.get(r.c, str(r.c)[-4:]), "local": r.local, "es": r.es,
            "home": r.team_a, "away": r.team_b, "journee": jn,
            "odds_over25": round(float(o_over), 2),
            "odds_under25": round(float(o_under), 2) if o_under else None,
            "p_mine": round(p, 3),
            "p_mine_cal": round(calib_over25(p), 3),
            "p_market": round(1.0 / float(o_over), 3),
            "edge": round(calib_over25(p) * float(o_over) - 1.0, 3),
            "lam_a": own["lam_a"], "lam_b": own["lam_b"],
            "seq_a": own.get("seq_a", ""), "seq_b": own.get("seq_b", ""),
            "season_a": own.get("season_a"), "season_b": own.get("season_b"),
            "cells": cells, "reels": reels})
    out.sort(key=lambda x: -x["p_mine_cal"])       # le plus SÛR d'abord (MA proba)
    return out[:top]


def special_scan(engine, leagues=None, minutes: int = 120, start_local: str | None = None,
                 end_local: str | None = None, n_recent: int = 200) -> list:
    """Scan des ligues CDM / ALL / POR — INDICATEUR D'AFFICHAGE, PAS une reco.

    Pour chaque match : la config de cotes (triplet plancher, ex. 2-3-3), le FAVORI
    (issue la + probable) et sa proba dévigée, le signal Over 2.5 et BTTS (marchés
    dévigés directs). Sert à LIRE le match selon la signature de la ligue :
      • ALL surbut → guette Over/BTTS ; POR bascule défensive ; CDM terrain neutre.
    RIEN À PARIER : tout est price (chasse exhaustive 76 500 cellules = 0 edge). Le
    « favori gagne le + souvent » ≈ 44% en config asymétrique, PAS « presque toujours ».
    D'abord les matchs à venir ; repli sur les derniers matchs réels si aucun.
    """
    lgs = leagues or list(SPECIAL3)
    now = datetime.now(timezone.utc)
    interval = bool(start_local and end_local)

    def _rows(df, recent):
        out = []
        for r in df.itertuples():
            oh, od, oa = float(r.oh), float(r.od), float(r.oa)
            if oh <= 1 or od <= 1 or oa <= 1:
                continue
            inv = 1 / oh + 1 / od + 1 / oa
            trip = [("1", oh, r.team_a), ("X", od, "nul"), ("2", oa, r.team_b)]
            fav = min(trip, key=lambda t: t[1])
            cfg = "-".join(str(min(int(x), 9)) for x in (oh, od, oa))
            n2 = sum(1 for x in (oh, od, oa) if int(x) == 2)   # nb de côtes à 2.x
            out.append({
                "match": f"{r.team_a} vs {r.team_b}",
                "local": r.es.tz_convert(MADA).strftime("%H:%M"),
                "lg": SPECIAL3.get(r.c, r.c[-4:]),
                "config": cfg, "deux_favoris": n2 >= 2,
                "fav_side": fav[0], "fav_name": fav[2], "fav_odds": round(fav[1], 2),
                "fav_p": round((1 / fav[1]) / inv, 4),
                "p_over": _devig_over25(r.xm), "p_btts": _devig_btts(r.xm),
                "recent": recent,
            })
        # les configs à UN seul favori clair d'abord (les + lisibles), puis proba fav
        out.sort(key=lambda x: (x["recent"], x["deux_favoris"], -x["fav_p"]))
        return out

    up = _upcoming_df(engine, lgs, minutes, start_local, end_local)
    if len(up):
        rows = _rows(up, recent=False)
        if rows or interval:
            return rows
    # repli : derniers matchs réels des 3 ligues
    ph = ",".join("?" * len(lgs))
    rec = pd.read_sql(f"""SELECT e.competition c, e.team_a, e.team_b, e.expected_start,
        o.odds_home oh, o.odds_draw od, o.odds_away oa, o.extra_markets xm FROM events e
        JOIN odds_snapshots o ON o.id=(SELECT MAX(id) FROM odds_snapshots WHERE event_id=e.id)
        WHERE e.competition IN ({ph}) AND e.expected_start IS NOT NULL
        ORDER BY e.expected_start DESC LIMIT {int(n_recent)}""", engine, params=tuple(lgs))
    if not len(rec):
        return []
    rec["es"] = pd.to_datetime(rec.expected_start, utc=True)
    rec = rec.drop_duplicates(["c", "team_a", "team_b", "expected_start"])
    return _rows(rec, recent=True)


def upcoming_all(engine, minutes: int = 6) -> list:
    """Matchs à venir des 9 LIGUES dans les `minutes` prochaines (boards marché, sans fit)
    -> alimente le combiné INTER-LIGUES."""
    now = datetime.now(timezone.utc)
    up = pd.read_sql("""SELECT e.competition c, e.team_a,e.team_b,e.expected_start,
        o.odds_home oh,o.odds_draw od,o.odds_away oa,o.extra_markets xm, e.id ev FROM events e
        JOIN odds_snapshots o ON o.id=(SELECT MAX(id) FROM odds_snapshots WHERE event_id=e.id)
        LEFT JOIN results r ON r.event_id=e.id
        WHERE r.id IS NULL AND e.expected_start IS NOT NULL
          AND e.competition LIKE 'InstantLeague-%'""", engine)
    if not len(up):
        return []
    up["es"] = pd.to_datetime(up.expected_start, utc=True)
    up = up[(up.es > now) & (up.es <= now + pd.Timedelta(minutes=minutes))]
    up = up.sort_values(["es", "ev"]).drop_duplicates(["c", "team_a", "team_b", "expected_start"])
    out = []
    for r in up.itertuples():
        if float(r.oh) <= 1 or float(r.oa) <= 1:
            continue
        tag = LEAGUE_TAGS.get(r.c, r.c[-4:])
        local = r.es.tz_convert(MADA).strftime("%H:%M")
        out.append({"match": f"[{tag} {local}] {r.team_a} v {r.team_b}",
                    "board": market_board(r.xm, r.oh, r.od, r.oa)})
    return out


def _p_over25_grille(g) -> float | None:
    """P(3 buts ou plus) lue sur une grille de scores, renormalisee."""
    try:
        g = np.asarray(g, float)
        idx = np.add.outer(np.arange(g.shape[0]), np.arange(g.shape[1]))
        tot = float(g.sum())
        return float(g[idx >= 3].sum() / tot) if tot > 0 else None
    except Exception:
        return None


def _pct_en_proba(v) -> float | None:
    return float(v) / 100.0 if isinstance(v, (int, float)) and v == v else None


def ou25_moteurs(v2, v5, marche) -> dict | None:
    """L'over/under 2,5 que recommandent les moteurs du trio (Olivio, 04/10).

    Chaque moteur donne sa P(over 2,5) : V2 sur sa grille de scores (Poisson
    + Dixon-Coles + marche du score exact), V5 sur la sienne (multi-marches),
    le marche sur les cotes devigees et calibrees par ligue. La
    recommandation est leur moyenne a POIDS EGAUX entre les moteurs PRESENTS
    -- la regle que le trio applique deja au score exact. Hors anglaise, V2
    et V5 n'ont pas les equipes : le marche reste seul, et l'accord le dit.

    Rend {"over": p, "sens": "Over"|"Under"|None, "accord": "2/3",
    "detail": {"V2": p, "V5": p, "Marché": p}} ou None sans aucun moteur.

    ⚠️ Le taux de reussite de cette moyenne n'est PAS mesure : seul le
    marche, pris seul, l'a ete. L'ecran doit le dire.
    """
    detail = {k: float(v) for k, v in (("V2", v2), ("V5", v5), ("Marché", marche))
              if isinstance(v, (int, float)) and v == v and 0.0 <= v <= 1.0}
    if not detail:
        return None
    over = sum(detail.values()) / len(detail)
    sens = "Over" if over > 0.5 else ("Under" if over < 0.5 else None)
    pour = sum(1 for v in detail.values()
               if (v > 0.5 if sens == "Over" else v < 0.5)) if sens else 0
    return {"over": round(over, 4), "sens": sens,
            "accord": f"{pour}/{len(detail)}",
            "detail": {k: round(v, 4) for k, v in detail.items()}}


def predict_one(engine, m5, v2model, team_a, team_b, oh, od, oa, extra_markets=None,
                lg: str = None) -> dict:
    oh, od, oa = float(oh), float(od), float(oa)
    sem = _sem(extra_markets)
    # --- V2 (grille blendée) ---
    v2top = []
    ph = pd_ = pa = None
    v2_o25 = v5_o25 = None
    try:
        p2 = predict_match_v2(v2model, team_a, team_b, oh, od, oa, sem)
        lh, la = p2.get("lam_h"), p2.get("lam_a")
        if lh:
            g2 = blended_score_grid(lh, la, v2model.rho, sem, v2model.score_market_weight)
            v2top = [(s, float(p)) for s, p in grid_top_k_scores(g2, 8)]
            v2_o25 = _p_over25_grille(g2)
        # ⚠️ CORRIGE LE 04/10 : ce code lisait `p_h_bl` / `p_d_bl` / `p_a_bl`,
        # cles que `predict_match_v2` n'a jamais rendues (les siennes sont
        # `p_h_blend`...). Le repli prenait donc TOUJOURS le Poisson pur : le
        # 1X2 « V2 » du round ignorait le melange avec le marche du score.
        ph = p2.get("p_h_blend", p2.get("p_h_pois"))
        pd_ = p2.get("p_d_blend", p2.get("p_d_pois"))
        pa = p2.get("p_a_blend", p2.get("p_a_pois"))
    except Exception:
        pass
    # --- V5 ---
    v5top = []
    try:
        p5 = predict_match_v5(m5, team_a, team_b, oh, od, oa, extra_markets=extra_markets)
        v5top = [(s, float(p)) for s, p in (p5.get("top5_scores_enriched") or [])]
        v5_o25 = p5.get("p_over_25_blend")
        if ph is None:
            ph = p5.get("p_h_blend"); pd_ = p5.get("p_d_blend"); pa = p5.get("p_a_blend")
    except Exception:
        pass
    # --- ARBITRE MARCHÉ (Score-exact devigé) ---
    mkttop = []
    try:
        gm = market_score_grid(sem)
        if gm is not None:
            mkttop = [(s, float(p)) for s, p in grid_top_k_scores(gm, 8)]
    except Exception:
        pass
    # --- CONSENSUS : poids égaux entre les moteurs PRÉSENTS ---
    sources = [s for s in (v2top, v5top, mkttop) if s]
    w = 1.0 / len(sources) if sources else 1.0
    cons = {}
    for src in sources:
        for sc, p in src:
            cons[sc] = cons.get(sc, 0.0) + w * p
    tt = sum(cons.values()) or 1.0
    cons = {k: v / tt for k, v in cons.items()}
    ctop = sorted(cons.items(), key=lambda kv: -kv[1])[:3]
    # DOUBLE MODE (backtest 9334 OOS) : calib aide le Top-1 (+0.3pp) mais coûte au
    # Top-3 (-0.4pp) -> Top-1 = distribution CALIBRÉE ; Top-3 = distribution BRUTE.
    cons_cal = _apply_calib(cons, lg)
    top1_cal = max(cons_cal.items(), key=lambda kv: kv[1]) if cons_cal else None
    # accord = les moteurs présents s'accordent-ils sur le top-1 ?
    tops = [src[0][0] for src in sources]
    n_agree = tops.count(max(set(tops), key=tops.count)) if tops else 0
    accord = f"{n_agree}/{len(tops)}"
    if ph is None:                    # ligues sans modèle d'équipes -> 1X2 dévigé (calibré)
        inv = 1/oh + 1/od + 1/oa
        ph, pd_, pa = (1/oh)/inv, (1/od)/inv, (1/oa)/inv
    o25_pct = _over25_calib(oh, od, oa, lg)
    return {"match": f"{team_a} v {team_b}", "team_a": team_a, "team_b": team_b,
            "cotes": [oh, od, oa], "x12": [round(ph, 3), round(pd_, 3), round(pa, 3)],
            "over25_pct": o25_pct,
            # La recommandation over/under 2,5 DES MOTEURS (04/10) : V2, V5 et
            # le marche, a poids egaux -- la regle du trio pour le score.
            "ou25_moteurs": ou25_moteurs(v2_o25, v5_o25, _pct_en_proba(o25_pct)),
            "v2_top3": [(s, round(p, 3)) for s, p in v2top[:3]],
            "v5_top3": [(s, round(p, 3)) for s, p in v5top[:3]],
            "market_top3": [(s, round(p, 3)) for s, p in mkttop[:3]],
            "consensus_top3": [(s, round(p, 3)) for s, p in ctop],
            "top1_calibre": (top1_cal[0], round(top1_cal[1], 3)) if top1_cal else None,
            "confidence": round(sum(p for _, p in ctop), 3),   # masse Top-3 = concentration
            "board": market_board(extra_markets, oh, od, oa),
            "accord": accord}


def predict_own(engine, team_a, team_b, lg: str = LG, n: int = 60,
                journee=None) -> dict | None:
    """Analyse PROPRE, sans regarder les cotes : Poisson attaque/défense sur la
    forme Bet261 des deux équipes (n derniers matchs virtuels, récents pondérés).
    Si `journee` est fournie (round_info du match à venir), la forme de la SAISON
    en cours (les journee-1 derniers matchs = le cycle actuel) est calculée et
    fusionnée 50/50 dans les taux de buts. Retour None si historique insuffisant."""
    def _hist(team):
        t = team.replace("'", "''")
        return pd.read_sql(f"""SELECT e.team_a ta, r.score_a sa, r.score_b sb
            FROM events e JOIN results r ON r.event_id=e.id
            WHERE e.competition='{lg}' AND r.score_a IS NOT NULL
              AND (e.team_a='{t}' OR e.team_b='{t}')
            ORDER BY e.id DESC LIMIT {int(n)}""", engine)
    ha, hb = _hist(team_a), _hist(team_b)
    if len(ha) < 8 or len(hb) < 8:
        return None
    def _rates(df, team):
        mine = np.where(df.ta == team, df.sa, df.sb).astype(float)
        opp = np.where(df.ta == team, df.sb, df.sa).astype(float)
        w = 0.5 ** (np.arange(len(mine)) / 20.0)   # forme : le recent pese plus (demi-vie 20)
        atk = float((mine * w).sum() / w.sum())
        dfc = float((opp * w).sum() / w.sum())
        seq = "".join("V" if a > b else ("N" if a == b else "D")
                      for a, b in zip(mine[:5], opp[:5]))
        return atk, dfc, seq
    atk_a, def_a, seq_a = _rates(ha, team_a)
    atk_b, def_b, seq_b = _rates(hb, team_b)
    # forme de la SAISON en cours : chaque equipe joue 1 match/journee, donc le
    # cycle actuel = les (journee-1) derniers matchs. Fusion 50/50 des taux.
    season_a = season_b = None
    try:
        jn = int(journee) if journee is not None else None
    except (TypeError, ValueError):
        jn = None

    def _rec(df, team, k):
        d = df.head(k)
        mine = np.where(d.ta == team, d.sa, d.sb).astype(int)
        opp = np.where(d.ta == team, d.sb, d.sa).astype(int)
        v = int((mine > opp).sum()); nl = int((mine == opp).sum())
        return {"v": v, "n": nl, "d": int((mine < opp).sum()),
                "bp": int(mine.sum()), "bc": int(opp.sum()),
                "pts": 3 * v + nl, "k": len(d)}
    if jn and jn >= 5:
        k = min(jn - 1, len(ha), len(hb))
        sa_atk, sa_def, _ = _rates(ha.head(k), team_a)
        sb_atk, sb_def, _ = _rates(hb.head(k), team_b)
        atk_a, def_a = (atk_a + sa_atk) / 2, (def_a + sa_def) / 2
        atk_b, def_b = (atk_b + sb_atk) / 2, (def_b + sb_def) / 2
        season_a, season_b = _rec(ha, team_a, k), _rec(hb, team_b, k)
    mu = max((atk_a + def_a + atk_b + def_b) / 4.0, 0.2)   # niveau moyen local
    lam_a = min(max(atk_a * def_b / mu, 0.15), 6.0)
    lam_b = min(max(atk_b * def_a / mu, 0.15), 6.0)
    K = K_GRID
    grid = _grille(lam_a, lam_b, K)
    ph = float(np.tril(grid, -1).sum())      # sa > sb
    pd_ = float(np.trace(grid))
    pav = float(np.triu(grid, 1).sum())
    flat = [(f"{i}-{j}", float(grid[i, j])) for i in range(K) for j in range(K)]
    flat.sort(key=lambda kv: -kv[1])
    p_o25 = float(sum(grid[i, j] for i in range(K) for j in range(K) if i + j >= 3))
    # distribution des TOTAUX telle que Bet261 la cote : 0..5 exacts, "6" = 6 et plus
    tot = [0.0] * 7
    for i in range(K):
        for j in range(K):
            tot[min(i + j, 6)] += float(grid[i, j])
    return {"x12": [round(ph, 3), round(pd_, 3), round(pav, 3)],
            "top3": [(s, round(p, 3)) for s, p in flat[:3]],
            "lam_a": round(lam_a, 2), "lam_b": round(lam_b, 2),
            "n_a": len(ha), "n_b": len(hb), "seq_a": seq_a, "seq_b": seq_b,
            "journee": jn, "season_a": season_a, "season_b": season_b,
            "p_over25": round(p_o25, 3),
            "totals": [round(x, 4) for x in tot]}


def predict_round(engine, m5, v2model, target_local=None, lg: str = LG) -> dict:
    now = datetime.now(timezone.utc)
    up = pd.read_sql(f"""SELECT e.team_a,e.team_b,e.expected_start,o.odds_home oh,o.odds_draw od,
        o.odds_away oa,o.extra_markets,e.id ev FROM events e
        JOIN odds_snapshots o ON o.id=(SELECT MAX(id) FROM odds_snapshots WHERE event_id=e.id)
        LEFT JOIN results r ON r.event_id=e.id
        WHERE r.id IS NULL AND e.expected_start IS NOT NULL AND e.competition='{lg}'""", engine)
    if not len(up):
        return {"target": None, "rounds": [], "matches": []}
    up["es"] = pd.to_datetime(up.expected_start, utc=True)
    up = up[up.es > now - pd.Timedelta(minutes=3)]
    up["local"] = up.es.dt.tz_convert(MADA).dt.strftime("%H:%M")
    up = up.sort_values(["es", "ev"]).drop_duplicates(["team_a", "team_b", "local"])
    rounds = sorted(up.local.unique())
    if not len(rounds):
        return {"target": None, "rounds": [], "matches": []}
    target = target_local if (target_local and target_local in rounds) else rounds[0]
    ms = up[up.local == target]
    matches = [predict_one(engine, m5, v2model, r.team_a, r.team_b, r.oh, r.od, r.oa, r.extra_markets, lg)
               for r in ms.itertuples() if float(r.oh) > 1 and float(r.oa) > 1]
    return {"target": target, "rounds": rounds, "matches": matches}


def main():
    e = create_engine(load_settings().db_url)
    print("fit V5 + V2…")
    m5, v2, n = fit(e)
    tgt = sys.argv[1] if len(sys.argv) > 1 else None
    res = predict_round(e, m5, v2, tgt)
    if not res["matches"]:
        print(f"Aucun match. Rounds : {res['rounds'][:8]}"); return
    print(f"\nROUND {res['target']} Mada — TRIO V2 + V5 + MARCHÉ (fit {n})\n")
    print(f"  {'match':<26}{'1X2':<16}{'Ov2.5':>6}  {'V2':<14}{'V5':<14}{'MARCHÉ':<14}{'CONSENSUS':<14}accord")
    print("  " + "-" * 116)
    f = lambda l: " ".join(f"{s}({p*100:.0f})" for s, p in l) if l else "-"
    for m in res["matches"]:
        ph, pd_, pa = m["x12"]
        x = f"1:{ph*100:.0f} X:{pd_*100:.0f} 2:{pa*100:.0f}"
        ov = f"{m['over25_pct']:.0f}%" if m["over25_pct"] is not None else "-"
        print(f"  {m['match'][:25]:<26}{x:<16}{ov:>6}  {f(m['v2_top3']):<14}{f(m['v5_top3']):<14}"
              f"{f(m['market_top3']):<14}{f(m['consensus_top3']):<14}{m['accord']}")


if __name__ == "__main__":
    main()
