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
import sys, json
from datetime import datetime, timedelta, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sqlalchemy import create_engine
from scraper.config import load_settings
from scraper.predictor_v2 import (fit_model_v2, predict_match_v2, blended_score_grid,
                                  grid_top_k_scores, market_score_grid)
from scraper.predictor_v5 import fit_model_v5, predict_match_v5
from scraper.market_inversion import exact_invert_1x2, apply_sim_deviations

MADA = timezone(timedelta(hours=3))
LG = "InstantLeague-8035"

_CALIB = None
try:
    _cp = Path(__file__).resolve().parents[1] / "data" / "vfoot_ml" / "score_calibration.json"
    if _cp.exists():
        _CALIB = np.asarray(json.loads(_cp.read_text(encoding="utf-8"))["correction"], float)
except Exception:
    _CALIB = None


def _apply_calib(d: dict) -> dict:
    """Applique la table de calibration 7x7 à une distribution {score: p}, renormalise."""
    if _CALIB is None or not d:
        return d
    out = {}
    for sc, p in d.items():
        try:
            h, a = map(int, sc.split("-"))
            f = float(_CALIB[h][a]) if (0 <= h < 7 and 0 <= a < 7) else 1.0
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


def _over25_calib(oh, od, oa):
    try:
        lh, la = exact_invert_1x2(oh, od, oa)
        g = np.asarray(apply_sim_deviations(lh, la, "cells"), float)[:7, :7]; g /= g.sum()
        if _CALIB is not None:
            g = g * _CALIB; g /= g.sum()
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


def predict_one(engine, m5, v2model, team_a, team_b, oh, od, oa, extra_markets=None) -> dict:
    oh, od, oa = float(oh), float(od), float(oa)
    sem = _sem(extra_markets)
    # --- V2 (grille blendée) ---
    v2top = []
    ph = pd_ = pa = None
    try:
        p2 = predict_match_v2(v2model, team_a, team_b, oh, od, oa, sem)
        lh, la = p2.get("lam_h"), p2.get("lam_a")
        if lh:
            g2 = blended_score_grid(lh, la, v2model.rho, sem, v2model.score_market_weight)
            v2top = [(s, float(p)) for s, p in grid_top_k_scores(g2, 8)]
        ph = p2.get("p_h_bl", p2.get("p_h_pois")); pd_ = p2.get("p_d_bl", p2.get("p_d_pois"))
        pa = p2.get("p_a_bl", p2.get("p_a_pois"))
    except Exception:
        pass
    # --- V5 ---
    v5top = []
    try:
        p5 = predict_match_v5(m5, team_a, team_b, oh, od, oa, extra_markets=extra_markets)
        v5top = [(s, float(p)) for s, p in (p5.get("top5_scores_enriched") or [])]
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
    cons_cal = _apply_calib(cons)
    top1_cal = max(cons_cal.items(), key=lambda kv: kv[1]) if cons_cal else None
    # accord = les moteurs présents s'accordent-ils sur le top-1 ?
    tops = [src[0][0] for src in sources]
    n_agree = tops.count(max(set(tops), key=tops.count)) if tops else 0
    accord = f"{n_agree}/{len(tops)}"
    if ph is None:                    # ligues sans modèle d'équipes -> 1X2 dévigé (calibré)
        inv = 1/oh + 1/od + 1/oa
        ph, pd_, pa = (1/oh)/inv, (1/od)/inv, (1/oa)/inv
    return {"match": f"{team_a} v {team_b}", "team_a": team_a, "team_b": team_b,
            "cotes": [oh, od, oa], "x12": [round(ph, 3), round(pd_, 3), round(pa, 3)],
            "over25_pct": _over25_calib(oh, od, oa),
            "v2_top3": [(s, round(p, 3)) for s, p in v2top[:3]],
            "v5_top3": [(s, round(p, 3)) for s, p in v5top[:3]],
            "market_top3": [(s, round(p, 3)) for s, p in mkttop[:3]],
            "consensus_top3": [(s, round(p, 3)) for s, p in ctop],
            "top1_calibre": (top1_cal[0], round(top1_cal[1], 3)) if top1_cal else None,
            "confidence": round(sum(p for _, p in ctop), 3),   # masse Top-3 = concentration
            "board": market_board(extra_markets, oh, od, oa),
            "accord": accord}


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
    matches = [predict_one(engine, m5, v2model, r.team_a, r.team_b, r.oh, r.od, r.oa, r.extra_markets)
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
