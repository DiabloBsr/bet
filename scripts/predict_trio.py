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


def head_to_head(engine, team_a: str, team_b: str, leagues: list | None = None, n: int = 30) -> list:
    """Tous les face-à-face directs entre 2 équipes (les deux orientations), du + récent au + ancien."""
    a = str(team_a).replace("'", "''"); b = str(team_b).replace("'", "''")
    d = pd.read_sql(f"""SELECT e.competition c, e.team_a, e.team_b, e.expected_start,
        r.score_a sa, r.score_b sb FROM events e JOIN results r ON r.event_id=e.id
        WHERE r.score_a IS NOT NULL {_lg_clause(leagues)}
        AND ((e.team_a='{a}' AND e.team_b='{b}') OR (e.team_a='{b}' AND e.team_b='{a}'))
        ORDER BY e.expected_start DESC LIMIT {int(n)}""", engine)
    out = []
    for r in d.itertuples():
        es = pd.to_datetime(r.expected_start, utc=True).tz_convert(MADA)
        out.append({"date": es.strftime("%d/%m %H:%M"), "home": r.team_a, "away": r.team_b,
                    "sa": int(r.sa), "sb": int(r.sb), "tot": int(r.sa + r.sb)})
    return out


def _upcoming_df(engine, leagues=None, minutes=120, start_local=None, end_local=None):
    now = datetime.now(timezone.utc)
    up = pd.read_sql("""SELECT e.competition c, e.team_a, e.team_b, e.expected_start,
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


def predict_one(engine, m5, v2model, team_a, team_b, oh, od, oa, extra_markets=None,
                lg: str = None) -> dict:
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
    cons_cal = _apply_calib(cons, lg)
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
            "over25_pct": _over25_calib(oh, od, oa, lg),
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
