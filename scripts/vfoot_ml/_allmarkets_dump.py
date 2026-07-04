"""Dataset TOUS-MARCHÉS : 21 marchés × 121 sélections, règlement + dévig + cellules.

Sorties :
  data/vfoot_ml/market_cells.csv   — cellules (market, selection, fav_band, phase) :
      n, k(hits), imp_sum(dévig), pnl_sum(hit*cote-1), odds_sum  -> calibration/ROI
  data/vfoot_ml/conjunctive_wide.csv — 1 ligne/match : probas dévig des composantes
      + cotes/probas des marchés CONJONCTIFS natifs (HT/FT, 1X2&Total, 1X2&G/NG)
      + les 4 prix du MÊME outcome 0-0  -> cohérence inter-marchés / arbitrage
Règlement partiel (NaN si indéterminable) : FTTS 1/2 quand les 2 marquent ;
Minute du 1er but par mi-temps via le score HT. 'Total de buts'='6' réglé comme 6+.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np, pandas as pd
from sqlalchemy import create_engine, text
from scraper.config import load_settings

LG = "InstantLeague-8035"
OUT = Path(__file__).resolve().parents[2] / "data" / "vfoot_ml"

eng = create_engine(load_settings().db_url)
df = pd.read_sql(text(f"""
    SELECT e.expected_start ts, e.team_a, e.team_b,
           o.odds_home oh, o.odds_draw od, o.odds_away oa, o.extra_markets xm,
           r.score_a sa, r.score_b sb, r.ht_score_a ha, r.ht_score_b hb
    FROM events e
    JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
    JOIN results r ON r.event_id=e.id
    WHERE r.ht_score_a IS NOT NULL AND e.competition='{LG}'
      AND o.odds_home>1 AND o.odds_draw>1 AND o.odds_away>1
    ORDER BY e.expected_start"""), eng)
df = df.drop_duplicates(["ts", "team_a", "team_b"]).reset_index(drop=True)
print(f"{len(df)} matchs réglés (FT+HT)")

# marchés dont les sélections forment une PARTITION (dévig proportionnel valide)
PARTITION = {"1X2", "Mi-tps 1X2", "Double Chance", "Mi-tps DC", "+/-", "G/NG",
             "G/NG equipe domicile", "G/NG equipe extérieur", "Pair/Impair",
             "HT/FT", "1X2 & Total", "1X2 & G/NG", "FTTS",
             "Minute du premier but", "Total de buts",
             "Les deux équipes marquent / 1ère mi temps",
             "Total equipe domicile", "Total equipe extérieur"}
# non-partition : Multi-Buts (chevauchants), Score exact / Mi-tps CS / 2e MT CS (tronqués)

CANON = ["1X2", "Mi-tps 1X2", "Double Chance", "Mi-tps DC", "Score exact", "Mi-tps CS",
         "+/-", "HT/FT", "Total de buts", "G/NG",
         "Les deux équipes marquent / 1ère mi temps", "1X2 & Total", "1X2 & G/NG",
         "Total equipe domicile", "Total equipe extérieur", "G/NG equipe domicile",
         "G/NG equipe extérieur", "Pair/Impair", "Minute du premier but", "FTTS",
         "Multi-Buts", "2ème mi-tps - CS"]


def canon_market(k: str):
    """Nom canonique robuste au mojibake (é -> octets exotiques)."""
    kn = k.replace("\x82", "é").replace("\xe9", "é")
    # cas domicile/extérieur : le préfixe 12 chars est identique -> distinguer explicitement
    if kn.startswith("Total equipe") or kn.startswith("Total équipe"):
        return "Total equipe domicile" if "dom" in kn else "Total equipe extérieur"
    if kn.startswith("G/NG equipe") or kn.startswith("G/NG équipe"):
        return "G/NG equipe domicile" if "dom" in kn else "G/NG equipe extérieur"
    for c in CANON:
        if kn[:12] == c[:12]:
            return c
    for c in CANON:  # fallback prefixe court
        if kn[:6] == c[:6]:
            return c
    return None


def r123(x, y):  # 1/X/2
    return "1" if x > y else ("2" if y > x else "X")


def settle(mkt, sel, sa, sb, ha, hb):
    """0/1 si réglable, None sinon."""
    tot, htot = sa + sb, ha + hb
    h2a, h2b = sa - ha, sb - hb
    ft, ht = r123(sa, sb), r123(ha, hb)
    if mkt == "1X2":
        return int(sel == ft)
    if mkt == "Mi-tps 1X2":
        return int(sel == ht)
    if mkt == "Double Chance":
        return int(ft in {"1X": "1X", "X2": "X2", "12": "12"}.get(sel, "") or ft in sel)
    if mkt == "Mi-tps DC":
        return int(ht in sel)
    if mkt == "Score exact":
        return int(sel == f"{sa}-{sb}")
    if mkt == "Mi-tps CS":
        return int(sel == f"{ha}-{hb}")
    if mkt == "2ème mi-tps - CS":
        return int(sel == f"{h2a}-{h2b}")
    if mkt == "+/-":
        return int(tot > 3.5) if sel.startswith(">") else int(tot < 3.5)
    if mkt == "Total equipe domicile":
        return int(sa > 3.5) if sel.startswith(">") else int(sa < 3.5)
    if mkt == "Total equipe extérieur":
        return int(sb > 3.5) if sel.startswith(">") else int(sb < 3.5)
    if mkt == "Total de buts":
        n = int(sel)
        return int(tot >= 6) if n == 6 else int(tot == n)
    if mkt == "G/NG":
        return int((sa > 0 and sb > 0) == (sel == "Oui"))
    if mkt == "G/NG equipe domicile":
        return int((sa > 0) == (sel == "Oui"))
    if mkt == "G/NG equipe extérieur":
        return int((sb > 0) == (sel == "Oui"))
    if mkt == "Les deux équipes marquent / 1ère mi temps":
        return int((ha > 0 and hb > 0) == (sel == "Oui"))
    if mkt == "Pair/Impair":
        return int((tot % 2 == 0) == sel.startswith("Pair"))
    if mkt == "HT/FT":
        return int(sel == f"{ht}/{ft}")
    if mkt == "1X2 & Total":
        p, t = sel.split("/")
        return int(p.strip() == ft and ((tot > 3.5) if ">" in t else (tot < 3.5)))
    if mkt == "1X2 & G/NG":
        btts = sa > 0 and sb > 0
        if sel.startswith("1 gagne et les deux"): return int(ft == "1" and btts)
        if sel.startswith("1 gagne et seulement"): return int(ft == "1" and sb == 0)
        if sel.startswith("2 gagne et les deux"): return int(ft == "2" and btts)
        if sel.startswith("2 gagne et seulement"): return int(ft == "2" and sa == 0)
        if sel.startswith("X et aucun"): return int(sa == 0 and sb == 0)
        if sel.startswith("X et les deux"): return int(ft == "X" and btts)
        return None
    if mkt == "Multi-Buts":
        if "0, 1 ou 2" in sel: return int(tot <= 2)
        if "1, 2 ou 3" in sel: return int(1 <= tot <= 3)
        if "2, 3 ou 4" in sel: return int(2 <= tot <= 4)
        if "supérieur" in sel or "sup" in sel: return int(tot > 4)
        return None
    if mkt == "FTTS":
        if sel.startswith("Pas"): return int(tot == 0)
        if tot == 0: return 0
        if sa > 0 and sb == 0: return int(sel == "1")
        if sb > 0 and sa == 0: return int(sel == "2")
        return None                     # les 2 marquent -> 1er buteur inconnu
    if mkt == "Minute du premier but":
        if sel.startswith("Pas"): return int(tot == 0)
        if tot == 0: return 0
        first_half = sel.split("-")[0] in ("1", "16", "31")
        if htot > 0:                     # 1er but en 1re MT
            return None if first_half else 0
        return 0 if first_half else None  # 1er but en 2e MT
    return None


cells = {}     # (mkt, sel, band, phase) -> [n, k, imp_sum, odds_sum, pnl_sum]
wide = []
cut_idx = len(df) // 2

for i, r in enumerate(df.itertuples()):
    phase = "train" if i < cut_idx else "test"
    try:
        xm = json.loads(r.xm) if isinstance(r.xm, str) else (r.xm or {})
    except Exception:
        xm = {}
    markets = {"1X2": {"1": r.oh, "X": r.od, "2": r.oa}}
    for k, v in (xm or {}).items():
        c = canon_market(k)
        if c and isinstance(v, dict):
            markets[c] = v
    inv = 1/r.oh + 1/r.od + 1/r.oa
    fav = max((1/r.oh)/inv, (1/r.oa)/inv)
    band = ("f<=.40" if fav <= .40 else "f<=.45" if fav <= .45 else "f<=.50" if fav <= .50
            else "f<=.57" if fav <= .57 else "f<=.65" if fav <= .65 else "f>.65")
    w = {"ts": r.ts, "sa": r.sa, "sb": r.sb, "ha": r.ha, "hb": r.hb, "phase": phase, "fav": round(fav, 4)}
    for mkt, sels in markets.items():
        valid = {s: o for s, o in sels.items() if isinstance(o, (int, float)) and 1 < o < 99.99}
        if not valid:
            continue
        tinv = sum(1/o for o in valid.values())
        # dévig seulement si la partition est COMPLÈTE (tinv ~ 1+marge) ;
        # jambe manquante/cappée -> proba brute 1/cote (juger par ROI)
        part = mkt in PARTITION and tinv >= 0.95
        for s, o in valid.items():
            imp = (1/o)/tinv if part else 1/o
            out = settle(mkt, s, r.sa, r.sb, r.ha, r.hb)
            if out is not None:
                for b in (band, "ALL"):
                    key = (mkt, s, b, phase)
                    a = cells.setdefault(key, [0, 0, 0.0, 0.0, 0.0])
                    a[0] += 1; a[1] += out; a[2] += imp; a[3] += o; a[4] += out*o - 1
            # colonnes wide : composantes + conjonctifs + prix du 0-0
            wk = None
            if mkt in ("1X2", "Mi-tps 1X2", "+/-", "G/NG", "Double Chance", "Mi-tps DC",
                       "Total de buts", "FTTS", "Pair/Impair",
                       "Les deux équipes marquent / 1ère mi temps",
                       "Total equipe domicile", "Total equipe extérieur",
                       "G/NG equipe domicile", "G/NG equipe extérieur"):
                wk = f"p|{mkt}|{s}"; w[wk] = round(imp, 5)
            if mkt in ("HT/FT", "1X2 & Total", "1X2 & G/NG", "Multi-Buts", "Minute du premier but"):
                w[f"o|{mkt}|{s}"] = o; w[f"p|{mkt}|{s}"] = round(imp, 5)
            if mkt in ("Score exact", "Mi-tps CS") and s == "0-0":
                w[f"o|{mkt}|0-0"] = o
    wide.append(w)

rows = [{"market": k[0], "selection": k[1], "fav_band": k[2], "phase": k[3],
         "n": v[0], "k": v[1], "imp_sum": round(v[2], 3), "odds_sum": round(v[3], 2),
         "pnl_sum": round(v[4], 3)} for k, v in cells.items()]
pd.DataFrame(rows).to_csv(OUT / "market_cells.csv", index=False)
W = pd.DataFrame(wide)
W.to_csv(OUT / "conjunctive_wide.csv", index=False)
nc = pd.DataFrame(rows)
print(f"market_cells.csv : {len(rows)} cellules | {nc.market.nunique()} marchés | "
      f"{nc.groupby(['market','selection']).ngroups} sélections")
print(f"conjunctive_wide.csv : {len(W)} matchs × {len(W.columns)} colonnes")
