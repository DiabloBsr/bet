"""Chasse exhaustive à une concordance +EV STABLE — A à Z, toutes ligues, en boucle.

Objectif : tester TOUS les paris possibles sous TOUTES les conditions raisonnables,
et ne retenir QUE ce qui est rentable sur le passé ET le futur (le seul filtre qui
survit au multiple-comparison — cf. le null de procédure +10.67% déjà établi).

Univers d'issues (offered odds réelles, marge incluse) :
  1X2 · Double Chance · G/NG · Pair/Impair · Total de buts · +/- · Multi-Buts
  Total dom/ext · 1X2&Total · 1X2&G/NG · Score exact (28) · Mi-tps 1X2 · HT/FT
  + dérivés : favori (min cote), outsider (max cote).

Dimensions de conditionnement (chaque re-boucle tout l'univers) :
  D0 inconditionnel
  D1 par CONFIG de cotes (triplet plancher, ex. 2-3-3)
  D2 par bande de cote du FAVORI
  D3 par SÉRIE d'équipe (les k derniers matchs : a gagné/perdu/marqué/muet/encaissé)

Filtre de survie (par cellule) :
  n_train >= 500, n_test >= 300, ROI_train > 0, (ROI_test - 1.96*SE_test) > 0.

Anti-mirage : le split chronologique train/test défait le null de procédure —
une cellule positive par hasard sur le train ne se répète pas sur le test. Tout
survivant est ré-audité (peut être un bug de prédicat, comme le +35% « supérieur à 4 »).

Sortie : survivants -> data/logs/hunt_survivors.jsonl ; résumé en stdout.
Tourne ligue par ligue (RAM ~1.3 Go) : une seule ligue en mémoire à la fois.

    python scripts/exhaustive_hunt.py [--leagues 8035,8043,...] [--min-tr 500] [--min-te 300]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import sqlite3
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import cross_market_check as cm  # noqa: E402  (prédicats de marché testés)

DB = ROOT / "data" / "virtual_sports.db"
OUT = ROOT / "data" / "logs" / "hunt_survivors.jsonl"
NOMS = {"8035": "ANG", "8036": "FRA", "8037": "ESP", "8042": "ITA", "8043": "ALL",
        "8044": "POR", "8056": "UCL", "8065": "CDM", "8060": "CAN"}


def _read(code: str):
    q = ("SELECT e.team_a, e.team_b, o.odds_home, o.odds_draw, o.odds_away, "
         "o.extra_markets, r.score_a, r.score_b, r.ht_score_a, r.ht_score_b "
         "FROM events e "
         "JOIN (SELECT event_id, MIN(id) mid FROM odds_snapshots GROUP BY event_id) f "
         "  ON f.event_id = e.id "
         "JOIN odds_snapshots o ON o.id = f.mid "
         "JOIN results r ON r.event_id = e.id "
         f"WHERE e.competition = 'InstantLeague-{code}' AND r.score_a IS NOT NULL "
         "ORDER BY e.id")
    for attempt in range(7):
        try:
            c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=250)
            c.execute("PRAGMA busy_timeout=250000")
            rows = c.execute(q).fetchall()
            c.close()
            return rows
        except sqlite3.Error as exc:
            if attempt == 6:
                raise
            print(f"    base occupee ({str(exc)[:40]}), essai {attempt+2}", flush=True)
            time.sleep(10)
    return []


def _outcomes(mk, oh, od, oa, sa, sb, ha, hb):
    """Toutes les (marche, label, cote_offerte, gagne?) d'un match."""
    out = []
    # 1X2 sec (colonnes)
    for lbl, o in (("1", oh), ("X", od), ("2", oa)):
        res = "1" if sa > sb else ("2" if sb > sa else "X")
        out.append(("1X2", lbl, o, int(lbl == res)))
    # dérivés favori / outsider (côté home/away)
    fav_i = 0 if oh <= oa else 2
    out.append(("FAV", "fav", (oh if fav_i == 0 else oa),
                int((sa > sb) if fav_i == 0 else (sb > sa))))
    out.append(("DOG", "dog", (oa if fav_i == 0 else oh),
                int((sb > sa) if fav_i == 0 else (sa > sb))))
    # marchés extra à prédicat testé
    for nom, cle, fab in cm.MARCHES:
        m = cm._market(mk, cle)
        if not isinstance(m, dict):
            continue
        for lbl, o in m.items():
            if not isinstance(o, (int, float)) or not 1 < o < 99:
                continue
            pred = fab(lbl)
            if pred is None:
                continue
            out.append((nom, str(lbl)[:24], o, int(pred(sa, sb))))
    # Score exact (partition 28)
    for k, v in (mk or {}).items():
        if str(k).replace("é", "e").startswith("Score exact") and isinstance(v, dict):
            for sc, o in v.items():
                if isinstance(o, (int, float)) and 1 < o < 99:
                    out.append(("ScoreExact", sc, o, int(sc == f"{sa}-{sb}")))
            break
    # Mi-tps 1X2 (si score mi-temps dispo)
    if ha is not None and hb is not None:
        for k, v in (mk or {}).items():
            if str(k).replace("é", "e").startswith("Mi-tps 1X2") and isinstance(v, dict):
                htr = "1" if ha > hb else ("2" if hb > ha else "X")
                for lbl, o in v.items():
                    if str(lbl).strip() in ("1", "X", "2") and isinstance(o, (int, float)) and 1 < o < 99:
                        out.append(("HT-1X2", lbl.strip(), o, int(lbl.strip() == htr)))
                break
    return out


def _config(oh, od, oa):
    def b(x):
        return min(int(x), 9)
    return f"{b(oh)}-{b(od)}-{b(oa)}"


def _fav_band(oh, oa):
    f = min(oh, oa)
    for lo, hi in [(1.0, 1.5), (1.5, 1.8), (1.8, 2.1), (2.1, 2.4), (2.4, 2.7),
                   (2.7, 3.2), (3.2, 5.0)]:
        if lo <= f < hi:
            return f"fav[{lo},{hi})"
    return "fav[5+]"


# --- séries d'équipe : état AVANT le match, mis à jour APRÈS ---------------------
STREAK_DEFS = {
    "W": lambda gf, ga: gf > ga,          # a gagné
    "L": lambda gf, ga: gf < ga,          # a perdu
    "GF": lambda gf, ga: gf > 0,          # a marqué
    "NG": lambda gf, ga: gf == 0,         # muet
    "CS": lambda gf, ga: ga == 0,         # clean sheet
    "GA": lambda gf, ga: ga > 0,          # a encaissé
}


def _streak_conds(state, side):
    """Conditions de série actives pour une équipe : 'H:NG>=3' etc."""
    conds = []
    hist = state  # deque des (gf, ga) récents, plus récent à droite
    for name, fn in STREAK_DEFS.items():
        run = 0
        for gf, ga in reversed(hist):
            if fn(gf, ga):
                run += 1
            else:
                break
        for k in (1, 2, 3, 4, 5):
            if run >= k:
                conds.append(f"{side}:{name}>={k}")
    return conds


def hunt_league(code, min_tr, min_te):
    nom = NOMS.get(code, code)
    rows = _read(code)
    rows = [r for r in rows if all(isinstance(x, (int, float)) and 1 < x < 99
                                   for x in (r[2], r[3], r[4]))
            and isinstance(r[6], int) and isinstance(r[7], int)]
    n = len(rows)
    if n < 3000:
        print(f"  {nom}: {n} matchs -> trop peu", flush=True)
        return []
    cut = int(n * 0.70)
    # accumulateurs GLISSANTS (RAM constante) : cle -> [n_tr,s_tr,ss_tr, n_te,s_te,ss_te]
    #   roi_i = win*cote-1 ; mean = s/n ; var = ss/n - mean^2
    acc = defaultdict(lambda: [0, 0.0, 0.0, 0, 0.0, 0.0])
    hist = defaultdict(lambda: deque(maxlen=6))     # team -> derniers (gf,ga)
    for i, (ta, tb, oh, od, oa, xm, sa, sb, ha, hb) in enumerate(rows):
        part = "tr" if i < cut else "te"
        try:
            mk = json.loads(xm) if isinstance(xm, str) else (xm or {})
        except Exception:
            mk = {}
        outs = _outcomes(mk, oh, od, oa, sa, sb, ha, hb)
        # conditions du match
        conds = ["D0", f"cfg:{_config(oh, od, oa)}", _fav_band(oh, oa)]
        conds += _streak_conds(hist[ta], "H")
        conds += _streak_conds(hist[tb], "A")
        off = 0 if part == "tr" else 3
        for nomk, lbl, o, win in outs:
            roi = win * o - 1
            base = f"{nomk}|{lbl}"
            for cond in conds:
                a = acc[(base, cond)]
                a[off] += 1
                a[off + 1] += roi
                a[off + 2] += roi * roi
        # maj des séries APRÈS le match
        hist[ta].append((sa, sb))
        hist[tb].append((sb, sa))

    survivors = []
    for (base, cond), a in acc.items():
        n_tr, s_tr, _ss_tr, n_te, s_te, ss_te = a
        if n_tr < min_tr or n_te < min_te:
            continue
        rtr = s_tr / n_tr
        if rtr <= 0:
            continue
        rte = s_te / n_te
        var = max(ss_te / n_te - rte * rte, 0.0)
        se = 1.96 * (var ** 0.5) / np.sqrt(n_te)
        if rte - se > 0:
            survivors.append({"league": nom, "bet": base, "cond": cond,
                              "n_tr": int(n_tr), "n_te": int(n_te),
                              "roi_tr": round(100 * rtr, 2), "roi_te": round(100 * rte, 2),
                              "ci95": round(100 * se, 2)})
    survivors.sort(key=lambda s: -(s["roi_te"] - s["ci95"]))
    print(f"  {nom}: {n} matchs, {len(acc)} cellules testees -> {len(survivors)} survivant(s) "
          f"train+test", flush=True)
    return survivors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leagues", type=str,
                    default="8060,8035,8043,8065,8044,8036,8037,8042,8056")
    ap.add_argument("--min-tr", type=int, default=500)
    ap.add_argument("--min-te", type=int, default=300)
    args = ap.parse_args()
    codes = [x.strip() for x in args.leagues.split(",") if x.strip()]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("", encoding="utf-8")
    print(f"CHASSE EXHAUSTIVE — {len(codes)} ligues, filtre ROI>0 train ET (test-IC95)>0", flush=True)
    total, tested = [], 0
    for code in codes:
        surv = hunt_league(code, args.min_tr, args.min_te)
        with OUT.open("a", encoding="utf-8") as fh:
            for s in surv:
                fh.write(json.dumps(s, ensure_ascii=False) + "\n")
        total += surv

    print("\n" + "=" * 78, flush=True)
    if not total:
        print("RESULTAT : AUCUN survivant. Mur total — aucun pari +EV stable, "
              "aucune ligue, aucune condition.", flush=True)
    else:
        total.sort(key=lambda s: -(s["roi_te"] - s["ci95"]))
        print(f"RESULTAT : {len(total)} cellule(s) survivante(s) train+test — A RE-AUDITER "
              "(prédicat ? bug ? bruit multiple-comparison résiduel ?) :", flush=True)
        for s in total[:40]:
            print(f"  {s['league']:<4} {s['bet']:<28} {s['cond']:<14} "
                  f"n_te={s['n_te']:>5} ROItr {s['roi_tr']:+.1f}% ROIte {s['roi_te']:+.1f}% "
                  f"(+-{s['ci95']:.1f})", flush=True)
    print(f"\nsurvivants -> {OUT}", flush=True)
    print("HUNT-DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
