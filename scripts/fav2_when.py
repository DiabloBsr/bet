"""Quand le favori côte-2 gagne-t-il ? — CDM/ALL/POR, tous les angles temporels.

Répond aux questions : config, match PRÉCÉDENT, ADVERSAIRE, SÉRIE, et surtout
FENÊTRES (5/10 matchs) et « les deux » = y a-t-il des RÉGIMES (les victoires se
regroupent-elles dans le temps au-delà de ce que les cotes prédisent ?).

3 volets :
  A. RÉGIMES (diagnostic, PAS pariable — regarde passé+futur) :
     - autocorrélation du RÉSIDU (win − p_dévigé) aux lags 1..5 : un résidu
       autocorrélé = les sur/sous-performances s'enchaînent (régime).
     - sur-dispersion : variance observée du nb de victoires par fenêtre de 5 et 10
       vs variance de Poisson-binomial Σp(1−p). Ratio ~1 = i.i.d. ; >1 = clustering.
  B. MATCH PRÉCÉDENT du favori (résultat, était-il favori, a-t-il marqué) — pariable.
  C. ADVERSAIRE : sa série (W/L/muet/CS…) — pariable.
  Filtre pariable : ROI>0 train ET (test−1.96·IC)>0 + résidu réel−dévigé.

    python scripts/fav2_when.py
"""
from __future__ import annotations

import sqlite3
import time
from collections import defaultdict, deque

import numpy as np

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "virtual_sports.db"
LEAGUES = {"8065": "CDM", "8043": "ALL", "8044": "POR"}


def _read(code):
    q = ("SELECT e.team_a, e.team_b, o.odds_home, o.odds_draw, o.odds_away, "
         "r.score_a, r.score_b FROM events e "
         "JOIN (SELECT event_id, MIN(id) mid FROM odds_snapshots GROUP BY event_id) f "
         "  ON f.event_id = e.id "
         "JOIN odds_snapshots o ON o.id = f.mid JOIN results r ON r.event_id = e.id "
         f"WHERE e.competition = 'InstantLeague-{code}' AND r.score_a IS NOT NULL "
         "ORDER BY e.id")
    for attempt in range(7):
        try:
            c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=250)
            c.execute("PRAGMA busy_timeout=250000")
            rows = c.execute(q).fetchall()
            c.close()
            return rows
        except sqlite3.Error:
            if attempt == 6:
                raise
            time.sleep(10)
    return []


DEFS = {"W": lambda gf, ga: gf > ga, "L": lambda gf, ga: gf < ga,
        "GF": lambda gf, ga: gf > 0, "NG": lambda gf, ga: gf == 0,
        "CS": lambda gf, ga: ga == 0}


def _streak(hist_team, name):
    fn = DEFS[name]
    run = 0
    for gf, ga in reversed(hist_team):
        if fn(gf, ga):
            run += 1
        else:
            break
    return run


def main():
    # séquences par ligue pour A ; accumulateurs pariables pour B/C
    acc = defaultdict(lambda: [0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0])  # n_tr,s,ss, n_te,s,ss, wte, pte
    for code, nom in LEAGUES.items():
        rows = [r for r in _read(code)
                if all(isinstance(x, (int, float)) and 1 < x < 99 for x in (r[2], r[3], r[4]))
                and isinstance(r[5], int) and isinstance(r[6], int)]
        n = len(rows)
        cut = int(n * 0.70)
        hist = defaultdict(lambda: deque(maxlen=8))
        prev = {}                       # team -> (résultat 'W/L/D', était_favori, a_marqué)
        seq_w, seq_p = [], []           # séquence chrono des paris côte-2-fav (win, p_devig)
        for i, (ta, tb, oh, od, oa, sa, sb) in enumerate(rows):
            fs = "H" if oh <= oa else "A"
            fo = oh if fs == "H" else oa
            if 2.0 <= fo < 3.0:
                ftm = ta if fs == "H" else tb
                otm = tb if fs == "H" else ta
                inv = 1 / oh + 1 / od + 1 / oa
                p = (1 / fo) / inv
                win = int((sa > sb) if fs == "H" else (sb > sa))
                roi = win * fo - 1
                seq_w.append(win); seq_p.append(p)
                off = 0 if i < cut else 3
                conds = ["ALL"]
                # B. match précédent du FAVORI
                pf = prev.get(ftm)
                if pf:
                    conds.append(f"favprev:{pf[0]}")
                    conds.append(f"favprevFav:{int(pf[1])}")
                    conds.append(f"favprevGF:{int(pf[2])}")
                # C. série de l'ADVERSAIRE (outsider)
                for nm in DEFS:
                    r = _streak(hist[otm], nm)
                    for k in (2, 3):
                        if r >= k:
                            conds.append(f"opp:{nm}>={k}")
                # série du favori (rappel)
                for nm in DEFS:
                    r = _streak(hist[ftm], nm)
                    for k in (2, 3):
                        if r >= k:
                            conds.append(f"fav:{nm}>={k}")
                for cond in conds:
                    a = acc[(nom, cond)]
                    a[off] += 1; a[off + 1] += roi; a[off + 2] += roi * roi
                    if off == 3:
                        a[6] += win; a[7] += p
            # maj historiques APRÈS
            hist[ta].append((sa, sb)); hist[tb].append((sb, sa))
            prev[ta] = ("W" if sa > sb else "L" if sa < sb else "D", oh <= oa, sa > 0)
            prev[tb] = ("W" if sb > sa else "L" if sb < sa else "D", oa < oh, sb > 0)

        # ---- A. RÉGIMES ----
        w = np.array(seq_w, float); p = np.array(seq_p, float); r = w - p
        print(f"=== {nom} — {len(w)} paris côte-2-favori ===")
        print("  A. RÉGIMES (résidu win−p_dévigé) :")
        for lag in (1, 2, 3, 5):
            if len(r) > lag + 10:
                ac = np.corrcoef(r[:-lag], r[lag:])[0, 1]
                print(f"     autocorr lag-{lag} : {ac:+.4f}", end="   ")
        print()
        for W in (5, 10):
            m = len(w) // W
            if m < 30:
                continue
            cnt = w[:m * W].reshape(m, W).sum(1)
            pw = p[:m * W].reshape(m, W)
            vth = (pw * (1 - pw)).sum(1).mean()
            ratio = cnt.var(ddof=1) / vth
            print(f"     fenêtre {W:>2} matchs : variance obs/théo = {ratio:.3f} "
                  f"({'clustering' if ratio > 1.1 else 'i.i.d.'})")
        print()

    # ---- B/C : conditions pariables survivantes ----
    print("=== B/C — conditions PARIABLES (match préc. / adversaire / série), filtre train+test ===")
    surv = 0
    for (nom, cond), a in sorted(acc.items()):
        n_tr, s_tr, _2, n_te, s_te, ss_te, wte, pte = a
        if n_tr < 300 or n_te < 200 or cond == "ALL":
            continue
        rtr = s_tr / n_tr; rte = s_te / n_te
        var = max(ss_te / n_te - rte * rte, 0.0)
        se = 1.96 * (var ** 0.5) / np.sqrt(n_te)
        if rtr > 0 and rte - se > 0:
            surv += 1
            resid = 100 * (wte / n_te - pte / n_te)
            print(f"  <<< {nom} {cond:<22} n_te={n_te} ROItr {100*rtr:+.1f}% "
                  f"ROIte {100*rte:+.1f}% (+-{100*se:.1f}) résidu {resid:+.1f}pp")
    if not surv:
        print("  AUCUNE condition pariable ne passe train+test (match préc., adversaire, série).")
    print("\nFAV2WHEN-DONE")


if __name__ == "__main__":
    main()
