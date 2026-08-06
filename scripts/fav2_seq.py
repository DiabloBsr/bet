"""Séquences, rounds, H2H croisés et TRAJECTOIRE de cote — CDM/ALL/POR.

Répond à 3 questions :
  A. CYCLE : la suite des résultats (par équipe et par round) suit-elle un motif
     répétitif (période, ex. VNVDD) ? Autocorrélation par lag 1..10 du résidu
     (résultat − attendu par la cote) + stabilité sur 3 tiers de l'historique.
  B. H2H CROISÉ : quand 2 équipes se rencontrent, leurs séries (victoires…)
     combinées prédisent-elles l'issue au-delà des cotes ? (pariable)
  C. TRAJECTOIRE DE COTE (« baisse de cote ») : si la cote d'une équipe a BAISSÉ
     depuis son match précédent (le marché la favorise +), gagne-t-elle le match
     courant plus que sa cote actuelle ne le dit ? Trend sur 1, 2, 3 matchs. (pariable)

Filtre pariable : ROI>0 train ET (test−1.96·IC)>0, + résidu réel−dévigé.
Un marché calibré price déjà la trajectoire → résidu attendu ≈ 0.

    python scripts/fav2_seq.py
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
    q = ("SELECT e.expected_start, e.team_a, e.team_b, o.odds_home, o.odds_draw, "
         "o.odds_away, r.score_a, r.score_b FROM events e "
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


def _acc_report(name, acc, min_tr=300, min_te=200):
    surv = 0
    for cond, a in sorted(acc.items()):
        n_tr, s_tr, _2, n_te, s_te, ss_te, wte, pte = a
        if n_tr < min_tr or n_te < min_te:
            continue
        rtr = s_tr / n_tr; rte = s_te / n_te
        var = max(ss_te / n_te - rte * rte, 0.0)
        se = 1.96 * (var ** 0.5) / np.sqrt(n_te)
        if rtr > 0 and rte - se > 0:
            surv += 1
            print(f"  <<< {name} {cond:<26} n_te={n_te} ROItr {100*rtr:+.1f}% "
                  f"ROIte {100*rte:+.1f}% (+-{100*se:.1f}) résidu "
                  f"{100*(wte/n_te - pte/n_te):+.1f}pp", flush=True)
    return surv


def main():
    tot_surv = 0
    accB = defaultdict(lambda: [0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0])
    accC = defaultdict(lambda: [0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0])
    for code, nom in LEAGUES.items():
        rows = [r for r in _read(code)
                if all(isinstance(x, (int, float)) and 1 < x < 99 for x in (r[3], r[4], r[5]))
                and isinstance(r[6], int) and isinstance(r[7], int)]
        n = len(rows)
        cut = int(n * 0.70)
        # historiques
        res_hist = defaultdict(list)       # team -> liste résidus résultat (win_norm - p)
        odds_hist = defaultdict(lambda: deque(maxlen=5))   # team -> dernières cotes
        wstreak = defaultdict(int)         # team -> série de victoires courante
        round_hw = defaultdict(list)       # round -> [home_win 0/1]
        for i, (es, ta, tb, oh, od, oa, sa, sb) in enumerate(rows):
            inv = 1 / oh + 1 / od + 1 / oa
            ph, pa = (1 / oh) / inv, (1 / oa) / inv
            hw = int(sa > sb); aw = int(sb > sa)
            # A. résidu de résultat par équipe (pour autocorr) : P(win) dévigé
            res_hist[ta].append(hw - ph)
            res_hist[tb].append(aw - pa)
            round_hw[str(es)].append(hw)
            # C. TRAJECTOIRE DE COTE — pour chaque équipe, sa cote a-t-elle baissé ?
            for team, o_now, p_now, win in ((ta, oh, ph, hw), (tb, oa, pa, aw)):
                oh_prev = odds_hist[team]
                if len(oh_prev) >= 1 and 1.05 < o_now < 30:
                    drop1 = o_now < oh_prev[-1]           # baisse vs match précédent
                    drop2 = len(oh_prev) >= 2 and o_now < oh_prev[-1] < oh_prev[-2]
                    roi = win * o_now - 1
                    off = 0 if i < cut else 3
                    keys = []
                    if drop1:
                        keys.append("baisse1")
                    if drop2:
                        keys.append("baisse2_consecutive")
                    if drop1 and wstreak[team] >= 2:
                        keys.append("baisse1+serieW>=2")
                    for k in keys:
                        a = accC[(nom, k)]
                        a[off] += 1; a[off+1] += roi; a[off+2] += roi*roi
                        if off == 3:
                            a[6] += win; a[7] += p_now
            # B. H2H CROISÉ : parier le favori selon (série favori × série adversaire)
            fs = "H" if oh <= oa else "A"
            fo = oh if fs == "H" else oa
            ftm, otm = (ta, tb) if fs == "H" else (tb, ta)
            if 1.2 < fo < 6:
                win = hw if fs == "H" else aw
                p_fav = (ph if fs == "H" else pa)
                roi = win * fo - 1
                off = 0 if i < cut else 3
                fw = wstreak[ftm]; ow = wstreak[otm]
                keys = [f"favW>={min(fw,3)}_oppW>={min(ow,3)}"] if (fw >= 2 or ow >= 2) else []
                if fw >= 2 and ow >= 2:
                    keys.append("les2_enserieW")
                if fw >= 2 and ow == 0:
                    keys.append("favW>=2_oppfroid")
                for k in keys:
                    a = accB[(nom, k)]
                    a[off] += 1; a[off+1] += roi; a[off+2] += roi*roi
                    if off == 3:
                        a[6] += win; a[7] += p_fav
            # maj APRÈS
            odds_hist[ta].append(oh); odds_hist[tb].append(oa)
            wstreak[ta] = wstreak[ta] + 1 if hw else 0
            wstreak[tb] = wstreak[tb] + 1 if aw else 0

        # ---- A. rapport CYCLE ----
        print(f"=== {nom} — n={n} ===", flush=True)
        # autocorr du résidu de résultat par équipe (concat, en sautant les frontières)
        allr = []
        for t, seq in res_hist.items():
            if len(seq) >= 12:
                allr.append(np.array(seq))
        if allr:
            acs = {lag: [] for lag in (1, 2, 3, 4, 5, 10)}
            for seq in allr:
                for lag in acs:
                    if len(seq) > lag + 3:
                        a1, a2 = seq[:-lag], seq[lag:]
                        if a1.std() > 0 and a2.std() > 0:
                            acs[lag].append(np.corrcoef(a1, a2)[0, 1])
            print("  A. CYCLE — autocorr moyenne du résidu de résultat (par équipe) :", flush=True)
            print("    " + "  ".join(f"lag{lag}:{np.mean(v):+.4f}" for lag, v in acs.items() if v),
                  flush=True)
        # round-level : over-dispersion du nb de victoires domicile par round
        cnts = [sum(v) for v in round_hw.values() if len(v) >= 6]
        if len(cnts) >= 50:
            # théorique : chaque round ~ Binomiale ; approx variance via moyenne de taille
            sizes = [len(v) for v in round_hw.values() if len(v) >= 6]
            pbar = np.mean([np.mean(v) for v in round_hw.values() if len(v) >= 6])
            vth = np.mean(sizes) * pbar * (1 - pbar)
            print(f"    round-level : variance obs nb victoires dom/round = {np.var(cnts,ddof=1):.2f} "
                  f"vs théo {vth:.2f} -> ratio {np.var(cnts,ddof=1)/vth:.3f}", flush=True)
        print(flush=True)

    print("=== B. H2H CROISÉ (séries des 2 équipes) — survivants train+test ===", flush=True)
    tot_surv += _acc_report("B", accB)
    print("=== C. BAISSE DE COTE (trajectoire) — survivants train+test ===", flush=True)
    tot_surv += _acc_report("C", accC)
    # afficher AUSSI le résidu global de la baisse de cote (diagnostic, même si non survivant)
    print("\n  Diagnostic baisse de cote (plein test, même si −EV) :", flush=True)
    for (nom, k), a in sorted(accC.items()):
        if a[3] < 200:
            continue
        rte = a[4] / a[3]; resid = 100 * (a[6] / a[3] - a[7] / a[3])
        print(f"    {nom} {k:<24} n_te={a[3]:>6} ROIte {100*rte:+.1f}% résidu {resid:+.2f}pp", flush=True)

    if not tot_surv:
        print("\nRESULTAT : aucun survivant pariable (cycle, H2H croisé, baisse de cote).", flush=True)
    print("FAV2SEQ-DONE", flush=True)


if __name__ == "__main__":
    main()
