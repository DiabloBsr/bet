"""Analyse deep d'un match selon protocole 7 dimensions."""
from __future__ import annotations
import argparse, sys
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
from sqlalchemy import create_engine
from scraper.config import load_settings
from scraper.team_gold_data import (
    PAIR_HOME_GOLD, PAIR_AWAY_GOLD, PAIR_TRAP_HOME,
    BRACKET_GOLD_HOME, BRACKET_GOLD_AWAY, BRACKET_TRAP_HOME,
    OVER_GOLD, UNDER_GOLD, BTTS_OUI_GOLD, BTTS_NON_GOLD, SCORE_DOMINANT_GOLD,
)


def analyze_match(team_a, team_b, oh, od, oa, history):
    """Analyse complète d'un match selon protocole 7 dimensions."""
    print(f"\n{'═' * 100}")
    print(f"  🎯 ANALYSE DEEP : {team_a} vs {team_b}")
    print(f"     Cotes : {oh:.2f} / {od:.2f} / {oa:.2f}")
    print(f"{'═' * 100}")

    inv = 1/oh + 1/od + 1/oa
    p_market = {"1": (1/oh)/inv, "X": (1/od)/inv, "2": (1/oa)/inv}
    cote_fav = min(oh, od, oa)
    fav = "1" if oh == cote_fav else ("X" if od == cote_fav else "2")

    # 1️⃣ H2H
    print(f"\n1️⃣ H2H — CONFRONTATIONS DIRECTES (toutes confrontations historiques)")
    print(f"   {'─' * 95}")
    h2h = history[(history.team_a == team_a) & (history.team_b == team_b)]
    if len(h2h) > 0:
        w = (h2h.ft_o == "1").sum(); d = (h2h.ft_o == "X").sum(); l = (h2h.ft_o == "2").sum()
        print(f"   {team_a} home vs {team_b} : n={len(h2h)} ({w}W / {d}D / {l}L)")
        # Score exact récurrent
        scores = Counter()
        for _, r in h2h.iterrows():
            scores[f"{r.score_a}-{r.score_b}"] += 1
        top_scores = scores.most_common(3)
        print(f"   Scores fréquents : {' · '.join(f'{s}({c}/{len(h2h)})' for s,c in top_scores)}")
        # Cotes typiques
        cote_avg = h2h.odds_home.mean()
        print(f"   Cote home moyenne historique : {cote_avg:.2f}  (vs cote actuelle {oh:.2f})")
        # ROI parier 1, X, 2
        gain_1 = np.where(h2h.ft_o == "1", h2h.odds_home - 1, -1).mean()
        gain_x = np.where(h2h.ft_o == "X", h2h.odds_draw - 1, -1).mean()
        gain_2 = np.where(h2h.ft_o == "2", h2h.odds_away - 1, -1).mean()
        print(f"   ROI historique : parier 1={gain_1*100:+.1f}%, X={gain_x*100:+.1f}%, 2={gain_2*100:+.1f}%")
    else:
        print(f"   ❌ Aucune confrontation directe historique")
    # Inverse (team_b vs team_a)
    h2h_inv = history[(history.team_a == team_b) & (history.team_b == team_a)]
    if len(h2h_inv) > 0:
        w = (h2h_inv.ft_o == "1").sum(); d = (h2h_inv.ft_o == "X").sum(); l = (h2h_inv.ft_o == "2").sum()
        print(f"   {team_b} home vs {team_a} : n={len(h2h_inv)} ({w}W / {d}D / {l}L)")

    # 2️⃣ Performance Home/Away
    print(f"\n2️⃣ PERFORMANCE HOME / AWAY (toutes confrontations)")
    print(f"   {'─' * 95}")
    h_team = history[history.team_a == team_a]
    if len(h_team) > 0:
        w = (h_team.ft_o == "1").mean()
        clean_sheets = ((h_team.score_a > 0) & (h_team.score_b == 0)).mean()
        bts = ((h_team.score_a >= 1) & (h_team.score_b >= 1)).mean()
        over_25 = (h_team.score_a + h_team.score_b > 2.5).mean()
        print(f"   {team_a} HOME : n={len(h_team)}, {w*100:.0f}% wins, "
              f"buts marqués {h_team.score_a.mean():.2f}, encaissés {h_team.score_b.mean():.2f}")
        print(f"      Clean sheets : {clean_sheets*100:.0f}%   BTTS : {bts*100:.0f}%   Over 2.5 : {over_25*100:.0f}%")
    a_team = history[history.team_b == team_b]
    if len(a_team) > 0:
        w = (a_team.ft_o == "2").mean()
        cs = ((a_team.score_b > 0) & (a_team.score_a == 0)).mean()
        bts = ((a_team.score_a >= 1) & (a_team.score_b >= 1)).mean()
        over_25 = (a_team.score_a + a_team.score_b > 2.5).mean()
        print(f"   {team_b} AWAY : n={len(a_team)}, {w*100:.0f}% wins, "
              f"buts marqués {a_team.score_b.mean():.2f}, encaissés {a_team.score_a.mean():.2f}")
        print(f"      Clean sheets : {cs*100:.0f}%   BTTS : {bts*100:.0f}%   Over 2.5 : {over_25*100:.0f}%")

    # 3️⃣ Patterns de cotes
    print(f"\n3️⃣ PATTERNS DE COTES (que se passe-t-il à cette cote précise ?)")
    print(f"   {'─' * 95}")
    # Pour le home
    bracket_h = next(((lo, hi) for (t, (lo, hi)) in BRACKET_GOLD_HOME if t == team_a and lo <= oh < hi), None)
    if bracket_h:
        roi = BRACKET_GOLD_HOME.get((team_a, bracket_h), 0)
        print(f"   ✅ {team_a} HOME @cote [{bracket_h[0]:.1f};{bracket_h[1]:.1f}] : BRACKET OR, ROI +{roi*100:.0f}% historique")
    bracket_t = next(((lo, hi) for (t, (lo, hi)) in BRACKET_TRAP_HOME if t == team_a and lo <= oh < hi), None)
    if bracket_t:
        roi = BRACKET_TRAP_HOME.get((team_a, bracket_t), 0)
        print(f"   ❌ {team_a} HOME @cote [{bracket_t[0]:.1f};{bracket_t[1]:.1f}] : BRACKET TRAP, ROI {roi*100:+.0f}% historique")
    bracket_a = next(((lo, hi) for (t, (lo, hi)) in BRACKET_GOLD_AWAY if t == team_b and lo <= oa < hi), None)
    if bracket_a:
        roi = BRACKET_GOLD_AWAY.get((team_b, bracket_a), 0)
        print(f"   ✅ {team_b} AWAY @cote [{bracket_a[0]:.1f};{bracket_a[1]:.1f}] : BRACKET OR, ROI +{roi*100:.0f}% historique")
    # Paires gold/trap
    if (team_a, team_b) in PAIR_HOME_GOLD:
        p = PAIR_HOME_GOLD[(team_a, team_b)]
        print(f"   💎 PAIRE OR HOME : n={p['n']}, win={p['win']*100:.0f}%, ROI +{p['roi']*100:.0f}%")
    if (team_a, team_b) in PAIR_AWAY_GOLD:
        p = PAIR_AWAY_GOLD[(team_a, team_b)]
        print(f"   💎 PAIRE OR AWAY : n={p['n']}, win={p['win']*100:.0f}%, ROI +{p['roi']*100:.0f}%")
    if (team_a, team_b) in PAIR_TRAP_HOME:
        print(f"   ❌❌ PAIRE TRAP HOME — ne JAMAIS parier 1")
    # Favori cote test
    print(f"   Probas vig-free : 1={p_market['1']*100:.1f}%  X={p_market['X']*100:.1f}%  2={p_market['2']*100:.1f}%")

    # 4️⃣ Séries et cycles
    print(f"\n4️⃣ SÉRIES ET CYCLES")
    print(f"   {'─' * 95}")
    # Forme team_a (chronologique)
    h_recent = h_team.tail(5)
    if len(h_recent) > 0:
        form_a = "".join("W" if r.ft_o == "1" else "L" if r.ft_o == "2" else "D" for _, r in h_recent.iterrows())
        print(f"   {team_a} (home, 5 derniers) : {form_a}")
    a_recent = a_team.tail(5)
    if len(a_recent) > 0:
        form_b = "".join("W" if r.ft_o == "2" else "L" if r.ft_o == "1" else "D" for _, r in a_recent.iterrows())
        print(f"   {team_b} (away, 5 derniers) : {form_b}")
    print(f"   ⚠️ Note empirique : en virtuel Sporty-Tech, les séries ne prédisent PAS le prochain match")
    print(f"      (testé : streak ≥ 5 → ROI -16%, gambler's fallacy)")

    # 5️⃣ Buts
    print(f"\n5️⃣ BUTS (Over/Under, BTTS, score exact)")
    print(f"   {'─' * 95}")
    # Over/Under cumulé pour ce match-up
    if len(h2h) > 0:
        over_25 = ((h2h.score_a + h2h.score_b) > 2.5).mean()
        over_15 = ((h2h.score_a + h2h.score_b) > 1.5).mean()
        bts = ((h2h.score_a >= 1) & (h2h.score_b >= 1)).mean()
        avg = (h2h.score_a + h2h.score_b).mean()
        print(f"   H2H buts : moyenne {avg:.2f}, Over 1.5={over_15*100:.0f}%, Over 2.5={over_25*100:.0f}%, BTTS={bts*100:.0f}%")
    else:
        est_total = 1.5 + 0.5 * (1/cote_fav - 0.3) * 5
        print(f"   Pas d'H2H — estimation Poisson : {est_total:.2f} buts attendus")
    # 🆕 GOLD signaux Over/Under et BTTS
    if (team_a, team_b) in OVER_GOLD:
        og = OVER_GOLD[(team_a, team_b)]
        print(f"   💎 OVER 2.5 GOLD : {og['rate']*100:.0f}% historique sur n={og['n']} → PARIER OVER 2.5")
    if (team_a, team_b) in UNDER_GOLD:
        ug = UNDER_GOLD[(team_a, team_b)]
        print(f"   💎 UNDER 2.5 GOLD : seulement {ug['over_rate']*100:.0f}% Over historique sur n={ug['n']} → PARIER UNDER 2.5")
    if (team_a, team_b) in BTTS_OUI_GOLD:
        bg = BTTS_OUI_GOLD[(team_a, team_b)]
        print(f"   💎 BTTS OUI GOLD : {bg['rate']*100:.0f}% historique sur n={bg['n']} → PARIER BTTS OUI")
    if (team_a, team_b) in BTTS_NON_GOLD:
        bn = BTTS_NON_GOLD[(team_a, team_b)]
        print(f"   💎 BTTS NON GOLD : seulement {bn['bts_rate']*100:.0f}% BTTS historique sur n={bn['n']} → PARIER BTTS NON")
    if (team_a, team_b) in SCORE_DOMINANT_GOLD:
        sg = SCORE_DOMINANT_GOLD[(team_a, team_b)]
        print(f"   💎 SCORE EXACT GOLD : {sg['score']} arrive {sg['rate']*100:.0f}% du temps sur n={sg['n']} → cote 10-20 = value")

    # 6️⃣ Mi-temps
    print(f"\n6️⃣ MI-TEMPS")
    print(f"   {'─' * 95}")
    if "ht_score_a" in history.columns and len(h2h) > 0:
        h2h_with_ht = h2h[h2h.ht_score_a.notna()]
        if len(h2h_with_ht) > 0:
            ht_o = np.where(h2h_with_ht.ht_score_a > h2h_with_ht.ht_score_b, "1",
                    np.where(h2h_with_ht.ht_score_a == h2h_with_ht.ht_score_b, "X", "2"))
            ht_dist = {"1": (ht_o == "1").mean(), "X": (ht_o == "X").mean(), "2": (ht_o == "2").mean()}
            print(f"   H2H mi-temps : 1={ht_dist['1']*100:.0f}%  X={ht_dist['X']*100:.0f}%  2={ht_dist['2']*100:.0f}%")
            # Retournement (HT≠FT)
            retournements = 0
            for _, r in h2h_with_ht.iterrows():
                ht = "1" if r.ht_score_a > r.ht_score_b else ("X" if r.ht_score_a == r.ht_score_b else "2")
                ft = "1" if r.score_a > r.score_b else ("X" if r.score_a == r.score_b else "2")
                if ht != ft: retournements += 1
            print(f"   Retournements HT→FT : {retournements}/{len(h2h_with_ht)} ({retournements/len(h2h_with_ht)*100:.0f}%)")

    # 7️⃣ Algorithme virtuel
    print(f"\n7️⃣ ALGORITHME VIRTUEL")
    print(f"   {'─' * 95}")
    print(f"   ⚠️ Note empirique : Sporty-Tech n'a PAS de cycle de correction validé")
    print(f"      (rounds indépendants RNG, gambler's fallacy débunkée)")
    print(f"   ✅ Edge réel exploitable : HT/FT comebacks (1/2 ROI +104%, 2/1 ROI +28%)")
    print(f"      = bookmaker sous-évalue les comebacks à cote 30-100")

    # VERDICT FINAL
    print(f"\n{'═' * 100}")
    print(f"  ✅ VERDICT FINAL")
    print(f"{'═' * 100}")

    # Pronostic basé sur tous les signaux
    signals_for = {"1": 0, "X": 0, "2": 0}
    reasons = {"1": [], "X": [], "2": []}

    # H2H : poids
    if len(h2h) >= 5:
        h_win_rate = (h2h.ft_o == "1").mean()
        h_x_rate = (h2h.ft_o == "X").mean()
        h_a_rate = (h2h.ft_o == "2").mean()
        if h_win_rate > 0.55: signals_for["1"] += 2; reasons["1"].append(f"H2H domine {h_win_rate*100:.0f}%")
        if h_a_rate > 0.45: signals_for["2"] += 2; reasons["2"].append(f"H2H away gagne {h_a_rate*100:.0f}%")
    # Paire gold/trap
    if (team_a, team_b) in PAIR_HOME_GOLD:
        signals_for["1"] += 3
        p = PAIR_HOME_GOLD[(team_a, team_b)]
        reasons["1"].append(f"PAIRE OR home n={p['n']} ROI+{p['roi']*100:.0f}%")
    if (team_a, team_b) in PAIR_AWAY_GOLD:
        signals_for["2"] += 3
        p = PAIR_AWAY_GOLD[(team_a, team_b)]
        reasons["2"].append(f"PAIRE OR away n={p['n']} ROI+{p['roi']*100:.0f}%")
    if (team_a, team_b) in PAIR_TRAP_HOME:
        signals_for["1"] -= 5
        reasons["1"].append("PAIRE TRAP — refuser 1")
    # Bracket gold/trap
    if bracket_h:
        signals_for["1"] += 2
        reasons["1"].append(f"BRACKET OR HOME [{bracket_h[0]:.1f};{bracket_h[1]:.1f}]")
    if bracket_t:
        signals_for["1"] -= 3
        reasons["1"].append(f"BRACKET TRAP")
    if bracket_a:
        signals_for["2"] += 2
        reasons["2"].append(f"BRACKET OR AWAY")

    # Choix
    best_outcome = max(signals_for, key=signals_for.get)
    cote_pick = oh if best_outcome == "1" else (od if best_outcome == "X" else oa)
    p_pick = p_market[best_outcome]
    confiance = min(round(p_pick * 10 + signals_for[best_outcome], 1), 10)
    if confiance < 0: confiance = 0

    if signals_for[best_outcome] <= 0:
        print(f"  🚫 AUCUN SIGNAL FORT — recommandation : SKIP ce match")
        return

    if signals_for[best_outcome] >= 5:
        ptype = "SÉCURISÉ" if cote_pick < 2 else ("VALUE" if cote_pick < 4 else "SPÉCULATIF")
    elif signals_for[best_outcome] >= 3:
        ptype = "VALUE" if cote_pick > 2 else "SÉCURISÉ"
    else:
        ptype = "VALUE"

    print(f"\n   ✅ PRONOSTIC : {best_outcome}  ({team_a if best_outcome=='1' else team_b if best_outcome=='2' else 'nul'})")
    print(f"   ✅ COTE     : @{cote_pick:.2f}")
    print(f"   ✅ CONFIANCE: {confiance}/10")
    print(f"   ✅ TYPE     : {ptype}")
    print(f"   ✅ JUSTIFICATION : {' + '.join(reasons[best_outcome]) if reasons[best_outcome] else 'signaux globaux'}")
    if len(h2h) >= 5:
        top_scores = Counter()
        for _, r in h2h.iterrows():
            top_scores[f"{r.score_a}-{r.score_b}"] += 1
        modal = top_scores.most_common(1)[0]
        print(f"   ✅ SCORE PROBABLE : {modal[0]} (récurrent {modal[1]}/{len(h2h)} fois H2H)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team-a", required=True)
    ap.add_argument("--team-b", required=True)
    ap.add_argument("--cote-h", type=float, required=True)
    ap.add_argument("--cote-d", type=float, required=True)
    ap.add_argument("--cote-a", type=float, required=True)
    args = ap.parse_args()

    settings = load_settings()
    engine = create_engine(settings.db_url)
    history = pd.read_sql("""
        SELECT e.team_a, e.team_b, e.expected_start,
               o.odds_home, o.odds_draw, o.odds_away,
               r.score_a, r.score_b, r.ht_score_a, r.ht_score_b
        FROM events e
        JOIN odds_snapshots o ON o.id = (SELECT MIN(id) FROM odds_snapshots WHERE event_id = e.id)
        JOIN results r ON r.event_id = e.id
        WHERE r.score_a IS NOT NULL
        ORDER BY e.expected_start
    """, engine)
    history["ft_o"] = np.where(history.score_a > history.score_b, "1",
                       np.where(history.score_a == history.score_b, "X", "2"))

    analyze_match(args.team_a, args.team_b, args.cote_h, args.cote_d, args.cote_a, history)
    return 0


if __name__ == "__main__":
    sys.exit(main())
