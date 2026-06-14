"""Re-prédiction du round 23:37 avec journée=8 forcée (MS_early au lieu de DS)."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scraper.strategy_engine import StrategyEngine, print_evaluation

JOURNEE = 8  # MS_early
MATCHES = [
    ("Fulham", "Manchester Red", 2.05, 3.67, 3.33, "0-1"),
    ("Burnley", "Leeds", 1.94, 3.30, 4.13, "1-1"),
    ("A. Villa", "Manchester Blue", 2.90, 3.60, 2.29, "0-5"),
    ("London Blues", "Newcastle", 1.96, 3.89, 3.41, "2-1"),
    ("C. Palace", "Sunderland", 1.70, 4.09, 4.41, "2-0"),
    ("London Reds", "Everton", 1.27, 5.53, 10.76, "2-3"),
    ("Wolverhampton", "Spurs", 2.63, 3.87, 2.37, "1-1"),
    ("Liverpool", "West Ham", 1.21, 6.65, 12.27, "2-0"),
    ("Brentford", "N. Forest", 1.55, 4.70, 4.94, "0-0"),
    ("Bournemouth", "Brighton", 2.49, 3.64, 2.61, "1-2"),
]


def main():
    engine = StrategyEngine()
    results = []
    total_pnl = 0.0
    wins = 0
    print(f"\n{'═'*100}")
    print(f"  🔄 REPLAY ROUND avec JOURNÉE={JOURNEE} (MS_early) — au lieu de DS")
    print(f"{'═'*100}\n")

    for ta, tb, oh, od, oa, actual_score in MATCHES:
        score_a, score_b = map(int, actual_score.split("-"))
        ft_o = "1" if score_a > score_b else ("X" if score_a == score_b else "2")
        ev = engine.evaluate(ta, tb, JOURNEE, oh, od, oa)
        print(f"┌─ {ta} vs {tb}  ({oh}/{od}/{oa})  RÉSULTAT {actual_score} ({ft_o})")
        for sig in ev.base_signals:
            print(f"│  🎯 {sig.reason}")
        for trap in ev.traps[:2]:
            print(f"│  ⚠️  {trap.reason}")
        if ev.recommended_picks:
            rp = ev.recommended_picks[0]
            won = rp["pick"] == ft_o
            pnl = (rp["cote"] - 1) if won else -1
            total_pnl += pnl
            wins += 1 if won else 0
            verdict = "✅ WIN" if won else "❌ LOSS"
            print(f"│  🟢 PICK : {rp['pick']} @{rp['cote']:.2f}  (strength {rp['strength']:.2f})  →  {verdict} {pnl:+.2f}u")
            results.append({"match": f"{ta} vs {tb}", "pick": rp["pick"], "cote": rp["cote"],
                            "won": won, "pnl": pnl})
        else:
            print(f"│  ⚪ SKIP (pas de conviction)")
        print(f"└{'─'*98}\n")

    # Summary
    print(f"{'═'*100}")
    print(f"  📊 RÉCAP MS_early (J{JOURNEE})")
    print(f"{'═'*100}")
    print(f"\n  Picks générés     : {len(results)}/{len(MATCHES)}")
    print(f"  Wins              : {wins}/{len(results)} ({wins/max(1,len(results))*100:.0f}%)")
    print(f"  P&L cumulé        : {total_pnl:+.2f}u")
    print(f"  vs DS (résultat live) : -4.04u")
    print(f"  Différence        : {total_pnl - (-4.04):+.2f}u")

    print(f"\n  📌 DÉTAIL PICKS :")
    for r in results:
        tag = "✅" if r["won"] else "❌"
        print(f"     {tag} {r['match']:<45}  {r['pick']} @{r['cote']:.2f}  → {r['pnl']:+.2f}u")


if __name__ == "__main__":
    main()
