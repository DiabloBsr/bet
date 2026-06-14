"""DEMO du strategy_engine — vérification sur cas test connus."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scraper.strategy_engine import StrategyEngine, print_evaluation, label_segment


def main():
    engine = StrategyEngine()

    # ===== Cas 1 : DS West Ham (DS PEAKER +20pp) home =====
    print("\n" + "🔥" * 50)
    print("  CAS 1 : DS — West Ham home (PEAKER +20pp) vs Manchester Red")
    print("🔥" * 50)
    ev = engine.evaluate("West Ham", "Manchester Red", journee=1,
                          odds_h=2.30, odds_d=3.60, odds_a=3.10)
    print_evaluation(ev)

    # ===== Cas 2 : DS Leeds home (DS DROPPER -15pp) =====
    print("\n" + "❄️ " * 50)
    print("  CAS 2 : DS — Leeds home (DROPPER -15pp, 5% WR!) vs C. Palace")
    print("❄️ " * 50)
    ev = engine.evaluate("Leeds", "C. Palace", journee=2,
                          odds_h=5.00, odds_d=3.80, odds_a=1.65)
    print_evaluation(ev)

    # ===== Cas 3 : DS — pari TRAP home équilibré =====
    print("\n" + "⚠️ " * 50)
    print("  CAS 3 : DS — Burnley vs Sunderland (cote home 2.00 = LÉGER FAVORI TRAP)")
    print("⚠️ " * 50)
    ev = engine.evaluate("Burnley", "Sunderland", journee=3,
                          odds_h=2.00, odds_d=3.50, odds_a=3.80)
    print_evaluation(ev)

    # ===== Cas 4 : FS Spurs home (PEAKER +23pp!) =====
    print("\n" + "🔥" * 50)
    print("  CAS 4 : FS — Spurs home (PEAKER +23pp, 76% WR!) vs N. Forest")
    print("🔥" * 50)
    ev = engine.evaluate("Spurs", "N. Forest", journee=36,
                          odds_h=1.85, odds_d=4.10, odds_a=4.40)
    print_evaluation(ev)

    # ===== Cas 5 : FS Everton home (DROPPER -17pp!) =====
    print("\n" + "❄️ " * 50)
    print("  CAS 5 : FS — Everton home (DROPPER -17pp, 18% WR!) vs Manchester Blue")
    print("❄️ " * 50)
    ev = engine.evaluate("Everton", "Manchester Blue", journee=37,
                          odds_h=3.50, odds_d=3.80, odds_a=2.05)
    print_evaluation(ev)

    # ===== Cas 6 : FS away favori modéré TRAP (-43% ROI!) =====
    print("\n" + "⚠️ " * 50)
    print("  CAS 6 : FS — Wolverhampton vs A. Villa (away cote 1.75 = TRAP -43% ROI!)")
    print("⚠️ " * 50)
    ev = engine.evaluate("Wolverhampton", "A. Villa", journee=35,
                          odds_h=3.60, odds_d=3.50, odds_a=1.75)
    print_evaluation(ev)

    # ===== Cas 7 : MS_early HOME long shot (ROI +44%!) =====
    print("\n" + "🎰" * 50)
    print("  CAS 7 : MS_early — Sunderland home @5.50 vs Newcastle (LONG SHOT GOLD!)")
    print("🎰" * 50)
    ev = engine.evaluate("Sunderland", "Newcastle", journee=8,
                          odds_h=5.50, odds_d=4.20, odds_a=1.55)
    print_evaluation(ev)

    # ===== Cas 8 : Match équilibré sans signaux particuliers =====
    print("\n" + "─" * 50)
    print("  CAS 8 : MS_mid — Brighton vs Newcastle (match neutre)")
    print("─" * 50)
    ev = engine.evaluate("Brighton", "Newcastle", journee=20,
                          odds_h=1.95, odds_d=3.70, odds_a=3.70)
    print_evaluation(ev)


if __name__ == "__main__":
    main()
