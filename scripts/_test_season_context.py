"""Test du module season_context : afficher saison courante + équipes en forme."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import create_engine
from scraper.config import load_settings
from scraper.season_context import detect_current_season, compute_season_stats


def main():
    settings = load_settings()
    engine = create_engine(settings.db_url)

    since_ts, season_id = detect_current_season(engine, lookback_seasons=20)
    print(f"🌍 Saison courante détectée : ID={season_id}")
    print(f"   Début : {since_ts}")
    print()

    stats = compute_season_stats(engine, since_ts)
    print(f"📊 {len(stats)} équipes en stats")
    print()
    print(f"   {'Équipe':<22} {'n_home':<8} {'WR saison':<12} {'WR global':<12} {'Effective':<12} {'Delta':<10} {'Conf'}")
    print("   " + "-" * 95)
    rows = sorted(stats.values(), key=lambda s: -(s.wr_home_effective - s.wr_home_global))
    for s in rows:
        delta = s.wr_home_effective - s.wr_home_global
        marker = ""
        if s.season_confidence >= 0.6 and delta > 0.05:
            marker = " 🔥 FORME"
        elif s.season_confidence >= 0.6 and delta < -0.05:
            marker = " ❄️  PERTE"
        print(f"   {s.team:<22} {s.n_home_season:<8} {s.wr_home_season*100:>5.1f}%       {s.wr_home_global*100:>5.1f}%       {s.wr_home_effective*100:>5.1f}%       {delta*100:>+5.1f}pp    {s.season_confidence:.2f}{marker}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
