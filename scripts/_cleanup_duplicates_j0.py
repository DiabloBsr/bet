"""Cleanup BDD : merger les duplicates (team_a, team_b, expected_start).

Stratégie :
1. Pour chaque groupe de duplicates :
   - Garder l'event avec le round_info > 0 ET un résultat (priorité MAX)
   - Sinon, garder celui avec round_info > 0
   - Sinon, garder le plus ancien (qui a probablement les premières cotes)
2. Migrer odds_snapshots et results vers l'event gardé
3. Supprimer les autres
4. Faire le bilan
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import create_engine, text
from scraper.config import load_settings


def main():
    settings = load_settings()
    engine = create_engine(settings.db_url)

    # 1. Bilan initial
    with engine.connect() as conn:
        total_before = conn.execute(text("SELECT COUNT(*) FROM events")).scalar()
        j0_before = conn.execute(text("SELECT COUNT(*) FROM events WHERE round_info='0'")).scalar()
        dupes_before = conn.execute(text("""
            SELECT COUNT(*) FROM (
                SELECT team_a, team_b, expected_start FROM events
                WHERE expected_start IS NOT NULL
                GROUP BY team_a, team_b, expected_start
                HAVING COUNT(*) > 1
            )
        """)).scalar()

    print(f"📊 Avant cleanup : {total_before:,} events ({j0_before:,} J0, {dupes_before:,} groupes dupes)")

    # 2. Identifier les groupes de duplicates
    print("\n🔍 Identification des groupes...")
    with engine.connect() as conn:
        dupes = conn.execute(text("""
            SELECT team_a, team_b, expected_start
            FROM events
            WHERE expected_start IS NOT NULL
            GROUP BY team_a, team_b, expected_start
            HAVING COUNT(*) > 1
        """)).all()

    print(f"   {len(dupes):,} groupes à traiter\n")

    # 3. Pour chaque groupe : choisir le keeper et migrer
    merged = 0
    deleted = 0
    promoted = 0  # nb de round_info promus de 0 → vrai journée
    with engine.begin() as conn:
        for ta, tb, ts in dupes:
            rows = conn.execute(text("""
                SELECT e.id, e.round_info, e.match_key, e.first_seen_at,
                       (SELECT COUNT(*) FROM odds_snapshots o WHERE o.event_id=e.id) as n_odds,
                       (SELECT COUNT(*) FROM results r WHERE r.event_id=e.id AND r.score_a IS NOT NULL) as has_result
                FROM events e
                WHERE e.team_a=:ta AND e.team_b=:tb AND e.expected_start=:ts
                ORDER BY
                  (CASE WHEN e.round_info != '0' AND e.round_info IS NOT NULL THEN 0 ELSE 1 END),
                  (SELECT COUNT(*) FROM results r WHERE r.event_id=e.id AND r.score_a IS NOT NULL) DESC,
                  e.first_seen_at
            """), {"ta": ta, "tb": tb, "ts": ts}).all()

            if len(rows) < 2: continue
            keeper = rows[0]
            losers = rows[1:]
            keeper_id = keeper[0]
            keeper_round = keeper[1]

            # Vérifier si un des loser a un round_info > 0 alors que le keeper a 0
            if (not keeper_round or keeper_round == "0"):
                for l in losers:
                    if l[1] and l[1] != "0":
                        # Promouvoir le round_info du loser vers keeper
                        conn.execute(text("UPDATE events SET round_info=:r WHERE id=:i"),
                                      {"r": l[1], "i": keeper_id})
                        keeper_round = l[1]
                        promoted += 1
                        break

            # Vérifier si keeper a déjà un result
            keeper_has_result = conn.execute(text(
                "SELECT COUNT(*) FROM results WHERE event_id=:k"), {"k": keeper_id}).scalar() > 0

            # Migrer odds_snapshots et results des losers vers keeper
            for l in losers:
                lid = l[0]
                # Toujours migrer les odds_snapshots
                conn.execute(text("UPDATE odds_snapshots SET event_id=:k WHERE event_id=:l"),
                              {"k": keeper_id, "l": lid})
                # Pour les results : si keeper a déjà un result, supprimer celui du loser
                # sinon migrer
                if keeper_has_result:
                    conn.execute(text("DELETE FROM results WHERE event_id=:l"), {"l": lid})
                else:
                    loser_has_result = conn.execute(text(
                        "SELECT COUNT(*) FROM results WHERE event_id=:l"), {"l": lid}).scalar() > 0
                    if loser_has_result:
                        conn.execute(text("UPDATE results SET event_id=:k WHERE event_id=:l"),
                                      {"k": keeper_id, "l": lid})
                        keeper_has_result = True
                conn.execute(text("DELETE FROM events WHERE id=:l"), {"l": lid})
                deleted += 1
            merged += 1

    print(f"✅ {merged:,} groupes mergés, {deleted:,} doublons supprimés, {promoted:,} round_info promus")

    # 4. Bilan final
    with engine.connect() as conn:
        total_after = conn.execute(text("SELECT COUNT(*) FROM events")).scalar()
        j0_after = conn.execute(text("SELECT COUNT(*) FROM events WHERE round_info='0'")).scalar()
        train_after = conn.execute(text("""
            SELECT COUNT(*) FROM events e
            JOIN odds_snapshots o ON o.event_id=e.id
            JOIN results r ON r.event_id=e.id
            WHERE r.score_a IS NOT NULL AND r.ht_score_a IS NOT NULL
            AND e.round_info != '0' AND e.round_info IS NOT NULL
        """)).scalar()

    print(f"\n📊 Après cleanup :")
    print(f"   Total events    : {total_after:,} (-{total_before - total_after:,})")
    print(f"   J0 events       : {j0_after:,} (-{j0_before - j0_after:,})")
    print(f"   TRAIN exploitable (cotes + FT + HT + journée connue) : {train_after:,}")


if __name__ == "__main__":
    main()
