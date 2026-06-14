"""Nettoyage complet de la BDD : backup + dédup + suppression données corrompues."""
from __future__ import annotations
import shutil
import sys
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sqlalchemy import create_engine, text
from scraper.config import load_settings


def main(dry_run=True):
    settings = load_settings()
    engine = create_engine(settings.db_url)
    db_path = Path(settings.db_url.replace("sqlite:///", ""))
    print(f"BDD : {db_path.resolve()}")

    # 1. BACKUP
    if not dry_run:
        backup_path = db_path.parent / f"virtual_sports_BACKUP_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        print(f"📦 Backup : {backup_path}")
        shutil.copy2(db_path, backup_path)
        print(f"   ✅ Backup créé ({backup_path.stat().st_size / 1024 / 1024:.1f} MB)\n")
    else:
        print("⚠️  DRY-RUN — aucune modification\n")

    actions = []

    # 2. Diagnostic AVANT nettoyage
    print("=" * 80)
    print("AVANT NETTOYAGE")
    print("=" * 80)
    with engine.connect() as conn:
        for table in ["events", "odds_snapshots", "results", "rankings_snapshots", "scrape_runs"]:
            n = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            print(f"  {table:<22} : {n:>8} lignes")

    print()
    print("=" * 80)
    print("NETTOYAGE - 6 ÉTAPES")
    print("=" * 80)

    with engine.begin() if not dry_run else engine.connect() as conn:
        # ÉTAPE 1 : Supprimer events sans odds_snapshot (inutilisables)
        n_orphan_events = conn.execute(text("""
            SELECT COUNT(*) FROM events e
            LEFT JOIN odds_snapshots o ON o.event_id = e.id
            WHERE o.id IS NULL
        """)).scalar()
        print(f"\n1️⃣  Events sans odds_snapshot      : {n_orphan_events} à supprimer")
        if not dry_run:
            # Supprimer aussi les results liés (déjà orphelins après)
            conn.execute(text("""
                DELETE FROM results
                WHERE event_id IN (
                    SELECT e.id FROM events e
                    LEFT JOIN odds_snapshots o ON o.event_id = e.id
                    WHERE o.id IS NULL
                )
            """))
            r = conn.execute(text("""
                DELETE FROM events
                WHERE id IN (
                    SELECT e.id FROM events e
                    LEFT JOIN odds_snapshots o ON o.event_id = e.id
                    WHERE o.id IS NULL
                )
            """))
            print(f"   ✅ Supprimé {r.rowcount} events")
        actions.append(("orphan_events", n_orphan_events))

        # ÉTAPE 2 : Supprimer results corrompus (finished_at >> expected_start)
        n_corrupt_results = conn.execute(text("""
            SELECT COUNT(*) FROM events e JOIN results r ON r.event_id = e.id
            WHERE e.expected_start IS NOT NULL
              AND r.finished_at IS NOT NULL
              AND ABS(JULIANDAY(r.finished_at) - JULIANDAY(e.expected_start)) > 1.0
        """)).scalar()
        print(f"\n2️⃣  Results avec finished_at corrompu (>1j décalage) : {n_corrupt_results} à supprimer")
        print(f"   → Cause : bug match_key dedup (vieux résultats attachés aux nouveaux events)")
        if not dry_run:
            r = conn.execute(text("""
                DELETE FROM results
                WHERE event_id IN (
                    SELECT e.id FROM events e JOIN results r ON r.event_id = e.id
                    WHERE e.expected_start IS NOT NULL
                      AND r.finished_at IS NOT NULL
                      AND ABS(JULIANDAY(r.finished_at) - JULIANDAY(e.expected_start)) > 1.0
                )
            """))
            print(f"   ✅ Supprimé {r.rowcount} results corrompus")
        actions.append(("corrupt_results", n_corrupt_results))

        # ÉTAPE 3 : Dédupliquer events (team_a, team_b, expected_start)
        n_dup_events = conn.execute(text("""
            SELECT COUNT(*) FROM (
                SELECT team_a, team_b, expected_start, COUNT(*) c
                FROM events
                WHERE expected_start IS NOT NULL
                GROUP BY team_a, team_b, expected_start
                HAVING c > 1
            )
        """)).scalar()
        print(f"\n3️⃣  Groupes d'events dupliqués : {n_dup_events}")
        if not dry_run:
            # Garder le plus récent (max id) pour chaque groupe
            r = conn.execute(text("""
                DELETE FROM events
                WHERE id IN (
                    SELECT e.id FROM events e
                    INNER JOIN (
                        SELECT team_a, team_b, expected_start, MAX(id) as keep_id
                        FROM events
                        WHERE expected_start IS NOT NULL
                        GROUP BY team_a, team_b, expected_start
                        HAVING COUNT(*) > 1
                    ) d ON e.team_a = d.team_a AND e.team_b = d.team_b
                       AND e.expected_start = d.expected_start
                       AND e.id != d.keep_id
                )
            """))
            print(f"   ✅ Supprimé {r.rowcount} events dupliqués")
        actions.append(("dup_events", n_dup_events))

        # ÉTAPE 4 : Supprimer events sans expected_start
        n_no_date = conn.execute(text("""
            SELECT COUNT(*) FROM events WHERE expected_start IS NULL
        """)).scalar()
        print(f"\n4️⃣  Events sans expected_start : {n_no_date}")
        if not dry_run:
            conn.execute(text("DELETE FROM results WHERE event_id IN (SELECT id FROM events WHERE expected_start IS NULL)"))
            r = conn.execute(text("DELETE FROM events WHERE expected_start IS NULL"))
            print(f"   ✅ Supprimé {r.rowcount} events")
        actions.append(("no_date", n_no_date))

        # ÉTAPE 5 : Nettoyer odds_snapshots orphelins
        n_orphan_odds = conn.execute(text("""
            SELECT COUNT(*) FROM odds_snapshots o
            LEFT JOIN events e ON e.id = o.event_id
            WHERE e.id IS NULL
        """)).scalar()
        print(f"\n5️⃣  Odds_snapshots orphelins : {n_orphan_odds}")
        if not dry_run:
            r = conn.execute(text("""
                DELETE FROM odds_snapshots
                WHERE event_id NOT IN (SELECT id FROM events)
            """))
            print(f"   ✅ Supprimé {r.rowcount} snapshots")
        actions.append(("orphan_odds", n_orphan_odds))

        # ÉTAPE 6 : Zombie scrape_runs
        n_zombie = conn.execute(text("""
            SELECT COUNT(*) FROM scrape_runs WHERE status = 'running'
        """)).scalar()
        print(f"\n6️⃣  Scrape_runs zombies (status=running) : {n_zombie}")
        if not dry_run:
            r = conn.execute(text("""
                UPDATE scrape_runs SET status='error', error_message='Zombie cleaned'
                WHERE status = 'running'
            """))
            print(f"   ✅ Marqué {r.rowcount} runs comme erreur")
        actions.append(("zombie_runs", n_zombie))

    # AFTER stats
    if not dry_run:
        print()
        print("=" * 80)
        print("APRÈS NETTOYAGE")
        print("=" * 80)
        with engine.connect() as conn:
            for table in ["events", "odds_snapshots", "results", "rankings_snapshots", "scrape_runs"]:
                n = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                print(f"  {table:<22} : {n:>8} lignes")

        # VACUUM (réduire taille BDD)
        print()
        print("⚙️  VACUUM (compactage)...")
        # SQLite needs autocommit for VACUUM
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.execute(text("VACUUM"))
        print(f"   ✅ BDD compactée. Taille finale : {db_path.stat().st_size / 1024 / 1024:.1f} MB")

    print()
    print("=" * 80)
    print("RÉSUMÉ DES ACTIONS")
    print("=" * 80)
    total = 0
    for name, n in actions:
        print(f"  {name:<25} : {n:>6}")
        total += n
    print(f"  {'TOTAL':<25} : {total:>6}")
    if dry_run:
        print()
        print("⚠️  DRY-RUN MODE. Pour exécuter vraiment, ajoutez --execute")
    return 0


if __name__ == "__main__":
    import sys as _s
    execute = "--execute" in _s.argv
    _s.exit(main(dry_run=not execute))
