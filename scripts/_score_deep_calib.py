"""Calibration deep score exact : top 3 par paire + patterns équipe + brackets cote."""
from __future__ import annotations
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
from sqlalchemy import create_engine
from scraper.config import load_settings


def bucket(c):
    if c < 1.3: return "<1.3"
    if c < 1.5: return "<1.5"
    if c < 1.8: return "<1.8"
    if c < 2.1: return "<2.1"
    if c < 2.5: return "<2.5"
    if c < 3.0: return "<3.0"
    if c < 4.0: return "<4.0"
    if c < 6.0: return "<6.0"
    return "6+"


def main():
    settings = load_settings()
    engine = create_engine(settings.db_url)
    full = pd.read_sql("""
        SELECT e.id, e.expected_start, e.team_a, e.team_b,
               o.odds_home, o.odds_draw, o.odds_away, o.extra_markets,
               r.score_a, r.score_b
        FROM events e
        JOIN odds_snapshots o ON o.id = (SELECT MIN(id) FROM odds_snapshots WHERE event_id = e.id)
        JOIN results r ON r.event_id = e.id
        WHERE r.score_a IS NOT NULL
    """, engine)
    full = full.drop_duplicates(["team_a", "team_b", "expected_start", "score_a", "score_b"]).copy()
    full["expected_start"] = pd.to_datetime(full.expected_start, utc=True)
    full["score"] = full.apply(lambda r: f"{int(r.score_a)}-{int(r.score_b)}", axis=1)
    full = full.sort_values("expected_start").reset_index(drop=True)
    print(f"=== CALIBRATION DEEP SCORE EXACT — n={len(full)} matchs ===\n")

    # Split temporel : 75% train, 25% test
    split = int(len(full) * 0.75)
    train = full.iloc[:split]
    test = full.iloc[split:]
    print(f"Train n={len(train)}, Test n={len(test)}")
    print()

    # ============ 1. TOP 3 SCORES PAR PAIRE ============
    print("=" * 110)
    print("1️⃣  TOP 3 SCORES PAR PAIRE (n>=8 sur train)")
    print("=" * 110)
    pair_scores_top3 = {}
    for ta in train.team_a.unique():
        for tb in train.team_b.unique():
            sub = train[(train.team_a == ta) & (train.team_b == tb)]
            if len(sub) < 8: continue
            scores_counter = Counter(sub.score)
            total = len(sub)
            top3 = [(s, c / total, c) for s, c in scores_counter.most_common(3)]
            pair_scores_top3[(ta, tb)] = {"n": total, "top3": top3}

    print(f"\n  {len(pair_scores_top3)} paires avec top 3 scores calculés")

    # ============ 2. PATTERNS PAR ÉQUIPE HOME ============
    print()
    print("=" * 110)
    print("2️⃣  PATTERNS SCORES HOME (par équipe favori cote 1.3-2.0)")
    print("=" * 110)
    team_home_scores = defaultdict(Counter)
    for _, r in train.iterrows():
        if 1.3 <= r.odds_home <= 2.0:
            team_home_scores[r.team_a][r.score] += 1
    # Top 5 scores par équipe en home
    for team in sorted(team_home_scores):
        cnt = team_home_scores[team]
        total = sum(cnt.values())
        if total < 15: continue
        top5 = cnt.most_common(5)
        print(f"  {team:<22} (n={total})  :", end="")
        for s, c in top5:
            print(f"  {s}={c/total*100:.0f}%", end="")
        print()

    # ============ 3. PATTERNS PAR ÉQUIPE AWAY FAVORI ============
    print()
    print("=" * 110)
    print("3️⃣  PATTERNS SCORES AWAY FAVORI (cote 1.3-2.0)")
    print("=" * 110)
    team_away_scores = defaultdict(Counter)
    for _, r in train.iterrows():
        if 1.3 <= r.odds_away <= 2.0:
            team_away_scores[r.team_b][r.score] += 1
    for team in sorted(team_away_scores):
        cnt = team_away_scores[team]
        total = sum(cnt.values())
        if total < 15: continue
        top5 = cnt.most_common(5)
        print(f"  {team:<22} (n={total})  :", end="")
        for s, c in top5:
            print(f"  {s}={c/total*100:.0f}%", end="")
        print()

    # ============ 4. DISTRIBUTION PAR BRACKET COTE ============
    print()
    print("=" * 110)
    print("4️⃣  DISTRIBUTION SCORES PAR BRACKET COTE (cote_h, cote_a) — n>=30 par bracket")
    print("=" * 110)
    bracket_scores = defaultdict(Counter)
    for _, r in train.iterrows():
        k = (bucket(r.odds_home), bucket(r.odds_away))
        bracket_scores[k][r.score] += 1
    print(f"\n  {'Bracket (h,a)':<18} {'n':<5} {'Top 5 scores'}")
    for k, cnt in sorted(bracket_scores.items()):
        total = sum(cnt.values())
        if total < 30: continue
        top5 = cnt.most_common(5)
        scores_str = "  ".join(f"{s}({c/total*100:.0f}%)" for s, c in top5)
        print(f"  {k[0]+','+k[1]:<18} n={total:<4} {scores_str}")

    # ============ 5. VALIDATION OUT-OF-SAMPLE — TOP 3 PAR PAIRE ============
    print()
    print("=" * 110)
    print("5️⃣  VALIDATION OUT-OF-SAMPLE (test sur 25% restant)")
    print("=" * 110)

    # Pour chaque match du test, prédire le top 1 et top 3 selon la paire
    results = {"top1_hit": 0, "top3_hit": 0, "total_with_pair": 0}
    by_rate = defaultdict(lambda: {"n": 0, "hit": 0})
    for _, r in test.iterrows():
        key = (r.team_a, r.team_b)
        if key not in pair_scores_top3: continue
        results["total_with_pair"] += 1
        top3 = pair_scores_top3[key]["top3"]
        if top3[0][0] == r.score:
            results["top1_hit"] += 1
        if any(s == r.score for s, _, _ in top3):
            results["top3_hit"] += 1
        # Stats par rate de top 1
        top1_rate = top3[0][1]
        if top1_rate >= 0.50: bk = "≥50%"
        elif top1_rate >= 0.40: bk = "40-49%"
        elif top1_rate >= 0.35: bk = "35-39%"
        elif top1_rate >= 0.30: bk = "30-34%"
        elif top1_rate >= 0.25: bk = "25-29%"
        else: bk = "<25%"
        by_rate[bk]["n"] += 1
        if top3[0][0] == r.score:
            by_rate[bk]["hit"] += 1

    if results["total_with_pair"] > 0:
        print(f"\n  Sur {results['total_with_pair']} matchs test avec paire connue:")
        print(f"  Top 1 accuracy : {results['top1_hit']/results['total_with_pair']*100:.1f}%")
        print(f"  Top 3 accuracy : {results['top3_hit']/results['total_with_pair']*100:.1f}%")
        print()
        print(f"  Accuracy par rate du Top 1 :")
        for bk in ["≥50%", "40-49%", "35-39%", "30-34%", "25-29%", "<25%"]:
            s = by_rate[bk]
            if s["n"] >= 5:
                print(f"    Top 1 rate {bk:<8} n={s['n']:<4} acc réelle = {s['hit']/s['n']*100:.1f}%")

    # ============ 6. EXPORT DES NOUVEAUX SIGNAUX ============
    print()
    print("=" * 110)
    print("6️⃣  EXPORT — PAIRES SCORE EXACT VALIDÉES (rate >= 30%, n>=10)")
    print("=" * 110)
    score_gold_validated = []
    for key, data in pair_scores_top3.items():
        ta, tb = key
        if data["n"] < 10: continue
        top1_score, top1_rate, top1_count = data["top3"][0]
        if top1_rate >= 0.30:
            score_gold_validated.append({"home": ta, "away": tb, "score": top1_score,
                                          "rate": top1_rate, "n": data["n"], "count": top1_count})
    score_gold_validated.sort(key=lambda x: -x["rate"])
    print(f"\n  {len(score_gold_validated)} paires SCORE GOLD validées:")
    for p in score_gold_validated[:30]:
        print(f"  ({p['home']!r:<22}, {p['away']!r:<22})  score={p['score']!r:<6} {p['rate']*100:.1f}% ({p['count']}/{p['n']})")

    # ============ 7. TOP 2 BACKUP SCORES (pour value bets multi) ============
    print()
    print("=" * 110)
    print("7️⃣  TOP 2 BACKUP SCORES (parier 2 scores = augmenter chance)")
    print("=" * 110)
    print(f"\n  {'Paire':<48} {'Top 1':<14} {'Top 2':<14} {'Combo %'}")
    print("-" * 110)
    combo_picks = []
    for key, data in pair_scores_top3.items():
        ta, tb = key
        if data["n"] < 10: continue
        top3 = data["top3"]
        if len(top3) < 2: continue
        s1, r1, _ = top3[0]
        s2, r2, _ = top3[1]
        combo_rate = r1 + r2
        if combo_rate >= 0.50:
            combo_picks.append({"home": ta, "away": tb, "s1": s1, "r1": r1, "s2": s2, "r2": r2,
                                  "combo": combo_rate, "n": data["n"]})
    combo_picks.sort(key=lambda x: -x["combo"])
    for p in combo_picks[:25]:
        s1str = f"{p['s1']}({p['r1']*100:.0f}%)"
        s2str = f"{p['s2']}({p['r2']*100:.0f}%)"
        print(f"  {p['home']+' vs '+p['away']:<48} {s1str:<14} {s2str:<14} {p['combo']*100:.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
