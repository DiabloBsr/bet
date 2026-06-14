"""Look for algorithmic patterns / regularities in match outcomes.

Sections :
  1. Sequence outcome (chaine de Markov, runs, longest streak)
  2. Distribution par numero de round
  3. Distribution par equipe (qui sur-performe ?)
  4. Sequences de scores (n-grammes les plus frequents)
  5. Periodicite / autocorrelation
  6. Tests d'independance (chi2, runs test)
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from scraper.config import load_settings


def _hr(t):
    print(f"\n{'='*78}\n  {t}\n{'='*78}")


def main():
    settings = load_settings()
    engine = create_engine(settings.db_url)

    df = pd.read_sql(
        """
        SELECT e.id, e.team_a, e.team_b, e.round_info, e.first_seen_at,
               r.score_a, r.score_b, r.scrape_run_id
        FROM events e
        JOIN results r ON r.event_id = e.id
        ORDER BY r.scrape_run_id, r.id
        """,
        engine,
    )
    if df.empty:
        print("aucune donnee")
        return 1

    df["outcome"] = df.apply(
        lambda r: "1" if r.score_a > r.score_b
        else ("X" if r.score_a == r.score_b else "2"),
        axis=1,
    )
    df["btts"] = (df["score_a"] > 0) & (df["score_b"] > 0)
    df["total"] = df["score_a"] + df["score_b"]
    df["score"] = df["score_a"].astype(str) + "-" + df["score_b"].astype(str)
    df["round_int"] = df["round_info"].astype(int)

    _hr("0. ECHANTILLON CHRONOLOGIQUE")
    print(f"  n = {len(df)} matchs ordonnes par scrape_run_id (ordre de capture)")
    print(f"  rounds couverts : {df['round_int'].min()} -> {df['round_int'].max()}")
    print(f"  equipes distinctes : {len(set(df['team_a']) | set(df['team_b']))}")

    # ===============================================================
    # 1. Sequence outcome — chaine de Markov + runs
    # ===============================================================
    _hr("1. SEQUENCE OUTCOME — Markov & runs")
    seq = df["outcome"].tolist()
    print(f"\n  Distribution iid attendue (sous H0 random) :")
    counts = Counter(seq)
    for o in ("1", "X", "2"):
        p = counts[o] / len(seq)
        print(f"    P({o}) = {p:.3f}  ({counts[o]}/{len(seq)})")

    print(f"\n  CHAINE DE MARKOV — P(next | current) :")
    print(f"  Si independant, P(next|*) doit egaler P(next) global")
    trans = Counter()
    for a, b in zip(seq, seq[1:]):
        trans[(a, b)] += 1
    print(f"  {'':<4} -> {'1':>8} {'X':>8} {'2':>8}    (effectif)")
    for state in ("1", "X", "2"):
        row_total = sum(trans[(state, n)] for n in ("1", "X", "2"))
        if row_total == 0:
            continue
        rates = [trans[(state, n)] / row_total * 100 for n in ("1", "X", "2")]
        marg = [counts[n] / len(seq) * 100 for n in ("1", "X", "2")]
        deltas = [r - m for r, m in zip(rates, marg)]
        delta_str = "  ".join(f"{d:+.1f}pp" for d in deltas)
        print(f"  {state:<4} -> {rates[0]:>7.1f}% {rates[1]:>7.1f}% {rates[2]:>7.1f}%    (n={row_total}) deltas vs marg : {delta_str}")

    # Runs : longest streak
    print(f"\n  RUNS (sequences identiques consecutives) :")
    for target in ("1", "X", "2"):
        longest = current = 0
        run_lens = []
        for x in seq:
            if x == target:
                current += 1
                longest = max(longest, current)
            else:
                if current > 0:
                    run_lens.append(current)
                current = 0
        if current > 0:
            run_lens.append(current)
        avg = np.mean(run_lens) if run_lens else 0
        print(f"    {target} : plus longue serie={longest}  | nb de runs={len(run_lens)}  | run moyen={avg:.2f}")

    # Wald-Wolfowitz runs test approximation
    print(f"\n  TEST DES RUNS (Wald-Wolfowitz simplifie sur 1 vs non-1) :")
    binary = [1 if x == "1" else 0 for x in seq]
    n1 = sum(binary); n0 = len(binary) - n1
    runs = 1 + sum(1 for a, b in zip(binary, binary[1:]) if a != b)
    expected_runs = (2 * n1 * n0) / (n1 + n0) + 1
    var_runs = (2 * n1 * n0 * (2 * n1 * n0 - n1 - n0)) / (((n1 + n0) ** 2) * (n1 + n0 - 1))
    z = (runs - expected_runs) / np.sqrt(var_runs)
    print(f"    runs observes : {runs} | attendu sous H0 : {expected_runs:.1f}")
    print(f"    z-score : {z:+.2f}  (|z|>1.96 = independance rejetee a 5%)")
    if abs(z) > 1.96:
        print(f"    ==> SIGNAL : la sequence n'est PAS i.i.d., elle a une structure")
    else:
        print(f"    ==> rien d'anormal, comportement compatible avec aleatoire")

    # ===============================================================
    # 2. Distribution par numero de round
    # ===============================================================
    _hr("2. DISTRIBUTION PAR NUMERO DE ROUND")
    by_round = df.groupby("round_int").agg(
        n=("outcome", "size"),
        pct_1=("outcome", lambda s: (s == "1").mean() * 100),
        pct_X=("outcome", lambda s: (s == "X").mean() * 100),
        pct_2=("outcome", lambda s: (s == "2").mean() * 100),
        avg_goals=("total", "mean"),
        btts=("btts", lambda s: s.mean() * 100),
    ).round(1)
    print(by_round.to_string())

    # ===============================================================
    # 3. Distribution par equipe
    # ===============================================================
    _hr("3. PERFORMANCE PAR EQUIPE")
    teams = defaultdict(lambda: {"as_home": 0, "as_away": 0,
                                  "wins_home": 0, "draws_home": 0, "losses_home": 0,
                                  "wins_away": 0, "draws_away": 0, "losses_away": 0,
                                  "goals_scored": 0, "goals_conceded": 0})
    for _, r in df.iterrows():
        ta, tb = r["team_a"], r["team_b"]
        outcome = r["outcome"]
        teams[ta]["as_home"] += 1
        teams[tb]["as_away"] += 1
        teams[ta]["goals_scored"] += int(r["score_a"])
        teams[ta]["goals_conceded"] += int(r["score_b"])
        teams[tb]["goals_scored"] += int(r["score_b"])
        teams[tb]["goals_conceded"] += int(r["score_a"])
        if outcome == "1":
            teams[ta]["wins_home"] += 1
            teams[tb]["losses_away"] += 1
        elif outcome == "X":
            teams[ta]["draws_home"] += 1
            teams[tb]["draws_away"] += 1
        else:
            teams[ta]["losses_home"] += 1
            teams[tb]["wins_away"] += 1

    rows = []
    for name, s in teams.items():
        n = s["as_home"] + s["as_away"]
        wins = s["wins_home"] + s["wins_away"]
        draws = s["draws_home"] + s["draws_away"]
        losses = s["losses_home"] + s["losses_away"]
        if n == 0:
            continue
        rows.append({
            "team": name,
            "n": n,
            "W": wins, "D": draws, "L": losses,
            "win_rate": wins / n * 100,
            "goals_for": s["goals_scored"],
            "goals_against": s["goals_conceded"],
            "diff": s["goals_scored"] - s["goals_conceded"],
            "ppm": (wins * 3 + draws) / n,  # points par match
        })
    tdf = pd.DataFrame(rows).sort_values("ppm", ascending=False).round(2)
    print(tdf.to_string(index=False))

    # ===============================================================
    # 4. N-grammes de scores
    # ===============================================================
    _hr("4. TOP SCORES + 3-GRAMMES DE SCORES")
    print("  Top 10 scores :")
    for sc, c in Counter(df["score"]).most_common(10):
        print(f"    {sc:<6} : {c} ({c/len(df)*100:.1f}%)")

    print("\n  Top 3-grammes de scores consecutifs (sequence de 3 matchs successifs) :")
    score_seq = df["score"].tolist()
    trigrams = Counter()
    for a, b, c in zip(score_seq, score_seq[1:], score_seq[2:]):
        trigrams[(a, b, c)] += 1
    for tri, c in trigrams.most_common(8):
        print(f"    {tri[0]} -> {tri[1]} -> {tri[2]}  : {c}x")

    print("\n  Top 3-grammes d'outcomes consecutifs :")
    out_trigrams = Counter()
    for a, b, c in zip(seq, seq[1:], seq[2:]):
        out_trigrams[(a, b, c)] += 1
    total_tri = sum(out_trigrams.values())
    for tri, c in out_trigrams.most_common(10):
        expected = (counts[tri[0]]/len(seq)) * (counts[tri[1]]/len(seq)) * (counts[tri[2]]/len(seq)) * total_tri
        ratio = c / expected if expected > 0 else 0
        flag = " <--" if abs(ratio - 1) > 0.3 and c >= 5 else ""
        print(f"    {tri[0]}{tri[1]}{tri[2]} : {c:>3}  | attendu iid={expected:>5.1f}  | ratio={ratio:.2f}{flag}")

    # ===============================================================
    # 5. Autocorrelation
    # ===============================================================
    _hr("5. AUTOCORRELATION — y a-t-il un cycle ?")
    # Convert outcome to numeric : 1 -> +1, X -> 0, 2 -> -1
    num = np.array([1 if o == "1" else (-1 if o == "2" else 0) for o in seq], dtype=float)
    num_centered = num - num.mean()
    n = len(num_centered)
    var = (num_centered ** 2).sum() / n

    print(f"  Autocorrelation r(k) sur la sequence des outcomes (1=+1, X=0, 2=-1)")
    print(f"  Sous H0 i.i.d., r(k) approx 0 +/- 1.96/sqrt(n) = +/-{1.96/np.sqrt(n):.3f}")
    print(f"  lag  | r(lag) | significant ?")
    for lag in (1, 2, 3, 5, 7, 10, 15, 20, 30, 42):
        if lag >= n: continue
        cov = (num_centered[:-lag] * num_centered[lag:]).sum() / n
        r = cov / var if var > 0 else 0
        sig = "  <-- significant" if abs(r) > 1.96 / np.sqrt(n) else ""
        print(f"  {lag:>3}  | {r:+.3f}{sig}")

    # Same on BTTS sequence
    print(f"\n  Autocorrelation r(k) sur la sequence BTTS (1=Oui, 0=Non)")
    btts_num = df["btts"].astype(int).values.astype(float)
    btts_c = btts_num - btts_num.mean()
    var_b = (btts_c ** 2).sum() / len(btts_c)
    for lag in (1, 2, 3, 5, 10):
        if lag >= len(btts_c): continue
        cov = (btts_c[:-lag] * btts_c[lag:]).sum() / len(btts_c)
        r = cov / var_b if var_b > 0 else 0
        sig = "  <-- significant" if abs(r) > 1.96 / np.sqrt(len(btts_c)) else ""
        print(f"  {lag:>3}  | {r:+.3f}{sig}")

    # Total goals autocorr
    print(f"\n  Autocorrelation r(k) sur la sequence du total de buts")
    tg = df["total"].values.astype(float)
    tg_c = tg - tg.mean()
    var_t = (tg_c ** 2).sum() / len(tg_c)
    for lag in (1, 2, 3, 5, 10):
        if lag >= len(tg_c): continue
        cov = (tg_c[:-lag] * tg_c[lag:]).sum() / len(tg_c)
        r = cov / var_t if var_t > 0 else 0
        sig = "  <-- significant" if abs(r) > 1.96 / np.sqrt(len(tg_c)) else ""
        print(f"  {lag:>3}  | {r:+.3f}{sig}")

    # ===============================================================
    # 6. Synthese
    # ===============================================================
    _hr("6. SYNTHESE")
    print("  Si autocorrelation et runs test rejettent H0 -> moteur non i.i.d.")
    print("  Si rounds 1-42 ont memes proportions -> generation par numero de round symetrique")
    print("  Si un n-gramme depasse 1.5x l'attendu -> sequence privilegiee")

    return 0


if __name__ == "__main__":
    sys.exit(main())
