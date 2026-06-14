"""PHASES 3+4 : Team strength + Pair patterns par segment.

Phase 3 : Pour chaque équipe, calculer perf home/away dans chaque segment
          → identifier qui PEAK en DS, MS, FS
Phase 4 : Pour chaque paire (a,b), identifier patterns spéciaux par segment
          → COMBO SCORE GOLD, PAIRE OR HOME/AWAY, PAIRE TRAP
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
from sqlalchemy import create_engine
from scraper.config import load_settings

SEGMENTS = [("DS", 1, 3), ("MS_early", 4, 12), ("MS_mid", 13, 25), ("MS_late", 26, 33), ("FS", 34, 38)]

def label_segment(j):
    if pd.isna(j): return None
    j = int(j)
    for name, lo, hi in SEGMENTS:
        if lo <= j <= hi: return name
    return None


def main():
    settings = load_settings()
    engine = create_engine(settings.db_url)
    df = pd.read_sql("""
        SELECT e.round_info, e.team_a, e.team_b,
               o.odds_home, o.odds_draw, o.odds_away,
               r.score_a, r.score_b, r.ht_score_a, r.ht_score_b
        FROM events e
        JOIN odds_snapshots o ON o.id = (SELECT MIN(id) FROM odds_snapshots WHERE event_id = e.id)
        JOIN results r ON r.event_id = e.id
        WHERE r.score_a IS NOT NULL AND r.ht_score_a IS NOT NULL
          AND e.round_info IS NOT NULL
    """, engine)
    df["journee"] = pd.to_numeric(df.round_info, errors="coerce")
    df["segment"] = df.journee.apply(label_segment)
    df = df[df.segment.notna()].copy()
    df["ft_o"] = np.where(df.score_a > df.score_b, "1",
                  np.where(df.score_a == df.score_b, "X", "2"))
    df["score"] = df.apply(lambda r: f"{int(r.score_a)}-{int(r.score_b)}", axis=1)
    df["total"] = df.score_a + df.score_b
    df["btts"] = ((df.score_a >= 1) & (df.score_b >= 1)).astype(int)
    df["won_home"] = (df.ft_o == "1").astype(int)
    df["won_away"] = (df.ft_o == "2").astype(int)
    df["drew"] = (df.ft_o == "X").astype(int)

    teams = sorted(set(df.team_a) | set(df.team_b))
    print(f"📊 Train : {len(df):,} matchs, {len(teams)} équipes\n")

    # ============ PHASE 3 — TEAM STRENGTH PER SEGMENT ============
    print("═" * 105)
    print("  PHASE 3 — STRENGTH PAR ÉQUIPE PAR SEGMENT (forme HOME / AWAY)")
    print("═" * 105)

    team_seg = {}
    for team in teams:
        team_seg[team] = {}
        for seg_name, _, _ in SEGMENTS:
            home = df[(df.segment == seg_name) & (df.team_a == team)]
            away = df[(df.segment == seg_name) & (df.team_b == team)]
            if len(home) < 5 or len(away) < 5:
                continue
            team_seg[team][seg_name] = {
                "n_home": int(len(home)),
                "wr_home": float(home.won_home.mean()),
                "gf_home": float(home.score_a.mean()),
                "ga_home": float(home.score_b.mean()),
                "n_away": int(len(away)),
                "wr_away": float(away.won_away.mean()),
                "gf_away": float(away.score_b.mean()),
                "ga_away": float(away.score_a.mean()),
            }

    # Compute baselines per team (toutes les saisons)
    baselines = {}
    for team in teams:
        h_all = df[df.team_a == team]
        a_all = df[df.team_b == team]
        baselines[team] = {
            "wr_home_all": float(h_all.won_home.mean()) if len(h_all) > 0 else 0,
            "wr_away_all": float(a_all.won_away.mean()) if len(a_all) > 0 else 0,
        }

    # Print : pour chaque équipe, delta par segment
    print(f"\n  {'Équipe':<22}", end="")
    for seg, _, _ in SEGMENTS:
        print(f" {seg:^16}", end="")
    print(f" {'GLOBAL':^10}")
    print(f"  {'='*22}", end="")
    for _ in SEGMENTS:
        print(" " + "="*16, end="")
    print(" " + "="*10)
    for team in teams:
        b = baselines[team]
        print(f"  {team:<22}", end="")
        for seg, _, _ in SEGMENTS:
            data = team_seg[team].get(seg)
            if data:
                delta = (data["wr_home"] - b["wr_home_all"]) * 100
                marker = "🔥" if delta >= 8 else ("❄️" if delta <= -8 else "  ")
                print(f" H{data['wr_home']*100:4.0f}%{marker} {delta:+3.0f}", end="")
            else:
                print(f" {'(n<5)':^16}", end="")
        print(f"   H={b['wr_home_all']*100:4.1f}%")

    # ============ Identifier qui PEAK en quel segment ============
    print()
    print("═" * 105)
    print("  🔥 ÉQUIPES QUI PEAK PAR SEGMENT (delta WR home ≥ +10pp vs global)")
    print("═" * 105)
    peak_per_seg = {}
    for seg_name, _, _ in SEGMENTS:
        peakers = []
        for team in teams:
            b = baselines[team]
            data = team_seg[team].get(seg_name)
            if not data: continue
            delta = data["wr_home"] - b["wr_home_all"]
            if delta >= 0.10:
                peakers.append((team, data["wr_home"], delta, data["n_home"]))
        peakers.sort(key=lambda x: -x[2])
        peak_per_seg[seg_name] = peakers[:5]
        print(f"\n  {seg_name}:")
        for team, wr, delta, n in peakers[:5]:
            print(f"    🔥 {team:<22}  WR home {wr*100:.0f}% (n={n})  Δ +{delta*100:.0f}pp")

    # ============ Identifier qui CHUTE par segment ============
    print()
    print("═" * 105)
    print("  ❄️  ÉQUIPES QUI CHUTENT PAR SEGMENT (delta WR home ≤ -10pp)")
    print("═" * 105)
    drop_per_seg = {}
    for seg_name, _, _ in SEGMENTS:
        droppers = []
        for team in teams:
            b = baselines[team]
            data = team_seg[team].get(seg_name)
            if not data: continue
            delta = data["wr_home"] - b["wr_home_all"]
            if delta <= -0.10:
                droppers.append((team, data["wr_home"], delta, data["n_home"]))
        droppers.sort(key=lambda x: x[2])
        drop_per_seg[seg_name] = droppers[:5]
        print(f"\n  {seg_name}:")
        for team, wr, delta, n in droppers[:5]:
            print(f"    ❄️  {team:<22}  WR home {wr*100:.0f}% (n={n})  Δ {delta*100:.0f}pp")

    # ============ PHASE 4 — PAIR PATTERNS PER SEGMENT ============
    print()
    print("═" * 105)
    print("  PHASE 4 — PAIRES OR PAR SEGMENT (n≥5, win rate ≥ 70%)")
    print("═" * 105)

    pair_patterns = {}
    for seg_name, _, _ in SEGMENTS:
        sub = df[df.segment == seg_name]
        pair_stats = []
        for (ta, tb), grp in sub.groupby(["team_a", "team_b"]):
            if len(grp) < 5: continue
            wr_h = grp.won_home.mean()
            wr_a = grp.won_away.mean()
            wr_x = grp.drew.mean()
            avg_g = grp.total.mean()
            modal_score = grp.score.value_counts().index[0]
            modal_rate = grp.score.value_counts().iloc[0] / len(grp)
            avg_cote_h = grp.odds_home.mean()
            roi_h = ((grp.won_home * (grp.odds_home - 1)) - (1 - grp.won_home)).mean()
            pair_stats.append({
                "pair": (ta, tb), "n": len(grp),
                "wr_h": wr_h, "wr_x": wr_x, "wr_a": wr_a,
                "avg_g": avg_g, "modal_score": modal_score, "modal_rate": modal_rate,
                "avg_cote_h": avg_cote_h, "roi_h": roi_h,
            })

        pair_patterns[seg_name] = {
            "paire_or_home": sorted([p for p in pair_stats if p["wr_h"] >= 0.7], key=lambda x: -x["roi_h"])[:15],
            "paire_or_away": sorted([p for p in pair_stats if p["wr_a"] >= 0.6], key=lambda x: -x["wr_a"])[:10],
            "paire_trap_home": sorted([p for p in pair_stats if p["wr_h"] <= 0.30 and p["avg_cote_h"] <= 2.5], key=lambda x: x["wr_h"])[:10],
            "score_combo_gold": [],
            "score_dominant": [],
        }

        # Score combo
        for p in pair_stats:
            grp = sub[(sub.team_a == p["pair"][0]) & (sub.team_b == p["pair"][1])]
            scores = grp.score.value_counts(normalize=True)
            if len(scores) >= 2:
                combo = scores.iloc[0] + scores.iloc[1]
                if combo >= 0.55:
                    pair_patterns[seg_name]["score_combo_gold"].append({
                        "pair": p["pair"], "n": p["n"],
                        "top1": scores.index[0], "r1": scores.iloc[0],
                        "top2": scores.index[1], "r2": scores.iloc[1], "combo": combo,
                    })
            # Dominant 30-44%
            if p["modal_rate"] >= 0.30 and p["modal_rate"] <= 0.44:
                pair_patterns[seg_name]["score_dominant"].append({
                    "pair": p["pair"], "n": p["n"],
                    "score": p["modal_score"], "rate": p["modal_rate"],
                })

        print(f"\n┌─ Segment {seg_name}")
        print(f"│  💎 PAIRES OR HOME ({len(pair_patterns[seg_name]['paire_or_home'])}) — top 5 :")
        for p in pair_patterns[seg_name]["paire_or_home"][:5]:
            print(f"│    {p['pair'][0]} vs {p['pair'][1]:<22}  WR {p['wr_h']*100:.0f}% (n={p['n']})  ROI {p['roi_h']*100:+.0f}%")
        print(f"│  💎 SCORE COMBO GOLD ({len(pair_patterns[seg_name]['score_combo_gold'])}) — top 5 :")
        for p in pair_patterns[seg_name]["score_combo_gold"][:5]:
            print(f"│    {p['pair'][0]} vs {p['pair'][1]:<22}  {p['top1']}+{p['top2']} = {p['combo']*100:.0f}% (n={p['n']})")
        print(f"│  ❌ PAIRE TRAP HOME ({len(pair_patterns[seg_name]['paire_trap_home'])}) — top 3 :")
        for p in pair_patterns[seg_name]["paire_trap_home"][:3]:
            print(f"│    {p['pair'][0]} vs {p['pair'][1]:<22}  WR {p['wr_h']*100:.0f}% (n={p['n']}, cote {p['avg_cote_h']:.2f})")

    # Save
    out_team = Path(__file__).parent.parent / "exports" / "strategy_phase_3_team.json"
    with open(out_team, "w", encoding="utf-8") as f:
        json.dump({
            "baselines": baselines,
            "team_per_segment": team_seg,
            "peak_per_segment": {k: [(t,wr,d,n) for t,wr,d,n in v] for k,v in peak_per_seg.items()},
            "drop_per_segment": {k: [(t,wr,d,n) for t,wr,d,n in v] for k,v in drop_per_seg.items()},
        }, f, indent=2, ensure_ascii=False)

    out_pair = Path(__file__).parent.parent / "exports" / "strategy_phase_4_pair.json"
    # Convert tuples to lists for JSON
    def serialize_pairs(d):
        return {seg: {cat: [{**p, "pair": list(p["pair"])} for p in plist] for cat, plist in segdata.items()}
                for seg, segdata in d.items()}
    with open(out_pair, "w", encoding="utf-8") as f:
        json.dump(serialize_pairs(pair_patterns), f, indent=2, ensure_ascii=False)
    print(f"\n💾 Configs sauvegardés : strategy_phase_3_team.json + strategy_phase_4_pair.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
