# -*- coding: utf-8 -*-
"""WF5 — AUDIT des buckets cote x segment saison du strategy engine.

Re-backtest complet sur InstantLeague-8035, cotes d'OUVERTURE (MIN(id) snapshot),
split temporel :
  - orig   : results.finished_at <  2026-06-08  (periode de calibration de COTE_EDGES)
  - recent : results.finished_at >= 2026-06-08  (hold-out forward)

Grille complete : 5 segments x 2 cotes (home/away) x 8 buckets canoniques = 80 cellules.
Pour chaque cellule : n, wins, WR, WR implicite (1/cote moyenne), edge (pp), ROI flat 1u,
z-score du ROI, par periode. Puis confrontation aux valeurs declarees dans COTE_EDGES.

Sortie : exports/wf5_bucket_audit.json (LECTURE SEULE sur la DB).
"""
import sys, json, math
sys.path.insert(0, ".")

import pandas as pd
from sqlalchemy import create_engine, text
from scraper.config import load_settings
from scraper.strategy_engine import label_segment, COTE_EDGES

SPLIT_DATE = "2026-06-08"
LEAGUE = "InstantLeague-8035"

# Buckets canoniques (memes bornes que COTE_EDGES, [min, max) )
BUCKETS = [
    ("favori_extreme_1.0_1.3", 1.00, 1.30),
    ("favori_solide_1.3_1.5",  1.30, 1.50),
    ("favori_modere_1.5_1.8",  1.50, 1.80),
    ("leger_favori_1.8_2.2",   1.80, 2.20),
    ("equilibre_2.2_2.7",      2.20, 2.70),
    ("non_favori_2.7_3.5",     2.70, 3.50),
    ("underdog_3.5_5",         3.50, 5.00),
    ("long_shot_5plus",        5.00, 50.0),
]

def bucket_of(cote: float):
    for name, lo, hi in BUCKETS:
        if lo <= cote < hi:
            return name
    return None


def main():
    eng = create_engine(load_settings().db_url)

    corr = json.load(open("exports/corrupted_events.json"))
    bad_ids = set(int(k) for k in corr["events"].keys())
    # soft_incomplete = {categorie: [ids...]} — on n'exclut que le noyau dur (473)
    # mais on note les categories soft pour info.
    soft = corr.get("soft_incomplete", {})
    print("soft_incomplete categories:", {k: len(v) for k, v in soft.items()})
    print(f"excluded corrupted ids: {len(bad_ids)}")

    # Cotes d'ouverture = snapshot MIN(id) par event. Colonnes minimales.
    q = f"""
    SELECT ev.id AS event_id,
           CAST(ev.round_info AS INT) AS journee,
           os.odds_home, os.odds_draw, os.odds_away,
           r.score_a, r.score_b,
           substr(r.finished_at, 1, 10) AS fin_date
    FROM events ev
    JOIN results r ON r.event_id = ev.id
    JOIN odds_snapshots os ON os.id = (
        SELECT MIN(os2.id) FROM odds_snapshots os2 WHERE os2.event_id = ev.id
    )
    WHERE ev.competition = '{LEAGUE}'
      AND r.score_a IS NOT NULL AND r.score_b IS NOT NULL
      AND ev.round_info GLOB '[0-9]*'
      AND CAST(ev.round_info AS INT) BETWEEN 1 AND 38
      AND os.odds_home IS NOT NULL AND os.odds_away IS NOT NULL
    """
    df = pd.read_sql(text(q), eng)
    print(f"rows loaded: {len(df)}")
    df = df[~df.event_id.isin(bad_ids)].copy()
    print(f"after corrupted exclusion: {len(df)}")

    df["segment"] = df.journee.map(label_segment)
    df["period"] = (df.fin_date >= SPLIT_DATE).map({False: "orig", True: "recent"})
    print(df.period.value_counts().to_dict())
    print(df.groupby(["period", "segment"]).size().unstack(fill_value=0))

    # Construire les paris : pour chaque match, 2 lignes (side=home, side=away)
    rows = []
    for t in df.itertuples(index=False):
        home_win = t.score_a > t.score_b
        away_win = t.score_b > t.score_a
        for side, cote, win in (("home", t.odds_home, home_win),
                                ("away", t.odds_away, away_win)):
            b = bucket_of(float(cote))
            if b is None:
                continue
            rows.append((t.period, t.segment, side, b, float(cote),
                         1 if win else 0, (cote - 1.0) if win else -1.0))
    bets = pd.DataFrame(rows, columns=["period", "segment", "side", "bucket",
                                       "cote", "win", "pnl"])
    print(f"bet-lines: {len(bets)}")

    def cell_stats(g):
        n = len(g)
        wr = g.win.mean()
        implied = (1.0 / g.cote).mean()
        roi = g.pnl.mean()
        sd = g.pnl.std(ddof=1) if n > 1 else float("nan")
        z = roi / (sd / math.sqrt(n)) if n > 1 and sd > 0 else float("nan")
        return dict(n=int(n), wins=int(g.win.sum()),
                    wr=round(wr, 4), implied=round(implied, 4),
                    edge_pp=round((wr - implied) * 100, 2),
                    roi_pct=round(roi * 100, 2),
                    z=round(z, 2) if not math.isnan(z) else None)

    grid = {}
    for (per, seg, side, buc), g in bets.groupby(["period", "segment", "side", "bucket"]):
        grid.setdefault((seg, side, buc), {})[per] = cell_stats(g)
    for key, g in bets.groupby(["segment", "side", "bucket"]):
        grid.setdefault(key, {})["full"] = cell_stats(g)

    # Confrontation aux cellules declarees dans COTE_EDGES
    declared = []
    for seg, cells in COTE_EDGES.items():
        for name, e in cells.items():
            buc = bucket_of((e["min"] + e["max"]) / 2.0)
            key = (seg, e["side"], buc)
            cell = grid.get(key, {})
            declared.append({
                "engine_name": name, "segment": seg, "side": e["side"],
                "bucket": buc, "declared_edge_pp": e["edge"] * 100,
                "declared_roi_pct": e["roi"] * 100,
                "is_trap": e["edge"] < 0,
                "orig": cell.get("orig"), "recent": cell.get("recent"),
                "full": cell.get("full"),
            })

    out = {
        "meta": {
            "league": LEAGUE, "split_date": SPLIT_DATE,
            "n_matches": int(len(df)),
            "n_orig": int((df.period == "orig").sum()),
            "n_recent": int((df.period == "recent").sum()),
            "n_corrupted_excluded": len(bad_ids),
            "note": "cotes d'ouverture (MIN snapshot id), mise flat 1u, ROI = mean(pnl)",
        },
        "declared_cells_audit": declared,
        "full_grid": [
            {"segment": k[0], "side": k[1], "bucket": k[2], **{p: v for p, v in d.items()}}
            for k, d in sorted(grid.items())
        ],
    }
    with open("exports/wf5_bucket_audit.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("written exports/wf5_bucket_audit.json")

    # ------- Rapport console -------
    print("\n================ AUDIT CELLULES DECLAREES (COTE_EDGES) ================")
    hdr = f"{'cellule':<42}{'decl.ROI':>9} | {'n_o':>4}{'ROI_o':>8}{'z_o':>6} | {'n_r':>4}{'ROI_r':>8}{'z_r':>6} | verdict"
    print(hdr); print("-" * len(hdr))
    for d in sorted(declared, key=lambda x: (x["segment"], -abs(x["declared_roi_pct"]))):
        o, r = d["orig"], d["recent"]
        def fmt(c):
            return (f"{c['n']:>4}{c['roi_pct']:>8.1f}{(c['z'] if c['z'] is not None else 0):>6.1f}"
                    if c else f"{'--':>4}{'--':>8}{'--':>6}")
        same_sign_o = o and (o["roi_pct"] * d["declared_roi_pct"] > 0)
        same_sign_r = r and (r["roi_pct"] * d["declared_roi_pct"] > 0)
        if r and r["n"] >= 100 and same_sign_r and abs(r["z"] or 0) >= 1.5:
            verdict = "REPLIQUE"
        elif r and r["n"] >= 100 and same_sign_r:
            verdict = "meme signe (faible)"
        elif r and r["n"] >= 100:
            verdict = "INVERSE"
        else:
            verdict = "n<100 recent"
        print(f"{d['segment']+'/'+d['engine_name']:<42}{d['declared_roi_pct']:>8.1f}% | {fmt(o)} | {fmt(r)} | {verdict}"
              f"{'' if same_sign_o or not o else '  [!orig deja inverse]'}")

    print("\n================ GRILLE COMPLETE (cellules n>=100 dans les 2 periodes) ================")
    hdr2 = f"{'segment':<9}{'side':<5}{'bucket':<24}{'n_o':>5}{'ROI_o':>8}{'z_o':>6}{'n_r':>5}{'ROI_r':>8}{'z_r':>6}  replication"
    print(hdr2); print("-" * len(hdr2))
    n_cells = n_both100 = n_repl = 0
    for k, d in sorted(grid.items()):
        o, r = d.get("orig"), d.get("recent")
        n_cells += 1
        if not (o and r and o["n"] >= 100 and r["n"] >= 100):
            continue
        n_both100 += 1
        sig_o = abs(o["z"] or 0) >= 2.0
        same = o["roi_pct"] * r["roi_pct"] > 0
        rep = "repl-signe" if same else "INVERSE"
        if sig_o:
            rep = ("REPLIQUE*" if same and abs(r["z"] or 0) >= 1.5 else
                   ("signe ok, z_r faible" if same else "SIG-ORIG->INVERSE"))
            if same and abs(r["z"] or 0) >= 1.5:
                n_repl += 1
        print(f"{k[0]:<9}{k[1]:<5}{k[2]:<24}{o['n']:>5}{o['roi_pct']:>8.1f}{o['z']:>6.1f}"
              f"{r['n']:>5}{r['roi_pct']:>8.1f}{r['z']:>6.1f}  {rep}")
    print(f"\ncells total={n_cells}, n>=100 both periods={n_both100}, "
          f"sig-orig(z>=2) AND replique(z_r>=1.5, meme signe)={n_repl}")


if __name__ == "__main__":
    main()
