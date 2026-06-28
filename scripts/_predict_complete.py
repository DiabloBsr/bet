"""Prédiction COMPLÈTE d'un round (ou [HH:MM]) — chemin validé (inversion -> lam ->
grille sim), SANS gold-data leaky. Par match : 1X2 dévigé + cote offerte, lam_h/a/tot,
loi du total (top-3 + modal + zone edge), BTTS, score exact V5/V2/V6/ENSEMBLE, et
détection de VALUE (grosses cotes seulement quand EV>0 vs cote offerte).
Usage: ./.venv/Scripts/python.exe scripts/_predict_complete.py [HH:MM]
"""
from __future__ import annotations
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sqlalchemy import create_engine
from scraper.config import load_settings
from scraper.predictor_v5 import fit_model_v5, predict_match_v5
from scraper.score_predictor_v2 import ScorePredictorV2
from scraper.score_predictor_v6 import predict_score_v6
from scraper.score_ensemble import ensemble_from_raw
from scraper.journee_inference import infer_current_journee
from scraper.market_inversion import (
    invert_markets, apply_sim_deviations, total_distribution, devig,
    total_buts_odds, score_exact_odds, parse_extra_markets,
)

MG = timezone(timedelta(hours=3))
COMMON = ["0-0","1-0","0-1","1-1","2-0","0-2","2-1","1-2","2-2","3-0","0-3","3-1","1-3"]
# seuils edge totaux validés (3-fold)
UNDER_LAM = 2.45   # lam_tot < 2.45 -> Under 3.5 (~76%)
OVER_LAM  = 3.13   # lam_tot >= 3.13 -> Over 2.5 (~72%)
EV_MIN = 0.03      # value = EV > +3% vs cote offerte


def f3(lst):
    return " ".join(f"{s}({p*100:.0f}%)" for s, p in lst[:3]) if lst else "-"


def main():
    s = load_settings(); e = create_engine(s.db_url); now = datetime.now(timezone.utc)
    hist = pd.read_sql("""SELECT e.team_a,e.team_b,o.odds_home,o.odds_draw,o.odds_away,
        r.score_a,r.score_b,r.ht_score_a,r.ht_score_b FROM events e
        JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
        JOIN results r ON r.ht_score_a IS NOT NULL AND r.event_id=e.id
        WHERE e.competition='InstantLeague-8035'""", e)
    m5 = fit_model_v5(hist, ht_history=hist.copy(), engine=e, form_alpha=0.0)
    v2 = ScorePredictorV2(e)
    up = pd.read_sql("""SELECT e.team_a,e.team_b,e.expected_start,e.round_info,o.odds_home oh,
        o.odds_draw od,o.odds_away oa,o.extra_markets,e.id ev FROM events e
        JOIN odds_snapshots o ON o.id=(SELECT MAX(id) FROM odds_snapshots WHERE event_id=e.id)
        LEFT JOIN results r ON r.event_id=e.id
        WHERE r.id IS NULL AND e.expected_start IS NOT NULL AND e.competition='InstantLeague-8035'""", e)
    up["es"] = pd.to_datetime(up.expected_start, utc=True)
    up = up[up.es > now - pd.Timedelta(minutes=3)]
    up["local"] = up.es.dt.tz_convert(MG).dt.strftime("%H:%M")
    up = up.sort_values(["es", "ev"]).drop_duplicates(["team_a", "team_b", "local"])
    rounds = sorted(up.local.unique())
    if not rounds:
        print("Aucun round futur en base — relance le scraper."); return
    TARGET = sys.argv[1] if len(sys.argv) > 1 else (rounds[1] if len(rounds) > 1 else rounds[0])
    if TARGET not in rounds:
        print(f"Round {TARGET} absent. Rounds dispo : {rounds}"); return
    ms = up[up.local == TARGET]
    print(f"now {now.astimezone(MG):%H:%M} | ROUND {TARGET} — prédiction complète ({len(ms)} matchs)\n")

    summary = {"under": [], "over": [], "3buts": [], "btts_oui": [], "btts_non": [], "value": []}
    for r in ms.itertuples():
        oh, od, oa = float(r.oh), float(r.od), float(r.oa)
        if oh <= 1 or oa <= 1:
            continue
        q1, qX, q2 = devig(oh, od, oa)
        inv = invert_markets(oh, od, oa, r.extra_markets)
        g = apply_sim_deviations(inv.lam_h, inv.lam_a, "cells")
        td = total_distribution(g)
        lt = inv.lam_h + inv.lam_a
        mt = int(td.argmax())
        btts = float(g[1:, 1:].sum())
        # 1X2 issue
        fav = "1 (dom)" if q1 > q2 else "2 (ext)"
        favp = max(q1, q2)
        # totaux top-3
        tot_order = sorted(range(len(td)), key=lambda k: -td[k])[:3]
        tot_str = " ".join(f"{'6+' if k==6 else k}({td[k]*100:.0f}%)" for k in tot_order)
        # zone edge
        if lt < UNDER_LAM:
            zone = "⬇ Under 3.5 (edge ~76%)"; summary["under"].append((f"{r.team_a} v {r.team_b}", lt))
        elif lt >= OVER_LAM:
            zone = "⬆ Over 2.5 (edge ~72%)"; summary["over"].append((f"{r.team_a} v {r.team_b}", lt))
        elif mt == 3 and 2.5 <= lt < 3.05:
            zone = "🎯 total 3 (modal)"; summary["3buts"].append((f"{r.team_a} v {r.team_b}", lt))
        else:
            zone = f"total modal {mt}"
        if btts >= 0.58:
            summary["btts_oui"].append((f"{r.team_a} v {r.team_b}", btts))
        elif btts <= 0.45:
            summary["btts_non"].append((f"{r.team_a} v {r.team_b}", btts))
        # scores
        p5 = predict_match_v5(m5, r.team_a, r.team_b, oh, od, oa, extra_markets=r.extra_markets)
        top5 = p5.get("top5_scores_enriched") or []
        v6 = predict_score_v6(oh, od, oa, r.extra_markets, top_n=20)
        try:
            grid = {sc: p for sc, p in top5}
            for sc in COMMON:
                grid.setdefault(sc, 0.01)
            tt = sum(grid.values()); grid = {k: v / tt for k, v in grid.items()}
            jrn = int(r.round_info) if str(r.round_info).isdigit() and r.round_info != "0" else (infer_current_journee(e, r.es) or 8)
            v2t = v2.predict(r.team_a, r.team_b, jrn, v5_score_grid=grid,
                             extra_markets=r.extra_markets, odds_h=oh, odds_a=oa, top_n=5)
            v2l = [(t[0], t[1]) for t in v2t]
        except Exception:
            v2t = []; v2l = []
        ens = ensemble_from_raw(top5, v2t if v2l else None, v6, top_n=3)
        # VALUE : EV>0 sur cote offerte (totaux + scores exacts), proba = grille sim
        em = parse_extra_markets(r.extra_markets)
        vals = []
        for k, cote in total_buts_odds(em).items():
            try:
                ki = int(k)
            except ValueError:
                continue
            p = float(td[min(ki, 6)])
            ev = p * cote - 1
            if ev > EV_MIN and p > 0.10:
                vals.append((f"Total {k} buts", cote, p, ev))
        for sc, cote in score_exact_odds(em).items():
            try:
                h, a = map(int, sc.split("-"))
            except ValueError:
                continue
            if h < g.shape[0] and a < g.shape[0]:
                p = float(g[h, a]); ev = p * cote - 1
                if ev > EV_MIN and p > 0.06:
                    vals.append((f"Score {sc}", cote, p, ev))
        vals.sort(key=lambda x: -x[3])
        for lbl, cote, p, ev in vals[:3]:
            summary["value"].append((f"{r.team_a} v {r.team_b}", lbl, cote, p, ev))

        print(f"┌ {r.team_a} v {r.team_b}")
        print(f"│ 1X2  : 1 {q1*100:.0f}% / X {qX*100:.0f}% / 2 {q2*100:.0f}%  (cotes {oh:.2f}/{od:.2f}/{oa:.2f})  fav {fav} {favp*100:.0f}%")
        print(f"│ buts : λh {inv.lam_h:.2f} λa {inv.lam_a:.2f} λtot {lt:.2f} | total {tot_str} | {zone}")
        print(f"│ BTTS : Oui {btts*100:.0f}% / Non {(1-btts)*100:.0f}%")
        print(f"│ score V5 : {f3(top5)}")
        print(f"│ score V2 : {f3(v2l)}")
        print(f"│ score V6 : {f3(v6['top'])}")
        print(f"│ score ENS: {f3(ens['top'])}  → Top1 {ens['modal']}")
        if vals:
            for lbl, cote, p, ev in vals[:3]:
                print(f"│ 💰 VALUE: {lbl} @{cote:.2f}  (p={p*100:.0f}%, EV {ev*100:+.0f}%)")
        print("└" + "─" * 64)

    # ===== RÉCAP =====
    print(f"\n{'='*66}\nRÉCAP ROUND {TARGET} — les paris les plus fiables\n{'='*66}")
    def show(key, title):
        rows = summary[key]
        if rows:
            print(f"\n{title}:")
            for row in sorted(rows, key=lambda x: x[1], reverse=("over" in key or "btts_oui" in key)):
                if key == "value":
                    m, lbl, cote, p, ev = row
                    print(f"   {m:<34} {lbl} @{cote:.2f} (EV {ev*100:+.0f}%)")
                else:
                    print(f"   {row[0]:<34} (λtot {row[1]:.2f})" if key in ("under","over","3buts")
                          else f"   {row[0]:<34} ({row[1]*100:.0f}%)")
    show("under", "⬇ UNDER 3.5 (le plus sûr sur les buts, ~76%)")
    show("over", "⬆ OVER 2.5 (~72%)")
    show("3buts", "🎯 TOTAL 3 buts (modal)")
    show("btts_oui", "BTTS Oui (≥58%)")
    show("btts_non", "BTTS Non (≤45%)")
    if summary["value"]:
        print("\n💰 VALUE (grosses cotes EV>0 — rares, le marché est efficient):")
        for m, lbl, cote, p, ev in sorted(summary["value"], key=lambda x: -x[4]):
            print(f"   {m:<34} {lbl} @{cote:.2f} (p={p*100:.0f}%, EV {ev*100:+.0f}%)")
    else:
        print("\n💰 VALUE : aucune cote +EV ce round (normal — marché efficient).")


if __name__ == "__main__":
    main()
