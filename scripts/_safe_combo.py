"""Combiné 1X2 des favoris les + sûrs par round (gates TIER1/ULTRA du moteur).
Sort, par round : les picks TIER1 + le combiné recommandé (cote × proba réelle).
Usage: ./.venv/Scripts/python.exe scripts/_safe_combo.py [n_rounds] [legs]"""
from __future__ import annotations
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from sqlalchemy import create_engine
from scraper.config import load_settings
from scraper.predictor_v5 import fit_model_v5, predict_match_v5
from scraper.tier1_picker import classify_pick

MG = timezone(timedelta(hours=3))
N_ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
MAX_LEGS = int(sys.argv[2]) if len(sys.argv) > 2 else 3

def main():
    s = load_settings(); e = create_engine(s.db_url)
    hist = pd.read_sql("""SELECT e.team_a,e.team_b,o.odds_home,o.odds_draw,o.odds_away,
        r.score_a,r.score_b,r.ht_score_a,r.ht_score_b FROM events e
        JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
        JOIN results r ON r.event_id=e.id WHERE r.ht_score_a IS NOT NULL AND e.competition='InstantLeague-8035'""", e)
    m5 = fit_model_v5(hist, ht_history=hist.copy(), engine=e, form_alpha=0.0)
    now = datetime.now(timezone.utc)
    up = pd.read_sql("""SELECT e.team_a,e.team_b,e.expected_start,o.odds_home,o.odds_draw,o.odds_away,o.extra_markets,e.id ev
        FROM events e JOIN odds_snapshots o ON o.id=(SELECT MAX(id) FROM odds_snapshots WHERE event_id=e.id)
        LEFT JOIN results r ON r.event_id=e.id
        WHERE r.id IS NULL AND e.expected_start IS NOT NULL AND e.competition='InstantLeague-8035'""", e)
    up["es"] = pd.to_datetime(up.expected_start, utc=True); up = up[up.es > now - pd.Timedelta(minutes=3)]
    up["local"] = up.es.dt.tz_convert(MG).dt.strftime("%H:%M")
    up = up.sort_values(["es","ev"]).drop_duplicates(["team_a","team_b","local"])
    rounds = sorted(up.local.unique())[:N_ROUNDS]
    TIER_RK = {"TIER_1_ULTRA":3,"TIER_1_STRICT":2,"TIER_1_STANDARD":1}
    print(f"now {now.astimezone(MG):%H:%M} | COMBINÉ favoris sûrs (TIER1) par round\n")
    for rd in rounds:
        ms = up[up.local == rd]; picks = []
        for r in ms.itertuples():
            oh,od,oa = float(r.odds_home),float(r.odds_draw),float(r.odds_away)
            if oh<=1 or oa<=1: continue
            p5 = predict_match_v5(m5, r.team_a, r.team_b, oh, od, oa, extra_markets=r.extra_markets)
            pk = p5.get("primary_pick"); pp = p5.get("primary_p") or 0
            pc = {"1":p5.get("p_h_cote"),"X":p5.get("p_d_cote"),"2":p5.get("p_a_cote")}.get(pk) or 0
            cote = {"1":oh,"X":od,"2":oa}.get(pk)
            gate = max(pp, pc)
            tp = classify_pick(pk, gate, cote, 0, 0, False)
            if tp and tp.tier in TIER_RK:
                picks.append(dict(m=f"{r.team_a} v {r.team_b}", pick=pk, cote=tp.cote,
                                  wr=tp.expected_wr, tier=tp.tier, rk=TIER_RK[tp.tier]))
        picks.sort(key=lambda x:(-x["rk"], x["cote"]))
        print(f"━━━ ROUND {rd} ━━━")
        if not picks:
            print("  (aucun favori TIER1)\n"); continue
        lab={"TIER_1_ULTRA":"🟢🟢🟢ULTRA","TIER_1_STRICT":"🟢🟢STRICT","TIER_1_STANDARD":"🟢STD"}
        for p in picks:
            print(f"  {lab[p['tier']]:<12} {p['m']:<32} {p['pick']} @{p['cote']:.2f}  (WR {p['wr']*100:.0f}%)")
        # combiné des MAX_LEGS plus sûrs
        legs = picks[:MAX_LEGS]
        if len(legs) >= 2:
            import functools, operator
            cote_c = functools.reduce(operator.mul, [l["cote"] for l in legs], 1)
            p_c = functools.reduce(operator.mul, [l["wr"] for l in legs], 1)
            print(f"  ➡️ COMBINÉ {len(legs)} jambes : {' + '.join(l['pick']+'@'+format(l['cote'],'.2f') for l in legs)}")
            print(f"     cote {cote_c:.2f} | proba réelle de toucher {p_c*100:.0f}% | EV {(p_c*cote_c-1)*100:+.0f}%")
        print()

if __name__ == "__main__":
    sys.exit(main())
