"""Prédiction COMPLÈTE pour un round précis : 1X2, HT, score, O/U, BTTS, combos."""
from __future__ import annotations
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
from scipy.stats import poisson
from sqlalchemy import create_engine
from scraper.config import load_settings
from scraper.predictor_v5 import fit_model_v5, predict_match_v5
from scraper.predictor_v10 import fit_model_v10, predict_v10
from scraper.team_gold_data import (
    PAIR_HOME_GOLD, PAIR_AWAY_GOLD, PAIR_TRAP_HOME,
    OVER_GOLD, UNDER_GOLD, BTTS_OUI_GOLD, BTTS_NON_GOLD,
    SCORE_COMBO_GOLD, SCORE_DOMINANT_GOLD,
    BRACKET_GOLD_HOME, bracket_match,
)
from scraper.season_context import detect_current_season, compute_season_stats, season_adjustment_factor
from scraper.strategy_engine import StrategyEngine, label_segment
from scraper.journee_inference import infer_current_journee
from scraper.tier1_picker import classify_pick, expected_outcome
from scraper.exotic_signals import evaluate_exotics
from scraper.score_predictor_v2 import ScorePredictorV2
from scraper.market_inversion import (
    invert_markets, apply_sim_deviations, grid_predictions,
    parse_extra_markets, total_buts_odds, score_exact_odds, exact_invert_1x2,
)
from scraper.halftime_model import ht_predictions

MG_TZ = timezone(timedelta(hours=3))
TARGET = "23:21"

# Registre des cles combinees (Track B) — optionnel, charge si present
import json as _json
from pathlib import Path as _Path
def _load_combokeys():
    p = _Path(__file__).resolve().parents[1] / "data" / "combokeys_registry.json"
    try:
        return _json.load(open(p, encoding="utf-8"))
    except Exception:
        return {"rules": {}, "binspec_ref": None}
def _load_binspec():
    p = _Path(__file__).resolve().parents[1] / "exports" / "combokeys_binspec.json"
    try:
        return _json.load(open(p, encoding="utf-8"))
    except Exception:
        return {}
def _load_json(name):
    p = _Path(__file__).resolve().parents[1] / "exports" / name
    try:
        return _json.load(open(p, encoding="utf-8"))
    except Exception:
        return None
def _band(v, bands):
    for lo, hi, lbl in bands:
        if lo <= v < hi:
            return lbl
    return None
def _totals_cal_lookup(cal, lam_tot, lam_diff):
    """Taux Over 1.5/2.5/3.5 empirique (calibre) pour la bande (lam_tot, lam_diff)."""
    if not cal:
        return None
    te = cal["_edges"]["lam_tot"]; de = cal["_edges"]["lam_diff"]
    def edge(v, edges):
        for i in range(len(edges) - 1):
            if edges[i] <= v < edges[i + 1]:
                return edges[i], edges[i + 1]
        return None
    bt = edge(lam_tot, te); bd = edge(lam_diff, de)
    if not bt:
        return None
    if bd:
        key = f"{float(bt[0])}|{float(bt[1])}|{float(bd[0])}|{float(bd[1])}"
        if key in cal["cells"]:
            return cal["cells"][key]
    # repli : marginal par bande de lam_tot SEUL (jamais la moyenne globale)
    lk = f"{float(bt[0])}|{float(bt[1])}"
    return (cal.get("_lamtot") or {}).get(lk)
def _narrow_lookup(nt, lam_tot, lam_diff, p_btts):
    if not nt or p_btts is None:
        return None
    b = nt["_bands"]
    tl = _band(lam_tot, b["tot"]); dl = _band(lam_diff, b["diff"]); bl = _band(p_btts, b["btts"])
    if not (tl and dl and bl):
        return None
    return nt["cells"].get(f"{tl}|{dl}|{bl}")


def main():
    settings = load_settings()
    engine = create_engine(settings.db_url)
    history = pd.read_sql("""
        SELECT e.team_a, e.team_b, o.odds_home, o.odds_draw, o.odds_away,
               r.score_a, r.score_b, r.ht_score_a, r.ht_score_b
        FROM events e
        JOIN odds_snapshots o ON o.id = (SELECT MIN(id) FROM odds_snapshots WHERE event_id = e.id)
        JOIN results r ON r.event_id = e.id
        WHERE r.ht_score_a IS NOT NULL AND e.competition = 'InstantLeague-8035'
    """, engine)
    print(f"V5/V10 (n_train={len(history)})\n")
    model_v5 = fit_model_v5(history, ht_history=history.copy(), engine=engine, form_alpha=0.0)
    model_v10 = fit_model_v10(history)

    # 🌍 CONTEXTE SAISON COURANTE (forme actuelle des équipes)
    # 🎯 STRATEGY ENGINE (segmenté DS/MS/FS, backtest +14.58% ROI)
    strat_engine = StrategyEngine()

    # 🎯 SCORE PREDICTOR V2 (Top 1 +9pp, Top 3 +15pp accuracy vs V5)
    score_v2 = ScorePredictorV2(engine)

    since_ts, season_id = detect_current_season(engine, lookback_seasons=20)
    season_stats = compute_season_stats(engine, since_ts)
    print(f"🌍 Saison récente (20 dernières) depuis {since_ts}")
    hot = [s for s in season_stats.values() if s.season_confidence >= 0.6 and (s.wr_home_effective - s.wr_home_global) > 0.05]
    cold = [s for s in season_stats.values() if s.season_confidence >= 0.6 and (s.wr_home_effective - s.wr_home_global) < -0.05]
    if hot:
        print(f"   🔥 EN FORME : {', '.join(f'{s.team}({s.wr_home_season*100:.0f}%/{s.n_home_season})' for s in sorted(hot, key=lambda x: -x.wr_home_effective+x.wr_home_global))}")
    if cold:
        print(f"   ❄️  EN PERTE : {', '.join(f'{s.team}({s.wr_home_season*100:.0f}%/{s.n_home_season})' for s in sorted(cold, key=lambda x: x.wr_home_effective-x.wr_home_global))}")
    print()

    h_all = pd.read_sql("""
        SELECT e.team_a, e.team_b, r.score_a, r.score_b
        FROM events e JOIN results r ON r.event_id = e.id WHERE r.score_a IS NOT NULL AND e.competition = 'InstantLeague-8035'
    """, engine)
    h_all = h_all.drop_duplicates(["team_a", "team_b", "score_a", "score_b"], keep="last").copy()
    h_all["score"] = h_all.apply(lambda r: f"{int(r.score_a)}-{int(r.score_b)}", axis=1)

    now_utc = datetime.now(timezone.utc)
    upcoming = pd.read_sql("""
        SELECT e.team_a, e.team_b, e.expected_start, e.round_info,
               o.odds_home, o.odds_draw, o.odds_away, o.extra_markets, e.id as ev_id
        FROM events e
        JOIN odds_snapshots o ON o.id = (SELECT MAX(id) FROM odds_snapshots WHERE event_id = e.id)
        LEFT JOIN results r ON r.event_id = e.id
        WHERE r.id IS NULL AND e.expected_start IS NOT NULL AND e.competition = 'InstantLeague-8035'
    """, engine)
    upcoming["expected_start"] = pd.to_datetime(upcoming.expected_start, utc=True)
    # Grace 3 min : un round qui vient de démarrer reste prédictible (résultat pas encore connu)
    upcoming = upcoming[upcoming.expected_start > now_utc - pd.Timedelta(minutes=3)].copy()
    upcoming["local"] = upcoming.expected_start.dt.tz_convert(MG_TZ).dt.strftime("%H:%M")
    # Dédup : préférer l'event avec une VRAIE journée (round_info != 0) avant le plus récent
    upcoming["has_round"] = upcoming.round_info.fillna("0").astype(str).ne("0")
    upcoming = (upcoming.sort_values(["has_round", "ev_id"], ascending=False)
                         .drop_duplicates(["team_a", "team_b", "local"]))
    matches = upcoming[upcoming.local == TARGET]
    if matches.empty:
        print(f"Round {TARGET} introuvable. Dispo : {sorted(upcoming.local.unique())[:6]}")
        return 1

    print(f"{'═' * 105}")
    print(f"  ⏰ ROUND {TARGET} Mada — {len(matches)} matchs")
    print(f"{'═' * 105}\n")

    safe_picks = []
    combokeys = _load_combokeys()
    ck_binspec = _load_binspec()
    totals_cal = _load_json("totals_calibration.json")
    narrow_tab = _load_json("narrow_table.json")

    for i, (_, m) in enumerate(matches.iterrows(), 1):
        pred5 = predict_match_v5(model_v5, m.team_a, m.team_b, m.odds_home, m.odds_draw, m.odds_away,
                                   extra_markets=m.extra_markets)
        pred10 = predict_v10(model_v10, m.team_a, m.team_b, m.odds_home, m.odds_draw, m.odds_away,
                              extra_markets=m.extra_markets)

        if pred5.get("lam_h_ht"):
            lam_h_ht = pred5["lam_h_ht"]; lam_a_ht = pred5["lam_a_ht"]
            lam_h_ft = lam_h_ht / model_v5.ht_lambda_ratio
            lam_a_ft = lam_a_ht / model_v5.ht_lambda_ratio
            lam_total = lam_h_ft + lam_a_ft
            ht_score = (int(round(lam_h_ht)), int(round(lam_a_ht)))
            # O/U + BTTS depuis la grille BLEND (market-contrainte) si dispo :
            # +1.8pp acc Over 2.5 OOS vs Poisson indépendant
            if pred5.get("p_over_25_blend") is not None:
                p_over_15 = pred5["p_over_15_blend"] * 100
                p_over_25 = pred5["p_over_25_blend"] * 100
                p_over_35 = pred5["p_over_35_blend"] * 100
                p_under_35 = (1 - pred5["p_over_35_blend"]) * 100
                p_btts = pred5["p_btts_blend"] * 100
            else:
                p_over_15 = (1 - poisson.cdf(1, lam_total)) * 100
                p_over_25 = (1 - poisson.cdf(2, lam_total)) * 100
                p_over_35 = (1 - poisson.cdf(3, lam_total)) * 100
                p_under_35 = poisson.cdf(3, lam_total) * 100
                p_btts = (1 - poisson.pmf(0, lam_h_ft)) * (1 - poisson.pmf(0, lam_a_ft)) * 100
        else:
            lam_total = 0; ht_score = (0,0)
            p_over_15 = p_over_25 = p_over_35 = p_under_35 = p_btts = 0

        top5 = pred5.get("top5_scores_enriched") or []
        score_ft = top5[0][0] if top5 else "?"
        top3_str = " · ".join(f"{s}({p*100:.0f}%)" for s, p in top5[:3])

        # 🎯 SCORE V2 ENSEMBLE (Top 1 22% vs V5 12%)
        try:
            v5_grid = {s: p for s, p in top5}
            common_scores = ["0-0","1-0","0-1","1-1","2-0","0-2","2-1","1-2","2-2","3-0","0-3","3-1","1-3","3-2","2-3","4-0","0-4"]
            for s in common_scores:
                if s not in v5_grid: v5_grid[s] = 0.01
            tot = sum(v5_grid.values())
            v5_grid = {s: p/tot for s, p in v5_grid.items()}
            try:
                jrn = int(m.round_info) if m.round_info and str(m.round_info).isdigit() and m.round_info != "0" else infer_current_journee(engine, m.expected_start) or 8
            except Exception:
                jrn = 8
            v2_top5 = score_v2.predict(m.team_a, m.team_b, jrn,
                                        v5_score_grid=v5_grid,
                                        extra_markets=m.extra_markets,
                                        odds_h=m.odds_home, odds_a=m.odds_away,
                                        top_n=5)
            v2_top3_str = " · ".join(f"{s}({p*100:.0f}%)" for s, p, _ in v2_top5[:3])
        except Exception as ex:
            v2_top5 = []
            v2_top3_str = ""
        ft_pick = pred5.get("primary_pick", "—")
        ft_p = (pred5.get("primary_p") or 0) * 100
        ht_pick = pred5.get("ht_pick", "—")
        p_h_ht = (pred5.get("p_h_ht") or 0) * 100
        p_d_ht = (pred5.get("p_d_ht") or 0) * 100
        p_a_ht = (pred5.get("p_a_ht") or 0) * 100

        # H2H
        h2h = h_all[(h_all.team_a == m.team_a) & (h_all.team_b == m.team_b)]
        h2h_n = len(h2h)
        h2h_str = "—"
        if h2h_n >= 5:
            top_h2h = Counter(h2h.score).most_common(3)
            h2h_str = " ".join(f"{s}({c}/{h2h_n})" for s, c in top_h2h)

        print(f"┌─ MATCH {i}  {m.team_a} vs {m.team_b}")
        print(f"│  Cotes 1X2 : {m.odds_home:.2f} / {m.odds_draw:.2f} / {m.odds_away:.2f}")
        print(f"│  H2H récents : {h2h_str}" if h2h_n >= 5 else "│  H2H récents : pas assez")
        print(f"│  ── HT ──")
        # 🕐 Modèle HT calibré (split 45/55 + DC rho=-0.15) — validé : HT 1X2 acc 47.7%,
        # score HT Top1 26%/Top3 59% (au plafond, 2× le score FT). Accuracy, pas +EV (marge).
        _hp = ht_predictions(*exact_invert_1x2(float(m.odds_home), float(m.odds_draw), float(m.odds_away)))
        _h = _hp["ht_1x2"]
        _hts = " · ".join(f"{s}({p*100:.0f}%)" for s, p in _hp["ht_top_scores"])
        _htft = " · ".join(f"{k}({p*100:.0f}%)" for k, p in _hp["htft_top"])
        print(f"│  🕐 HT 1X2 : 1 {_h['1']*100:.0f}% / X {_h['X']*100:.0f}% / 2 {_h['2']*100:.0f}%  → pick {_hp['ht_pick']}  (modèle calibré, acc ~48%)")
        print(f"│  🕐 Score HT : {_hts}   ·   HT/FT : {_htft}")
        print(f"│  ── FT ──")
        print(f"│  Pick FT  : {ft_pick} ({ft_p:.0f}%)  Score modal : {score_ft}")
        print(f"│  Top 3 scores V5 : {top3_str}")
        if v2_top3_str:
            print(f"│  🎯 Top 3 V2 ENSEMBLE : {v2_top3_str}")
        print(f"│  ── BUTS ──")
        print(f"│  Buts attendus : {lam_total:.2f}")
        print(f"│  Over 1.5 : {p_over_15:.0f}%  Over 2.5 : {p_over_25:.0f}%  Over 3.5 : {p_over_35:.0f}%")
        print(f"│  Under 3.5 : {p_under_35:.0f}%  BTTS : {p_btts:.0f}%")
        # 🌍 SAISON COURANTE
        sa = season_adjustment_factor(m.team_a, m.team_b, season_stats)
        sh = season_stats.get(m.team_a)
        if sh and sh.season_confidence > 0:
            print(f"│  Saison home : {sh.wr_home_season*100:.0f}% wr ({sh.n_home_season} matchs), Δ vs global {(sh.wr_home_effective - sh.wr_home_global)*100:+.1f}pp")
            if sa["note"]:
                print(f"│  {sa['note']}")
        print(f"│  ── SIGNAUX ──")

        # 🎯 STRATEGY ENGINE (segmenté DS/MS/FS — backtesté +14.58% ROI)
        try:
            journee_int = int(m.round_info) if m.round_info and str(m.round_info).isdigit() else None
        except Exception:
            journee_int = None
        # J0 = round non assigné → inférer depuis derniers matchs finis
        if not journee_int or journee_int == 0:
            inferred = infer_current_journee(engine, m.expected_start)
            eval_journee = inferred if inferred else 8  # default MS_early si tout échoue
        else:
            eval_journee = journee_int
        se_eval = strat_engine.evaluate(m.team_a, m.team_b, eval_journee,
                                         float(m.odds_home), float(m.odds_draw), float(m.odds_away))
        for sig in se_eval.base_signals:
            print(f"│  🎯 {sig.reason}")
        for trap in se_eval.traps[:3]:
            print(f"│  ⚠️  {trap.reason}")
        if se_eval.recommended_picks:
            rp = se_eval.recommended_picks[0]
            tag = "🟢🟢🟢" if rp["strength"] >= 1.5 else ("🟢🟢" if rp["strength"] >= 1.0 else "🟢")
            j_label = f"J{journee_int}" if journee_int else "J0 (preseason→DS)"
            print(f"│  {tag} STRATEGY ENGINE → {rp['pick']} @{rp['cote']:.2f}  (strength {rp['strength']:.2f}, segment {se_eval.segment}, {j_label})")
            safe_picks.append(("STRATEGY", f"{m.team_a} vs {m.team_b}", rp["pick"], rp["cote"], rp["strength"]/2))
        if se_eval.score_signals:
            top_scores = ", ".join(f"{s['score']}({s['rate']*100:.0f}%)" for s in se_eval.score_signals[:3])
            print(f"│  📊 Scores profile {se_eval.profile} en {se_eval.segment} : {top_scores}")

        # 🔑 INVERSION MARCHÉ — grille latente (1X2 = Poisson pur) + déviations simulateur.
        # Backtest OOS : la calibration 'cells' atteint le plafond empirique (Top1 12.0%,
        # Top3 30.6%) et bat le pricing brut (+2.9pt Top3, McNemar p=0.005), history-free.
        inv = invert_markets(float(m.odds_home), float(m.odds_draw), float(m.odds_away), m.extra_markets)
        em_parsed = parse_extra_markets(m.extra_markets)
        grid_sim = apply_sim_deviations(inv.lam_h, inv.lam_a, "cells")
        gp = grid_predictions(grid_sim, top_k=3)
        tb_odds = total_buts_odds(em_parsed)
        se_odds = score_exact_odds(em_parsed)
        mlt = gp["most_likely_total"]
        tot_cote = tb_odds.get(str(mlt))
        tot_prob = gp["total_dist"][str(mlt)]
        tot_ev = (tot_prob * tot_cote - 1) if tot_cote else None
        print(f"│  ── INVERSION MARCHÉ ──")
        incoh = "  ⚠️ marchés incohérents (mispricing?)" if inv.fit_quality == "inconsistent" else ""
        print(f"│  🔑 μ moteur : ({inv.lam_h:.2f}, {inv.lam_a:.2f})  fit={inv.fit_quality}  résidu={inv.residual}{incoh}")
        tot_lbl = f"{mlt}+" if mlt >= 6 else f"{mlt}"
        tot_str = f"@{tot_cote:.2f} EV{tot_ev*100:+.0f}%" if tot_cote else "(cote n/d)"
        print(f"│  🔑 Total le + probable : {tot_lbl} buts ({tot_prob*100:.0f}%)  {tot_str}")
        sc_parts = []
        for s, p in gp["top_scores"]:
            c = se_odds.get(s)
            if c:
                sc_parts.append(f"{s}({p*100:.0f}%)@{c:.1f} EV{(p*c-1)*100:+.0f}%")
            else:
                sc_parts.append(f"{s}({p*100:.0f}%)")
        print(f"│  🔑 Top-3 score (sim) : {' · '.join(sc_parts)}")
        # Over calibrés (taux EMPIRIQUE par bande λ_tot×λ_diff) — à comparer à la cote affichée
        lam_tot_v = inv.lam_h + inv.lam_a
        lam_diff_v = inv.lam_h - inv.lam_a
        cal = _totals_cal_lookup(totals_cal, lam_tot_v, lam_diff_v)
        if cal:
            o15 = cal.get("over1.5"); o25 = cal.get("over2.5"); o35 = cal.get("over3.5")
            gl = (totals_cal or {}).get("_global", {})
            d25 = (o25 - gl.get("over2.5", o25)) if o25 is not None else 0
            print(f"│  🔑 Over calibré (réel) : O1.5 {o15*100:.0f}% · O2.5 {o25*100:.0f}% · O3.5 {o35*100:.0f}%"
                  f"   (cote rentable Over2.5 si ≥ {1/o25:.2f})")
        # Chaînage : total × dominance × BTTS → 1 score (ACCURACY ; EV indicatif OOS, NON confirmé)
        p_btts_band = (gp["btts_oui"])
        nb = _narrow_lookup(narrow_tab, lam_tot_v, lam_diff_v, p_btts_band)
        if nb:
            ev_raw = nb.get("ev")
            evs = f" · EV OOS {float(ev_raw):+.0f}% (n={nb['n']}, non confirmé)" if ev_raw not in (None, "None") else ""
            cts = f"@{nb['cote']}" if nb.get("cote") not in (None, "None") else ""
            print(f"│  🔑 Chaînage → score le + probable : {nb['score']} ({nb['rate']}%) {cts}{evs}")
        # NB : EV inversion/chaînage = indicatifs (le minage rigoureux a trouvé 0 clé +EV
        # après Bonferroni). On N'ALIMENTE PAS le RÉCAP des picks (info seulement).
        # Cles combinees CONFIRMED/WATCH (Track B) touchees par ce match
        if combokeys.get("rules"):
            # signaux du match (memes definitions que l'extract)
            dc_x2 = (em_parsed.get("Double Chance") or {}).get("X2")
            tb_full = total_buts_odds(em_parsed)
            from scraper.market_inversion import devig_market as _dvm
            tb_dev = _dvm(tb_full) if len(tb_full) >= 4 else {}
            gng_m = em_parsed.get("G/NG") or {}
            gng_dev = _dvm({"Oui": gng_m.get("Oui"), "Non": gng_m.get("Non")}) if gng_m.get("Oui") and gng_m.get("Non") else {}
            ck_sig = {
                "fav": min(m.odds_home, m.odds_away), "dog": max(m.odds_home, m.odds_away),
                "od": float(m.odds_draw), "odds_ratio": max(m.odds_home, m.odds_away) / min(m.odds_home, m.odds_away),
                "lam_tot": inv.lam_h + inv.lam_a, "lam_diff": inv.lam_h - inv.lam_a,
                "residual": inv.residual, "dc_x2_cote": float(dc_x2) if dc_x2 else None,
                "p_total_eq3": tb_dev.get("3"), "p_btts": gng_dev.get("Oui"),
            }
            for ck_name, rule in combokeys["rules"].items():
                if rule.get("status") not in ("CONFIRMED", "WATCH"):
                    continue
                dfn = rule.get("definition") or {}
                ok = True
                for s, bounds in dfn.items():
                    v = ck_sig.get(s)
                    if v is None or bounds is None or not (bounds[0] <= v < bounds[1]):
                        ok = False; break
                if ok and dfn:
                    print(f"│  🔑 CLÉ {rule['status']} : {rule['predict']} @~{rule.get('expected_cote')} "
                          f"(hit {rule.get('expected_hit_rate',0)*100:.0f}%, EV{rule.get('ev',0):+.0f}%, n={rule.get('n_te')})")
                    if rule.get("ev", 0) > 5 and rule["status"] == "CONFIRMED":
                        safe_picks.append(("CLE_COMBINEE", f"{m.team_a} vs {m.team_b}",
                                           rule["predict"], rule.get("expected_cote"), rule.get("expected_hit_rate", 0)))

        # 🎯 TIER 1 PICKER — calibré 72-82% WR
        # Gate sur max(p_blend, p_cote calibrée) : la p_cote (devig + calibration) est
        # mieux calibrée en haut de distribution → +48% de volume à WR égal, ROI +4pp OOS
        cote_ft_val = {"1": m.odds_home, "X": m.odds_draw, "2": m.odds_away}.get(ft_pick)
        p_cote_pick = {"1": pred5.get("p_h_cote"), "X": pred5.get("p_d_cote"),
                        "2": pred5.get("p_a_cote")}.get(ft_pick) or 0
        gate_p = max(ft_p / 100, p_cote_pick)
        n_traps_on_ft = sum(1 for t in se_eval.traps if t.pick == ft_pick)
        n_gold_on_ft = sum(1 for s in se_eval.base_signals
                            if s.pick == ft_pick and ("PEAK" in s.category or "COTE_BUCKET" in s.category))
        form_drop = bool(sh and sh.season_confidence >= 0.6 and (sh.wr_home_effective - sh.wr_home_global) < -0.10 and ft_pick == "1")
        tier_pick = classify_pick(ft_pick, gate_p, cote_ft_val, n_traps_on_ft, n_gold_on_ft, form_drop)
        if tier_pick:
            tier_label = {"TIER_1_ULTRA": "🟢🟢🟢 TIER 1 ULTRA",
                          "TIER_1_STRICT": "🟢🟢 TIER 1 STRICT",
                          "TIER_1_STANDARD": "🟢 TIER 1 STANDARD",
                          "TIER_2_MODERATE": "🟡 TIER 2"}.get(tier_pick.tier, tier_pick.tier)
            print(f"│  {tier_label} → {tier_pick.pick} @{tier_pick.cote:.2f}  (WR attendu {tier_pick.expected_wr*100:.0f}%, {tier_pick.reason})")
            safe_picks.append((tier_pick.tier, f"{m.team_a} vs {m.team_b}", tier_pick.pick, tier_pick.cote, tier_pick.expected_wr))

        # 🎰 EXOTIC SIGNALS — hautes cotes validées walk-forward 2 périodes (2026-06-11)
        journee_reliable = bool(journee_int and journee_int >= 1)
        exotic_picks = evaluate_exotics(eval_journee, float(m.odds_home), float(m.odds_away),
                                         extra_markets=m.extra_markets,
                                         journee_reliable=journee_reliable)
        for ep in exotic_picks:
            cote_str = f"@{ep.cote:.2f}" if ep.cote else "(cote marché indispo)"
            print(f"│  🎰 EXOTIC {ep.signal_id} : {ep.market} '{ep.selection}' {cote_str}  mise {ep.stake}u")
            print(f"│     {ep.reason}")
            if ep.cote:
                safe_picks.append((f"EXOTIC_{ep.signal_id}", f"{m.team_a} vs {m.team_b}",
                                    f"{ep.market} {ep.selection}", ep.cote, ep.expected_wr))

        # Garde-fou : home en grosse perte de forme → bloque PAIRE OR HOME et MULTI 1
        home_in_cold = sh and sh.season_confidence >= 0.6 and (sh.wr_home_effective - sh.wr_home_global) < -0.08

        # PAIRE OR HOME
        if (m.team_a, m.team_b) in PAIR_HOME_GOLD:
            p = PAIR_HOME_GOLD[(m.team_a, m.team_b)]
            if home_in_cold:
                print(f"│  ⚠️  PAIRE OR HOME bloquée : {m.team_a} en perte saison (skip)")
            else:
                print(f"│  💎💎 PAIRE OR HOME : 1 @{m.odds_home:.2f}  ({p['win']*100:.0f}% wins n={p['n']}, ROI+{p['roi']*100:.0f}%)")
                safe_picks.append(("PAIRE_OR_HOME", f"{m.team_a} vs {m.team_b}", "1", m.odds_home, p['win']))

        # PAIRE OR AWAY
        if (m.team_a, m.team_b) in PAIR_AWAY_GOLD:
            p = PAIR_AWAY_GOLD[(m.team_a, m.team_b)]
            sb = season_stats.get(m.team_b)
            away_in_cold = sb and sb.season_confidence >= 0.6 and (sb.wr_home_effective - sb.wr_home_global) < -0.10
            if m.odds_away <= p["cote"] * p.get("max_cote_factor", 1.05) and not away_in_cold:
                print(f"│  💎 PAIRE OR AWAY : 2 @{m.odds_away:.2f}  ({p['win']*100:.0f}% wins n={p['n']})")
                safe_picks.append(("PAIRE_OR_AWAY", f"{m.team_a} vs {m.team_b}", "2", m.odds_away, p['win']))
            elif away_in_cold:
                print(f"│  ⚠️  PAIRE OR AWAY bloquée : {m.team_b} en perte saison")

        # PAIRE TRAP
        if (m.team_a, m.team_b) in PAIR_TRAP_HOME:
            print(f"│  ❌❌ PAIRE TRAP HOME — JAMAIS parier 1")

        # BRACKET HOME
        br_h = bracket_match(m.team_a, m.odds_home, BRACKET_GOLD_HOME)
        if br_h is not None and br_h > 0.10:
            print(f"│  ⭐ BRACKET GOLD HOME : {m.team_a} @{m.odds_home:.2f}  ROI+{br_h*100:.0f}%")

        # OVER/UNDER GOLD
        if (m.team_a, m.team_b) in OVER_GOLD:
            og = OVER_GOLD[(m.team_a, m.team_b)]
            print(f"│  💎 OVER 2.5 GOLD : {og['rate']*100:.0f}% (n={og['n']})")
            safe_picks.append(("OVER", f"{m.team_a} vs {m.team_b}", "Over 2.5", None, og['rate']))
        if (m.team_a, m.team_b) in UNDER_GOLD:
            ug = UNDER_GOLD[(m.team_a, m.team_b)]
            print(f"│  💎 UNDER 2.5 GOLD : {(1-ug['over_rate'])*100:.0f}% (n={ug['n']})")
            safe_picks.append(("UNDER", f"{m.team_a} vs {m.team_b}", "Under 2.5", None, 1-ug['over_rate']))

        # BTTS
        if (m.team_a, m.team_b) in BTTS_OUI_GOLD:
            bg = BTTS_OUI_GOLD[(m.team_a, m.team_b)]
            if m.odds_home >= bg.get('min_cote_h', 1.8):
                print(f"│  💎 BTTS OUI : {bg['rate']*100:.0f}% (n={bg['n']})")
                safe_picks.append(("BTTS_OUI", f"{m.team_a} vs {m.team_b}", "BTTS Oui", None, bg['rate']))
        if (m.team_a, m.team_b) in BTTS_NON_GOLD:
            bn = BTTS_NON_GOLD[(m.team_a, m.team_b)]
            print(f"│  💎 BTTS NON : {(1-bn['bts_rate'])*100:.0f}% (n={bn['n']})")
            safe_picks.append(("BTTS_NON", f"{m.team_a} vs {m.team_b}", "BTTS Non", None, 1-bn['bts_rate']))

        # SCORE COMBO
        if (m.team_a, m.team_b) in SCORE_COMBO_GOLD:
            c = SCORE_COMBO_GOLD[(m.team_a, m.team_b)]
            print(f"│  💎💎 COMBO SCORE : {c['top1']}({c['r1']*100:.0f}%) + {c['top2']}({c['r2']*100:.0f}%) = {c['combo']*100:.0f}% (n={c['n']})")
            safe_picks.append(("COMBO_SCORE", f"{m.team_a} vs {m.team_b}", f"{c['top1']} + {c['top2']}", None, c['combo']))

        # SCORE DOMINANT
        if (m.team_a, m.team_b) in SCORE_DOMINANT_GOLD and (m.team_a, m.team_b) not in SCORE_COMBO_GOLD:
            s = SCORE_DOMINANT_GOLD[(m.team_a, m.team_b)]
            if 0.30 <= s['rate'] <= 0.44:
                print(f"│  💎 SCORE SWEET : {s['score']} {s['rate']*100:.0f}% (n={s['n']})")
                safe_picks.append(("SCORE_SWEET", f"{m.team_a} vs {m.team_b}", s['score'], None, s['rate']))

        # MULTI-SIGNAL V10 (bloque si home en perte de forme et MULTI = 1)
        for outcome in ["1", "X", "2"]:
            agg = pred10["agg"][outcome]
            ev = pred10["ev_1x2"][outcome]
            conf = pred10["confidence"][outcome]
            cote = pred10["cotes"][outcome]
            if agg.get("has_pair_trap"): continue
            if agg["n_pos"] >= 2 and ev > 0.05 and not agg.get("has_pair_gold") and cote < 3:
                if outcome == "1" and home_in_cold:
                    print(f"│  ⚠️  MULTI 1 bloqué : {m.team_a} en perte saison")
                    continue
                print(f"│  🔥🔥 MULTI : {outcome} @{cote:.2f}  EV+{ev*100:.0f}% conf {conf}/10")
                safe_picks.append(("MULTI", f"{m.team_a} vs {m.team_b}", outcome, cote, conf/10))
        print(f"└{'─' * 100}")
        print()

    # RÉCAP
    print(f"\n{'═' * 105}")
    print(f"  📋 RÉCAP DES PICKS RECOMMANDÉS")
    print(f"{'═' * 105}\n")
    if not safe_picks:
        print("  Aucun signal fort détecté")
    else:
        # Group par type
        by_type = {}
        for typ, match, pari, cote, rate in safe_picks:
            by_type.setdefault(typ, []).append((match, pari, cote, rate))
        for typ, picks in by_type.items():
            print(f"  {typ} ({len(picks)}):")
            for match, pari, cote, rate in picks:
                cote_str = f"@{cote:.2f}" if cote else ""
                print(f"     • {match:<40} {pari} {cote_str}  ({rate*100:.0f}%)")
            print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
