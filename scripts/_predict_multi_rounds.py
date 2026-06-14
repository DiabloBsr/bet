"""Prédit plusieurs rounds en un seul passage, avec signal DS et saison."""
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
from scraper.season_context import detect_current_season, compute_season_stats

MG_TZ = timezone(timedelta(hours=3))
TARGETS = ["17:59", "18:01", "18:03", "18:05"]

# Tables DS (de _analyze_ds.py — équipes avec delta >=8pp ou <=-8pp)
DS_HOME_STRONG = {
    "Everton": 0.407, "West Ham": 0.286, "London Blues": 0.128,
    "A. Villa": 0.118, "Manchester Blue": 0.117,
}
DS_HOME_WEAK = {
    "Brentford": -0.285, "Newcastle": -0.146, "Spurs": -0.142,
    "London Reds": -0.134, "Fulham": -0.123, "C. Palace": -0.10,
    "Burnley": -0.093,
}
# Pour les away, on regarde les WR away en DS — équipes faibles away DS = signal pour home pick
DS_AWAY_WEAK = {  # équipes qui perdent souvent away en DS
    "Fulham": -0.123,  # idem
    "Newcastle": -0.146,
    "Brentford": -0.285,
    "Spurs": -0.142,
    "London Reds": -0.134,
}


def ds_lens(team_h: str, team_a: str) -> dict:
    """Calcule un score DS pour un match (positif = home avantagé)."""
    home_boost = DS_HOME_STRONG.get(team_h, 0)
    home_penalty = DS_HOME_WEAK.get(team_h, 0)  # déjà négatif
    away_penalty = DS_AWAY_WEAK.get(team_a, 0)  # déjà négatif
    # boost home = home strong + away weak
    delta = home_boost + home_penalty - away_penalty
    note = ""
    if home_boost > 0:
        note += f"🔥{team_h} +{home_boost*100:.0f}pp DS  "
    if home_penalty < 0:
        note += f"❄️{team_h} {home_penalty*100:.0f}pp DS  "
    if away_penalty < 0:
        note += f"❄️{team_a} away {away_penalty*100:.0f}pp DS"
    return {"delta": delta, "note": note.strip()}


def main():
    settings = load_settings()
    engine = create_engine(settings.db_url)
    history = pd.read_sql("""
        SELECT e.team_a, e.team_b, o.odds_home, o.odds_draw, o.odds_away,
               r.score_a, r.score_b, r.ht_score_a, r.ht_score_b
        FROM events e
        JOIN odds_snapshots o ON o.id = (SELECT MIN(id) FROM odds_snapshots WHERE event_id = e.id)
        JOIN results r ON r.event_id = e.id
        WHERE r.ht_score_a IS NOT NULL
    """, engine)
    print(f"V5/V10 (n_train={len(history)})")
    model_v5 = fit_model_v5(history, ht_history=history.copy(), engine=engine, form_alpha=0.0)
    model_v10 = fit_model_v10(history)

    h_all = pd.read_sql("""
        SELECT e.team_a, e.team_b, r.score_a, r.score_b
        FROM events e JOIN results r ON r.event_id = e.id WHERE r.score_a IS NOT NULL
    """, engine)
    h_all = h_all.drop_duplicates(["team_a", "team_b", "score_a", "score_b"], keep="last").copy()
    h_all["score"] = h_all.apply(lambda r: f"{int(r.score_a)}-{int(r.score_b)}", axis=1)

    # Saison context
    since_ts, _ = detect_current_season(engine, lookback_seasons=20)
    season_stats = compute_season_stats(engine, since_ts)

    now_utc = datetime.now(timezone.utc)
    upcoming = pd.read_sql("""
        SELECT e.team_a, e.team_b, e.expected_start, e.round_info,
               o.odds_home, o.odds_draw, o.odds_away, o.extra_markets, e.id as ev_id
        FROM events e
        JOIN odds_snapshots o ON o.id = (SELECT MAX(id) FROM odds_snapshots WHERE event_id = e.id)
        LEFT JOIN results r ON r.event_id = e.id
        WHERE r.id IS NULL AND e.expected_start IS NOT NULL
    """, engine)
    upcoming["expected_start"] = pd.to_datetime(upcoming.expected_start, utc=True)
    upcoming = upcoming[upcoming.expected_start > now_utc].copy()
    upcoming["local"] = upcoming.expected_start.dt.tz_convert(MG_TZ).dt.strftime("%H:%M")
    upcoming = upcoming.sort_values("ev_id", ascending=False).drop_duplicates(["team_a", "team_b", "local"])

    all_picks = []
    for TARGET in TARGETS:
        round_matchs = upcoming[upcoming.local == TARGET]
        if len(round_matchs) == 0:
            print(f"\n⚠️  Round {TARGET} introuvable")
            continue
        print(f"\n{'═'*100}")
        print(f"  ⏰ ROUND {TARGET} Mada — {len(round_matchs)} matchs  (J{round_matchs.iloc[0].round_info})")
        print(f"{'═'*100}")

        for _, m in round_matchs.iterrows():
            pred5 = predict_match_v5(model_v5, m.team_a, m.team_b, m.odds_home, m.odds_draw, m.odds_away, extra_markets=m.extra_markets)
            pred10 = predict_v10(model_v10, m.team_a, m.team_b, m.odds_home, m.odds_draw, m.odds_away, extra_markets=m.extra_markets)
            ds = ds_lens(m.team_a, m.team_b)

            # Score / buts via FT lambdas
            if pred5.get("lam_h_ht"):
                lam_h_ht = pred5["lam_h_ht"]; lam_a_ht = pred5["lam_a_ht"]
                lam_h_ft = lam_h_ht / model_v5.ht_lambda_ratio
                lam_a_ft = lam_a_ht / model_v5.ht_lambda_ratio
                lam_total = lam_h_ft + lam_a_ft
                ht_score = (int(round(lam_h_ht)), int(round(lam_a_ht)))
                p_over_15 = (1 - poisson.cdf(1, lam_total)) * 100
                p_over_25 = (1 - poisson.cdf(2, lam_total)) * 100
                p_over_35 = (1 - poisson.cdf(3, lam_total)) * 100
                p_under_35 = poisson.cdf(3, lam_total) * 100
                p_btts = (1 - poisson.pmf(0, lam_h_ft)) * (1 - poisson.pmf(0, lam_a_ft)) * 100
            else:
                lam_total = 0; ht_score = (0, 0)
                p_over_15 = p_over_25 = p_over_35 = p_under_35 = p_btts = 0

            top5 = pred5.get("top5_scores_enriched") or []
            score_ft = top5[0][0] if top5 else "?"
            top3_str = ", ".join(f"{s}({p*100:.0f}%)" for s, p in top5[:3])
            ft_pick = pred5.get("primary_pick", "—")
            ft_p = pred5.get("primary_p", 0) * 100
            ht_pick = pred5.get("ht_pick", "—")
            p_h_ht = pred5.get("p_h_ht", 0) * 100
            p_d_ht = pred5.get("p_d_ht", 0) * 100
            p_a_ht = pred5.get("p_a_ht", 0) * 100

            print(f"\n┌─ {m.team_a} vs {m.team_b}  ({m.odds_home}/{m.odds_draw}/{m.odds_away})")
            print(f"│  HT: {ht_pick} ({p_h_ht:.0f}/{p_d_ht:.0f}/{p_a_ht:.0f})  score HT {ht_score[0]}-{ht_score[1]}")
            print(f"│  FT: {ft_pick} ({ft_p:.0f}%) score {score_ft}  |  Top3: {top3_str}")
            print(f"│  Buts {lam_total:.2f}  O2.5 {p_over_25:.0f}%  O3.5 {p_over_35:.0f}%  U3.5 {p_under_35:.0f}%  BTTS {p_btts:.0f}%")
            if ds["note"]:
                print(f"│  🌍 DS : {ds['note']}  (delta {ds['delta']*100:+.0f}pp)")

            picks_this = []

            # Flags DS contextuels
            home_in_ds_cold = (DS_HOME_WEAK.get(m.team_a, 0) <= -0.08)
            away_in_ds_hot = (DS_HOME_STRONG.get(m.team_b, 0) >= 0.30)  # adversaire CRUSH en DS

            # 🆕 FT HIGH CONFIDENCE — pick FT ≥60% ET cote ≤1.95 → auto pick
            if ft_p >= 60 and ft_pick in ("1", "2"):
                cote_pick = m.odds_home if ft_pick == "1" else m.odds_away
                if cote_pick <= 1.95:
                    picks_this.append(("FT_HIGH", m.team_a, m.team_b, ft_pick, cote_pick, ft_p/100))
                    print(f"│  ⭐ FT HIGH CONF : {ft_pick} @{cote_pick}  ({ft_p:.0f}%)")

            # 🆕 DOUBLE CHANCE : si pick FT faible (<50%) ET cote_X basse (<3.6) → 1X/X2
            #     MAIS bloque si l'adversaire est en DS form (+30pp) — leçon West Ham vs Everton
            if ft_p < 50 and m.odds_draw < 3.6:
                if ft_pick == "1":
                    p_1x = (pred5.get("p_h_blend", 0) + pred5.get("p_d_blend", 0)) * 100
                    if p_1x >= 65 and not away_in_ds_hot:
                        cote_1x = 1 / (p_1x/100) * 0.95
                        picks_this.append(("DOUBLE_CH", m.team_a, m.team_b, "1X", round(cote_1x, 2), p_1x/100))
                        print(f"│  🛡️  DOUBLE CHANCE 1X : ~@{cote_1x:.2f}  ({p_1x:.0f}%)")
                    elif away_in_ds_hot:
                        print(f"│  ⚠️  Double Chance 1X bloqué : {m.team_b} en DS form forte (+{DS_HOME_STRONG[m.team_b]*100:.0f}pp)")
                elif ft_pick == "2":
                    p_x2 = (pred5.get("p_d_blend", 0) + pred5.get("p_a_blend", 0)) * 100
                    home_in_ds_hot = (DS_HOME_STRONG.get(m.team_a, 0) >= 0.30)
                    if p_x2 >= 65 and not home_in_ds_hot:
                        cote_x2 = 1 / (p_x2/100) * 0.95
                        picks_this.append(("DOUBLE_CH", m.team_a, m.team_b, "X2", round(cote_x2, 2), p_x2/100))
                        print(f"│  🛡️  DOUBLE CHANCE X2 : ~@{cote_x2:.2f}  ({p_x2:.0f}%)")

            # 🆕 SIGNAL DS — DOWNGRADÉ après échecs Journée 10 (Everton/West Ham KO)
            # Désormais seul delta>=30pp ET cote home <= 2.5 valide DS GOLD
            if ds["delta"] >= 0.30 and m.odds_home <= 2.5:
                picks_this.append(("DS_HOME", m.team_a, m.team_b, "1", m.odds_home, ds["delta"]))
                print(f"│  💎 DS HOME (réduit) : 1 @{m.odds_home}  (delta DS {ds['delta']*100:.0f}pp)")
            elif ds["delta"] <= -0.20:
                picks_this.append(("DS_AWAY", m.team_a, m.team_b, "2", m.odds_away, -ds["delta"]))
                print(f"│  💎 DS AWAY : 2 @{m.odds_away}  (delta DS {ds['delta']*100:.0f}pp)")
            elif ds["delta"] >= 0.15:
                # Note informative sans pick automatique
                print(f"│  ℹ️  DS bonus home modéré (mise prudente)")

            # 🆕 PAIRE OR HOME : BLOQUE si home en perte DS (leçon C. Palace 1:1)
            if (m.team_a, m.team_b) in PAIR_HOME_GOLD:
                p = PAIR_HOME_GOLD[(m.team_a, m.team_b)]
                if home_in_ds_cold:
                    print(f"│  ⚠️  PAIRE OR HOME bloquée : {m.team_a} en perte DS ({DS_HOME_WEAK[m.team_a]*100:.0f}pp)")
                else:
                    print(f"│  💎 PAIRE OR HOME : 1 @{m.odds_home}  ({p['win']*100:.0f}% wins n={p['n']})")
                    picks_this.append(("PAIRE_OR_H", m.team_a, m.team_b, "1", m.odds_home, p['win']))

            # PAIRE OR AWAY
            if (m.team_a, m.team_b) in PAIR_AWAY_GOLD:
                p = PAIR_AWAY_GOLD[(m.team_a, m.team_b)]
                if m.odds_away <= p["cote"] * p.get("max_cote_factor", 1.05):
                    print(f"│  💎 PAIRE OR AWAY : 2 @{m.odds_away}  ({p['win']*100:.0f}% wins n={p['n']})")
                    picks_this.append(("PAIRE_OR_A", m.team_a, m.team_b, "2", m.odds_away, p['win']))

            # 🆕 OVER 2.5 FILTRÉ : raised à 3.5+ buts modèle (Fulham-Liv 3.37 buts → 1:1)
            if (m.team_a, m.team_b) in OVER_GOLD:
                og = OVER_GOLD[(m.team_a, m.team_b)]
                if lam_total >= 3.5:
                    print(f"│  💎 OVER 2.5 GOLD : {og['rate']*100:.0f}%  (modèle {lam_total:.2f} buts ✓)")
                    picks_this.append(("OVER", m.team_a, m.team_b, "Over 2.5", None, og['rate']))
                else:
                    print(f"│  ⚠️  Over 2.5 GOLD historique {og['rate']*100:.0f}% MAIS modèle {lam_total:.2f} buts (<3.5) — skip")
            # 🆕 UNDER 2.5 GOLD : renforcé (a marché J8)
            if (m.team_a, m.team_b) in UNDER_GOLD:
                ug = UNDER_GOLD[(m.team_a, m.team_b)]
                rate = 1 - ug['over_rate']
                print(f"│  💎💎 UNDER 2.5 GOLD : {rate*100:.0f}%")
                picks_this.append(("UNDER", m.team_a, m.team_b, "Under 2.5", None, rate))

            # BTTS
            if (m.team_a, m.team_b) in BTTS_OUI_GOLD:
                bg = BTTS_OUI_GOLD[(m.team_a, m.team_b)]
                if m.odds_home >= bg.get('min_cote_h', 1.8):
                    print(f"│  💎 BTTS OUI : {bg['rate']*100:.0f}%")
                    picks_this.append(("BTTS_OUI", m.team_a, m.team_b, "BTTS Oui", None, bg['rate']))
            if (m.team_a, m.team_b) in BTTS_NON_GOLD:
                bn = BTTS_NON_GOLD[(m.team_a, m.team_b)]
                print(f"│  💎 BTTS NON : {(1-bn['bts_rate'])*100:.0f}%")
                picks_this.append(("BTTS_NON", m.team_a, m.team_b, "BTTS Non", None, 1-bn['bts_rate']))

            # 🆕 SCORE COMBO : filtré à n>=8 ET combo >= 60%
            if (m.team_a, m.team_b) in SCORE_COMBO_GOLD:
                c = SCORE_COMBO_GOLD[(m.team_a, m.team_b)]
                if c.get('n', 0) >= 8 and c['combo'] >= 0.60:
                    print(f"│  💎 COMBO SCORE : {c['top1']}+{c['top2']} = {c['combo']*100:.0f}% (n={c.get('n')})")
                    picks_this.append(("COMBO", m.team_a, m.team_b, f"{c['top1']}+{c['top2']}", None, c['combo']))
                else:
                    print(f"│  ⚠️  COMBO score {c['top1']}+{c['top2']} {c['combo']*100:.0f}% mais critères réduits — skip")

            # 🆕 MULTI V10 : doublé en mise virtuelle si conf >= 8/10
            for outcome in ["1", "X", "2"]:
                agg = pred10["agg"][outcome]
                ev = pred10["ev_1x2"][outcome]
                conf = pred10["confidence"][outcome]
                cote = pred10["cotes"][outcome]
                if agg.get("has_pair_trap"): continue
                if agg["n_pos"] >= 2 and ev > 0.05 and not agg.get("has_pair_gold") and cote < 3:
                    marker = "🔥🔥 MULTI++" if conf >= 8 else "🔥 MULTI"
                    cat = "MULTI_HIGH" if conf >= 8 else "MULTI"
                    print(f"│  {marker} : {outcome} @{cote:.2f}  EV+{ev*100:.0f}% conf {conf}/10")
                    picks_this.append((cat, m.team_a, m.team_b, outcome, cote, conf/10))

            for p in picks_this:
                all_picks.append((TARGET, *p))

    # RECAP global
    print(f"\n\n{'═'*100}")
    print(f"  📋 RÉCAP TOUS LES ROUNDS — TOP PICKS PAR CATÉGORIE")
    print(f"{'═'*100}\n")
    by_cat = {}
    for round_t, cat, ha, hb, pick, cote, conf in all_picks:
        by_cat.setdefault(cat, []).append((round_t, ha, hb, pick, cote, conf))
    for cat in ["MULTI_HIGH", "FT_HIGH", "DOUBLE_CH", "UNDER", "BTTS_NON", "DS_HOME", "DS_AWAY", "PAIRE_OR_H", "PAIRE_OR_A", "OVER", "BTTS_OUI", "COMBO", "MULTI"]:
        if cat in by_cat:
            print(f"  {cat} ({len(by_cat[cat])}):")
            for r, ha, hb, pk, c, cf in sorted(by_cat[cat], key=lambda x: -x[5]):
                cstr = f"@{c:.2f}" if c else ""
                print(f"     {r}  {ha:<18} vs {hb:<18}  {pk:<12} {cstr}  ({cf*100:.0f}%)")
            print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
