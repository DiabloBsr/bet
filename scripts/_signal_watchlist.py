"""SIGNAL WATCHLIST — paper-trading forward des signaux limites (z entre 1 et 2).

Principe (campagne fondamentale 2026-06-11) : ces signaux sont trop faibles pour être
prouvés sur 2 semaines de données, trop intéressants pour être jetés. On FIGE leur
définition aujourd'hui, puis on les évalue UNIQUEMENT sur les matchs postérieurs au
gel (données jamais vues par aucune analyse = estimateur non contaminé).

PROMOTION  → "BETTABLE" si z forward >= 2.0 ET n >= 80 ET ROI > 0
DÉMOTION   → "DEAD"     si n >= 80 ET (ROI <= -10% OU z <= -1.5)
Sinon      → "WATCHING"

Usage : PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe scripts/_signal_watchlist.py
(à lancer 1×/jour ou après chaque grosse session de scraping)
"""
from __future__ import annotations
import sys, json, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
from sqlalchemy import create_engine
from scraper.config import load_settings

REGISTRY_PATH = Path(__file__).resolve().parents[1] / "data" / "watchlist_registry.json"

# ═══════════════════════════════════════════════════════════════════════════
# REGISTRE — définitions FIGÉES le 2026-06-11. NE PLUS MODIFIER les définitions
# (sinon le test forward est contaminé). Ajouter de nouveaux signaux = OK.
# ═══════════════════════════════════════════════════════════════════════════
FROZEN_AT = "2026-06-11T17:00:00"

SIGNALS = {
    "fade_serie_5plus": {
        "desc": "Équipe en série de >=5 victoires intra-saison → parier l'ADVERSAIRE au 1X2",
        "origin": "OOS: équipe en série WR 44.6% vs 53.7% pricé (z=-1.37, n=116)",
    },
    "fade_serie_5plus_draw": {
        "desc": "Équipe en série de >=5 victoires intra-saison → parier le NUL",
        "origin": "variante fade via X (cote ~4) — non sélectionnée sur train, à pister",
    },
    "sous_regime_rebond": {
        "desc": "Équipe avec WR saison courante <= WR all-time -25pp (>=8 matchs saison) → parier SUR elle",
        "origin": "OOS: WR 54.2% vs 43.6% pricé (z=1.81, ROI +40%, n=72) — non répliqué train",
    },
    "standings_pos_gap5": {
        "desc": "Underdog de cote MAIS mieux classé de >=5 places (classement intra-saison, J>=6) → back l'underdog",
        "origin": "OOS: ROI +18.4% (n=176, z=1.34) — signe inversé sur train",
    },
    "standings_pts_gap5": {
        "desc": "Underdog de cote MAIS avance de >=5 points (intra-saison, J>=6) → back l'underdog",
        "origin": "OOS: ROI +14.4% (n=180, z=1.14)",
    },
    "value_home_vs_alltime": {
        "desc": "Cote home >= 2.5 alors que WR home all-time de l'équipe >= 45% → back home",
        "origin": "OOS: ROI +9.8% (n=280) — mispricing potentiel du niveau réel",
    },
    # ── Ajouts campagne reverse-engineering (2026-06-12, gel à frozen_at_v2) ──
    "value_jitter_pair": {
        "desc": "Cote 1X2 publiée > juste cote de la paire (freq hist. train, n_prior>=8, EV est. >= 0.98) → back ce côté",
        "origin": "identity: OOS +18.6% (n=345, p=0.065) ; mécanisme jitter prouvé (99% variance = bruit de publication)",
    },
    "mitps_longshot_global": {
        "desc": "Mi-tps 1X2 : sélection home avec cote 1X2 home >= 4.0 (longshot) → back '1' mi-temps, tous segments",
        "origin": "htft7: P(W>=15)=0.03, ROI +59.8% (n=136) ; généralisation du S3",
    },
    "follow_drift": {
        "desc": "Si la cote 1X2 du DERNIER snapshot dévie de l'ouverture de >=0.03 logit vers un côté → back ce côté au dernier prix",
        "origin": "templates: OOS30 +11.5% (n=197) ; season: +5.3% (p=0.045) ; le close corrige le jitter (corr -0.70)",
    },
}
FROZEN_AT_V2 = "2026-06-12T01:30:00"   # gel des 3 signaux reverse-engineering


def _z_score(wins: int, n: int, p_implied: float) -> float:
    """Z-score du WR observé vs la proba implicite moyenne des cotes prises."""
    if n == 0: return 0.0
    p_obs = wins / n
    se = math.sqrt(p_implied * (1 - p_implied) / n)
    return (p_obs - p_implied) / se if se > 0 else 0.0


def load_dataset(engine) -> pd.DataFrame:
    df = pd.read_sql("""
        SELECT e.id, e.round_info, e.team_a, e.team_b, e.expected_start,
               o.odds_home, o.odds_draw, o.odds_away, o.extra_markets,
               ol.odds_home AS last_h, ol.odds_draw AS last_d, ol.odds_away AS last_a,
               r.score_a, r.score_b, r.ht_score_a, r.ht_score_b
        FROM events e
        JOIN odds_snapshots o ON o.id = (SELECT MIN(id) FROM odds_snapshots WHERE event_id = e.id)
        JOIN odds_snapshots ol ON ol.id = (SELECT MAX(id) FROM odds_snapshots WHERE event_id = e.id)
        JOIN results r ON r.event_id = e.id
        WHERE r.score_a IS NOT NULL AND e.round_info IS NOT NULL AND e.round_info != '0' AND e.competition = 'InstantLeague-8035'
        ORDER BY e.expected_start
    """, engine)
    df["journee"] = pd.to_numeric(df.round_info, errors="coerce")
    df = df.dropna(subset=["journee"]).copy()
    df["journee"] = df.journee.astype(int)
    df["expected_start"] = pd.to_datetime(df.expected_start)
    df = df.drop_duplicates(["team_a", "team_b", "expected_start"]).reset_index(drop=True)
    df["ft"] = np.where(df.score_a > df.score_b, "1",
                np.where(df.score_a == df.score_b, "X", "2"))

    # season_id : nouveau segment quand la journée redescend (trié par expected_start)
    season_id, cur = [], 0
    prev_j = None
    for j in df.journee:
        if prev_j is not None and j < prev_j - 3:
            cur += 1
        season_id.append(cur)
        prev_j = j
    df["season_id"] = season_id
    return df


def build_prematch_features(df: pd.DataFrame) -> pd.DataFrame:
    """Séries, WR saison, WR all-time, points/position intra-saison — info pré-match only."""
    # état par équipe
    alltime = {}   # team -> {"wins": int, "n": int}  (tous matchs home+away)
    season = {}    # (season_id, team) -> {"wins","n","streak","pts"}
    rows = []
    for _, m in df.iterrows():
        sid = m.season_id
        fa = season.setdefault((sid, m.team_a), {"wins": 0, "n": 0, "streak": 0, "pts": 0})
        fb = season.setdefault((sid, m.team_b), {"wins": 0, "n": 0, "streak": 0, "pts": 0})
        aa = alltime.setdefault(m.team_a, {"wins": 0, "n": 0})
        ab = alltime.setdefault(m.team_b, {"wins": 0, "n": 0})

        # positions intra-saison (rang par points parmi les équipes vues cette saison)
        season_teams = {t: s for (s_id, t), s in season.items() if s_id == sid}
        ranked = sorted(season_teams.items(), key=lambda kv: -kv[1]["pts"])
        pos = {t: i + 1 for i, (t, _) in enumerate(ranked)}

        rows.append({
            "idx": m.name,
            "streak_h": fa["streak"], "streak_a": fb["streak"],
            "season_wr_h": fa["wins"] / fa["n"] if fa["n"] else None,
            "season_wr_a": fb["wins"] / fb["n"] if fb["n"] else None,
            "season_n_h": fa["n"], "season_n_a": fb["n"],
            "alltime_wr_h": aa["wins"] / aa["n"] if aa["n"] else None,
            "alltime_wr_a": ab["wins"] / ab["n"] if ab["n"] else None,
            "alltime_n_h": aa["n"], "alltime_n_a": ab["n"],
            "pts_h": fa["pts"], "pts_a": fb["pts"],
            "pos_h": pos.get(m.team_a), "pos_a": pos.get(m.team_b),
        })

        # update post-match
        if m.ft == "1":
            fa["wins"] += 1; fa["streak"] = fa["streak"] + 1 if fa["streak"] >= 0 else 1
            fa["pts"] += 3
            fb["streak"] = min(fb["streak"], 0) - 1
            aa["wins"] += 1
        elif m.ft == "2":
            fb["wins"] += 1; fb["streak"] = fb["streak"] + 1 if fb["streak"] >= 0 else 1
            fb["pts"] += 3
            fa["streak"] = min(fa["streak"], 0) - 1
            ab["wins"] += 1
        else:
            fa["streak"] = 0; fb["streak"] = 0
            fa["pts"] += 1; fb["pts"] += 1
        fa["n"] += 1; fb["n"] += 1
        aa["n"] += 1; ab["n"] += 1

    feats = pd.DataFrame(rows).set_index("idx")
    return df.join(feats)


def apply_signals(d: pd.DataFrame) -> dict:
    """Applique chaque signal figé. Retourne {signal: DataFrame des paris(pick, cote, won)}."""
    out = {}
    devig = lambda r: (1/r.odds_home + 1/r.odds_draw + 1/r.odds_away)

    # 1. fade_serie_5plus : équipe en série >=5 → back l'adversaire
    bets = []
    for _, r in d.iterrows():
        if r.streak_h >= 5:
            bets.append({"pick": "2", "cote": r.odds_away, "won": r.ft == "2", "ts": r.expected_start})
        elif r.streak_a >= 5:
            bets.append({"pick": "1", "cote": r.odds_home, "won": r.ft == "1", "ts": r.expected_start})
    out["fade_serie_5plus"] = pd.DataFrame(bets)

    # 2. fade_serie_5plus_draw : équipe en série >=5 → back le nul
    bets = []
    for _, r in d.iterrows():
        if r.streak_h >= 5 or r.streak_a >= 5:
            bets.append({"pick": "X", "cote": r.odds_draw, "won": r.ft == "X", "ts": r.expected_start})
    out["fade_serie_5plus_draw"] = pd.DataFrame(bets)

    # 3. sous_regime_rebond : WR saison <= WR all-time - 25pp (n_saison >= 8, n_alltime >= 30)
    bets = []
    for _, r in d.iterrows():
        if (r.season_n_h >= 8 and r.alltime_n_h >= 30 and r.season_wr_h is not None
                and r.alltime_wr_h is not None and r.season_wr_h <= r.alltime_wr_h - 0.25):
            bets.append({"pick": "1", "cote": r.odds_home, "won": r.ft == "1", "ts": r.expected_start})
        if (r.season_n_a >= 8 and r.alltime_n_a >= 30 and r.season_wr_a is not None
                and r.alltime_wr_a is not None and r.season_wr_a <= r.alltime_wr_a - 0.25):
            bets.append({"pick": "2", "cote": r.odds_away, "won": r.ft == "2", "ts": r.expected_start})
    out["sous_regime_rebond"] = pd.DataFrame(bets)

    # 4. standings_pos_gap5 : underdog cote mais position meilleure de >=5 places (J>=6)
    bets = []
    for _, r in d.iterrows():
        if r.journee < 6 or r.pos_h is None or r.pos_a is None: continue
        fav_home = r.odds_home < r.odds_away
        if fav_home and (r.pos_h - r.pos_a) >= 5:    # away mieux classé mais underdog
            bets.append({"pick": "2", "cote": r.odds_away, "won": r.ft == "2", "ts": r.expected_start})
        elif (not fav_home) and (r.pos_a - r.pos_h) >= 5:  # home mieux classé mais underdog
            bets.append({"pick": "1", "cote": r.odds_home, "won": r.ft == "1", "ts": r.expected_start})
    out["standings_pos_gap5"] = pd.DataFrame(bets)

    # 5. standings_pts_gap5 : underdog cote mais avance >=5 points (J>=6)
    bets = []
    for _, r in d.iterrows():
        if r.journee < 6: continue
        fav_home = r.odds_home < r.odds_away
        if fav_home and (r.pts_a - r.pts_h) >= 5:
            bets.append({"pick": "2", "cote": r.odds_away, "won": r.ft == "2", "ts": r.expected_start})
        elif (not fav_home) and (r.pts_h - r.pts_a) >= 5:
            bets.append({"pick": "1", "cote": r.odds_home, "won": r.ft == "1", "ts": r.expected_start})
    out["standings_pts_gap5"] = pd.DataFrame(bets)

    # 6. value_home_vs_alltime : cote home >= 2.5 mais WR home all-time >= 45% (n>=30)
    bets = []
    for _, r in d.iterrows():
        if r.odds_home >= 2.5 and r.alltime_n_h >= 30 and r.alltime_wr_h is not None and r.alltime_wr_h >= 0.45:
            bets.append({"pick": "1", "cote": r.odds_home, "won": r.ft == "1", "ts": r.expected_start})
    out["value_home_vs_alltime"] = pd.DataFrame(bets)

    # ── 7. value_jitter_pair : EV_est = freq_prior_paire × cote >= 0.98 (n_prior >= 8) ──
    import math as _math
    pair_hist: dict = {}   # (ta,tb) -> {"1": n, "X": n, "2": n, "n": n}
    bets = []
    for _, r in d.iterrows():
        key = (r.team_a, r.team_b)
        h = pair_hist.get(key)
        if h and h["n"] >= 8:
            for side, cote in (("1", r.odds_home), ("2", r.odds_away)):
                freq = h[side] / h["n"]
                if freq * cote >= 0.98:
                    bets.append({"pick": side, "cote": cote, "won": r.ft == side, "ts": r.expected_start})
        if h is None:
            h = pair_hist[key] = {"1": 0, "X": 0, "2": 0, "n": 0}
        h[r.ft] += 1; h["n"] += 1
    out["value_jitter_pair"] = pd.DataFrame(bets)

    # ── 8. mitps_longshot_global : odds_home >= 4.0 → back 'Mi-tps 1X2' '1' ──
    def _em(x):
        if isinstance(x, str):
            try: return json.loads(x)
            except Exception: return {}
        return x if isinstance(x, dict) else {}
    bets = []
    for _, r in d.iterrows():
        if r.odds_home < 4.0 or pd.isna(r.ht_score_a): continue
        em = _em(r.extra_markets)
        mt = em.get("Mi-tps 1X2")
        cote = mt.get("1") if isinstance(mt, dict) else None
        if not isinstance(cote, (int, float)) or cote <= 1.01: continue
        won = int(r.ht_score_a) > int(r.ht_score_b)
        bets.append({"pick": "HT 1", "cote": float(cote), "won": won, "ts": r.expected_start})
    out["mitps_longshot_global"] = pd.DataFrame(bets)

    # ── 9. follow_drift : |delta logit(pH)| open→last >= 0.03 → back le côté du mouvement ──
    bets = []
    for _, r in d.iterrows():
        if pd.isna(r.last_h) or pd.isna(r.last_a): continue
        try:
            p_open = (1/r.odds_home) / (1/r.odds_home + 1/r.odds_draw + 1/r.odds_away)
            p_last = (1/r.last_h) / (1/r.last_h + 1/r.last_d + 1/r.last_a)
            delta = _math.log(p_last/(1-p_last)) - _math.log(p_open/(1-p_open))
        except (ValueError, ZeroDivisionError):
            continue
        if abs(delta) < 0.03: continue
        if delta > 0:   # marché monte sur home
            bets.append({"pick": "1", "cote": float(r.last_h), "won": r.ft == "1", "ts": r.expected_start})
        else:
            bets.append({"pick": "2", "cote": float(r.last_a), "won": r.ft == "2", "ts": r.expected_start})
    out["follow_drift"] = pd.DataFrame(bets)

    return out


def main():
    settings = load_settings()
    engine = create_engine(settings.db_url)
    frozen_ts = pd.Timestamp(FROZEN_AT)

    print(f"📡 SIGNAL WATCHLIST — gel des définitions : {FROZEN_AT}")
    df = load_dataset(engine)
    print(f"   Dataset : {len(df):,} matchs ({df.expected_start.min()} → {df.expected_start.max()})")
    d = build_prematch_features(df)
    all_bets = apply_signals(d)

    # Charger/initialiser le registre
    registry = {}
    if REGISTRY_PATH.exists():
        registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))

    print(f"\n{'═'*100}")
    print(f"  {'SIGNAL':<28} {'PÉRIODE':<10} {'n':<6} {'WR':<8} {'cote moy':<9} {'ROI':<9} {'z':<7} {'STATUT'}")
    print(f"{'═'*100}")

    V2_SIGNALS = {"value_jitter_pair", "mitps_longshot_global", "follow_drift"}
    frozen_v2_ts = pd.Timestamp(FROZEN_AT_V2)

    report = {}
    for sig_name, meta in SIGNALS.items():
        bets = all_bets.get(sig_name, pd.DataFrame())
        if bets.empty:
            print(f"  {sig_name:<28} {'—':<10} 0")
            continue
        sig_frozen = frozen_v2_ts if sig_name in V2_SIGNALS else frozen_ts
        for period, sub in [("HISTO*", bets[bets.ts < sig_frozen]), ("FORWARD", bets[bets.ts >= sig_frozen])]:
            n = len(sub)
            if n == 0:
                print(f"  {sig_name:<28} {period:<10} 0      —        —         —         —      —")
                continue
            wins = int(sub.won.sum())
            wr = wins / n
            avg_cote = sub.cote.mean()
            roi = (sub.won * (sub.cote - 1) - (~sub.won)).mean()
            p_impl = (1 / sub.cote).mean() * 0.93   # devig approx (overround ~7%)
            z = _z_score(wins, n, p_impl)
            status = ""
            if period == "FORWARD":
                if z >= 2.0 and n >= 80 and roi > 0:
                    status = "🟢 PROMOTED → BETTABLE"
                elif n >= 80 and (roi <= -0.10 or z <= -1.5):
                    status = "🔴 DEMOTED → DEAD"
                else:
                    status = f"👁 WATCHING ({max(0, 80-n)} matchs restants avant verdict)"
                report[sig_name] = {"n": n, "wr": wr, "roi": float(roi), "z": z, "status": status}
            print(f"  {sig_name:<28} {period:<10} {n:<6} {wr*100:5.1f}%  {avg_cote:6.2f}   {roi*100:+6.1f}%  {z:+5.2f}  {status}")

    print(f"\n  (* HISTO = données pré-gel, CONTAMINÉES par la sélection — référence seulement.")
    print(f"     Seule la ligne FORWARD compte pour la promotion.)")

    # Sauvegarder l'état
    registry["frozen_at"] = FROZEN_AT
    registry["last_run"] = pd.Timestamp.now().isoformat()
    registry["signals"] = {k: {**SIGNALS[k], **report.get(k, {})} for k in SIGNALS}
    REGISTRY_PATH.parent.mkdir(exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n💾 Registre : {REGISTRY_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
