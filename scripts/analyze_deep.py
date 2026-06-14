"""Deep structured analysis on the joined dataset (cotes + score + extra markets).

Sections:
  A. Overview
  B. 1X2 — calibration globale + matrice cote-domicile x cote-exterieur
  C. Goals — total, Under/Over 2.5/3.5, distribution conditionnelle
  D. BTTS — global, par bracket de favori, value
  E. Half-time / Full-time : pattern HT->FT, comebacks
  F. Timing du premier but
  G. Combos de marches (e.g. fav home <1.5 + BTTS Non + Under 3.5)
  H. Synthese des signaux les plus actionables
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from scraper.config import load_settings


def _parse(em):
    if not em:
        return {}
    try:
        return em if isinstance(em, dict) else json.loads(em)
    except Exception:
        return {}


def _bucket_cote(c: float) -> str:
    if c < 1.3: return "[1.0-1.3]"
    if c < 1.5: return "[1.3-1.5]"
    if c < 1.8: return "[1.5-1.8]"
    if c < 2.1: return "[1.8-2.1]"
    if c < 2.5: return "[2.1-2.5]"
    if c < 3.0: return "[2.5-3.0]"
    if c < 4.0: return "[3.0-4.0]"
    if c < 6.0: return "[4.0-6.0]"
    return "[6.0+]"


def _hr(title: str) -> None:
    print(f"\n{'=' * 78}")
    print(f"  {title}")
    print("=" * 78)


def main() -> int:
    settings = load_settings()
    engine = create_engine(settings.db_url)

    df = pd.read_sql(
        """
        SELECT e.id, e.team_a, e.team_b, e.round_info,
               o.odds_home, o.odds_draw, o.odds_away, o.extra_markets,
               r.score_a, r.score_b
        FROM events e
        JOIN odds_snapshots o ON o.event_id = e.id
        JOIN results r ON r.event_id = e.id
        """,
        engine,
    )
    if df.empty:
        print("aucun match joint")
        return 1

    df["em"] = df["extra_markets"].apply(_parse)
    df["total_goals"] = df["score_a"] + df["score_b"]
    df["btts"] = (df["score_a"] > 0) & (df["score_b"] > 0)
    df["outcome"] = df.apply(
        lambda r: "1" if r.score_a > r.score_b
        else ("X" if r.score_a == r.score_b else "2"),
        axis=1,
    )
    df["fav_side"] = df.apply(
        lambda r: "1" if r.odds_home <= r.odds_away else "2", axis=1
    )
    df["fav_odds"] = df[["odds_home", "odds_away"]].min(axis=1)
    df["dog_odds"] = df[["odds_home", "odds_away"]].max(axis=1)
    df["fav_won"] = df["fav_side"] == df["outcome"]
    df["home_b"] = df["odds_home"].apply(_bucket_cote)
    df["away_b"] = df["odds_away"].apply(_bucket_cote)

    # HT/FT
    def _parse_ht(em):
        ht = em.get("halfTimeScore")
        if not ht or ":" not in str(ht):
            return None, None
        try:
            h, a = str(ht).split(":")
            return int(h.strip()), int(a.strip())
        except Exception:
            return None, None
    df[["ht_a", "ht_b"]] = df["em"].apply(lambda x: pd.Series(_parse_ht(x)))
    df["ht_total"] = df["ht_a"] + df["ht_b"]
    df["ht_outcome"] = df.apply(
        lambda r: None if pd.isna(r.ht_a)
        else ("1" if r.ht_a > r.ht_b
              else ("X" if r.ht_a == r.ht_b else "2")),
        axis=1,
    )
    df["second_half_goals"] = df["total_goals"] - df["ht_total"]
    df["comeback_home"] = (df["ht_outcome"] == "2") & (df["outcome"] == "1")
    df["comeback_away"] = (df["ht_outcome"] == "1") & (df["outcome"] == "2")
    df["leader_held"] = (df["ht_outcome"] == df["outcome"]) & (df["ht_outcome"] != "X")

    # First goal timing
    def _first_goal(em):
        gls = em.get("goals") or []
        mins = [g.get("minute") for g in gls if isinstance(g, dict) and g.get("minute") is not None]
        return min(mins) if mins else None
    df["first_goal_min"] = df["em"].apply(_first_goal)

    # Market odds extraction
    def _btts(em):
        g = em.get("G/NG") or {}
        return g.get("Oui"), g.get("Non")
    df[["btts_oui_odds", "btts_non_odds"]] = df["em"].apply(lambda x: pd.Series(_btts(x)))

    def _ou(em):
        m = em.get("+/-") or {}
        return m.get("> 3.5"), m.get("< 3.5")
    df[["over_35_odds", "under_35_odds"]] = df["em"].apply(lambda x: pd.Series(_ou(x)))

    # =================================================================
    # A. Overview
    # =================================================================
    _hr("A. ECHANTILLON")
    print(f"  matchs joints (cotes + score) : {len(df)}")
    print(f"  matchs avec HT score           : {df['ht_outcome'].notna().sum()}")
    print(f"  matchs avec liste de buts      : {df['first_goal_min'].notna().sum()}")
    print(f"  rounds couverts                : {df['round_info'].nunique()}")
    print(f"  competitions                   : {df['team_a'].nunique() + df['team_b'].nunique()} equipes distinctes")

    # =================================================================
    # B. 1X2 par cote brackets
    # =================================================================
    _hr("B. 1X2 — DISTRIBUTION GLOBALE")
    out_freq = df["outcome"].value_counts(normalize=True).reindex(["1", "X", "2"]) * 100
    print(f"  1 (domicile) : {out_freq.get('1', 0):.1f}%")
    print(f"  X (nul)      : {out_freq.get('X', 0):.1f}%")
    print(f"  2 (exterieur): {out_freq.get('2', 0):.1f}%")
    print(f"\n  Le favori (cote la plus basse) gagne : {df['fav_won'].mean()*100:.1f}%")
    print(f"  Le favori est le domicile dans       : {(df['fav_side']=='1').mean()*100:.1f}% des matchs")

    _hr("B.1 1X2 par bracket de cote DOMICILE")
    h_b = df.groupby("home_b").agg(
        n=("outcome", "size"),
        pct_1=("outcome", lambda s: (s == "1").mean() * 100),
        pct_X=("outcome", lambda s: (s == "X").mean() * 100),
        pct_2=("outcome", lambda s: (s == "2").mean() * 100),
        avg_cote_1=("odds_home", "mean"),
        avg_goals=("total_goals", "mean"),
    ).round(2)
    h_b["ROI_pari_1"] = (
        (df.assign(payoff=df.apply(lambda r: r.odds_home - 1 if r.outcome == "1" else -1, axis=1))
           .groupby("home_b")["payoff"].mean() * 100)
    ).round(1)
    print(h_b.to_string())

    _hr("B.2 1X2 par bracket de cote EXTERIEUR")
    a_b = df.groupby("away_b").agg(
        n=("outcome", "size"),
        pct_1=("outcome", lambda s: (s == "1").mean() * 100),
        pct_X=("outcome", lambda s: (s == "X").mean() * 100),
        pct_2=("outcome", lambda s: (s == "2").mean() * 100),
        avg_cote_2=("odds_away", "mean"),
        avg_goals=("total_goals", "mean"),
    ).round(2)
    a_b["ROI_pari_2"] = (
        (df.assign(payoff=df.apply(lambda r: r.odds_away - 1 if r.outcome == "2" else -1, axis=1))
           .groupby("away_b")["payoff"].mean() * 100)
    ).round(1)
    print(a_b.to_string())

    _hr("B.3 MATRICE — cote DOMICILE x cote EXTERIEUR — % victoire domicile")
    pivot_home_win = pd.pivot_table(
        df.assign(home_won=(df["outcome"] == "1").astype(int) * 100),
        values="home_won", index="home_b", columns="away_b",
        aggfunc="mean", fill_value=np.nan,
    ).round(0)
    n_matrix = pd.pivot_table(
        df.assign(one=1), values="one", index="home_b", columns="away_b",
        aggfunc="sum", fill_value=0,
    )
    print("  Probabilite que le DOMICILE gagne (cellule = % | n entre parentheses)")
    for i in pivot_home_win.index:
        line = f"  {i:<10}"
        for j in pivot_home_win.columns:
            v = pivot_home_win.loc[i, j]
            n = int(n_matrix.loc[i, j])
            cell = f"{v:>3.0f}%(n{n})" if not pd.isna(v) and n >= 3 else (f" -(n{n})" if n > 0 else "   .   ")
            line += f" {cell:<9}"
        print(line)
    print("  " + " " * 10 + " " + " ".join(f"{j:<9}" for j in pivot_home_win.columns))

    _hr("B.4 Cas specifiques demandes : 1.5-2.0 vs 2.9+")
    sub = df[(df["odds_home"].between(1.5, 2.0, inclusive="left"))
             & (df["odds_away"] >= 2.9)]
    if len(sub) >= 3:
        print(f"  n = {len(sub)} matchs (home dans [1.5;2.0[ et away >= 2.9)")
        print(f"  1: {(sub['outcome']=='1').mean()*100:.0f}%  X: {(sub['outcome']=='X').mean()*100:.0f}%  2: {(sub['outcome']=='2').mean()*100:.0f}%")
        print(f"  Moyenne buts : {sub['total_goals'].mean():.2f}")
        print(f"  BTTS Oui : {sub['btts'].mean()*100:.0f}%")
        print(f"  Under 3.5 buts : {(sub['total_goals']<3.5).mean()*100:.0f}%")
        roi_1 = sub.apply(lambda r: r.odds_home - 1 if r.outcome == "1" else -1, axis=1).mean() * 100
        print(f"  ROI pari sur 1 : {roi_1:+.1f}%")
    else:
        print(f"  echantillon trop petit (n={len(sub)})")

    sub2 = df[(df["odds_away"].between(1.5, 2.0, inclusive="left"))
              & (df["odds_home"] >= 2.9)]
    if len(sub2) >= 3:
        print(f"\n  Cas miroir : away dans [1.5;2.0[ et home >= 2.9 (n={len(sub2)})")
        print(f"  1: {(sub2['outcome']=='1').mean()*100:.0f}%  X: {(sub2['outcome']=='X').mean()*100:.0f}%  2: {(sub2['outcome']=='2').mean()*100:.0f}%")
        print(f"  BTTS Oui : {sub2['btts'].mean()*100:.0f}%")
        print(f"  Under 3.5 buts : {(sub2['total_goals']<3.5).mean()*100:.0f}%")
        roi_2 = sub2.apply(lambda r: r.odds_away - 1 if r.outcome == "2" else -1, axis=1).mean() * 100
        print(f"  ROI pari sur 2 : {roi_2:+.1f}%")

    # =================================================================
    # C. Goals
    # =================================================================
    _hr("C. NOMBRE DE BUTS — distribution & Under/Over")
    print(f"  Moyenne buts par match  : {df['total_goals'].mean():.2f}")
    print(f"  Mediane                 : {df['total_goals'].median():.1f}")
    print(f"  Buts domicile / ext     : {df['score_a'].mean():.2f} / {df['score_b'].mean():.2f}")
    print(f"  Matchs sans but (0-0)   : {(df['total_goals']==0).mean()*100:.1f}%")
    print("\n  Distribution exacte :")
    for g in range(0, 9):
        n = (df["total_goals"] == g).sum()
        pct = n / len(df) * 100
        bar = "#" * int(pct / 2)
        print(f"    {g} buts : {n:>3} ({pct:>4.1f}%) {bar}")
    print(f"\n  Under 1.5 : {(df['total_goals']<1.5).mean()*100:.1f}%")
    print(f"  Under 2.5 : {(df['total_goals']<2.5).mean()*100:.1f}%")
    print(f"  Under 3.5 : {(df['total_goals']<3.5).mean()*100:.1f}%")
    print(f"  Over  2.5 : {(df['total_goals']>2.5).mean()*100:.1f}%")
    print(f"  Over  3.5 : {(df['total_goals']>3.5).mean()*100:.1f}%")
    print(f"  Over  4.5 : {(df['total_goals']>4.5).mean()*100:.1f}%")

    _hr("C.1 Total buts par tranche de COTE DU FAVORI")
    df["fav_b"] = df["fav_odds"].apply(_bucket_cote)
    g_by_fav = df.groupby("fav_b").agg(
        n=("total_goals", "size"),
        avg_goals=("total_goals", "mean"),
        pct_under_25=("total_goals", lambda s: (s < 2.5).mean() * 100),
        pct_over_25=("total_goals", lambda s: (s > 2.5).mean() * 100),
        pct_under_35=("total_goals", lambda s: (s < 3.5).mean() * 100),
        pct_over_35=("total_goals", lambda s: (s > 3.5).mean() * 100),
        pct_0_0=("total_goals", lambda s: (s == 0).mean() * 100),
    ).round(2)
    print(g_by_fav.to_string())

    _hr("C.2 Over/Under 3.5 — backtest sur les cotes du marche")
    if df["over_35_odds"].notna().any():
        ovd = df.dropna(subset=["over_35_odds", "under_35_odds"]).copy()
        ovd["over_payoff"] = ovd.apply(
            lambda r: r.over_35_odds - 1 if r.total_goals > 3.5 else -1, axis=1
        )
        ovd["under_payoff"] = ovd.apply(
            lambda r: r.under_35_odds - 1 if r.total_goals < 3.5 else -1, axis=1
        )
        print(f"  n={len(ovd)} matchs")
        print(f"  Over 3.5  : hit={(ovd['total_goals']>3.5).mean()*100:.1f}%  cote moy={ovd['over_35_odds'].mean():.2f}  ROI={ovd['over_payoff'].mean()*100:+.1f}%")
        print(f"  Under 3.5 : hit={(ovd['total_goals']<3.5).mean()*100:.1f}%  cote moy={ovd['under_35_odds'].mean():.2f}  ROI={ovd['under_payoff'].mean()*100:+.1f}%")

    # =================================================================
    # D. BTTS
    # =================================================================
    _hr("D. BTTS — analyse profonde")
    print(f"  BTTS Oui (les deux marquent) : {df['btts'].mean()*100:.1f}%")
    print(f"  BTTS Non                     : {(~df['btts']).mean()*100:.1f}%")

    _hr("D.1 BTTS par bracket de cote DOMICILE")
    btts_h = df.groupby("home_b").agg(
        n=("btts", "size"),
        btts_oui_rate=("btts", lambda s: s.mean() * 100),
        avg_goals=("total_goals", "mean"),
    ).round(2)
    print(btts_h.to_string())

    _hr("D.2 BTTS — backtest sur les cotes du marche")
    if df["btts_oui_odds"].notna().any():
        bd = df.dropna(subset=["btts_oui_odds", "btts_non_odds"]).copy()
        bd["oui_payoff"] = bd.apply(lambda r: r.btts_oui_odds - 1 if r.btts else -1, axis=1)
        bd["non_payoff"] = bd.apply(lambda r: r.btts_non_odds - 1 if not r.btts else -1, axis=1)
        print(f"  n={len(bd)}")
        print(f"  BTTS Oui : hit={bd['btts'].mean()*100:.1f}%  cote moy={bd['btts_oui_odds'].mean():.2f}  ROI={bd['oui_payoff'].mean()*100:+.1f}%")
        print(f"  BTTS Non : hit={(~bd['btts']).mean()*100:.1f}%  cote moy={bd['btts_non_odds'].mean():.2f}  ROI={bd['non_payoff'].mean()*100:+.1f}%")

    # =================================================================
    # E. HT/FT
    # =================================================================
    _hr("E. MI-TEMPS / FIN — patterns")
    htf = df.dropna(subset=["ht_outcome"])
    print(f"  n avec HT score : {len(htf)}")
    print(f"  Buts moyens 1ere mi-temps : {htf['ht_total'].mean():.2f}")
    print(f"  Buts moyens 2eme mi-temps : {htf['second_half_goals'].mean():.2f}")
    print(f"  HT 0-0 : {(htf['ht_total']==0).mean()*100:.0f}%")
    print(f"  Le leader HT garde sa victoire : {htf['leader_held'].mean()*100:.0f}%")
    print(f"  Comeback domicile (HT perdant -> FT gagnant) : {htf['comeback_home'].mean()*100:.0f}%  ({htf['comeback_home'].sum()}/{len(htf)})")
    print(f"  Comeback exterieur                          : {htf['comeback_away'].mean()*100:.0f}%  ({htf['comeback_away'].sum()}/{len(htf)})")
    print("\n  Matrice HT -> FT :")
    ht_ft = pd.crosstab(htf["ht_outcome"], htf["outcome"], margins=True, margins_name="tot")
    print(ht_ft.to_string())

    # =================================================================
    # F. Timing du premier but
    # =================================================================
    _hr("F. TIMING DU PREMIER BUT")
    fg = df.dropna(subset=["first_goal_min"]).copy()
    if not fg.empty:
        print(f"  n={len(fg)} (matchs avec au moins un but)")
        print(f"  Minute moyenne du premier but : {fg['first_goal_min'].mean():.1f}")
        print(f"  Mediane                       : {fg['first_goal_min'].median():.0f}")
        print(f"  But avant 15e min  : {(fg['first_goal_min']<=15).mean()*100:.0f}%")
        print(f"  But avant 30e min  : {(fg['first_goal_min']<=30).mean()*100:.0f}%")
        print(f"  But avant 45e min  : {(fg['first_goal_min']<=45).mean()*100:.0f}%")
        print(f"  Premier but au 2T  : {(fg['first_goal_min']>45).mean()*100:.0f}%")

    # =================================================================
    # G. Combos
    # =================================================================
    _hr("G. COMBOS DE MARCHES — strategies croisees")

    # G.1 Gros favori home (<1.5) -> outcomes
    big_fav_h = df[df["odds_home"] < 1.5]
    print(f"\n  G.1 Gros favori DOMICILE (cote <1.5) : n={len(big_fav_h)}")
    if len(big_fav_h) >= 5:
        print(f"     Victoire home : {(big_fav_h['outcome']=='1').mean()*100:.0f}%")
        print(f"     Nul           : {(big_fav_h['outcome']=='X').mean()*100:.0f}%")
        print(f"     Defaite       : {(big_fav_h['outcome']=='2').mean()*100:.0f}%")
        print(f"     Buts moyens   : {big_fav_h['total_goals'].mean():.2f}")
        print(f"     BTTS Oui      : {big_fav_h['btts'].mean()*100:.0f}%")
        print(f"     Under 3.5     : {(big_fav_h['total_goals']<3.5).mean()*100:.0f}%")
        print(f"     Score modal   : {big_fav_h.apply(lambda r: f'{r.score_a}-{r.score_b}', axis=1).value_counts().head(3).to_dict()}")

    # G.2 Gros favori away (<1.5)
    big_fav_a = df[df["odds_away"] < 1.5]
    print(f"\n  G.2 Gros favori EXTERIEUR (cote <1.5) : n={len(big_fav_a)}")
    if len(big_fav_a) >= 5:
        print(f"     Victoire away : {(big_fav_a['outcome']=='2').mean()*100:.0f}%")
        print(f"     Nul           : {(big_fav_a['outcome']=='X').mean()*100:.0f}%")
        print(f"     Defaite       : {(big_fav_a['outcome']=='1').mean()*100:.0f}%")
        print(f"     BTTS Oui      : {big_fav_a['btts'].mean()*100:.0f}%")

    # G.3 Match equilibre (toutes cotes 1X2 entre 2.0 et 3.5)
    bal = df[(df["odds_home"].between(2.0, 3.5)) & (df["odds_away"].between(2.0, 3.5))]
    print(f"\n  G.3 Match EQUILIBRE (cotes 1 et 2 toutes deux dans [2.0;3.5]) : n={len(bal)}")
    if len(bal) >= 5:
        print(f"     1: {(bal['outcome']=='1').mean()*100:.0f}%  X: {(bal['outcome']=='X').mean()*100:.0f}%  2: {(bal['outcome']=='2').mean()*100:.0f}%")
        print(f"     Buts moyens : {bal['total_goals'].mean():.2f}  | BTTS Oui : {bal['btts'].mean()*100:.0f}%")
        # ROI sur le nul
        if bal["odds_draw"].notna().any():
            roi_x = bal.apply(lambda r: r.odds_draw - 1 if r.outcome == "X" else -1, axis=1).mean() * 100
            print(f"     ROI pari sur le nul X : {roi_x:+.1f}%")

    # G.4 Combo Under 3.5 + BTTS Non
    if df["under_35_odds"].notna().any() and df["btts_non_odds"].notna().any():
        combo = df.dropna(subset=["under_35_odds", "btts_non_odds"]).copy()
        combo["both_hit"] = (combo["total_goals"] < 3.5) & (~combo["btts"])
        combo_odds = combo["under_35_odds"] * combo["btts_non_odds"]  # cote combinee
        roi_combo = (combo["both_hit"] * combo_odds).mean() - 1
        print(f"\n  G.4 Combo Under 3.5 ET BTTS Non (n={len(combo)})")
        print(f"     Hit rate : {combo['both_hit'].mean()*100:.1f}%")
        print(f"     Cote combinee moyenne : {combo_odds.mean():.2f}")
        print(f"     ROI theorique : {roi_combo*100:+.1f}%")

    # =================================================================
    # H. Synthese
    # =================================================================
    _hr("H. SYNTHESE — signaux les plus actionables")
    signals = []
    if df["btts_non_odds"].notna().any():
        bd = df.dropna(subset=["btts_non_odds"])
        roi = bd.apply(lambda r: r.btts_non_odds - 1 if not r.btts else -1, axis=1).mean() * 100
        signals.append(("BTTS Non sur tous les matchs", len(bd), roi))
    if df["under_35_odds"].notna().any():
        ud = df.dropna(subset=["under_35_odds"])
        roi = ud.apply(lambda r: r.under_35_odds - 1 if r.total_goals < 3.5 else -1, axis=1).mean() * 100
        signals.append(("Under 3.5 sur tous les matchs", len(ud), roi))

    big_fav_h_full = df[df["odds_home"] < 1.5]
    if len(big_fav_h_full) >= 10:
        roi = big_fav_h_full.apply(lambda r: r.odds_home - 1 if r.outcome == "1" else -1, axis=1).mean() * 100
        signals.append(("Toujours pari domicile quand cote <1.5", len(big_fav_h_full), roi))

    # Combo
    if df["under_35_odds"].notna().any() and df["btts_non_odds"].notna().any():
        combo = df.dropna(subset=["under_35_odds", "btts_non_odds"]).copy()
        combo["both_hit"] = (combo["total_goals"] < 3.5) & (~combo["btts"])
        combo_odds = combo["under_35_odds"] * combo["btts_non_odds"]
        roi = (combo["both_hit"] * combo_odds).mean() - 1
        signals.append(("Combo Under 3.5 + BTTS Non", len(combo), roi * 100))

    signals.sort(key=lambda x: -x[2])
    print(f"  {'Strategie':<48} {'n':>5}  {'ROI':>8}")
    for name, n, roi in signals:
        flag = "++" if roi > 5 and n >= 50 else ("~" if abs(roi) < 3 else "  ")
        print(f"  {flag} {name:<46} {n:>5}  {roi:>+7.1f}%")
    print()
    print("  Legende : ++ = signal positif avec assez de donnees | ~ = trop proche de zero")

    return 0


if __name__ == "__main__":
    sys.exit(main())
