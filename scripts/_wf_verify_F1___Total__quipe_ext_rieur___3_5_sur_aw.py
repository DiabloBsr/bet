# -*- coding: utf-8 -*-
"""
VERIFICATION ADVERSARIALE — from scratch, indépendante du script du mineur.

Signal F1 : marché 'Total equipe extérieur', sélection '> 3.5' (équipe ext.
marque 4+), quand profil cote = away_slight (1.6 <= oddsA < 2.2 ET oddsH >= 2.5,
cotes d'ouverture = snapshot MIN(id)) ET segment = MS_mid (J13-25).
Cote sélection exclue si < 1.01 ou > 35.

Protocole : walk-forward 3 fenêtres
  W1: train [0%,50%)  -> test [50%,66%)
  W2: train [0%,66%)  -> test [66%,83%)
  W3: train [0%,83%)  -> test [83%,100%)
Le "signal" (la cellule mkt x sel x prof x seg) est re-vérifié sur chaque train
avec le critère d'acceptation pré-enregistré du mineur (roi_tr >= +10%,
wins_tr >= 8, n_tr >= 40) avant d'être parié sur la fenêtre test.

Sensibilités :
  - scores BRUTS (aucune reconstruction goals_json)
  - test sans les lignes reconstruites
  - bootstrap 20k sur l'agrégat des 3 fenêtres test
"""
import sys, json
sys.path.insert(0, '.')
import numpy as np
from sqlalchemy import create_engine, text
from scraper.config import load_settings

RULE_MKT, RULE_SEL = "Total equipe extérieur", "> 3.5"
MIN_ODD, MAX_ODD = 1.01, 35.0


# ------------------------------------------------------------------ data
def load():
    eng = create_engine(load_settings().db_url)
    q = """
    SELECT e.id, e.round_info, e.expected_start,
           os.odds_home, os.odds_away, os.extra_markets,
           r.score_a, r.score_b, r.ht_score_a, r.ht_score_b, r.goals_json
    FROM events e
    JOIN (SELECT event_id, MIN(id) AS mid
          FROM odds_snapshots GROUP BY event_id) f ON f.event_id = e.id
    JOIN odds_snapshots os ON os.id = f.mid
    JOIN results r ON r.event_id = e.id
    WHERE e.round_info IS NOT NULL AND e.round_info != '0'
      AND r.score_a IS NOT NULL AND r.score_b IS NOT NULL
      AND os.extra_markets IS NOT NULL
    ORDER BY e.expected_start, e.id
    """
    with eng.connect() as c:
        return c.execute(text(q)).fetchall()


def clean_away_score(sa, sb, hta, htb, gj):
    """Réimplémentation indépendante de la politique 2-sur-3 pré-enregistrée.
    Retourne (sb_corrige, reconstructed:bool) ou None (ligne ambiguë -> drop)."""
    goals = None
    if gj:
        try:
            goals = json.loads(gj) if isinstance(gj, str) else gj
        except (json.JSONDecodeError, TypeError):
            goals = None
    if goals:
        last = max(goals, key=lambda g: g["minute"])
        g_sa, g_sb = int(last["homeScore"]), int(last["awayScore"])
        if g_sa == sa and g_sb == sb and len(goals) == sa + sb:
            return sb, False
        if hta is None or htb is None:
            return None
        gh = sum(1 for x in goals if x["minute"] <= 45 and x["team"] == "Home")
        ga = sum(1 for x in goals if x["minute"] <= 45 and x["team"] == "Away")
        if (gh, ga) == (hta, htb) and len(goals) == g_sa + g_sb:
            return g_sb, True
        return None
    if goals == []:
        if sa + sb == 0 and (hta, htb) in ((0, 0), (None, None)):
            return 0, False
        return None
    if sa + sb == 0:
        if (hta, htb) in ((0, 0), (None, None)):
            return 0, False
        return None
    return sb, False


def build():
    """Une ligne par évènement exploitable, ordonnée temporellement.
    bet=True si la cellule s'applique ET la cote est exploitable."""
    rows = load()
    evs = []        # politique nettoyée (celle du mineur)
    evs_raw = []    # sensibilité : scores bruts, zéro reconstruction/drop
    n_amb = n_cap = 0
    for (eid, ri, _es, oh, oa, em, sa, sb, hta, htb, gj) in rows:
        if isinstance(em, str):
            try:
                em = json.loads(em)
            except json.JSONDecodeError:
                continue
        if not isinstance(em, dict):
            continue
        try:
            j = int(ri)
        except (TypeError, ValueError):
            continue
        if j <= 0:
            continue
        in_seg = 13 <= j <= 25
        in_prof = (oh is not None and oa is not None
                   and 1.6 <= oa < 2.2 and oh >= 2.5)
        odd = None
        d = em.get(RULE_MKT)
        if isinstance(d, dict) and d.get(RULE_SEL) is not None:
            try:
                o = float(d[RULE_SEL])
                if MIN_ODD <= o <= MAX_ODD:
                    odd = o
                elif in_seg and in_prof:
                    n_cap += 1
            except (TypeError, ValueError):
                pass
        cell = in_seg and in_prof and odd is not None
        # version brute (toutes lignes gardées, score officiel)
        evs_raw.append((eid, cell, odd, int(sb >= 4)))
        # version nettoyée
        cs = clean_away_score(sa, sb, hta, htb, gj)
        if cs is None:
            n_amb += 1
            continue
        sb_c, reco = cs
        evs.append((eid, cell, odd, int(sb_c >= 4), reco))
    print(f"évènements exploitables: {len(rows)} | drops ambigus: {n_amb} | "
          f"cellule mais cote cappée/horslimites: {n_cap}")
    return evs, evs_raw


# ------------------------------------------------------------------ metrics
def met(bets):
    """bets = list of (odd, won)."""
    n = len(bets)
    if n == 0:
        return dict(n=0, wins=0, wr=np.nan, cote=np.nan, roi=np.nan)
    odds = np.array([b[0] for b in bets], float)
    won = np.array([b[1] for b in bets], float)
    pnl = won * (odds - 1) - (1 - won)
    return dict(n=n, wins=int(won.sum()), wr=won.mean(), cote=odds.mean(),
                roi=pnl.mean(), pnl=pnl)


def fmt(m):
    if m["n"] == 0:
        return "n=  0"
    return (f"n={m['n']:4d} wins={m['wins']:3d} wr={m['wr']*100:5.1f}% "
            f"cote={m['cote']:5.2f} roi={m['roi']*100:+7.1f}%")


def run_wf(evs, label, has_reco):
    print(f"\n=== WALK-FORWARD 3 FENÊTRES — {label} ===")
    N = len(evs)
    bounds = [(0.0, 0.50, 0.66), (0.0, 0.66, 0.83), (0.0, 0.83, 1.00)]
    agg, agg_gated = [], []
    for w, (a, b, c) in enumerate(bounds, 1):
        tr = evs[int(N * a):int(N * b)]
        te = evs[int(N * b):int(N * c)]
        tr_bets = [(e[2], e[3]) for e in tr if e[1]]
        te_bets = [(e[2], e[3]) for e in te if e[1]]
        mtr, mte = met(tr_bets), met(te_bets)
        accepted = (mtr["n"] >= 40 and mtr["wins"] >= 8 and mtr["roi"] >= 0.10)
        print(f" W{w} train[{a:.0%}-{b:.0%}] {fmt(mtr)}  "
              f"-> accepté={'OUI' if accepted else 'NON'}")
        print(f"    test [{b:.0%}-{c:.0%}] {fmt(mte)}")
        if has_reco:
            te_nr = [(e[2], e[3]) for e in te if e[1] and not e[4]]
            print(f"    test sans reconstruits: {fmt(met(te_nr))}")
        agg += te_bets
        if accepted:
            agg_gated += te_bets
    ma = met(agg)
    print(f" AGRÉGÉ 3 fenêtres test : {fmt(ma)}")
    if agg_gated and len(agg_gated) != len(agg):
        print(f" AGRÉGÉ (fenêtres acceptées seulement) : {fmt(met(agg_gated))}")
    elif not agg_gated:
        print(" (aucune fenêtre acceptée par le critère train)")
    # bootstrap sur l'agrégat
    if ma["n"] > 0:
        rng = np.random.default_rng(7)
        pnl = ma["pnl"]
        boots = pnl[rng.integers(0, len(pnl), size=(20000, len(pnl)))].mean(1)
        print(f" bootstrap 20k : P(roi<=0) = {(boots <= 0).mean():.4f}  "
              f"IC90% roi = [{np.percentile(boots,5)*100:+.1f}%, "
              f"{np.percentile(boots,95)*100:+.1f}%]")
    return ma


def main():
    evs, evs_raw = build()

    # -- sanity : base rate & répartition des cotes de la cellule
    cell = [e for e in evs if e[1]]
    all_odds = np.array([e[2] for e in cell])
    print(f"\ncellule totale (toutes périodes): n={len(cell)}, "
          f"wins={sum(e[3] for e in cell)}, "
          f"wr={np.mean([e[3] for e in cell])*100:.1f}%, "
          f"cote moy={all_odds.mean():.2f}, "
          f"implicite moy={np.mean(1/all_odds)*100:.1f}%")
    base = np.mean([e[3] for e in evs])
    print(f"base rate sb>=4 (tous matchs nettoyés): {base*100:.2f}%")
    qs = np.percentile(all_odds, [0, 25, 50, 75, 100])
    print("cotes cellule percentiles 0/25/50/75/100:",
          " ".join(f"{x:.2f}" for x in qs))

    # -- réplication du split 70/30 du mineur (contrôle de cohérence)
    N = len(evs)
    cut = int(N * 0.70)
    print("\n--- contrôle : réplication split 70/30 du mineur ---")
    print(" train:", fmt(met([(e[2], e[3]) for e in evs[:cut] if e[1]])))
    print(" OOS  :", fmt(met([(e[2], e[3]) for e in evs[cut:] if e[1]])))

    # -- walk-forward principal (politique de score du mineur)
    main_m = run_wf(evs, "scores nettoyés (politique 2-sur-3)", has_reco=True)

    # -- sensibilité : scores bruts, aucune reconstruction ni drop
    run_wf([e + (False,) for e in evs_raw],
           "scores BRUTS (aucune reconstruction)", has_reco=False)

    print("\n=== RÉSUMÉ AGRÉGÉ (politique principale) ===")
    print(fmt(main_m))


if __name__ == "__main__":
    main()
