# -*- coding: utf-8 -*-
"""
ADVERSARIAL VERIFICATION of F2:
  Market 'Total equipe domicile', selection '> 3.5' (home scores 4+),
  profile home_slight (1.6 <= oddsH < 2.2 AND oddsA >= 2.5, opening 1X2),
  selection odd in [5.0, 8.0). All segments.

Re-implemented FROM SCRATCH (only the SQL query shape and the definition
were taken from the miner's script; all settlement / split logic rewritten).

Protocol: walk-forward 3 windows on event-level temporal order:
  W1: train 0-50%   -> test 50.0-66.7%
  W2: train 0-66.7% -> test 66.7-83.3%
  W3: train 0-83.3% -> test 83.3-100%
Signal re-checked on each train (miner's own pre-registered acceptance:
roi_train >= +10% AND wins_train >= 8); test evaluated regardless.

Extra adversarial checks:
  - settlement on RAW official score (no majority-vote reconstruction)
  - reproduction of miner's 70/30 split (sanity: must match n=50 W=13)
  - duplicate event detection
  - temporal clustering of wins
"""
import sys, json
sys.path.insert(0, '.')
import numpy as np
from sqlalchemy import create_engine, text
from scraper.config import load_settings

MARKET = "Total equipe domicile"
SEL = "> 3.5"
ODD_LO, ODD_HI = 5.0, 8.0
MIN_ODD, MAX_ODD = 1.01, 35.0  # miner's global sanity filter


def load_rows():
    eng = create_engine(load_settings().db_url)
    q = """
    SELECT e.id, e.round_info, e.expected_start,
           os.odds_home, os.odds_away, os.extra_markets,
           r.score_a, r.score_b, r.ht_score_a, r.ht_score_b, r.goals_json
    FROM events e
    JOIN (SELECT event_id, MIN(id) mid FROM odds_snapshots GROUP BY event_id) f
         ON f.event_id = e.id
    JOIN odds_snapshots os ON os.id = f.mid
    JOIN results r ON r.event_id = e.id
    WHERE e.round_info IS NOT NULL AND e.round_info != '0'
      AND r.score_a IS NOT NULL AND r.score_b IS NOT NULL
      AND os.extra_markets IS NOT NULL
    ORDER BY e.expected_start, e.id
    """
    with eng.connect() as c:
        return c.execute(text(q)).fetchall()


def is_home_slight(oh, oa):
    return (oh is not None and oa is not None
            and 1.6 <= oh < 2.2 and oa >= 2.5)


def clean_home_score(sa, sb, hta, htb, gj):
    """My own re-implementation of the majority-vote 2-of-3 policy.
    Returns cleaned home score, or None -> drop row (ambiguous)."""
    goals = None
    if gj:
        try:
            goals = json.loads(gj) if isinstance(gj, str) else gj
        except json.JSONDecodeError:
            goals = None
    if goals:
        last = max(goals, key=lambda g: g["minute"])
        g_sa, g_sb = int(last["homeScore"]), int(last["awayScore"])
        if g_sa == sa and g_sb == sb and len(goals) == sa + sb:
            return sa
        if hta is None or htb is None:
            return None
        gh = sum(1 for x in goals if x["minute"] <= 45 and x["team"] == "Home")
        ga = sum(1 for x in goals if x["minute"] <= 45 and x["team"] == "Away")
        if (gh, ga) == (hta, htb) and len(goals) == g_sa + g_sb:
            return g_sa
        return None
    if goals == []:
        if sa + sb == 0 and (hta, htb) in ((0, 0), (None, None)):
            return 0
        return None
    if sa + sb == 0:
        if (hta, htb) in ((0, 0), (None, None)):
            return 0
        return None
    return sa  # gj null, score>0: score markets still settle on official


def build():
    rows = load_rows()
    ids = [r[0] for r in rows]
    n_dup = len(ids) - len(set(ids))
    print(f"loaded rows={len(rows)}  duplicate event ids={n_dup}")

    # event universes (temporal order preserved by the SQL ORDER BY)
    events_raw, events_clean = [], []   # ordered eids per variant
    bets_raw, bets_clean = [], []       # (eid, idx_later, odd, won, expected_start)
    sel_seen = set()
    for (eid, ri, es, oh, oa, em, sa, sb, hta, htb, gj) in rows:
        if isinstance(em, str):
            try:
                em = json.loads(em)
            except json.JSONDecodeError:
                continue
        if not isinstance(em, dict):
            continue
        d = em.get(MARKET)
        odd = None
        if isinstance(d, dict):
            sel_seen.update(d.keys())
            o = d.get(SEL)
            try:
                o = float(o) if o is not None else None
            except (TypeError, ValueError):
                o = None
            if o is not None and MIN_ODD <= o <= MAX_ODD:
                odd = o
        qual = (odd is not None and ODD_LO <= odd < ODD_HI
                and is_home_slight(oh, oa))

        # variant RAW: official score, no drops
        events_raw.append(eid)
        if qual:
            bets_raw.append([eid, len(events_raw) - 1, odd, int(sa >= 4), es])

        # variant CLEAN: miner's policy
        csa = clean_home_score(sa, sb, hta, htb, gj)
        if csa is None:
            continue
        events_clean.append(eid)
        if qual:
            bets_clean.append([eid, len(events_clean) - 1, odd,
                               int(csa >= 4), es])
    print(f"market selections seen: {sorted(sel_seen)}")
    print(f"events raw={len(events_raw)} clean={len(events_clean)}  "
          f"bets raw={len(bets_raw)} clean={len(bets_clean)}")
    return events_raw, bets_raw, events_clean, bets_clean


def met(bets):
    n = len(bets)
    if n == 0:
        return dict(n=0, wins=0, wr=np.nan, odd=np.nan, roi=np.nan)
    won = np.array([b[3] for b in bets], float)
    odd = np.array([b[2] for b in bets], float)
    pnl = won * (odd - 1) - (1 - won)
    return dict(n=n, wins=int(won.sum()), wr=won.mean(),
                odd=odd.mean(), roi=pnl.mean(), pnl=pnl)


def fmt(m):
    if m["n"] == 0:
        return "n=  0  (vide)"
    return (f"n={m['n']:3d} wins={m['wins']:3d} wr={m['wr']*100:5.1f}% "
            f"cote={m['odd']:5.2f} roi={m['roi']*100:+7.1f}%")


def run_walkforward(label, n_events, bets):
    print(f"\n=== WALK-FORWARD 3 FENETRES — variante {label} ===")
    bounds = [(0.0, 0.50, 2/3), (0.0, 2/3, 5/6), (0.0, 5/6, 1.0)]
    test_pool = []
    rois = []
    for i, (a, b, c) in enumerate(bounds, 1):
        i0, i1, i2 = int(n_events*a), int(n_events*b), int(n_events*c)
        tr = [x for x in bets if i0 <= x[1] < i1]
        te = [x for x in bets if i1 <= x[1] < i2]
        mtr, mte = met(tr), met(te)
        # miner's pre-registered acceptance recomputed on THIS train
        selected = (mtr["n"] > 0 and mtr["roi"] >= 0.10 and mtr["wins"] >= 8)
        print(f" W{i} train[0-{b*100:4.1f}%]: {fmt(mtr)}  "
              f"-> signal retenu sur train? {'OUI' if selected else 'NON'}")
        print(f"    test [{b*100:4.1f}-{c*100:5.1f}%]: {fmt(mte)}")
        test_pool += te
        rois.append(mte["roi"])
    agg = met(test_pool)
    pos = sum(1 for r in rois if not np.isnan(r) and r > 0)
    print(f" AGREGE (3 tests poolés 50-100%): {fmt(agg)}  "
          f"fenetres ROI>0: {pos}/3")
    if agg["n"]:
        rng = np.random.default_rng(7)
        pnl = agg["pnl"]
        boots = pnl[rng.integers(0, len(pnl), (10000, len(pnl)))].mean(axis=1)
        print(f" bootstrap P(roi<=0) = {(boots <= 0).mean():.3f}")
    return rois, agg, test_pool


def main():
    ev_raw, bets_raw, ev_clean, bets_clean = build()

    # --- sanity: reproduce miner's 70/30 split (variant CLEAN) -------------
    cut = int(len(ev_clean) * 0.70)
    oos = [b for b in bets_clean if b[1] >= cut]
    tr = [b for b in bets_clean if b[1] < cut]
    print(f"\n=== REPRODUCTION split mineur 70/30 (CLEAN) ===")
    print(f" train: {fmt(met(tr))}")
    print(f" OOS  : {fmt(met(oos))}   (claim: n=50 wins=13 wr=26.0% roi=+84.3%)")

    rois_c, agg_c, pool_c = run_walkforward("CLEAN (score reconstruit)",
                                            len(ev_clean), bets_clean)
    rois_r, agg_r, pool_r = run_walkforward("RAW (score officiel brut)",
                                            len(ev_raw), bets_raw)

    # --- temporal clustering of wins in pooled test (clean) ----------------
    print("\n=== Dispersion temporelle des wins (tests poolés, CLEAN) ===")
    wins = [(b[4], b[2]) for b in pool_c if b[3] == 1]
    for es, o in wins:
        print(f"  win @ {es}  cote={o:.2f}")

    # --- verdict ------------------------------------------------------------
    pos = sum(1 for r in rois_c if not np.isnan(r) and r > 0)
    roi_agg = agg_c["roi"]
    if pos >= 2 and roi_agg >= 0.08:
        v = "CONFIRMED"
    elif roi_agg > 0:
        v = "WEAKENED"
    else:
        v = "REFUTED"
    print(f"\nVERDICT (regles imposees, variante CLEAN): {v} "
          f"(fenetres ROI>0: {pos}/3, roi agrege={roi_agg*100:+.1f}%)")


if __name__ == "__main__":
    main()
