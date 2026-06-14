# -*- coding: utf-8 -*-
"""
ADVERSARIAL VERIFICATION of F3:
  Market 'Total de buts', selection '1' (exactly 1 goal in the match),
  profile home_slight (1.6 <= odds_home < 2.2 AND odds_away >= 2.5),
  segment MS_early (J4-12).

Re-implemented FROM SCRATCH (only the rule definition is taken from the
miner's script, none of its code is imported).

Protocol (different split from the miner's single 70/30):
  Walk-forward, 3 windows over events sorted by expected_start:
    W1: train [0%,50%)  -> test [50%,66%)
    W2: train [0%,66%)  -> test [66%,83%)
    W3: train [0%,83%)  -> test [83%,100%)
  On each train we recompute the cell's train metrics and check whether the
  miner's own pre-registered selection gate (roi_tr>=10%, wins_tr>=8,
  n_tr>=40) would have picked it. Test ROI is reported per window and pooled.

Robustness variants:
  A) RAW scores: official score_a/score_b, no reconstruction, no drops.
  B) CLEAN: my own re-implementation of the majority-vote 2-of-3 cleaning.

Extra checks:
  - duplicate events (same teams + same expected_start)
  - replication of the miner's 70/30 OOS claim (n=132, wr=19.7%, roi=+45.6%)
  - implied vs realized P(total==1) on the cell
  - bootstrap p-value on pooled walk-forward test pnl
"""
import sys, json
sys.path.insert(0, '.')
import numpy as np
from sqlalchemy import create_engine, text
from scraper.config import load_settings

MIN_ODD, MAX_ODD = 1.01, 35.0


# ----------------------------------------------------------------- helpers
def profile(oh, oa):
    if oh is None or oa is None:
        return "other"
    if oh < 1.3 and oa > 7:
        return "home_crush"
    if oh < 1.6 and oa > 4:
        return "home_strong"
    if 1.6 <= oh < 2.2 and oa >= 2.5:
        return "home_slight"
    return "other"


def seg(round_info):
    try:
        j = int(round_info)
    except (TypeError, ValueError):
        return None
    if j <= 0:
        return None
    if j <= 3:
        return "DS"
    if j <= 12:
        return "MS_early"
    if j <= 25:
        return "MS_mid"
    if j <= 33:
        return "MS_late"
    return "FS"


def clean_score(sa, sb, hta, htb, gj):
    """My own re-implementation of the majority-vote cleaning.
    Returns (sa, sb) or None to drop."""
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
            return sa, sb
        if hta is None or htb is None:
            return None
        gh = sum(1 for x in goals if x["minute"] <= 45 and x["team"] == "Home")
        ga = sum(1 for x in goals if x["minute"] <= 45 and x["team"] == "Away")
        if (gh, ga) == (hta, htb) and len(goals) == g_sa + g_sb:
            return g_sa, g_sb
        return None
    if goals == []:
        if sa + sb == 0 and (hta, htb) in ((0, 0), (None, None)):
            return 0, 0
        return None
    # gj null
    if sa + sb == 0:
        if (hta, htb) in ((0, 0), (None, None)):
            return 0, 0
        return None
    return sa, sb


# ----------------------------------------------------------------- load
def load():
    eng = create_engine(load_settings().db_url)
    q = """
    SELECT e.id, e.team_a, e.team_b, e.expected_start, e.round_info,
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


def build():
    rows = load()
    # duplicate check
    seen, dups = {}, 0
    for r in rows:
        k = (r[1], r[2], str(r[3]))
        if k in seen:
            dups += 1
        seen[k] = r[0]
    print(f"events loaded: {len(rows)}  duplicate (teamA,teamB,start): {dups}")

    raw, clean = [], []   # (eid, is_cell, odd, won)
    n_cell_no_odds = 0
    for (eid, ta, tb, es, ri, oh, oa, em, sa, sb, hta, htb, gj) in rows:
        if isinstance(em, str):
            try:
                em = json.loads(em)
            except json.JSONDecodeError:
                continue
        if not isinstance(em, dict):
            continue
        is_cell = (profile(oh, oa) == "home_slight"
                   and seg(ri) == "MS_early")
        odd = None
        d = em.get("Total de buts")
        if isinstance(d, dict):
            try:
                odd = float(d.get("1"))
            except (TypeError, ValueError):
                odd = None
        if odd is not None and not (MIN_ODD <= odd <= MAX_ODD):
            odd = None
        if is_cell and odd is None:
            n_cell_no_odds += 1

        # variant A: raw official score
        raw.append((eid, is_cell, odd, int(sa + sb == 1)))
        # variant B: cleaned
        cs = clean_score(sa, sb, hta, htb, gj)
        if cs is not None:
            csa, csb = cs
            clean.append((eid, is_cell, odd, int(csa + csb == 1)))
    print(f"cell events without usable 'Total de buts'->'1' odds: "
          f"{n_cell_no_odds}")
    return raw, clean


def stats(bets):
    """bets: list of (odd, won). Returns n, wins, wr, avg_odd, roi, pnl array"""
    n = len(bets)
    if n == 0:
        return 0, 0, np.nan, np.nan, np.nan, np.array([])
    odds = np.array([b[0] for b in bets], dtype=float)
    won = np.array([b[1] for b in bets], dtype=float)
    pnl = won * (odds - 1) - (1 - won)
    return n, int(won.sum()), won.mean(), odds.mean(), pnl.mean(), pnl


def run_variant(name, events):
    """events: list of (eid, is_cell, odd, won) ordered temporally."""
    print("\n" + "=" * 74)
    print(f"VARIANT {name}: {len(events)} events")
    N = len(events)

    def cell_bets(lo, hi):
        return [(o, w) for (eid, ic, o, w) in events[lo:hi]
                if ic and o is not None]

    # --- replication of miner's 70/30 split -------------------------------
    cut = int(N * 0.70)
    n, w, wr, av, roi, _ = stats(cell_bets(cut, N))
    print(f"  [replication 70/30] OOS n={n} wins={w} wr={wr*100:.1f}% "
          f"cote={av:.2f} roi={roi*100:+.1f}%  "
          f"(claim: n=132 wr=19.7% cote=7.74 roi=+45.6%)")

    # --- implied vs realized on full cell ---------------------------------
    all_cell = cell_bets(0, N)
    n, w, wr, av, roi, _ = stats(all_cell)
    imp = np.mean([1 / o for (o, _w) in all_cell]) if all_cell else np.nan
    print(f"  [full-history cell] n={n} realized={wr*100:.1f}% "
          f"implied={imp*100:.1f}% cote={av:.2f} roi={roi*100:+.1f}%")

    # --- walk-forward 3 windows --------------------------------------------
    wins_pos = 0
    pooled = []
    print(f"  {'win':4s} {'train':>26s}   {'gate':5s} {'test':>40s}")
    for i, (tr_hi, te_lo, te_hi) in enumerate((
            (0.50, 0.50, 0.66),
            (0.66, 0.66, 0.83),
            (0.83, 0.83, 1.00)), 1):
        a, b, c = int(N * tr_hi), int(N * te_lo), int(N * te_hi)
        ntr, wtr, wrtr, avtr, roitr, _ = stats(cell_bets(0, a))
        gate = "PASS" if (ntr >= 40 and wtr >= 8 and roitr >= 0.10) else "fail"
        nte, wte, wrte, avte, roite, pnl = stats(cell_bets(b, c))
        pooled.extend(pnl.tolist())
        if nte > 0 and roite > 0:
            wins_pos += 1
        print(f"  W{i:<3d} n={ntr:4d} w={wtr:3d} roi={roitr*100:+6.1f}%   "
              f"{gate:5s} n={nte:4d} w={wte:3d} wr={wrte*100:5.1f}% "
              f"cote={avte:5.2f} roi={roite*100:+6.1f}%")

    pooled = np.array(pooled)
    n_agg = len(pooled)
    roi_agg = pooled.mean() if n_agg else np.nan
    wr_agg = (pooled > 0).mean() if n_agg else np.nan
    # avg odds of pooled test bets
    test_bets = (cell_bets(int(N * .50), int(N * .66))
                 + cell_bets(int(N * .66), int(N * .83))
                 + cell_bets(int(N * .83), N))
    av_agg = np.mean([o for (o, _w) in test_bets]) if test_bets else np.nan
    # bootstrap
    p_le0 = np.nan
    if n_agg:
        rng = np.random.default_rng(7)
        idx = rng.integers(0, n_agg, size=(20000, n_agg))
        p_le0 = (pooled[idx].mean(axis=1) <= 0).mean()
    print(f"  AGGREGATE 3 windows: n={n_agg} wins={int((pooled>0).sum())} "
          f"wr={wr_agg*100:.1f}% cote={av_agg:.2f} roi={roi_agg*100:+.1f}% "
          f"windows_roi>0: {wins_pos}/3  bootstrap P(roi<=0)={p_le0:.4f}")
    return dict(n=n_agg, wr=wr_agg, roi=roi_agg, avg=av_agg,
                wins_pos=wins_pos, p=p_le0)


def main():
    raw, clean = build()
    res_raw = run_variant("A-RAW (official scores, no cleaning)", raw)
    res_clean = run_variant("B-CLEAN (majority-vote reconstruction)", clean)

    print("\n" + "=" * 74)
    print("VERDICT INPUTS")
    for nm, r in (("RAW", res_raw), ("CLEAN", res_clean)):
        print(f"  {nm:6s} agg_roi={r['roi']*100:+.1f}%  windows>0={r['wins_pos']}/3 "
              f"n={r['n']} wr={r['wr']*100:.1f}% p={r['p']:.4f}")


if __name__ == "__main__":
    main()
