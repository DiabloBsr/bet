# -*- coding: utf-8 -*-
"""
ADVERSARIAL VERIFICATION of F4 — "Total de buts" = '6' on away_slight.

Claim (miner): OOS 70/30 -> n=136, WR=5.9%, avg odds=27.04, ROI=+65.3%.

Re-implemented from scratch. Walk-forward with 3 windows:
  W1: train 0-50%   -> test 50-66%
  W2: train 0-66%   -> test 66-83%
  W3: train 0-83%   -> test 83-100%
Signal is re-checked on each train (does it still look positive?) and
evaluated on each disjoint test window. Aggregate = pooled test bets.

Adversarial checks:
  A) engine truncation claim: any match with total goals > 6?
  B) '6' settled as ==6 vs >=6 — must be identical if (A) holds
  C) sensitivity to score reconstruction (official scores only vs majority-vote)
  D) duplicate events double-counting
  E) reproduction of the miner's 70/30 numbers (definition match)
"""
import sys, json
sys.path.insert(0, '.')
import numpy as np
from sqlalchemy import create_engine, text
from scraper.config import load_settings

MIN_ODD, MAX_ODD = 1.01, 35.0


def load_rows():
    eng = create_engine(load_settings().db_url)
    q = """
    SELECT e.id, e.match_key, e.round_info, e.expected_start,
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


def is_away_slight(oh, oa):
    if oh is None or oa is None:
        return False
    return (1.6 <= oa < 2.2) and (oh >= 2.5)


def clean_score(sa, sb, hta, htb, gj):
    """My own re-implementation of the majority-vote 2-of-3 policy.
    Returns (sa, sb) or None to drop the row as ambiguous."""
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
    if sa + sb == 0:
        if (hta, htb) in ((0, 0), (None, None)):
            return 0, 0
        return None
    return sa, sb


def build():
    rows = load_rows()
    seen = set()
    all_events = []   # (eid, bet_or_None) preserving temporal order
    tot_gt6_official = 0
    tot_gt6_clean = 0
    dup_match_keys = 0
    seen_mk = set()
    for (eid, mk, ri, es, oh, oa, em, sa, sb, hta, htb, gj) in rows:
        if eid in seen:
            continue
        seen.add(eid)
        key = (mk, str(es))
        if key in seen_mk:
            dup_match_keys += 1
        seen_mk.add(key)
        if isinstance(em, str):
            try:
                em = json.loads(em)
            except json.JSONDecodeError:
                continue
        if not isinstance(em, dict):
            continue
        if sa + sb > 6:
            tot_gt6_official += 1
        cs = clean_score(sa, sb, hta, htb, gj)
        ev = {"eid": eid, "es": str(es)}
        # bet definition: market 'Total de buts', selection '6', away_slight
        bet = None
        if is_away_slight(oh, oa):
            d = em.get("Total de buts")
            if isinstance(d, dict):
                o = d.get("6")
                try:
                    o = float(o)
                except (TypeError, ValueError):
                    o = None
                if o is not None and MIN_ODD <= o <= MAX_ODD:
                    won_official_eq = int(sa + sb == 6)
                    won_official_ge = int(sa + sb >= 6)
                    if cs is not None:
                        csa, csb = cs
                        if csa + csb > 6:
                            tot_gt6_clean += 1
                        won_clean = int(csa + csb >= 6)
                        reconstructed = (csa != sa or csb != sb)
                    else:
                        won_clean = None      # dropped under miner policy
                        reconstructed = None
                    bet = {"odd": o,
                           "won_off_eq": won_official_eq,
                           "won_off_ge": won_official_ge,
                           "won_clean": won_clean,
                           "reco": reconstructed,
                           "dropped_clean": cs is None}
        ev["bet"] = bet
        ev["clean_dropped"] = cs is None
        all_events.append(ev)
    return all_events, tot_gt6_official, tot_gt6_clean, dup_match_keys


def stats(bets, key):
    b = [x for x in bets if x[key] is not None]
    n = len(b)
    if n == 0:
        return None
    wins = sum(x[key] for x in b)
    pnl = np.array([x[key] * (x["odd"] - 1) - (1 - x[key]) for x in b])
    return n, wins, wins / n, float(np.mean([x["odd"] for x in b])), pnl


def fmt(s):
    if s is None:
        return "n=0"
    n, w, wr, avg, pnl = s
    return (f"n={n:4d} wins={w:3d} wr={wr*100:5.1f}% cote={avg:6.2f} "
            f"roi={pnl.mean()*100:+7.1f}%")


def main():
    events, gt6_off, gt6_cln, dups = build()
    n_ev = len(events)
    print(f"events loaded (unique eid, temporal order): {n_ev}")
    print(f"[check A] matches with official total > 6 : {gt6_off}")
    print(f"[check A] F4 bets with cleaned total > 6  : {gt6_cln}")
    print(f"[check D] duplicate (match_key, expected_start): {dups}")

    bets_all = [e["bet"] for e in events if e["bet"] is not None]
    print(f"\nF4 qualifying bets (away_slight, '6' priced in [1.01,35]): "
          f"{len(bets_all)}")
    eq = sum(1 for b in bets_all
             if b["won_off_eq"] != b["won_off_ge"])
    print(f"[check B] bets where ==6 differs from >=6 (official): {eq}")
    n_drop = sum(1 for b in bets_all if b["dropped_clean"])
    n_reco = sum(1 for b in bets_all if b["reco"])
    print(f"[check C] F4 bets dropped by cleaning: {n_drop} ; "
          f"reconstructed scores among kept: {n_reco}")

    # ---- check E: reproduce the miner's 70/30 split -------------------------
    # The miner split on unique eids of the *bet-rows* dataframe (all markets),
    # which is ~ all cleaned events in temporal order. Approximate with all
    # events that survive cleaning.
    kept = [e for e in events if not e["clean_dropped"]]
    cut = int(len(kept) * 0.70)
    oos_ids = set(e["eid"] for e in kept[cut:])
    oos_bets = [e["bet"] for e in kept[cut:]
                if e["bet"] is not None and e["bet"]["won_clean"] is not None]
    print(f"\n[check E] my 70/30 reproduction (clean settlement, miner-style):")
    s = stats(oos_bets, "won_clean")
    print(f"   OOS  {fmt(s)}   (miner claims n=136 wr=5.9% cote=27.04 "
          f"roi=+65.3%)")
    tr_bets = [e["bet"] for e in kept[:cut]
               if e["bet"] is not None and e["bet"]["won_clean"] is not None]
    print(f"   train {fmt(stats(tr_bets, 'won_clean'))}   "
          f"(miner claims train roi=+11.6%)")

    # ---- walk-forward 3 windows ---------------------------------------------
    # Windows defined on the FULL temporal event list (kept after cleaning).
    N = len(kept)
    windows = [
        ("W1 train 0-50%  test 50-66%",  0.50, 0.6667),
        ("W2 train 0-66%  test 66-83%",  0.6667, 0.8333),
        ("W3 train 0-83%  test 83-100%", 0.8333, 1.0),
    ]

    def bets_in(lo, hi, key):
        seg = kept[int(N * lo):int(N * hi)]
        return [e["bet"] for e in seg
                if e["bet"] is not None and e["bet"][key] is not None]

    for settle_key, label in (("won_clean", "CLEAN (miner settlement)"),
                              ("won_off_eq", "OFFICIAL scores, ==6 strict")):
        print(f"\n===== WALK-FORWARD — settlement: {label} =====")
        agg_pnl, agg_n, agg_w, agg_odds, pos = [], 0, 0, [], 0
        per_win = []
        for name, a, b in windows:
            tr = bets_in(0.0, a, settle_key)
            te = bets_in(a, b, settle_key)
            str_ = stats(tr, settle_key)
            ste = stats(te, settle_key)
            tr_roi = str_[4].mean() if str_ else float("nan")
            tr_w = str_[1] if str_ else 0
            qualifies = (str_ is not None and tr_roi >= 0.10 and tr_w >= 8)
            print(f" {name}")
            print(f"   train: {fmt(str_)}  -> miner gate roi>=10% & wins>=8: "
                  f"{'PASS' if qualifies else 'FAIL'}")
            print(f"   test : {fmt(ste)}")
            if ste is not None:
                n, w, wr, avg, pnl = ste
                agg_pnl.append(pnl)
                agg_n += n
                agg_w += w
                agg_odds += [x["odd"] for x in te]
                if pnl.mean() > 0:
                    pos += 1
                per_win.append(pnl.mean())
            else:
                per_win.append(float("nan"))
        pnl = np.concatenate(agg_pnl) if agg_pnl else np.array([])
        roi = pnl.mean() if len(pnl) else float("nan")
        print(f"\n AGGREGATE [{label}]: n={agg_n} wins={agg_w} "
              f"wr={agg_w/agg_n*100:.1f}% cote={np.mean(agg_odds):.2f} "
              f"roi={roi*100:+.1f}%  | windows ROI>0: {pos}/3 "
              f"({', '.join(f'{r*100:+.1f}%' for r in per_win)})")
        if len(pnl):
            rng = np.random.default_rng(7)
            idx = rng.integers(0, len(pnl), size=(20000, len(pnl)))
            boots = pnl[idx].mean(axis=1)
            print(f"   bootstrap 20k: P(roi<=0)={float((boots <= 0).mean()):.3f}")

    # ---- per-window win detail (where do the rare wins land?) ---------------
    print("\n===== WIN LOCATIONS (clean settlement, full timeline) =====")
    for i, e in enumerate(kept):
        b = e["bet"]
        if b is not None and b["won_clean"]:
            print(f"   win at event-rank {i}/{N} ({i/N*100:.1f}%) "
                  f"odd={b['odd']:.2f} reco={b['reco']} es={e['es']}")


if __name__ == "__main__":
    main()
