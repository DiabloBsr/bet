# -*- coding: utf-8 -*-
"""
Walk-forward analysis of exotic goals markets (Bet261 virtual football).

Markets covered (settled from score_a/score_b/ht/goals_json, priced from
opening extra_markets snapshot):
  - G/NG (BTTS)                       - G/NG equipe domicile / extérieur
  - Total equipe domicile/extérieur   - +/- (O/U 3.5, only line offered)
  - Multi-Buts (0-2 / 1-3 / 2-4 / 5+) - Minute du premier but
  - Pair/Impair                       - FTTS
  - Les deux équipes marquent / 1ère mi temps (HT BTTS)
  - Total de buts (exact, '6' treated as 6+ -- assumption)

Anti-leakage protocol:
  * sort by expected_start, train = first 70%, OOS = last 30%
  * candidate cells (market x selection x conditioner) selected on TRAIN ONLY
    with pre-registered thresholds, then frozen and evaluated on OOS
  * report only OOS metrics; n_oos < 30 flagged 'instable'

Data-quality policy (pre-registered, applied to ALL rows before split):
  392/8153 results have score_a/score_b contradicted by goals_json; in 303 of
  them the HT score corroborates goals_json -> the official score field is the
  corrupted one. Majority vote 2-of-3 (score / HT / goals timeline):
  * gj consistent with score              -> keep official score
  * gj inconsistent, HT agrees with gj    -> RECONSTRUCT score from gj
  * gj inconsistent, HT does not agree    -> DROP row (ambiguous)
  * gj empty/null, score 0-0, HT 0-0      -> keep (true 0-0, goals=[])
  * gj empty list but score > 0           -> DROP (ambiguous)
  * gj null, score > 0                    -> keep score markets, skip
                                             minute/FTTS markets
"""
import sys, json
sys.path.insert(0, '.')
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import chisquare
from sqlalchemy import create_engine, text
from scraper.config import load_settings


# ---------------------------------------------------------------- profiles
def classify_profile(odds_h, odds_a):
    if odds_h is None or odds_a is None:
        return "other"
    if odds_h < 1.3 and odds_a > 7:  return "home_crush"
    if odds_h < 1.6 and odds_a > 4:  return "home_strong"
    if 1.6 <= odds_h < 2.2 and odds_a >= 2.5: return "home_slight"
    if 1.9 <= odds_h < 2.5 and 1.9 <= odds_a < 2.5: return "balanced"
    if 1.6 <= odds_a < 2.2 and odds_h >= 2.5: return "away_slight"
    if odds_a < 1.6 and odds_h > 4:  return "away_strong"
    if odds_a < 1.3 and odds_h > 7:  return "away_crush"
    return "other"


def seg_of(round_info):
    try:
        j = int(round_info)
    except (TypeError, ValueError):
        return None
    if j <= 0: return None
    if j <= 3: return "DS"
    if j <= 12: return "MS_early"
    if j <= 25: return "MS_mid"
    if j <= 33: return "MS_late"
    return "FS"


def fg_bucket(minute):
    if minute <= 15: return "1-15"
    if minute <= 30: return "16-30"
    if minute <= 45: return "31-45"
    if minute <= 60: return "46-60"
    if minute <= 75: return "61-75"
    return "76-90"


# ---------------------------------------------------------------- load data
def load_rows():
    eng = create_engine(load_settings().db_url)
    q = """
    SELECT e.id, e.round_info, e.expected_start,
           os.odds_home, os.odds_draw, os.odds_away, os.extra_markets,
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


MIN_ODD, MAX_ODD = 1.01, 35.0   # 100.0 = capped/unavailable; <1.01 = display quirk


def settle(em, sa, sb, hta, htb, goals):
    """Return list of (market, selection, odds, won:0/1).
    Every offered selection of every covered market is settled, so we can
    measure calibration of the whole book, not just sides we like."""
    out = []
    tot = sa + sb
    btts = int(sa > 0 and sb > 0)

    def add(market, sel, won):
        d = em.get(market)
        if not isinstance(d, dict):
            return
        o = d.get(sel)
        if o is None:
            return
        try:
            o = float(o)
        except (TypeError, ValueError):
            return
        if not (MIN_ODD <= o <= MAX_ODD):
            return
        out.append((market, sel, o, int(won)))

    # BTTS
    add("G/NG", "Oui", btts)
    add("G/NG", "Non", 1 - btts)
    # team to score / clean sheets
    add("G/NG equipe domicile", "Oui", sa > 0)
    add("G/NG equipe domicile", "Non", sa == 0)
    add("G/NG equipe extérieur", "Oui", sb > 0)
    add("G/NG equipe extérieur", "Non", sb == 0)
    # team totals (line 3.5 only)
    add("Total equipe domicile", "> 3.5", sa >= 4)
    add("Total equipe domicile", "< 3.5", sa <= 3)
    add("Total equipe extérieur", "> 3.5", sb >= 4)
    add("Total equipe extérieur", "< 3.5", sb <= 3)
    # match O/U 3.5
    add("+/-", "> 3.5", tot >= 4)
    add("+/-", "< 3.5", tot <= 3)
    # Multi-Buts
    add("Multi-Buts", "Le total de buts est de 0, 1 ou 2", tot <= 2)
    add("Multi-Buts", "Le total de buts est de 1, 2 ou 3", 1 <= tot <= 3)
    add("Multi-Buts", "Le total de buts est de 2, 3 ou 4", 2 <= tot <= 4)
    add("Multi-Buts", "Le total de buts est supérieur à 4", tot >= 5)
    # Pair / Impair
    add("Pair/Impair", "Pair", tot % 2 == 0)
    add("Pair/Impair", "Impair", tot % 2 == 1)
    # Total de buts exact ('6' assumed = 6+)
    for k in map(str, range(0, 6)):
        add("Total de buts", k, tot == int(k))
    add("Total de buts", "6", tot >= 6)
    # HT BTTS
    if hta is not None and htb is not None:
        ht_btts = int(hta > 0 and htb > 0)
        add("Les deux équipes marquent / 1ère mi temps", "Oui", ht_btts)
        add("Les deux équipes marquent / 1ère mi temps", "Non", 1 - ht_btts)
    # goals_json driven markets
    if goals is not None:
        if tot == 0:
            add("Minute du premier but", "Pas de but", 1)
            for b in ("1-15", "16-30", "31-45", "46-60", "61-75", "76-90"):
                add("Minute du premier but", b, 0)
            add("FTTS", "Pas de but", 1)
            add("FTTS", "1", 0)
            add("FTTS", "2", 0)
        elif goals:  # non-empty list consistent with tot>0
            first = min(goals, key=lambda g: g["minute"])
            fb = fg_bucket(first["minute"])
            add("Minute du premier but", "Pas de but", 0)
            for b in ("1-15", "16-30", "31-45", "46-60", "61-75", "76-90"):
                add("Minute du premier but", b, b == fb)
            ft = "1" if first["team"] == "Home" else "2"
            add("FTTS", "Pas de but", 0)
            add("FTTS", "1", ft == "1")
            add("FTTS", "2", ft == "2")
    return out


def clean_score(sa, sb, hta, htb, gj):
    """Majority-vote 2-of-3 score reconstruction.
    Returns (sa, sb, goals_or_None) or None to drop the row."""
    goals = None
    if gj:
        try:
            goals = json.loads(gj) if isinstance(gj, str) else gj
        except json.JSONDecodeError:
            goals = None
    if goals:  # non-empty timeline
        last = max(goals, key=lambda g: g["minute"])
        g_sa, g_sb = int(last["homeScore"]), int(last["awayScore"])
        consistent = (g_sa == sa and g_sb == sb and len(goals) == sa + sb)
        if consistent:
            return sa, sb, goals
        # arbitrate with HT
        if hta is None or htb is None:
            return None
        gh = sum(1 for x in goals if x["minute"] <= 45 and x["team"] == "Home")
        ga = sum(1 for x in goals if x["minute"] <= 45 and x["team"] == "Away")
        if (gh, ga) == (hta, htb) and len(goals) == g_sa + g_sb:
            return g_sa, g_sb, goals          # trust timeline + HT
        return None                            # ambiguous -> drop
    if goals == []:  # empty timeline
        if sa + sb == 0 and (hta, htb) in ((0, 0), (None, None)):
            return 0, 0, []
        return None                            # empty gj vs score>0 -> drop
    # gj null
    if sa + sb == 0:
        if (hta, htb) in ((0, 0), (None, None)):
            return 0, 0, []                    # consistent 0-0
        return None
    return sa, sb, None                        # score markets only


def build_dataset():
    rows = load_rows()
    recs = []
    n_drop = 0
    for (eid, ri, _es, oh, od, oa, em, sa, sb, hta, htb, gj) in rows:
        if isinstance(em, str):
            try:
                em = json.loads(em)
            except json.JSONDecodeError:
                continue
        if not isinstance(em, dict):
            continue
        cs = clean_score(sa, sb, hta, htb, gj)
        if cs is None:
            n_drop += 1
            continue
        sa, sb, goals = cs
        prof = classify_profile(oh, oa)
        seg = seg_of(ri)
        for (mkt, sel, odd, won) in settle(em, sa, sb, hta, htb, goals):
            recs.append((eid, prof, seg, oh, oa, mkt, sel, odd, won))
    print(f"dropped ambiguous rows: {n_drop}")
    df = pd.DataFrame(recs, columns=["eid", "prof", "seg", "oh", "oa",
                                     "mkt", "sel", "odd", "won"])
    # temporal split on EVENTS (rows already ordered by expected_start)
    eids = df["eid"].drop_duplicates().tolist()   # preserves load order
    cut = int(len(eids) * 0.70)
    train_ids = set(eids[:cut])
    df["split"] = np.where(df["eid"].isin(train_ids), "train", "oos")
    # OOS halves for stability check
    oos_ids = eids[cut:]
    h2 = set(oos_ids[len(oos_ids) // 2:])
    df["oos_half"] = np.where(df["eid"].isin(h2), 2, 1)
    return df, len(eids), cut


def metrics(g):
    n = len(g)
    if n == 0:
        return None
    wr = g["won"].mean()
    avg = g["odd"].mean()
    roi = (g["won"] * (g["odd"] - 1) - (1 - g["won"])).mean()
    return n, wr, avg, roi


def fmt(m):
    n, wr, avg, roi = m
    return f"n={n:4d} wr={wr*100:5.1f}% cote={avg:5.2f} roi={roi*100:+6.1f}%"


def main():
    df, n_events, cut = build_dataset()
    tr = df[df.split == "train"]
    oo = df[df.split == "oos"]
    print(f"events={n_events}  train={cut}  oos={n_events-cut}")
    print(f"bet-rows train={len(tr)}  oos={len(oo)}")

    # ============== 0. Pair/Impair structural test (chi2 on TRAIN) =========
    print("\n" + "=" * 78)
    print("0) PAIR/IMPAIR — chi2 vs 50/50 (train) puis ROI OOS")
    pi_tr = tr[(tr.mkt == "Pair/Impair") & (tr.sel == "Pair")]
    pair_n = int(pi_tr["won"].sum()); tot_n = len(pi_tr)
    chi = chisquare([pair_n, tot_n - pair_n])
    print(f"   train: Pair {pair_n}/{tot_n} = {pair_n/tot_n*100:.1f}%  "
          f"chi2 p={chi.pvalue:.4f}")
    for sel in ("Pair", "Impair"):
        for sp, d in (("train", tr), ("oos", oo)):
            g = d[(d.mkt == "Pair/Impair") & (d.sel == sel)]
            print(f"   {sel:7s} {sp:5s} {fmt(metrics(g))}")

    # ============== 1. global calibration per market ========================
    print("\n" + "=" * 78)
    print("1) CALIBRATION GLOBALE par marché (train) : implied vs realized")
    for mkt in sorted(tr.mkt.unique()):
        g = tr[tr.mkt == mkt]
        imp = (1 / g["odd"]).mean()
        real = g["won"].mean()
        roi = (g["won"] * (g["odd"] - 1) - (1 - g["won"])).mean()
        print(f"   {mkt[:42]:42s} n={len(g):5d} implied={imp*100:5.1f}% "
              f"real={real*100:5.1f}% roi_blind={roi*100:+6.1f}%")

    # ============== 2. candidate strategies =================================
    # Conditioners (pre-registered): ALL / prof / seg / prof x seg / odds-bin
    def odds_bin(o):
        edges = [1.0, 1.3, 1.6, 2.0, 2.5, 3.5, 5.0, 8.0, 35.0]
        for i in range(len(edges) - 1):
            if edges[i] <= o < edges[i + 1]:
                return f"[{edges[i]},{edges[i+1]})"
        return "?"

    df["obin"] = df["odd"].map(odds_bin)

    def favside(d):
        return np.select(
            [d["oh"] < 1.3, d["oh"] < 1.6, d["oa"] < 1.6],
            ["Hfav<1.3", "Hfav<1.6", "Afav<1.6"], default="nofav")

    df["fav"] = favside(df)
    tr = df[df.split == "train"]
    oo = df[df.split == "oos"]

    conds = [
        ("ALL",      lambda d: pd.Series("ALL", index=d.index)),
        ("prof",     lambda d: d["prof"]),
        ("seg",      lambda d: d["seg"]),
        ("profxseg", lambda d: d["prof"] + "|" + d["seg"]),
        ("obin",     lambda d: d["obin"]),
        ("profxobin", lambda d: d["prof"] + "|" + d["obin"]),
        ("fav",      lambda d: d["fav"]),
        ("favxseg",  lambda d: d["fav"] + "|" + d["seg"]),
    ]

    MIN_TRAIN = {"ALL": 200, "prof": 80, "seg": 80, "profxseg": 40,
                 "obin": 80, "profxobin": 40, "fav": 80, "favxseg": 40}

    cands = []
    for cname, cfun in conds:
        key_tr = cfun(tr)
        grp = tr.groupby([tr.mkt, tr.sel, key_tr])
        for (mkt, sel, cval), g in grp:
            m = metrics(g)
            if m is None:
                continue
            n, wr, avg, roi = m
            if n < MIN_TRAIN[cname]:
                continue
            wins = int(g["won"].sum())
            # pre-registered acceptance: value (roi>=10%) or accuracy (wr>=78%)
            if (roi >= 0.10 and wins >= 8) or (wr >= 0.78 and roi > -0.02):
                cands.append((cname, cval, mkt, sel, n, wr, avg, roi))

    print("\n" + "=" * 78)
    print(f"2) {len(cands)} cellules candidates retenues sur train -> eval OOS")
    key_oo = {c: f(oo) for c, f in conds}
    results = []
    for (cname, cval, mkt, sel, ntr, wrtr, avgtr, roitr) in cands:
        mask = (oo.mkt == mkt) & (oo.sel == sel) & (key_oo[cname] == cval)
        g = oo[mask]
        m = metrics(g)
        if m is None:
            results.append((cname, cval, mkt, sel, ntr, wrtr, roitr,
                            0, np.nan, np.nan, np.nan, 0, np.nan, np.nan))
            continue
        n, wr, avg, roi = m
        wins = int(g["won"].sum())
        roi_h = []
        for h in (1, 2):
            gh = g[g.oos_half == h]
            roi_h.append((gh["won"] * (gh["odd"] - 1) - (1 - gh["won"])).mean()
                         if len(gh) else np.nan)
        results.append((cname, cval, mkt, sel, ntr, wrtr, roitr,
                        n, wr, avg, roi, wins, roi_h[0], roi_h[1]))

    res = pd.DataFrame(results, columns=[
        "cond", "cval", "mkt", "sel", "n_tr", "wr_tr", "roi_tr",
        "n_oos", "wr_oos", "avg_cote", "roi_oos", "wins_oos",
        "roi_h1", "roi_h2"])
    res = res.sort_values("roi_oos", ascending=False)

    # selection-noise gauge: average OOS roi over ALL value candidates
    val_c = res[res.roi_tr >= 0.10]
    pooled = val_c["roi_oos"] * val_c["n_oos"]
    print(f"   gauge bruit de sélection: {len(val_c)} candidats 'value', "
          f"roi_oos pondéré moyen = {pooled.sum()/val_c['n_oos'].sum()*100:+.1f}%")

    print("\n--- TOP OOS (roi_oos >= +10%, n_oos >= 30) ---")
    good = res[(res.roi_oos >= 0.10) & (res.n_oos >= 30)]
    for _, r in good.iterrows():
        stab = "STABLE" if (r.roi_h1 > 0 and r.roi_h2 > 0) else "h-split!"
        print(f"   [{r['cond']}={r['cval']}] {r['mkt'][:30]} | {r['sel'][:32]:32s} "
              f"tr(n={r.n_tr},wr={r.wr_tr*100:.0f}%,roi={r.roi_tr*100:+.0f}%) "
              f"OOS n={r.n_oos:4.0f} w={r.wins_oos:3.0f} wr={r.wr_oos*100:5.1f}% "
              f"cote={r.avg_cote:5.2f} roi={r.roi_oos*100:+6.1f}% "
              f"[h1={r.roi_h1*100:+.0f}% h2={r.roi_h2*100:+.0f}% {stab}]")

    print("\n--- HIGH ACCURACY OOS (wr_oos >= 75%, n_oos >= 30) ---")
    acc = res[(res.wr_oos >= 0.75) & (res.n_oos >= 30)].sort_values(
        "wr_oos", ascending=False)
    for _, r in acc.head(25).iterrows():
        print(f"   [{r['cond']}={r['cval']}] {r['mkt'][:30]} | {r['sel'][:32]:32s} "
              f"tr(wr={r.wr_tr*100:.0f}%) OOS n={r.n_oos:4.0f} "
              f"wr={r.wr_oos*100:5.1f}% cote={r.avg_cote:5.2f} "
              f"roi={r.roi_oos*100:+6.1f}%")

    print("\n--- HIGH ODDS OOS (avg_cote >= 2.0, roi_oos >= +15%, n_oos >= 30) ---")
    ho = res[(res.avg_cote >= 2.0) & (res.roi_oos >= 0.15) & (res.n_oos >= 30)]
    for _, r in ho.iterrows():
        stab = "STABLE" if (r.roi_h1 > 0 and r.roi_h2 > 0) else "h-split!"
        print(f"   [{r['cond']}={r['cval']}] {r['mkt'][:30]} | {r['sel'][:32]:32s} "
              f"tr(n={r.n_tr},roi={r.roi_tr*100:+.0f}%) OOS n={r.n_oos:4.0f} "
              f"w={r.wins_oos:3.0f} wr={r.wr_oos*100:5.1f}% cote={r.avg_cote:5.2f} "
              f"roi={r.roi_oos*100:+6.1f}% [{stab}]")

    res.to_csv("scripts/_wf_goals_markets_results.csv", index=False)
    print(f"\nfull table -> scripts/_wf_goals_markets_results.csv "
          f"({len(res)} cells)")

    # ============== 3. focus questions from the brief =======================
    print("\n" + "=" * 78)
    print("3) QUESTIONS DU BRIEF (train / OOS séparés)")

    def show(label, mask_fn):
        for sp, d in (("train", tr), ("oos", oo)):
            g = d[mask_fn(d)]
            m = metrics(g)
            print(f"   {label[:58]:58s} {sp:5s} "
                  f"{fmt(m) if m else 'n=0'}")

    print("\n3a) BTTS Oui par profil x FS:")
    for p in ("home_crush", "home_strong", "away_slight", "balanced"):
        show(f"G/NG Oui | {p} | FS",
             lambda d, p=p: (d.mkt == "G/NG") & (d.sel == "Oui")
             & (d.prof == p) & (d.seg == "FS"))

    print("\n3b) Clean sheet markets sur favoris extrêmes:")
    show("NG extérieur (home clean sheet) | home_crush",
         lambda d: (d.mkt == "G/NG equipe extérieur") & (d.sel == "Non")
         & (d.prof == "home_crush"))
    show("Oui extérieur (l'outsider marque) | home_crush",
         lambda d: (d.mkt == "G/NG equipe extérieur") & (d.sel == "Oui")
         & (d.prof == "home_crush"))
    show("NG domicile (away clean sheet) | away_crush+away_strong",
         lambda d: (d.mkt == "G/NG equipe domicile") & (d.sel == "Non")
         & (d.prof.isin(["away_crush", "away_strong"])))
    show("Oui domicile (home marque) | away_crush+away_strong",
         lambda d: (d.mkt == "G/NG equipe domicile") & (d.sel == "Oui")
         & (d.prof.isin(["away_crush", "away_strong"])))

    print("\n3c) Total equipe domicile > 3.5 | home_crush / home_strong:")
    for p in ("home_crush", "home_strong"):
        show(f"Tot dom > 3.5 | {p}",
             lambda d, p=p: (d.mkt == "Total equipe domicile")
             & (d.sel == "> 3.5") & (d.prof == p))

    print("\n3d) Multi-Buts (toutes cellules globales):")
    for sel in ("Le total de buts est de 0, 1 ou 2",
                "Le total de buts est de 1, 2 ou 3",
                "Le total de buts est de 2, 3 ou 4",
                "Le total de buts est supérieur à 4"):
        show(f"MB '{sel[-12:]}'",
             lambda d, s=sel: (d.mkt == "Multi-Buts") & (d.sel == s))

    print("\n3e) Minute du premier but (global):")
    for sel in ("1-15", "16-30", "31-45", "46-60", "61-75", "76-90",
                "Pas de but"):
        show(f"1er but {sel}",
             lambda d, s=sel: (d.mkt == "Minute du premier but")
             & (d.sel == s))

    print("\n3f) O/U 3.5 par profil:")
    for p in ("home_crush", "home_strong", "home_slight", "balanced",
              "away_slight", "away_strong", "away_crush", "other"):
        show(f"> 3.5 | {p}",
             lambda d, p=p: (d.mkt == "+/-") & (d.sel == "> 3.5")
             & (d.prof == p))


if __name__ == "__main__":
    main()
