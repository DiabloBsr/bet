# -*- coding: utf-8 -*-
"""WF5 — AUDIT STATIONNARITE des edges coeur sur InstantLeague-8035. READ-ONLY.

Recalcule par semaine ISO et par jour (focus 5 derniers jours actifs) :
  - E1  : favori home <=1.50 -> FTTS '1' (settle via goals_json, garde-fous wf4)
  - E2  : favori extreme [1.10,1.20] -> 1X2 backFAV
  - T1p : proxy TIER1 ULTRA = favori (un cote ou l'autre) <=1.30 -> 1X2 backFAV
          (1/1.30 = 0.769 >= 0.75 ~ gate proba; le vrai V5 n'est pas reproductible ici)
  - FAVb: base favoris <=1.50 backFAV (population de reference pour le CUSUM)

CUSUM (2 variantes, ordonnees par expected_start) :
  A) calibration : x_i = win_i - p_norm_i (p = implied margin-removed) -> derive vs marche
  B) regime      : bridge centre sur la moyenne empirique -> rupture de WR interne
  p-value asymptotique Kolmogorov sur max|bridge|.

Cotes d'OUVERTURE uniquement (snapshot MIN(id) par event). Exclut corrupted_events.
Sortie: exports/wf5_stationarity.json
"""
import sys, json, math, collections
sys.path.insert(0, ".")
from scraper.config import load_settings
from sqlalchemy import create_engine, text

LG = "InstantLeague-8035"
e = create_engine(load_settings().db_url)

_corr = json.load(open("exports/corrupted_events.json", encoding="utf-8"))
CORRUPTED = set(int(k) for k in _corr["events"].keys())

# ---------------------------------------------------------------- helpers
def iso_week(ds):           # ds = 'YYYY-MM-DD...'
    import datetime as dt
    d = dt.date(int(ds[0:4]), int(ds[5:7]), int(ds[8:10]))
    y, w, _ = d.isocalendar()
    return "%d-W%02d" % (y, w)

def day(ds):
    return ds[0:10]

def kolmogorov_p(lam):
    """P(sup|B_bridge| > lam), asymptotic Kolmogorov distribution."""
    if lam <= 0:
        return 1.0
    s = 0.0
    for k in range(1, 101):
        s += (-1) ** (k - 1) * math.exp(-2.0 * k * k * lam * lam)
    return max(0.0, min(1.0, 2.0 * s))

def evaluate(bets):
    """bets: list of (won:bool, odds:float). 1u flat."""
    n = len(bets)
    if n == 0:
        return dict(n=0)
    wins = sum(1 for w, _ in bets if w)
    odds_sum = sum(o for _, o in bets)
    profit = sum((o - 1.0) if w else -1.0 for w, o in bets)
    profits = [(o - 1.0) if w else -1.0 for w, o in bets]
    mu = profit / n
    var = sum((p - mu) ** 2 for p in profits) / max(1, n - 1)
    se = math.sqrt(var / n) if n > 1 else None
    z = (mu / se) if se else None
    return dict(n=n, wins=wins, wr=round(wins / n, 4),
                roi_pct=round(100 * mu, 2),
                roi_se_pct=(round(100 * se, 2) if se else None),
                roi_z=(round(z, 2) if z is not None else None),
                avg_odds=round(odds_sum / n, 3),
                implied_wr=round(n / odds_sum, 4))

def by_period(rows, keyfun):
    out = collections.OrderedDict()
    for r in sorted(rows, key=lambda x: x["xs"]):
        k = keyfun(r["xs"])
        out.setdefault(k, []).append((r["won"], r["odds"]))
    return collections.OrderedDict((k, evaluate(v)) for k, v in out.items())

def cusum(rows, p_key):
    """rows sorted by xs; rows[i]['won'] bool, rows[i][p_key] = ref prob.
    Returns calibration CUSUM (vs p) and regime CUSUM (vs empirical mean)."""
    rows = sorted(rows, key=lambda x: x["xs"])
    n = len(rows)
    if n < 50:
        return dict(n=n, note="n<50, CUSUM non calcule")
    xs = [1.0 if r["won"] else 0.0 for r in rows]
    ps = [r[p_key] for r in rows]
    # A) vs market-implied
    devs = [x - p for x, p in zip(xs, ps)]
    var_h0 = sum(p * (1 - p) for p in ps)
    S, peak_a, arg_a = 0.0, 0.0, 0
    for i, d in enumerate(devs):
        S += d
        b = abs(S) / math.sqrt(var_h0)
        if b > peak_a:
            peak_a, arg_a = b, i
    final_dev = sum(devs)
    # B) regime change vs own mean (Brownian bridge)
    mu = sum(xs) / n
    var_e = mu * (1 - mu) * n
    S2, peak_b, arg_b = 0.0, 0.0, 0
    Stot = sum(x - mu for x in xs)  # ~0
    cum = 0.0
    for i, x in enumerate(xs):
        cum += x - mu
        b = abs(cum - (i + 1) / n * Stot) / math.sqrt(var_e)
        if b > peak_b:
            peak_b, arg_b = b, i
    # WR before/after detected breakpoint (variant B)
    cut = arg_b + 1
    wr_pre = sum(xs[:cut]) / cut if cut else None
    wr_post = sum(xs[cut:]) / (n - cut) if n - cut > 0 else None
    return dict(
        n=n,
        calib=dict(max_bridge=round(peak_a, 3), p_value=round(kolmogorov_p(peak_a), 4),
                   at_index=arg_a, at_date=rows[arg_a]["xs"][:16],
                   final_excess_wins=round(final_dev, 1),
                   mean_wr=round(sum(xs) / n, 4), mean_implied=round(sum(ps) / n, 4)),
        regime=dict(max_bridge=round(peak_b, 3), p_value=round(kolmogorov_p(peak_b), 4),
                    at_index=arg_b, at_date=rows[arg_b]["xs"][:16],
                    wr_pre=round(wr_pre, 4) if wr_pre is not None else None,
                    wr_post=round(wr_post, 4) if wr_post is not None else None,
                    n_pre=cut, n_post=n - cut),
    )

# ---------------------------------------------------------------- PASS A : 1X2 (leger)
base = []   # all finished with opening 1x2
skipped = collections.Counter()
with e.connect() as c:
    res = c.execute(text("""
        SELECT e.id, e.expected_start, r.score_a, r.score_b,
               o.odds_home, o.odds_draw, o.odds_away
        FROM events e
        JOIN results r ON r.event_id = e.id
        JOIN odds_snapshots o ON o.event_id = e.id
        WHERE e.competition = :lg
          AND o.id = (SELECT MIN(o2.id) FROM odds_snapshots o2 WHERE o2.event_id = e.id)
    """), {"lg": LG})
    while True:
        chunk = res.fetchmany(2000)
        if not chunk:
            break
        for (eid, xs, sa, sb, oh, od, oa) in chunk:
            if eid in CORRUPTED:
                skipped["corrupted"] += 1
                continue
            if oh is None or od is None or oa is None:
                skipped["no_1x2"] += 1
                continue
            oh, od, oa = float(oh), float(od), float(oa)
            if min(oh, od, oa) < 1.01:
                skipped["bad_odds"] += 1
                continue
            if sa is None or sb is None:
                skipped["no_score"] += 1
                continue
            base.append(dict(eid=eid, xs=str(xs), sa=int(sa), sb=int(sb),
                             oh=oh, od=od, oa=oa))

print("PASS A: %d rows ok, skipped=%s" % (len(base), dict(skipped)), file=sys.stderr)

booksum = lambda r: 1 / r["oh"] + 1 / r["od"] + 1 / r["oa"]

# E2 rows
e2_rows = []
for r in base:
    fav_home = r["oh"] <= r["oa"]
    fc = r["oh"] if fav_home else r["oa"]
    if 1.10 <= fc <= 1.20:
        won = (r["sa"] > r["sb"]) if fav_home else (r["sb"] > r["sa"])
        e2_rows.append(dict(xs=r["xs"], won=won, odds=fc,
                            p_norm=(1 / fc) / booksum(r)))

# T1 proxy rows (fav <=1.30)
t1_rows = []
for r in base:
    fav_home = r["oh"] <= r["oa"]
    fc = r["oh"] if fav_home else r["oa"]
    if fc <= 1.30:
        won = (r["sa"] > r["sb"]) if fav_home else (r["sb"] > r["sa"])
        t1_rows.append(dict(xs=r["xs"], won=won, odds=fc,
                            p_norm=(1 / fc) / booksum(r)))

# FAV base rows (fav <=1.50) — population CUSUM
fav_rows = []
for r in base:
    fav_home = r["oh"] <= r["oa"]
    fc = r["oh"] if fav_home else r["oa"]
    if fc <= 1.50:
        won = (r["sa"] > r["sb"]) if fav_home else (r["sb"] > r["sa"])
        fav_rows.append(dict(xs=r["xs"], won=won, odds=fc,
                             p_norm=(1 / fc) / booksum(r)))

# ---------------------------------------------------------------- PASS B : E1 (FTTS, chunks)
e1_rows = []
e1_skip = collections.Counter()
with e.connect() as c:
    res = c.execute(text("""
        SELECT e.id, e.expected_start, r.score_a, r.score_b, r.goals_json,
               o.extra_markets, o.odds_home
        FROM events e
        JOIN results r ON r.event_id = e.id
        JOIN odds_snapshots o ON o.event_id = e.id
        WHERE e.competition = :lg
          AND o.odds_home <= 1.5
          AND o.id = (SELECT MIN(o2.id) FROM odds_snapshots o2 WHERE o2.event_id = e.id)
    """), {"lg": LG})
    while True:
        chunk = res.fetchmany(400)
        if not chunk:
            break
        for (eid, xs, sa, sb, gj, em, oh) in chunk:
            if eid in CORRUPTED:
                e1_skip["corrupted"] += 1
                continue
            if oh is None or sa is None or sb is None:
                e1_skip["missing"] += 1
                continue
            try:
                m = json.loads(em) if em else {}
            except Exception:
                m = {}
            ftts = m.get("FTTS")
            if not isinstance(ftts, dict) or "1" not in ftts or "2" not in ftts \
               or "Pas de but" not in ftts:
                e1_skip["no_ftts"] += 1
                continue
            try:
                f1, f2, fng = float(ftts["1"]), float(ftts["2"]), float(ftts["Pas de but"])
            except Exception:
                e1_skip["bad_ftts"] += 1
                continue
            if max(f1, f2, fng) >= 99.5 or min(f1, f2, fng) < 1.01:
                e1_skip["capped"] += 1
                continue
            total = int(sa) + int(sb)
            if total == 0:
                outcome = "NG"
            else:
                try:
                    g = json.loads(gj) if gj else None
                except Exception:
                    g = None
                if not isinstance(g, list) or len(g) == 0:
                    e1_skip["goals_null"] += 1
                    continue
                if len(g) != total:
                    e1_skip["goals_mismatch"] += 1
                    continue
                firsts = [x for x in g if (x.get("homeScore", 0) + x.get("awayScore", 0)) == 1.0]
                if len(firsts) != 1 or firsts[0].get("team") not in ("Home", "Away"):
                    e1_skip["goals_ambiguous"] += 1
                    continue
                outcome = "1" if firsts[0]["team"] == "Home" else "2"
            p_norm = (1 / f1) / (1 / f1 + 1 / f2 + 1 / fng)
            e1_rows.append(dict(xs=str(xs), won=(outcome == "1"), odds=f1, p_norm=p_norm))

print("PASS B (E1): %d rows ok, skipped=%s" % (len(e1_rows), dict(e1_skip)), file=sys.stderr)

# ---------------------------------------------------------------- aggregates
LAST5 = ["2026-06-06", "2026-06-07", "2026-06-10", "2026-06-11", "2026-06-12"]

def panel(rows):
    daily_all = by_period(rows, day)
    return dict(
        full=evaluate([(r["won"], r["odds"]) for r in rows]),
        weekly=by_period(rows, iso_week),
        daily=daily_all,
        last5_active_days={d: daily_all.get(d, dict(n=0)) for d in LAST5},
        cusum=cusum(rows, "p_norm"),
    )

out = dict(
    generated_at="2026-06-12",
    league=LG,
    pass_a_rows=len(base), pass_a_skipped=dict(skipped),
    pass_b_skipped=dict(e1_skip),
    E1_ftts_homefav_le150=panel(e1_rows),
    E2_extremefav_110_120=panel(e2_rows),
    T1_proxy_fav_le130=panel(t1_rows),
    FAV_base_le150=panel(fav_rows),
)
json.dump(out, open("exports/wf5_stationarity.json", "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)

# ---------------------------------------------------------------- console
def show(name, p):
    print("\n=== %s ===" % name)
    f = p["full"]
    print("FULL   n=%5d wr=%.3f impl=%.3f roi=%+.2f%% (se %.2f, z %s) odds %.3f" % (
        f["n"], f["wr"], f["implied_wr"], f["roi_pct"], f["roi_se_pct"] or 0, f["roi_z"], f["avg_odds"]))
    for k, v in p["weekly"].items():
        if v.get("n"):
            print("  %s n=%5d wr=%.3f roi=%+7.2f%% z=%s" % (k, v["n"], v["wr"], v["roi_pct"], v["roi_z"]))
    print("  -- 5 derniers jours actifs --")
    for d in LAST5:
        v = p["last5_active_days"][d]
        if v.get("n"):
            print("  %s n=%5d wr=%.3f roi=%+7.2f%% z=%s" % (d, v["n"], v["wr"], v["roi_pct"], v["roi_z"]))
        else:
            print("  %s n=0" % d)
    cs = p["cusum"]
    if "calib" in cs:
        print("  CUSUM calib : max|B|=%.3f p=%.4f @%s (excess wins %+0.1f, wr %.3f vs impl %.3f)" % (
            cs["calib"]["max_bridge"], cs["calib"]["p_value"], cs["calib"]["at_date"],
            cs["calib"]["final_excess_wins"], cs["calib"]["mean_wr"], cs["calib"]["mean_implied"]))
        print("  CUSUM regime: max|B|=%.3f p=%.4f @%s  wr_pre=%.3f(n=%d) wr_post=%.3f(n=%d)" % (
            cs["regime"]["max_bridge"], cs["regime"]["p_value"], cs["regime"]["at_date"],
            cs["regime"]["wr_pre"], cs["regime"]["n_pre"], cs["regime"]["wr_post"], cs["regime"]["n_post"]))

show("E1 FTTS home fav<=1.50", out["E1_ftts_homefav_le150"])
show("E2 favori extreme [1.10-1.20] 1X2", out["E2_extremefav_110_120"])
show("T1 proxy favori<=1.30 1X2", out["T1_proxy_fav_le130"])
show("FAV base <=1.50 1X2", out["FAV_base_le150"])
