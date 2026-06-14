# -*- coding: utf-8 -*-
"""WF4 E1 (FTTS favori) sur les 8 nouvelles ligues. READ-ONLY.
Opening odds = snapshot MIN(o.id) par event. Settlement FTTS depuis goals_json
(garde-fou: len(goals_json)==score_a+score_b; 0-0 => 'Pas de but' sans goals_json).
Sortie: exports/wf4_ftts_newleagues.json
"""
import sys, json, math, collections
sys.path.insert(0, ".")
from scraper.config import load_settings
from sqlalchemy import create_engine, text

# scipy import crashes (OpenBLAS alloc) in this env -> normal approximation only
HAVE_SCIPY = False
st = None

e = create_engine(load_settings().db_url)
NEW = ["InstantLeague-8036", "InstantLeague-8037", "InstantLeague-8042",
       "InstantLeague-8043", "InstantLeague-8044", "InstantLeague-8056",
       "InstantLeague-8060", "InstantLeague-8065"]

_corr = json.load(open("exports/corrupted_events.json", encoding="utf-8"))
CORRUPTED = set(int(k) for k in _corr["events"].keys())

# ---------- load + settle ----------
rows_all = []
avail = collections.defaultdict(lambda: collections.Counter())
with e.connect() as c:
    for lg in NEW:
        rows = c.execute(text("""
            SELECT e.id, e.expected_start, r.score_a, r.score_b, r.goals_json,
                   o.extra_markets, o.odds_home, o.odds_draw, o.odds_away
            FROM events e
            JOIN results r ON r.event_id = e.id
            JOIN odds_snapshots o ON o.event_id = e.id
            WHERE e.competition = :lg
              AND o.id = (SELECT MIN(o2.id) FROM odds_snapshots o2 WHERE o2.event_id = e.id)
        """), {"lg": lg}).fetchall()
        for (eid, xs, sa, sb, gj, em, oh, od, oa) in rows:
            A = avail[lg]
            A["candidates"] += 1
            if eid in CORRUPTED:
                A["skip_corrupted"] += 1
                continue
            if oh is None or od is None or oa is None:
                A["skip_no_1x2"] += 1
                continue
            try:
                m = json.loads(em) if em else {}
            except Exception:
                m = {}
            ftts = m.get("FTTS")
            if not ftts or "1" not in ftts or "2" not in ftts or "Pas de but" not in ftts:
                A["skip_no_ftts"] += 1
                continue
            if any(float(ftts[k]) >= 99.5 for k in ("1", "2", "Pas de but")):
                A["skip_capped"] += 1
                continue
            total = (sa or 0) + (sb or 0)
            # settle first team to score
            if total == 0:
                outcome = "NG"
            else:
                g = None
                try:
                    g = json.loads(gj) if gj else None
                except Exception:
                    g = None
                if not isinstance(g, list) or len(g) == 0:
                    A["skip_goals_null"] += 1
                    continue
                if len(g) != total:
                    A["skip_goals_mismatch"] += 1
                    continue
                firsts = [x for x in g if (x.get("homeScore", 0) + x.get("awayScore", 0)) == 1.0]
                if len(firsts) != 1 or firsts[0].get("team") not in ("Home", "Away"):
                    A["skip_goals_ambiguous"] += 1
                    continue
                outcome = "1" if firsts[0]["team"] == "Home" else "2"
            A["settled"] += 1
            rows_all.append(dict(lg=lg, eid=eid, xs=str(xs), oh=float(oh), od=float(od),
                                 oa=float(oa), f1=float(ftts["1"]), f2=float(ftts["2"]),
                                 fng=float(ftts["Pas de but"]), outcome=outcome))

rows_all.sort(key=lambda r: r["xs"])
print("settled rows: %d" % len(rows_all), file=sys.stderr)


def pval_profit(profits):
    """two-sided t-test of mean profit vs 0 (normal approx fallback)."""
    n = len(profits)
    if n < 2:
        return None
    mu = sum(profits) / n
    var = sum((p - mu) ** 2 for p in profits) / (n - 1)
    if var == 0:
        return None
    tstat = mu / math.sqrt(var / n)
    if HAVE_SCIPY:
        return float(2 * st.t.sf(abs(tstat), n - 1))
    return float(math.erfc(abs(tstat) / math.sqrt(2)))


def evaluate(bets, sel_key):
    """bets: rows; sel_key in ('1','2','NG'); bet 1u at offered FTTS odds."""
    n = len(bets)
    if n == 0:
        return dict(n=0)
    odds_map = {"1": "f1", "2": "f2", "NG": "fng"}
    profits, wins, odds_sum = [], 0, 0.0
    for b in bets:
        o = b[odds_map[sel_key]]
        odds_sum += o
        if b["outcome"] == sel_key:
            profits.append(o - 1.0)
            wins += 1
        else:
            profits.append(-1.0)
    avg_odds = odds_sum / n
    wr = wins / n
    roi = sum(profits) / n
    return dict(n=n, wins=wins, wr=round(wr, 4), roi_pct=round(100 * roi, 2),
                avg_odds=round(avg_odds, 3), implied_p=round(1 / avg_odds, 4),
                calib_ratio=round(wr * avg_odds, 4), pvalue=pval_profit(profits))


# ---------- strategy grid (each evaluated combination = 1 scanned test) ----------
def fsel(side, lo, hi):
    key = "oh" if side == "home" else "oa"
    return [r for r in rows_all if lo < r[key] <= hi]


tests = []


def add(name, bets, sel):
    tests.append((name, evaluate(bets, sel)))


# E1 base + mirror
add("E1_home_fav<=1.5_FTTS1", fsel("home", 0, 1.5), "1")
add("away_fav<=1.5_FTTS2", fsel("away", 0, 1.5), "2")
# mid-odds extensions (objectif cote >= 1.6)
add("home_1.5-2.0_FTTS1", fsel("home", 1.5, 2.0), "1")
add("away_1.5-2.0_FTTS2", fsel("away", 1.5, 2.0), "2")
# contrarian
add("home_fav<=1.5_FTTS2_contrarian", fsel("home", 0, 1.5), "2")
add("away_fav<=1.5_FTTS1_contrarian", fsel("away", 0, 1.5), "1")
add("home_1.5-2.0_FTTS2_contrarian", fsel("home", 1.5, 2.0), "2")
add("away_1.5-2.0_FTTS1_contrarian", fsel("away", 1.5, 2.0), "1")
# finer bands, fav side
for lo, hi in [(1.0, 1.2), (1.2, 1.35), (1.35, 1.5), (1.5, 1.7), (1.7, 1.85), (1.85, 2.0)]:
    add("home_%s-%s_FTTS1" % (lo, hi), fsel("home", lo, hi), "1")
for lo, hi in [(1.0, 1.35), (1.35, 1.5), (1.5, 1.7), (1.7, 2.0)]:
    add("away_%s-%s_FTTS2" % (lo, hi), fsel("away", lo, hi), "2")
# selection par cote FTTS directe (cible cote>=1.6), fav = side with lower 1x2 odds
hf = [r for r in rows_all if r["oh"] < r["oa"] and 1.6 <= r["f1"] <= 2.2]
add("homefav_FTTSodds1.6-2.2_FTTS1", hf, "1")
af = [r for r in rows_all if r["oa"] < r["oh"] and 1.6 <= r["f2"] <= 2.2]
add("awayfav_FTTSodds1.6-2.2_FTTS2", af, "2")


# combined both-side fav (union, bet fav side)
def evaluate_mixed(pairs):
    n = len(pairs)
    if n == 0:
        return dict(n=0)
    profits, wins, odds_sum = [], 0, 0.0
    for r, sel in pairs:
        o = r["f1"] if sel == "1" else r["f2"]
        odds_sum += o
        if r["outcome"] == sel:
            profits.append(o - 1)
            wins += 1
        else:
            profits.append(-1)
    return dict(n=n, wins=wins, wr=round(wins / n, 4), roi_pct=round(100 * sum(profits) / n, 2),
                avg_odds=round(odds_sum / n, 3), implied_p=round(n / odds_sum, 4),
                calib_ratio=round((wins / n) * (odds_sum / n), 4), pvalue=pval_profit(profits))


both = [(r, "1") for r in fsel("home", 0, 1.5)] + [(r, "2") for r in fsel("away", 0, 1.5)]
tests.append(("anyfav<=1.5_FTTS_favside", evaluate_mixed(both)))
both_mid = [(r, "1") for r in fsel("home", 1.5, 2.0)] + [(r, "2") for r in fsel("away", 1.5, 2.0)]
tests.append(("anyfav_1.5-2.0_FTTS_favside", evaluate_mixed(both_mid)))

# ---------- per-league breakdown for E1 base & key variants ----------
per_league = {}
for lg in NEW:
    sub = [r for r in rows_all if r["lg"] == lg]
    per_league[lg] = {
        "E1_home_fav<=1.5_FTTS1": evaluate([r for r in sub if r["oh"] <= 1.5], "1"),
        "home_1.5-2.0_FTTS1": evaluate([r for r in sub if 1.5 < r["oh"] <= 2.0], "1"),
        "away_fav<=1.5_FTTS2": evaluate([r for r in sub if r["oa"] <= 1.5], "2"),
    }

# ---------- temporal robustness split 70/30 on pooled (info only) ----------
cut = int(0.7 * len(rows_all))
early, late = rows_all[:cut], rows_all[cut:]
robust = {}
for nm, flt, sel in [("E1_home_fav<=1.5_FTTS1", lambda r: r["oh"] <= 1.5, "1"),
                     ("home_1.5-2.0_FTTS1", lambda r: 1.5 < r["oh"] <= 2.0, "1"),
                     ("away_fav<=1.5_FTTS2", lambda r: r["oa"] <= 1.5, "2")]:
    robust[nm] = {"early70": evaluate([r for r in early if flt(r)], sel),
                  "late30": evaluate([r for r in late if flt(r)], sel)}

out = {
    "generated_at": "2026-06-12",
    "n_settled_rows": len(rows_all),
    "availability_per_league": {k: dict(v) for k, v in avail.items()},
    "n_tests_scanned": len(tests),
    "tests_pooled_newleagues": {nm: res for nm, res in tests},
    "per_league": per_league,
    "temporal_robustness_70_30": robust,
}
json.dump(out, open("exports/wf4_ftts_newleagues.json", "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)

print(json.dumps({"availability": {k: dict(v) for k, v in avail.items()}}, indent=1))
for nm, res in tests:
    print("%-42s n=%5d wr=%s roi%%=%s avg_odds=%s calib=%s p=%s" % (
        nm, res.get("n", 0), res.get("wr"), res.get("roi_pct"),
        res.get("avg_odds"), res.get("calib_ratio"), res.get("pvalue")))
