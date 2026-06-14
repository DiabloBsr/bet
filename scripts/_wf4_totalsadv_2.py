# WF4 TOTALS - ADVERSARIAL re-verification du finding "Over 3.5 n'est PAS sous-price"
# READ-ONLY DB. Nouveau fichier. Resultats -> exports/wf4_totalsadv2.json
# Checks: (a) fresh re-extraction vs pkl, (b) look-ahead (fetched_at vs expected_start),
# (c) settlement vs goals_json, (d) sous-periodes, (e) walk-forward 70/30 sur 8035,
# (f) bootstrap CI, (g) fresh-OOS (events absents du pkl), (h) rescan segments alternatifs.
import sys, json, pickle, math, random
sys.path.insert(0, ".")
import numpy as np
from scipy.stats import poisson, norm
from scipy.optimize import least_squares
from collections import Counter, defaultdict
from scraper.config import load_settings
from sqlalchemy import create_engine, text

random.seed(42); np.random.seed(42)
e = create_engine(load_settings().db_url)
OUT = {}

with open("exports/corrupted_events.json", "r", encoding="utf-8") as f:
    CORRUPT = set(int(k) for k in json.load(f)["events"].keys())

LEAGUES = ["InstantLeague-8035", "InstantLeague-8036", "InstantLeague-8037",
           "InstantLeague-8042", "InstantLeague-8043", "InstantLeague-8044",
           "InstantLeague-8056", "InstantLeague-8060", "InstantLeague-8065"]
NEW = {"InstantLeague-8036", "InstantLeague-8037", "InstantLeague-8042",
       "InstantLeague-8043", "InstantLeague-8044"}

comps = "(" + ",".join("'" + c + "'" for c in LEAGUES) + ")"
SQL = f"""
SELECT ev.id, ev.competition, ev.expected_start,
       r.score_a, r.score_b, r.ht_score_a, r.ht_score_b, r.goals_json,
       o.odds_home, o.odds_draw, o.odds_away, o.extra_markets, o.captured_at
FROM events ev
JOIN results r ON r.event_id = ev.id
JOIN odds_snapshots o ON o.event_id = ev.id
WHERE o.id = (SELECT MIN(o2.id) FROM odds_snapshots o2 WHERE o2.event_id = ev.id)
  AND ev.competition IN {comps}
"""

with e.connect() as conn:
    rows = conn.execute(text(SQL)).fetchall()
print(f"fresh raw rows: {len(rows)}")

# survivorship: events de ces ligues avec cotes mais SANS result
with e.connect() as conn:
    n_no_res = conn.execute(text(
        f"SELECT COUNT(*) FROM events ev WHERE ev.competition IN {comps} "
        "AND EXISTS (SELECT 1 FROM odds_snapshots o WHERE o.event_id=ev.id) "
        "AND NOT EXISTS (SELECT 1 FROM results r WHERE r.event_id=ev.id)"
    )).scalar()
    n_with_odds = conn.execute(text(
        f"SELECT COUNT(*) FROM events ev WHERE ev.competition IN {comps} "
        "AND EXISTS (SELECT 1 FROM odds_snapshots o WHERE o.event_id=ev.id)"
    )).scalar()
OUT["survivorship"] = {"events_with_odds": int(n_with_odds),
                       "with_odds_no_result": int(n_no_res),
                       "pct_missing": round(100.0 * n_no_res / max(n_with_odds, 1), 2)}
print("survivorship:", OUT["survivorship"])

# pkl du finding original (pour reutiliser lh/la + detecter le fresh-OOS)
with open("exports/wf4_totals_data.pkl", "rb") as f:
    PKL = pickle.load(f)
pkl_lam = {r["eid"]: (r["lh"], r["la"]) for r in PKL}
pkl_ids = set(pkl_lam.keys())
print(f"pkl events: {len(PKL)}")

GMAX = 13
ar = np.arange(GMAX + 1)

def grid_probs(lh, la):
    return np.outer(poisson.pmf(ar, lh), poisson.pmf(ar, la))

def invert_lambdas(oh, od, oa):
    imp = np.array([1 / oh, 1 / od, 1 / oa]); fair = imp / imp.sum()
    def resid(x):
        lh, la = np.exp(x)
        g = grid_probs(lh, la)
        return [np.tril(g, -1).sum() - fair[0], np.trace(g) - fair[1]]
    diff0 = math.log(max(fair[0], 1e-6) / max(fair[2], 1e-6)) * 0.55
    x0 = [math.log(max(0.2, 1.4 + diff0 / 2)), math.log(max(0.2, 1.4 - diff0 / 2))]
    sol = least_squares(resid, x0, xtol=1e-12, ftol=1e-12)
    lh, la = np.exp(sol.x)
    return float(lh), float(la), float(max(abs(v) for v in resid(sol.x)))

# ---------- build fresh dataset (memes guards que le builder, recodes independamment) ----------
data = []
n_corrupt = n_guard = n_badodds = n_settle_mismatch = n_la_after_start = 0
look_ahead_rows = 0
for row in rows:
    (eid, comp, exp_start, sa, sb, hta, htb, gj, oh, od, oa, xm, fat) = row
    if eid in CORRUPT:
        n_corrupt += 1; continue
    if sa is None or sb is None:
        continue
    if hta is not None and htb is not None and (hta > sa or htb > sb):
        n_guard += 1; continue
    gj_len = None
    if gj:
        try:
            gl = json.loads(gj)
            if isinstance(gl, list):
                gj_len = len(gl)
        except Exception:
            pass
    if gj_len is not None and gj_len != sa + sb:
        n_settle_mismatch += 1; continue
    if not oh or not od or not oa or oh <= 1 or od <= 1 or oa <= 1:
        n_badodds += 1; continue
    try:
        xmd = json.loads(xm) if xm else {}
    except Exception:
        xmd = {}
    m = xmd.get("+/-") or {}
    ou_o, ou_u = m.get("> 3.5"), m.get("< 3.5")
    # look-ahead: snapshot d'ouverture APRES expected_start ?
    la_flag = False
    if fat and exp_start and str(fat) >= str(exp_start):
        la_flag = True; look_ahead_rows += 1
    data.append(dict(eid=eid, comp=comp, start=str(exp_start), fat=str(fat),
                     sa=sa, sb=sb, tot=sa + sb, oh=oh, od=od, oa=oa,
                     ou_o=ou_o, ou_u=ou_u, la_flag=la_flag, gj_len=gj_len))
print(f"fresh kept {len(data)} | corrupt {n_corrupt} | guard {n_guard} | "
      f"badodds {n_badodds} | settle_mismatch {n_settle_mismatch}")
OUT["fresh_extraction"] = {"kept": len(data), "corrupt_excluded": n_corrupt,
                           "guard": n_guard, "badodds": n_badodds,
                           "settle_mismatch_excluded": n_settle_mismatch}

# ---------- (b) look-ahead ----------
OUT["look_ahead"] = {"opening_snapshot_after_start": look_ahead_rows,
                     "pct": round(100.0 * look_ahead_rows / max(len(data), 1), 3)}
print("look-ahead:", OUT["look_ahead"])

# ---------- (c) settlement: goals_json coverage + coherence ----------
n_gj = sum(1 for r in data if r["gj_len"] is not None)
OUT["settlement"] = {"with_goals_json": n_gj, "pct": round(100.0 * n_gj / len(data), 1),
                     "note": "mismatch len(goals_json)!=sa+sb excluded by guard; count above"}
print("settlement:", OUT["settlement"])

# ---------- helpers ----------
def roi_stats(bets):
    if not bets:
        return dict(n=0, wr=0, roi=0, odds=0, p=1.0)
    r = np.array([(o - 1) if w else -1.0 for w, o in bets])
    n = len(r); roi = float(r.mean())
    se = r.std(ddof=1) / math.sqrt(n) if n > 1 else 1e9
    p = float(2 * (1 - norm.cdf(abs(roi) / se))) if se > 0 else 1.0
    return dict(n=n, wr=float(np.mean([w for w, _ in bets])), roi=roi,
                odds=float(np.mean([o for _, o in bets])), p=p)

def over_bets(sub):
    return [((r["tot"] >= 4), r["ou_o"]) for r in sub
            if r["ou_o"] and 1 < r["ou_o"] < 100]

# ---------- (e') reproduction blanket sur data fraiche ----------
print("\n=== blanket Over 3.5 (fresh) ===")
OUT["blanket_fresh"] = {}
for scope, sub in [("pooled-9", data),
                   ("8035", [r for r in data if r["comp"] == "InstantLeague-8035"]),
                   ("newleagues", [r for r in data if r["comp"] != "InstantLeague-8035"])]:
    s = roi_stats(over_bets(sub))
    OUT["blanket_fresh"][scope] = s
    print(f"{scope:12s} n={s['n']:5d} WR={s['wr']:.4f} ROI={s['roi']*100:+.2f}% "
          f"odds={s['odds']:.3f} p={s['p']:.4f}")

# exclusion des rows look-ahead (robustesse)
s = roi_stats(over_bets([r for r in data if not r["la_flag"]]))
OUT["blanket_fresh"]["pooled-9_noLA"] = s
print(f"pooled-9 sans look-ahead rows: n={s['n']} ROI={s['roi']*100:+.2f}% p={s['p']:.4f}")

# ---------- (f) bootstrap CI 95% pooled over ----------
bets = over_bets(data)
rarr = np.array([(o - 1) if w else -1.0 for w, o in bets])
bs = np.array([rarr[np.random.randint(0, len(rarr), len(rarr))].mean() for _ in range(10000)])
OUT["bootstrap_pooled_over"] = {"roi": float(rarr.mean()),
                                "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                "p_roi_pos": float((bs > 0).mean())}
print("\nbootstrap pooled over:", OUT["bootstrap_pooled_over"])

# 8035 seul
b35 = over_bets([r for r in data if r["comp"] == "InstantLeague-8035"])
r35 = np.array([(o - 1) if w else -1.0 for w, o in b35])
bs35 = np.array([r35[np.random.randint(0, len(r35), len(r35))].mean() for _ in range(10000)])
OUT["bootstrap_8035_over"] = {"roi": float(r35.mean()),
                              "ci95": [float(np.percentile(bs35, 2.5)), float(np.percentile(bs35, 97.5))],
                              "p_roi_pos": float((bs35 > 0).mean())}
print("bootstrap 8035 over:", OUT["bootstrap_8035_over"])

# ---------- (d) sous-periodes (demi-mois) ----------
print("\n=== Over 3.5 par demi-mois ===")
OUT["subperiods"] = {}
per = defaultdict(list)
for r in data:
    d = r["start"][:10]
    key = d[:8] + ("A" if d[8:10] < "16" else "B")
    per[key].append(r)
for k in sorted(per):
    s = roi_stats(over_bets(per[k]))
    OUT["subperiods"][k] = s
    print(f"{k} n={s['n']:5d} ROI={s['roi']*100:+.2f}% p={s['p']:.3f}")

# ---------- (e) walk-forward 70/30 sur 8035 ----------
print("\n=== walk-forward 8035 (70/30 par expected_start) ===")
d35 = sorted([r for r in data if r["comp"] == "InstantLeague-8035"], key=lambda r: r["start"])
cut = int(0.7 * len(d35))
train, test = d35[:cut], d35[cut:]
st_tr, st_te = roi_stats(over_bets(train)), roi_stats(over_bets(test))
OUT["walkforward_8035"] = {"train": st_tr, "test": st_te,
                           "cut_date": d35[cut]["start"] if cut < len(d35) else "?"}
print(f"train n={st_tr['n']} ROI={st_tr['roi']*100:+.2f}% | test n={st_te['n']} ROI={st_te['roi']*100:+.2f}% p={st_te['p']:.3f}")

# ---------- (g) fresh-OOS: events absents du pkl du finding ----------
oos = [r for r in data if r["eid"] not in pkl_ids]
print(f"\n=== fresh-OOS (events ajoutes depuis le cache du finding): {len(oos)} ===")
s = roi_stats(over_bets(oos))
OUT["fresh_oos"] = {"n_events": len(oos), **s}
print(f"OOS over: n={s['n']} WR={s['wr']:.4f} ROI={s['roi']*100:+.2f}% odds={s['odds']:.3f} p={s['p']:.4f}")
per_l = Counter(r["comp"] for r in oos)
OUT["fresh_oos"]["per_league_counts"] = dict(per_l)

# ---------- (h) rescan segments ALTERNATIFS (edges differents des leurs) ----------
# lambdas: reuse pkl, invert seulement les nouveaux
print("\n=== rescan segments alternatifs ===")
n_inv = 0
for r in data:
    if r["eid"] in pkl_lam:
        r["lh"], r["la"] = pkl_lam[r["eid"]]
    else:
        lh, la, err = invert_lambdas(r["oh"], r["od"], r["oa"])
        r["lh"], r["la"] = (lh, la) if err <= 1e-5 else (None, None)
        n_inv += 1
print(f"inverted fresh lambdas: {n_inv}")

scan = []
# (1) deciles d'implied over (proba implicite de la cote offerte) x 3 groupes
def grp_of(c):
    return "8035" if c == "InstantLeague-8035" else ("dom" if c in NEW else "cup")
bets_all = [(r, (r["tot"] >= 4), r["ou_o"]) for r in data if r["ou_o"] and 1 < r["ou_o"] < 100]
imp = np.array([1 / o for _, _, o in bets_all])
deciles = np.percentile(imp, np.arange(0, 101, 10))
for g in ["8035", "dom", "cup", "all"]:
    for i in range(10):
        sel = [(w, o) for (r, w, o), v in zip(bets_all, imp)
               if deciles[i] <= v <= deciles[i + 1] + (1e-12 if i == 9 else 0)
               and v < deciles[i + 1] + (1 if i == 9 else 0)
               and (g == "all" or grp_of(r["comp"]) == g)]
        s = roi_stats(sel)
        if s["n"] >= 100:
            scan.append((f"impdec{i}_{g}", s))
# (2) bandes lambda_total avec edges DIFFERENTS (0.15 pas)
ed = [0, 2.0, 2.35, 2.65, 2.95, 3.25, 3.55, 99]
for g in ["8035", "dom", "cup", "all"]:
    for i in range(len(ed) - 1):
        sel = [( (r["tot"] >= 4), r["ou_o"]) for r in data
               if r.get("lh") and ed[i] <= r["lh"] + r["la"] < ed[i + 1]
               and r["ou_o"] and 1 < r["ou_o"] < 100
               and (g == "all" or grp_of(r["comp"]) == g)]
        s = roi_stats(sel)
        if s["n"] >= 100:
            scan.append((f"lam[{ed[i]}-{ed[i+1]})_{g}", s))
# (3) bandes de cote offerte over
oed = [1.0, 1.6, 2.0, 2.5, 3.0, 4.0, 6.0, 10.0, 999]
for g in ["8035", "dom", "cup", "all"]:
    for i in range(len(oed) - 1):
        sel = [((r["tot"] >= 4), r["ou_o"]) for r in data
               if r["ou_o"] and oed[i] <= r["ou_o"] < oed[i + 1]
               and (g == "all" or grp_of(r["comp"]) == g)]
        s = roi_stats(sel)
        if s["n"] >= 100:
            scan.append((f"odds[{oed[i]}-{oed[i+1]})_{g}", s))

pos = [(k, s) for k, s in scan if s["roi"] > 0]
pos_sig = [(k, s) for k, s in pos if s["p"] <= 0.01 and s["n"] >= 150]
scan_sorted = sorted(scan, key=lambda t: -t[1]["roi"])
print(f"segments scannes: {len(scan)} | ROI>0: {len(pos)} | ROI>0 & p<=0.01 & n>=150: {len(pos_sig)}")
for k, s in scan_sorted[:8]:
    print(f"  TOP {k:24s} n={s['n']:5d} ROI={s['roi']*100:+.2f}% p={s['p']:.3f}")
OUT["alt_scan"] = {"n_segments": len(scan), "n_positive": len(pos),
                   "n_positive_significant": len(pos_sig),
                   "top8": [{"seg": k, **s} for k, s in scan_sorted[:8]]}

# ---------- marge recheck ----------
m = [1 / r["ou_o"] + 1 / r["ou_u"] - 1 for r in data
     if r["ou_o"] and r["ou_u"] and r["ou_o"] > 1 and r["ou_u"] > 1]
OUT["margin_recheck"] = {"mean_pct": round(float(np.mean(m)) * 100, 3),
                         "std_pct": round(float(np.std(m)) * 100, 3), "n": len(m)}
print("\nmarge +/- 3.5 recheck:", OUT["margin_recheck"])

with open("exports/wf4_totalsadv2.json", "w", encoding="utf-8") as f:
    json.dump(OUT, f, indent=2, ensure_ascii=False)
print("\nsaved -> exports/wf4_totalsadv2.json")
