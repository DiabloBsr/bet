# WF4 - Calibration des favoris 1.05-1.60, pooled 9 ligues + familles + walk-forward 8035
# Cotes d'OUVERTURE uniquement (snapshot MIN(id) par event). Lecture seule.
import sys, json, math
sys.path.insert(0, ".")
from scraper.config import load_settings
from sqlalchemy import create_engine, text
from scipy.stats import binomtest

e = create_engine(load_settings().db_url)

# --- corrupted ids (couvre 8035 uniquement) ---
with open("exports/corrupted_events.json", "r", encoding="utf-8") as f:
    corr = json.load(f)
CORRUPT = set(int(k) for k in corr["events"].keys())

CHAMP = {"InstantLeague-8035", "InstantLeague-8036", "InstantLeague-8037",
         "InstantLeague-8042", "InstantLeague-8043", "InstantLeague-8044"}
CUP = {"InstantLeague-8056", "InstantLeague-8060", "InstantLeague-8065"}
ALL9 = CHAMP | CUP

SQL = """
SELECT e.id, e.competition, e.expected_start,
       o.odds_home, o.odds_draw, o.odds_away,
       r.score_a, r.score_b, r.ht_score_a, r.ht_score_b, r.goals_json
FROM events e
JOIN results r ON r.event_id = e.id
JOIN odds_snapshots o ON o.event_id = e.id
WHERE o.id = (SELECT MIN(o2.id) FROM odds_snapshots o2 WHERE o2.event_id = e.id)
"""

rows = []
with e.connect() as c:
    raw = c.execute(text(SQL)).fetchall()

n_corrupt_excl = 0
n_guard_excl = 0
n_oddsbad = 0
for r in raw:
    (eid, comp, exp_start, oh, od, oa, sa, sb, ha, hb, gj) = r
    if comp not in ALL9:
        continue
    if eid in CORRUPT:
        n_corrupt_excl += 1
        continue
    # garde-fou maison (nouvelles ligues non auditees): HT > FT ou goals_json incoherent
    bad = False
    if ha is not None and hb is not None:
        if ha > sa or hb > sb:
            bad = True
    if not bad and gj:
        try:
            g = json.loads(gj)
            if isinstance(g, list) and len(g) > 0 and len(g) != (sa + sb):
                bad = True
        except Exception:
            pass
    if bad:
        n_guard_excl += 1
        continue
    if not oh or not od or not oa or oh <= 1.0 or od <= 1.0 or oa <= 1.0:
        n_oddsbad += 1
        continue
    rows.append(dict(eid=eid, comp=comp, start=exp_start, oh=oh, od=od, oa=oa, sa=sa, sb=sb))

print(f"raw={len(raw)} kept={len(rows)} corrupt_excl={n_corrupt_excl} guard_excl={n_guard_excl} oddsbad={n_oddsbad}")

# --- construire les paris favoris ---
bets = []  # un par event avec favori dans [1.05, 1.60)
for r in rows:
    if r["oh"] < r["oa"]:
        fav_odds, win = r["oh"], r["sa"] > r["sb"]
    elif r["oa"] < r["oh"]:
        fav_odds, win = r["oa"], r["sb"] > r["sa"]
    else:
        continue  # cotes egales: pas de favori
    if not (1.05 <= fav_odds < 1.60):
        continue
    s = 1.0 / r["oh"] + 1.0 / r["od"] + 1.0 / r["oa"]
    imp_raw = 1.0 / fav_odds
    imp_norm = imp_raw / s
    bets.append(dict(eid=r["eid"], comp=r["comp"], start=r["start"], odds=fav_odds,
                     win=bool(win), imp_raw=imp_raw, imp_norm=imp_norm))

print(f"bets favoris 1.05-1.60: {len(bets)}")

# --- walk-forward split 8035 (70/30 par expected_start) ---
starts_8035 = sorted(b["start"] for b in bets if b["comp"] == "InstantLeague-8035")
cut = starts_8035[int(len(starts_8035) * 0.70)] if starts_8035 else None
print(f"8035 walk-forward cut at expected_start = {cut}")

def scope_of(b):
    out = ["pooled-9"]
    out.append("champ" if b["comp"] in CHAMP else "cup")
    if b["comp"] != "InstantLeague-8035":
        out.append("pooled-newleagues")
    else:
        out.append("8035-test" if b["start"] >= cut else "8035-train")
    return out

BUCKETS = [(round(1.05 + 0.05 * i, 2), round(1.10 + 0.05 * i, 2)) for i in range(11)]

def agg(bs):
    n = len(bs)
    if n == 0:
        return None
    wins = sum(1 for b in bs if b["win"])
    wr = wins / n
    avg_odds = sum(b["odds"] for b in bs) / n
    imp_raw = sum(b["imp_raw"] for b in bs) / n
    imp_norm = sum(b["imp_norm"] for b in bs) / n
    profit = sum((b["odds"] - 1.0) if b["win"] else -1.0 for b in bs)
    roi = profit / n
    # p-value vs break-even (ROI>0): wins ~ Binom(n, mean(1/odds))
    p_be = binomtest(wins, n, imp_raw, alternative="greater").pvalue
    # p-value vs implicite normalise (exces de calibration, 2-sided)
    p_imp = binomtest(wins, n, imp_norm, alternative="two-sided").pvalue
    return dict(n=n, wins=wins, wr=round(wr, 4), avg_odds=round(avg_odds, 4),
                imp_raw=round(imp_raw, 4), imp_norm=round(imp_norm, 4),
                roi_pct=round(100 * roi, 3), p_breakeven=round(p_be, 6),
                p_vs_implied_norm=round(p_imp, 6))

scopes = {}
for b in bets:
    for s in scope_of(b):
        scopes.setdefault(s, []).append(b)

results = {}
n_tests = 0
for sname, bs in sorted(scopes.items()):
    tbl = {}
    for lo, hi in BUCKETS:
        sub = [b for b in bs if lo <= b["odds"] < hi]
        a = agg(sub)
        if a:
            n_tests += 1
            tbl[f"[{lo:.2f}-{hi:.2f})"] = a
    # zones agregees
    for lo, hi, label in [(1.05, 1.30, "ZONE[1.05-1.30)"), (1.30, 1.60, "ZONE[1.30-1.60)"),
                          (1.10, 1.20, "ZONE[1.10-1.20)_E2")]:
        sub = [b for b in bs if lo <= b["odds"] < hi]
        a = agg(sub)
        if a:
            n_tests += 1
            tbl[label] = a
    results[sname] = tbl

out = dict(cut_8035=str(cut), n_bets=len(bets), n_tests_scanned=n_tests,
           excl=dict(corrupt=n_corrupt_excl, guard=n_guard_excl, oddsbad=n_oddsbad),
           buckets=results)
with open("exports/wf4_favcalib.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)

for sname in ["pooled-9", "champ", "cup", "pooled-newleagues", "8035-train", "8035-test"]:
    print(f"\n=== {sname} ===")
    print(f"{'bucket':22s} {'n':>5s} {'wr':>7s} {'imp_n':>7s} {'avg_o':>6s} {'roi%':>7s} {'p_be':>9s} {'p_imp':>9s}")
    for k, a in results.get(sname, {}).items():
        print(f"{k:22s} {a['n']:5d} {a['wr']:7.3f} {a['imp_norm']:7.3f} {a['avg_odds']:6.3f} "
              f"{a['roi_pct']:7.2f} {a['p_breakeven']:9.5f} {a['p_vs_implied_norm']:9.5f}")
print(f"\nn_tests_scanned={n_tests}")
