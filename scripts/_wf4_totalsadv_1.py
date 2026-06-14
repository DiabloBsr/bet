# WF4 TOTALS - verification ADVERSARIALE du finding "cap dur total<=6"
# READ-ONLY. Checks: corrupted inclus, goals_json clipping, sous-periodes, par ligue,
# lignes offertes >=6.5, recalcul attendu 7+ conservatif.
import sys, json, pickle
sys.path.insert(0, ".")
import numpy as np
from scipy.stats import poisson
from collections import Counter, defaultdict
from scraper.config import load_settings
from sqlalchemy import create_engine, text

e = create_engine(load_settings().db_url)

with open("exports/corrupted_events.json", "r", encoding="utf-8") as f:
    corr = json.load(f)
CORRUPT = set(int(k) for k in corr["events"].keys())
print(f"corrupted ids: {len(CORRUPT)}")

with e.connect() as conn:
    rows = conn.execute(text(
        "SELECT r.event_id, ev.competition, ev.expected_start, r.score_a, r.score_b, "
        "r.ht_score_a, r.ht_score_b, r.goals_json "
        "FROM results r JOIN events ev ON ev.id = r.event_id"
    )).fetchall()
print(f"all results rows: {len(rows)}")

# 1) max total avec ET sans corrupted, par ligue, par semaine
mx_all = max(r[3] + r[4] for r in rows if r[3] is not None and r[4] is not None)
clean = [r for r in rows if r[0] not in CORRUPT and r[3] is not None and r[4] is not None]
mx_clean = max(r[3] + r[4] for r in clean)
corrupt_rows = [r for r in rows if r[0] in CORRUPT and r[3] is not None and r[4] is not None]
mx_corrupt = max((r[3] + r[4] for r in corrupt_rows), default=-1)
print(f"max_total all={mx_all} clean={mx_clean} corrupted-only={mx_corrupt} (n_corrupt_with_result={len(corrupt_rows)})")

per_league = defaultdict(lambda: [0, 0])  # n, max
per_week = defaultdict(lambda: [0, 0])
for r in clean:
    t = r[3] + r[4]
    pl = per_league[r[1]]; pl[0] += 1; pl[1] = max(pl[1], t)
    wk = str(r[2])[:10] if r[2] else "?"
    wk = wk[:8] + ("A" if wk[8:10] < "16" else "B")  # demi-mois
    pw = per_week[wk]; pw[0] += 1; pw[1] = max(pw[1], t)
print("par ligue (n, max_total):", {k: tuple(v) for k, v in sorted(per_league.items())})
print("par demi-mois (n, max_total):", {k: tuple(v) for k, v in sorted(per_week.items())})

# 2) goals_json: longueur max observee (clipping scraper ?) + mismatches
gl_lens = Counter(); mism = 0; ng = 0
for r in clean:
    if not r[7]:
        continue
    try:
        gl = json.loads(r[7])
    except Exception:
        continue
    if isinstance(gl, list):
        ng += 1
        gl_lens[len(gl)] += 1
        if len(gl) != r[3] + r[4]:
            mism += 1
print(f"goals_json parses: {ng} | len>6: {sum(v for k, v in gl_lens.items() if k > 6)} | mismatch len!=total: {mism}")
print("dist longueurs goals_json:", dict(sorted(gl_lens.items())))

# 3) lignes offertes dans extra_markets: existe-t-il une ligne dont le payoff depend de 7+ ?
with open("exports/wf4_totals_data.pkl", "rb") as f:
    D = pickle.load(f)
print(f"\npkl events: {len(D)}")
# echantillon de structure des marches totals dans extra_markets brut
with e.connect() as conn:
    xms = conn.execute(text(
        "SELECT o.extra_markets FROM odds_snapshots o "
        "WHERE o.id IN (SELECT MIN(id) FROM odds_snapshots GROUP BY event_id) LIMIT 800"
    )).fetchall()
lines = Counter(); totx_sels = Counter()
for (xm,) in xms:
    try:
        d = json.loads(xm) if xm else {}
    except Exception:
        continue
    for sel in (d.get("+/-") or {}):
        lines[sel] += 1
    for sel in (d.get("Total de buts") or {}):
        totx_sels[sel] += 1
print("selections '+/-':", dict(lines))
print("selections 'Total de buts':", dict(totx_sels))

# 4) attendu 7+ conservatif: trois modeles
ar = np.arange(14)
n7_grid = 0.0
for r in D:
    g = np.outer(poisson.pmf(ar, r["lh"]), poisson.pmf(ar, r["la"]))
    tot = np.add.outer(ar, ar)
    n7_grid += g[tot >= 7].sum()
print(f"\nattendu 7+ sous grille 1X2 (n={len(D)}): {n7_grid:.1f} | observe: "
      f"{sum(1 for r in D if r['tot'] >= 7)}")
# modele conservatif min: Poisson simple calee sur la moyenne REELLE des totaux
mu_real = np.mean([r["tot"] for r in D])
p7_pois = 1 - poisson.cdf(6, mu_real)
print(f"mu_real={mu_real:.3f} -> Poisson simple P(7+)={p7_pois:.4f} -> attendu {p7_pois*len(D):.0f}")
# modele tail-ratio: extrapolation geometrique du ratio P(6)/P(5) observe
cnt = Counter(r["tot"] for r in D)
ratio65 = cnt[6] / cnt[5]
exp7_tail = cnt[6] * ratio65  # decroissance au moins aussi rapide
print(f"P(6)/P(5) observe={ratio65:.3f} -> attendu 7 (extrapolation geometrique): {exp7_tail:.0f}")

# 5) distribution reelle vs grille brute (verif des +22/+32/+38%)
exp_cnt = np.zeros(14)
for r in D:
    g = np.outer(poisson.pmf(ar, r["lh"]), poisson.pmf(ar, r["la"]))
    tot = np.add.outer(ar, ar)
    for t in range(14):
        exp_cnt[t] += g[tot == t].sum()
obs_cnt = np.array([cnt.get(t, 0) for t in range(14)])
print("\ntotal: observe / attendu_grille (ratio)")
for t in range(8):
    if exp_cnt[t] > 0:
        print(f"  {t}: {obs_cnt[t]:6d} / {exp_cnt[t]:8.1f}  ({obs_cnt[t]/exp_cnt[t]:.3f})")
