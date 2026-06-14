# WF4 TOTALS - step 5:
# (a) value-jitter par paire sur +/- 3.5 : parier le cote dont la cote ouverte est genereuse
#     vs la moyenne historique de la paire (uniquement occurrences PASSEES -> pas de fuite).
# (b) scan Total de buts exact x lambda bucket x groupe, walk-forward 8035.
import sys, pickle, math
sys.path.insert(0, ".")
import numpy as np
from scipy.stats import norm

with open("exports/wf4_totals_data.pkl", "rb") as f:
    D = pickle.load(f)

def roi_stats(bets):
    if not bets:
        return 0, 0.0, 0.0, 0.0, 1.0
    r = np.array([(o - 1) if w else -1.0 for w, o in bets])
    n = len(r); roi = float(r.mean())
    wr = float(np.mean([w for w, _ in bets])); ao = float(np.mean([o for _, o in bets]))
    se = r.std(ddof=1) / math.sqrt(n) if n > 1 else 1e9
    p = 2 * (1 - norm.cdf(abs(roi) / se)) if se > 0 else 1.0
    return n, wr, roi, ao, float(p)

n_tests = 0

# need team names for pairs -> re-query (read-only)
from scraper.config import load_settings
from sqlalchemy import create_engine, text
e = create_engine(load_settings().db_url)
with e.connect() as conn:
    teams = dict((row[0], (row[1], row[2])) for row in conn.execute(
        text("SELECT id, team_a, team_b FROM events")).fetchall())

for r in D:
    r["pair"] = (r["comp"],) + teams[r["eid"]]

# ---------- (a) value-jitter paire sur +/- 3.5 ----------
print("=== VALUE-JITTER PAIRE +/- 3.5 ===")
def pair_jitter(sub, thrs):
    """sub trie par start; pour chaque event avec >=3 occurrences passees de la paire,
    delta = pf_pair_mean_past - pf_now (pf = implied fair p_over).
    delta>thr -> over value; delta<-thr -> under value."""
    sub = sorted(sub, key=lambda r: (r["start"], r["eid"]))
    hist = {}
    rows = []  # (delta, won_over, o_over, won_under, o_under)
    for r in sub:
        if r["ou_o"] and r["ou_u"] and r["ou_o"] < 100 and r["ou_u"] < 100:
            io, iu = 1 / r["ou_o"], 1 / r["ou_u"]
            pf = io / (io + iu)
            h = hist.get(r["pair"], [])
            if len(h) >= 3:
                rows.append((float(np.mean(h)) - pf, r["tot"] >= 4, r["ou_o"],
                             r["tot"] <= 3, r["ou_u"], r["start"]))
            hist.setdefault(r["pair"], []).append(pf)
    out = {}
    for thr in thrs:
        ob = [(w, o) for d, w, o, _, _, _ in rows if d > thr]
        ub = [(w2, o2) for d, _, _, w2, o2, _ in rows if d < -thr]
        out[thr] = (roi_stats(ob), roi_stats(ub), rows)
    return out

L35 = [r for r in D if r["comp"] == "InstantLeague-8035"]
starts = sorted(r["start"] for r in L35)
cut = starts[int(0.7 * len(starts))]
train = [r for r in L35 if r["start"] < cut]
test = [r for r in L35 if r["start"] >= cut]
print(f"8035: train {len(train)} / test {len(test)} (cut {cut})")

THRS = [0.01, 0.02, 0.03, 0.04]
# train (history within train only)
res_tr = pair_jitter(train, THRS)
print("-- TRAIN 8035 --")
for thr in THRS:
    (no, wro, roio, aoo, po), (nu, wru, roiu, aou, pu), _ = res_tr[thr]
    n_tests += 2
    print(f"thr={thr:.2f} OVER n={no:4d} ROI={roio*100:+.2f}% (p={po:.3f}) | UNDER n={nu:4d} ROI={roiu*100:+.2f}% (p={pu:.3f})")

# full-series run, then report only bets with start >= cut (history may include train = legal, c'est du passe)
res_full = pair_jitter(L35, THRS)
print("-- TEST 8035 (bets posterieurs au cut, historique = tout le passe) --")
for thr in THRS:
    _, _, rows = res_full[thr]
    ob = [(w, o) for d, w, o, _, _, s in rows if d > thr and s >= cut]
    ub = [(w2, o2) for d, _, _, w2, o2, s in rows if d < -thr and s >= cut]
    no, wro, roio, aoo, po = roi_stats(ob)
    nu, wru, roiu, aou, pu = roi_stats(ub)
    n_tests += 2
    print(f"thr={thr:.2f} OVER n={no:4d} WR={wro:.3f} ROI={roio*100:+.2f}% odds={aoo:.2f} (p={po:.3f}) | UNDER n={nu:4d} WR={wru:.3f} ROI={roiu*100:+.2f}% odds={aou:.2f} (p={pu:.3f})")

# nouvelles ligues pooled
NEWALL = [r for r in D if r["comp"] != "InstantLeague-8035"]
res_nl = pair_jitter(NEWALL, THRS)
print("-- POOLED NEW LEAGUES (8 ligues, historique passe par paire) --")
for thr in THRS:
    (no, wro, roio, aoo, po), (nu, wru, roiu, aou, pu), rows = res_nl[thr]
    n_tests += 2
    print(f"thr={thr:.2f} OVER n={no:4d} WR={wro:.3f} ROI={roio*100:+.2f}% odds={aoo:.2f} (p={po:.3f}) | UNDER n={nu:4d} WR={wru:.3f} ROI={roiu*100:+.2f}% odds={aou:.2f} (p={pu:.3f})")
print(f"(rows dispo nouvelles ligues avec >=3 occurrences passees: {len(res_nl[THRS[0]][2])})")

# ---------- (b) Total de buts exact x lambda x groupe ----------
print("\n=== TOTAL DE BUTS EXACT x lambda bucket (8035 train -> test) ===")
edges = [0, 2.4, 2.8, 3.2, 99]
cands = []
for sel in ["0", "1", "2", "3", "4", "5", "6"]:
    for i in range(len(edges) - 1):
        bets_tr = []
        for r in train:
            o = r["totx"].get(sel)
            if not o or o <= 1 or o >= 100:
                continue
            if not (edges[i] <= r["lh"] + r["la"] < edges[i + 1]):
                continue
            won = (r["tot"] == int(sel)) if sel != "6" else (r["tot"] >= 6)
            bets_tr.append((won, o))
        n, wr, roi, ao, p = roi_stats(bets_tr)
        n_tests += 1
        if n >= 100 and roi > 0.04:
            cands.append((sel, i, n, roi, p))
            print(f"TRAIN cand: sel={sel} lam[{edges[i]}-{edges[i+1]}) n={n} ROI={roi*100:+.2f}% p={p:.4f}")
for sel, i, ntr, roitr, ptr in cands:
    bets_te = []
    for r in test:
        o = r["totx"].get(sel)
        if not o or o <= 1 or o >= 100:
            continue
        if not (edges[i] <= r["lh"] + r["la"] < edges[i + 1]):
            continue
        won = (r["tot"] == int(sel)) if sel != "6" else (r["tot"] >= 6)
        bets_te.append((won, o))
    n, wr, roi, ao, p = roi_stats(bets_te)
    n_tests += 1
    print(f"  TEST: sel={sel} lam[{edges[i]}-{edges[i+1]}) n={n} WR={wr:.3f} ROI={roi*100:+.2f}% odds={ao:.2f} p={p:.4f}")
if not cands:
    print("  aucun candidat train (ROI>4%, n>=100)")

print(f"\nn_tests this script: {n_tests}")
