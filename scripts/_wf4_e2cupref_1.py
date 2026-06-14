# WF4 - REFUTATION du finding "favori de coupe gagne moins a cote egale"
# Tests adversariaux:
#  T1 confound temporel: champ restreint a la MEME fenetre temporelle que les coupes
#  T2 confound 8035: new-champ (8036/37/42/43/44) vs coupes (memes dates de naissance)
#  T3 par ligue de coupe (effet porte par une seule ligue ?)
#  T4 overround 1X2 par famille (marge mecanique vs mispricing)
#  T5 test stratifie par bucket 0.05 (Mantel-Haenszel sur WR) -> tue l'effet composition
#  T6 bootstrap (10k) sur la difference de ROI, stratifie par bucket, new-champ vs coupe
#  T7 sous-periodes: coupes coupees en 2 moities temporelles
# Lecture seule. Sortie: exports/wf4_e2cupref.json
import sys, json, math, random
sys.path.insert(0, ".")
from scraper.config import load_settings
from sqlalchemy import create_engine, text

NEWCHAMP = {"InstantLeague-8036", "InstantLeague-8037", "InstantLeague-8042",
            "InstantLeague-8043", "InstantLeague-8044"}
CUP = {"InstantLeague-8056", "InstantLeague-8060", "InstantLeague-8065"}
CHAMP = NEWCHAMP | {"InstantLeague-8035"}

corrupted = set(int(k) for k in json.load(open("exports/corrupted_events.json"))["events"].keys())
eng = create_engine(load_settings().db_url)

SQL = text("""
SELECT ev.id, ev.competition, ev.expected_start, ev.round_info,
       r.score_a, r.score_b,
       o.odds_home, o.odds_draw, o.odds_away
FROM events ev
JOIN results r ON r.event_id = ev.id
JOIN (SELECT event_id, MIN(id) AS mid FROM odds_snapshots GROUP BY event_id) m
     ON m.event_id = ev.id
JOIN odds_snapshots o ON o.id = m.mid
""")
rows = []
with eng.connect() as c:
    for r in c.execute(SQL):
        rows.append(dict(r._mapping))

bets = []
for r in rows:
    if r["id"] in corrupted:
        continue
    sa, sb = r["score_a"], r["score_b"]
    if sa is None or sb is None:
        continue
    oh, od, oa = r["odds_home"], r["odds_draw"], r["odds_away"]
    if not oh or not od or not oa or min(oh, od, oa) < 1.0 or oh == oa:
        continue
    if oh < oa:
        fav_odds, win = oh, 1 if sa > sb else 0
    else:
        fav_odds, win = oa, 1 if sb > sa else 0
    if not (1.05 <= fav_odds < 1.60):
        continue
    ovr = 1/oh + 1/od + 1/oa
    bets.append({"comp": r["competition"], "ts": str(r["expected_start"]),
                 "odds": fav_odds, "win": win, "ovr": ovr})

cup_bets = [b for b in bets if b["comp"] in CUP]
champ_bets = [b for b in bets if b["comp"] in CHAMP]
newchamp_bets = [b for b in bets if b["comp"] in NEWCHAMP]
cup_min_ts = min(b["ts"] for b in cup_bets)
cup_max_ts = max(b["ts"] for b in cup_bets)
champ_win = [b for b in champ_bets if b["ts"] >= cup_min_ts]
b8035_win = [b for b in champ_win if b["comp"] == "InstantLeague-8035"]


def stat(bl):
    n = len(bl)
    if n == 0:
        return dict(n=0)
    w = sum(b["win"] for b in bl)
    roi = sum(b["odds"]*b["win"]-1 for b in bl)/n
    return dict(n=n, wr=round(w/n, 4), roi_pct=round(100*roi, 2),
                avg_odds=round(sum(b["odds"] for b in bl)/n, 4),
                avg_ovr=round(sum(b["ovr"] for b in bl)/n, 4))


def sf(z):
    return 0.5*math.erfc(z/math.sqrt(2.0))


out = {"cup_window": [cup_min_ts, cup_max_ts]}

# T1/T2/T3: scopes
scopes = {
    "cup-all": cup_bets,
    "champ-all": champ_bets,
    "champ-samewindow": champ_win,
    "8035-samewindow": b8035_win,
    "newchamp-all": newchamp_bets,
}
for lg in sorted(CUP):
    scopes["cup-" + lg.split("-")[1]] = [b for b in cup_bets if b["comp"] == lg]
out["scopes"] = {k: stat(v) for k, v in scopes.items()}

# T5: stratifie par bucket 0.05 -- Mantel-Haenszel-like z sur WR (cup vs newchamp et cup vs champ-samewindow)
def strat_test(A, B, label):
    # H0: meme WR par bucket. z combine: sum(wA - nA*pPool) / sqrt(sum var)
    num, den, per = 0.0, 0.0, []
    for i in range(11):
        lo, hi = 1.05+0.05*i, 1.10+0.05*i
        a = [b for b in A if lo <= b["odds"] < hi]
        bb = [b for b in B if lo <= b["odds"] < hi]
        na, nb = len(a), len(bb)
        if na < 10 or nb < 10:
            continue
        wa, wb = sum(x["win"] for x in a), sum(x["win"] for x in bb)
        pp = (wa+wb)/(na+nb)
        num += wa - na*pp
        den += pp*(1-pp)*na*nb/(na+nb)
        ra = sum(x["odds"]*x["win"]-1 for x in a)/na
        rb = sum(x["odds"]*x["win"]-1 for x in bb)/nb
        per.append(dict(lo=round(lo, 2), nA=na, nB=nb, wrA=round(wa/na, 3), wrB=round(wb/nb, 3),
                        roiA=round(100*ra, 2), roiB=round(100*rb, 2),
                        cup_worse_wr=wa/na < wb/nb, cup_worse_roi=ra < rb))
    z = num/math.sqrt(den) if den > 0 else 0.0
    nwr = sum(1 for p in per if p["cup_worse_wr"])
    nroi = sum(1 for p in per if p["cup_worse_roi"])
    return dict(label=label, z_strat=round(z, 3), p_one_sided=float(f"{sf(-z):.3e}"),
                buckets=per, n_buckets=len(per),
                buckets_cup_worse_wr=nwr, buckets_cup_worse_roi=nroi)


out["strat_cup_vs_newchamp"] = strat_test(cup_bets, newchamp_bets, "cup vs newchamp (meme fenetre)")
out["strat_cup_vs_champwin"] = strat_test(cup_bets, champ_win, "cup vs champ-samewindow")
out["strat_cup_vs_champall"] = strat_test(cup_bets, champ_bets, "cup vs champ-all (replication du finding)")

# T6: bootstrap stratifie par bucket sur diff de ROI (cup - newchamp), 10k
random.seed(42)
def buckets_of(bl):
    d = {}
    for b in bl:
        i = int((b["odds"]-1.05)/0.05)
        d.setdefault(i, []).append(b["odds"]*b["win"]-1)
    return d

bc, bn = buckets_of(cup_bets), buckets_of(newchamp_bets)
common = sorted(set(bc) & set(bn))
NB = 10000
diffs = []
for _ in range(NB):
    sc, nc, sn_, nn = 0.0, 0, 0.0, 0
    for i in common:
        xs = bc[i]; ys = bn[i]
        sc += sum(random.choice(xs) for _ in xs); nc += len(xs)
        sn_ += sum(random.choice(ys) for _ in ys); nn += len(ys)
    diffs.append(100*(sc/nc - sn_/nn))
diffs.sort()
obs = 100*(sum(x for i in common for x in bc[i])/sum(len(bc[i]) for i in common)
           - sum(x for i in common for x in bn[i])/sum(len(bn[i]) for i in common))
out["bootstrap_roi_diff_cup_minus_newchamp"] = dict(
    obs=round(obs, 2), ci95=[round(diffs[int(0.025*NB)], 2), round(diffs[int(0.975*NB)], 2)],
    p_diff_ge0=round(sum(1 for d in diffs if d >= 0)/NB, 4))

# T7: sous-periodes coupes (2 moities temporelles), vs newchamp memes moities
cup_sorted = sorted(cup_bets, key=lambda b: b["ts"])
half_ts = cup_sorted[len(cup_sorted)//2]["ts"]
out["half_ts"] = half_ts
for tag, lo_t, hi_t in [("H1", cup_min_ts, half_ts), ("H2", half_ts, "9999")]:
    cs = [b for b in cup_bets if lo_t <= b["ts"] < hi_t]
    ns = [b for b in newchamp_bets if lo_t <= b["ts"] < hi_t]
    out[f"period_{tag}"] = dict(cup=stat(cs), newchamp=stat(ns))

with open("exports/wf4_e2cupref.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)

print(json.dumps(out, indent=1, ensure_ascii=False))
