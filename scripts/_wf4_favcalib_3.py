# WF4 - Calibration des favoris 1X2 (verification independante + raffinements)
# - Cote d'OUVERTURE = snapshot MIN(id) par event. Lecture seule.
# - Scopes: 8035 walk-forward 70/30 (test only), pooled-newleagues, pooled-9, champ, coupe,
#           par ligue, fav-home vs fav-away, 8035 J0 vs J1+.
# - Buckets fins 0.05 sur [1.05,1.60) + bins 0.02 sur [1.10,1.26) + zones.
# Sortie: exports/wf4_favcalib3.json
import sys, json, math
sys.path.insert(0, ".")
from scraper.config import load_settings
from sqlalchemy import create_engine, text

NEW = {"InstantLeague-8036", "InstantLeague-8037", "InstantLeague-8042",
       "InstantLeague-8043", "InstantLeague-8044", "InstantLeague-8056",
       "InstantLeague-8060", "InstantLeague-8065"}
CHAMP = {"InstantLeague-8035", "InstantLeague-8036", "InstantLeague-8037",
         "InstantLeague-8042", "InstantLeague-8043", "InstantLeague-8044"}
CUP = {"InstantLeague-8056", "InstantLeague-8060", "InstantLeague-8065"}

corrupted = set(int(k) for k in json.load(open("exports/corrupted_events.json"))["events"].keys())
eng = create_engine(load_settings().db_url)

SQL = text("""
SELECT ev.id, ev.competition, ev.expected_start, ev.round_info,
       r.score_a, r.score_b, r.ht_score_a, r.ht_score_b, r.goals_json,
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
print(f"raw joined rows: {len(rows)}")

bets = []
excl = {"corrupt": 0, "ht_guard": 0, "goals_guard": 0, "odds_bad": 0, "tie": 0}
for r in rows:
    if r["id"] in corrupted:
        excl["corrupt"] += 1; continue
    sa, sb = r["score_a"], r["score_b"]
    if sa is None or sb is None:
        continue
    ha, hb = r["ht_score_a"], r["ht_score_b"]
    if ha is not None and hb is not None and (ha > sa or hb > sb):
        excl["ht_guard"] += 1; continue
    if r["goals_json"]:
        try:
            g = json.loads(r["goals_json"])
            if isinstance(g, list) and len(g) > 0 and len(g) != sa + sb:
                excl["goals_guard"] += 1; continue
        except Exception:
            pass
    oh, od, oa = r["odds_home"], r["odds_draw"], r["odds_away"]
    if not oh or not od or not oa or min(oh, od, oa) < 1.0:
        excl["odds_bad"] += 1; continue
    if oh == oa:
        excl["tie"] += 1; continue
    if oh < oa:
        fav_odds, side, win = oh, "H", 1 if sa > sb else 0
    else:
        fav_odds, side, win = oa, "A", 1 if sb > sa else 0
    ovr = 1/oh + 1/od + 1/oa
    bets.append({"comp": r["competition"], "ts": str(r["expected_start"]),
                 "round": r["round_info"], "odds": fav_odds, "side": side,
                 "win": win, "impl": (1/fav_odds)/ovr})
print(f"bets: {len(bets)}  excl={excl}")

b8035 = sorted([b for b in bets if b["comp"] == "InstantLeague-8035"], key=lambda b: b["ts"])
cut = int(len(b8035)*0.70)
b8035_test = b8035[cut:]
print(f"8035 bets={len(b8035)} test={len(b8035_test)} cut_ts={b8035[cut]['ts']}")

scopes = {
    "8035-wf-test": b8035_test,
    "8035-full": b8035,
    "8035-J0": [b for b in b8035 if b["round"] == "0"],
    "8035-J1plus": [b for b in b8035 if b["round"] != "0"],
    "pooled-newleagues": [b for b in bets if b["comp"] in NEW],
    "pooled-9": bets,
    "family-championnat": [b for b in bets if b["comp"] in CHAMP],
    "family-coupe": [b for b in bets if b["comp"] in CUP],
    "pooled-9-favHOME": [b for b in bets if b["side"] == "H"],
    "pooled-9-favAWAY": [b for b in bets if b["side"] == "A"],
}
for lg in sorted({b["comp"] for b in bets}):
    scopes["lg-" + lg.split("-")[1]] = [b for b in bets if b["comp"] == lg]

FINE5 = [(round(1.05+0.05*i, 2), round(1.10+0.05*i, 2)) for i in range(11)]
FINE2 = [(round(1.10+0.02*i, 2), round(1.12+0.02*i, 2)) for i in range(8)]
ZONES = [(1.05, 1.30), (1.30, 1.60), (1.10, 1.20), (1.13, 1.20), (1.15, 1.20), (1.05, 1.60)]


def sf(z):  # one-sided upper tail
    return 0.5*math.erfc(z/math.sqrt(2.0))


def ev(blist, lo, hi):
    sel = [b for b in blist if lo <= b["odds"] < hi]
    n = len(sel)
    if n < 5:
        return None
    w = sum(b["win"] for b in sel)
    avg_o = sum(b["odds"] for b in sel)/n
    roi = sum(b["odds"]*b["win"]-1 for b in sel)/n
    mu_be = sum(1/b["odds"] for b in sel)
    va_be = sum((1/b["odds"])*(1-1/b["odds"]) for b in sel)
    z_be = (w-mu_be)/math.sqrt(va_be) if va_be > 0 else 0
    mu_im = sum(b["impl"] for b in sel)
    va_im = sum(b["impl"]*(1-b["impl"]) for b in sel)
    z_im = (w-mu_im)/math.sqrt(va_im) if va_im > 0 else 0
    return dict(lo=lo, hi=hi, n=n, wins=w, wr=round(w/n, 4), avg_odds=round(avg_o, 4),
                be_wr=round(mu_be/n, 4), impl_wr=round(mu_im/n, 4),
                roi_pct=round(100*roi, 2),
                p_be=float(f"{sf(z_be):.3e}"), p_impl=float(f"{sf(z_im):.3e}"))


out = {"n_bets": len(bets), "excl": excl, "cut_8035": b8035[cut]["ts"], "scopes": {}}
n_tests = 0
for sn, bl in scopes.items():
    res = []
    for lo, hi in FINE5 + FINE2 + ZONES:
        n_tests += 1
        r = ev(bl, lo, hi)
        if r:
            res.append(r)
    out["scopes"][sn] = {"n": len(bl), "buckets": res}
out["n_tests_scanned"] = n_tests
with open("exports/wf4_favcalib3.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)

for sn in scopes:
    s = out["scopes"][sn]
    print(f"\n=== {sn} (n={s['n']}) ===")
    print(f"{'bucket':>13} {'n':>5} {'WR':>6} {'BE':>6} {'IMPL':>6} {'avgO':>6} {'ROI%':>7} {'pBE':>8} {'pIMPL':>8}")
    for b in s["buckets"]:
        print(f"[{b['lo']:.2f},{b['hi']:.2f}) {b['n']:>5} {b['wr']:>6.3f} {b['be_wr']:>6.3f} "
              f"{b['impl_wr']:>6.3f} {b['avg_odds']:>6.3f} {b['roi_pct']:>7.2f} "
              f"{b['p_be']:>8.1e} {b['p_impl']:>8.1e}")
print(f"\nn_tests_scanned={n_tests}")
