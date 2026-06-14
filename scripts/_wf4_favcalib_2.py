# WF4 - Raffinements calibration favoris:
#  A) split favori domicile/exterieur par bucket
#  B) buckets fins 0.02 dans 1.11-1.27
#  C) marches derives sur la zone d'exces [1.15-1.20): 1X2&Total, HT/FT fav/fav, Mi-tps 1X2 fav
# Cotes d'OUVERTURE (MIN(o.id)), settlement via results (+ht) - lecture seule.
import sys, json
sys.path.insert(0, ".")
from scraper.config import load_settings
from sqlalchemy import create_engine, text
from scipy.stats import binomtest

e = create_engine(load_settings().db_url)
with open("exports/corrupted_events.json", "r", encoding="utf-8") as f:
    CORRUPT = set(int(k) for k in json.load(f)["events"].keys())

CHAMP = {"InstantLeague-8035", "InstantLeague-8036", "InstantLeague-8037",
         "InstantLeague-8042", "InstantLeague-8043", "InstantLeague-8044"}
CUP = {"InstantLeague-8056", "InstantLeague-8060", "InstantLeague-8065"}
ALL9 = CHAMP | CUP

SQL = """
SELECT e.id, e.competition, e.expected_start,
       o.odds_home, o.odds_draw, o.odds_away, o.extra_markets,
       r.score_a, r.score_b, r.ht_score_a, r.ht_score_b, r.goals_json
FROM events e
JOIN results r ON r.event_id = e.id
JOIN odds_snapshots o ON o.event_id = e.id
WHERE o.id = (SELECT MIN(o2.id) FROM odds_snapshots o2 WHERE o2.event_id = e.id)
"""
rows = []
with e.connect() as c:
    raw = c.execute(text(SQL)).fetchall()
for r in raw:
    (eid, comp, exp_start, oh, od, oa, em, sa, sb, ha, hb, gj) = r
    if comp not in ALL9 or eid in CORRUPT:
        continue
    if ha is not None and hb is not None and (ha > sa or hb > sb):
        continue
    if gj:
        try:
            g = json.loads(gj)
            if isinstance(g, list) and len(g) > 0 and len(g) != (sa + sb):
                continue
        except Exception:
            pass
    if not oh or not od or not oa or oh <= 1.0 or od <= 1.0 or oa <= 1.0:
        continue
    rows.append(dict(eid=eid, comp=comp, start=exp_start, oh=oh, od=od, oa=oa,
                     em=em, sa=sa, sb=sb, ha=ha, hb=hb))

# split walk-forward 8035 (meme cut que script 1: 70% par expected_start des favoris)
def fav_of(r):
    if r["oh"] < r["oa"]:
        return "H", r["oh"]
    if r["oa"] < r["oh"]:
        return "A", r["oa"]
    return None, None

starts_8035 = sorted(r["start"] for r in rows if r["comp"] == "InstantLeague-8035"
                     and fav_of(r)[1] and 1.05 <= fav_of(r)[1] < 1.60)
cut = starts_8035[int(len(starts_8035) * 0.70)]

def agg(bs, label, store, ntests):
    n = len(bs)
    if n == 0:
        return ntests
    wins = sum(1 for b in bs if b["win"])
    avg_odds = sum(b["odds"] for b in bs) / n
    be = sum(1.0 / b["odds"] for b in bs) / n
    profit = sum((b["odds"] - 1.0) if b["win"] else -1.0 for b in bs)
    p_be = binomtest(wins, n, be, alternative="greater").pvalue
    store[label] = dict(n=n, wr=round(wins / n, 4), avg_odds=round(avg_odds, 4),
                        roi_pct=round(100 * profit / n, 3), p_breakeven=round(p_be, 6))
    return ntests + 1

out = {}
n_tests = 0

# ---------- A) split domicile / exterieur ----------
BUCKETS = [(round(1.05 + 0.05 * i, 2), round(1.10 + 0.05 * i, 2)) for i in range(11)]
for scope_name, comps in [("pooled-9", ALL9), ("champ", CHAMP), ("cup", CUP)]:
    tbl = {}
    for side in ("H", "A"):
        for lo, hi in BUCKETS:
            bs = []
            for r in rows:
                if r["comp"] not in comps:
                    continue
                s, fo = fav_of(r)
                if s != side or fo is None or not (lo <= fo < hi):
                    continue
                win = (r["sa"] > r["sb"]) if side == "H" else (r["sb"] > r["sa"])
                bs.append(dict(odds=fo, win=win))
            n_tests = agg(bs, f"fav{side}[{lo:.2f}-{hi:.2f})", tbl, n_tests)
    out[f"sideSplit:{scope_name}"] = tbl

# ---------- B) buckets fins 0.02 sur 1.11-1.27, pooled-9 ----------
tbl = {}
fine = [(round(1.11 + 0.02 * i, 2), round(1.13 + 0.02 * i, 2)) for i in range(8)]
for lo, hi in fine:
    bs = []
    for r in rows:
        s, fo = fav_of(r)
        if fo is None or not (lo <= fo < hi):
            continue
        win = (r["sa"] > r["sb"]) if s == "H" else (r["sb"] > r["sa"])
        bs.append(dict(odds=fo, win=win))
    n_tests = agg(bs, f"[{lo:.2f}-{hi:.2f})", tbl, n_tests)
out["fine:pooled-9"] = tbl

# ---------- C) marches derives, favori dans [1.15-1.20) ----------
# selections testees (cote depuis extra_markets du MEME snapshot d'ouverture)
def get_market(em, mname, sel):
    try:
        d = json.loads(em)
        return d.get(mname, {}).get(sel)
    except Exception:
        return None

derived = {
    "1X2&Total fav/<3.5": lambda r, s: ("1X2 & Total", ("1 / < 3.5" if s == "H" else "2 / < 3.5"),
        ((r["sa"] > r["sb"]) if s == "H" else (r["sb"] > r["sa"])) and (r["sa"] + r["sb"] <= 3)),
    "1X2&Total fav/>3.5": lambda r, s: ("1X2 & Total", ("1 / > 3.5" if s == "H" else "2 / > 3.5"),
        ((r["sa"] > r["sb"]) if s == "H" else (r["sb"] > r["sa"])) and (r["sa"] + r["sb"] >= 4)),
    "HT/FT fav/fav": lambda r, s: ("HT/FT", ("1/1" if s == "H" else "2/2"),
        (r["ha"] is not None) and (((r["ha"] > r["hb"]) and (r["sa"] > r["sb"])) if s == "H"
                                   else ((r["hb"] > r["ha"]) and (r["sb"] > r["sa"])))),
    "Mi-tps 1X2 fav": lambda r, s: ("Mi-tps 1X2", ("1" if s == "H" else "2"),
        (r["ha"] is not None) and ((r["ha"] > r["hb"]) if s == "H" else (r["hb"] > r["ha"]))),
    "1X2&G/NG fav+gg": lambda r, s: ("1X2 & G/NG",
        ("1 gagne et les deux équipes marquent" if s == "H" else "2 gagne et les deux équipes marquent"),
        ((r["sa"] > r["sb"]) if s == "H" else (r["sb"] > r["sa"])) and r["sa"] > 0 and r["sb"] > 0),
}
for zone_lo, zone_hi, zlabel in [(1.15, 1.20, "fav[1.15-1.20)"), (1.30, 1.60, "fav[1.30-1.60)")]:
    for scope_name, comps in [("pooled-9", ALL9)]:
        tbl = {}
        for dname, fn in derived.items():
            bs = []
            nomkt = 0
            for r in rows:
                if r["comp"] not in comps:
                    continue
                s, fo = fav_of(r)
                if fo is None or not (zone_lo <= fo < zone_hi):
                    continue
                mname, sel, win = fn(r, s)
                if ("HT" in dname or "Mi-tps" in dname) and r["ha"] is None:
                    continue
                odds = get_market(r["em"], mname, sel)
                if odds is None or odds <= 1.0 or odds >= 100.0:
                    nomkt += 1
                    continue
                bs.append(dict(odds=odds, win=bool(win)))
            n_tests = agg(bs, f"{dname} (nomkt={nomkt})", tbl, n_tests)
        out[f"derived:{zlabel}:{scope_name}"] = tbl

# ---------- C bis) memes derives sur 8035-test uniquement (walk-forward) ----------
tbl = {}
for dname, fn in derived.items():
    bs = []
    for r in rows:
        if r["comp"] != "InstantLeague-8035" or r["start"] < cut:
            continue
        s, fo = fav_of(r)
        if fo is None or not (1.15 <= fo < 1.20):
            continue
        if ("HT" in dname or "Mi-tps" in dname) and r["ha"] is None:
            continue
        mname, sel, win = fn(r, s)
        odds = get_market(r["em"], mname, sel)
        if odds is None or odds <= 1.0 or odds >= 100.0:
            continue
        bs.append(dict(odds=odds, win=bool(win)))
    n_tests = agg(bs, dname, tbl, n_tests)
out["derived:fav[1.15-1.20):8035-test"] = tbl

out["n_tests_scanned_script2"] = n_tests
with open("exports/wf4_favcalib_refine.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)

for sec, tbl in out.items():
    if not isinstance(tbl, dict):
        continue
    print(f"\n=== {sec} ===")
    for k, a in tbl.items():
        print(f"{k:42s} n={a['n']:5d} wr={a['wr']:.3f} avg_o={a['avg_odds']:6.3f} "
              f"roi={a['roi_pct']:7.2f}% p_be={a['p_breakeven']:.5f}")
print(f"\nn_tests_scanned_script2={n_tests}")
