# WF4 - ADVERSARIAL re-check du finding favcalib [1.15,1.20) champ-only
# 1) integrite: duplicats results / odds, events corrompus
# 2) scan buckets adjacents (cherry-picking de zone ?)
# 3) walk-forward strict 70/30 par expected_start (8035 et pooled-6)
# 4) increment depuis le finding (reconstruction: 354 premiers bets pooled-6 par ts)
# 5) bootstrap ROI pooled-6 courant
# Lecture seule. Sortie: exports/wf4_favcalib5_adversarial.json
import sys, json, random
sys.path.insert(0, ".")
from scraper.config import load_settings
from sqlalchemy import create_engine, text
from scipy.stats import binomtest

NEWCHAMP = {"InstantLeague-8036", "InstantLeague-8037", "InstantLeague-8042",
            "InstantLeague-8043", "InstantLeague-8044"}
CHAMP6 = NEWCHAMP | {"InstantLeague-8035"}

corrupted = set(int(k) for k in json.load(open("exports/corrupted_events.json"))["events"].keys())
eng = create_engine(load_settings().db_url)

with eng.connect() as c:
    # duplicats results ?
    dup_r = c.execute(text("SELECT COUNT(*) - COUNT(DISTINCT event_id) FROM results")).scalar()
    # events avec >1 snapshot min ? (impossible par construction MIN(id), juste sanity)
    n_events = c.execute(text("SELECT COUNT(*) FROM events")).scalar()
    print(f"results dup rows (total - distinct event_id) = {dup_r} ; events = {n_events}")

SQL = text("""
SELECT ev.id, ev.competition, ev.expected_start,
       r.score_a, r.score_b, r.ht_score_a, r.ht_score_b, r.goals_json,
       o.odds_home, o.odds_draw, o.odds_away
FROM events ev
JOIN results r ON r.event_id = ev.id
JOIN (SELECT event_id, MIN(id) AS mid FROM odds_snapshots GROUP BY event_id) m ON m.event_id = ev.id
JOIN odds_snapshots o ON o.id = m.mid
""")
bets = []
seen = set()
ndup_bets = 0
with eng.connect() as c:
    for r in c.execute(SQL):
        d = dict(r._mapping)
        if d["id"] in corrupted: continue
        if d["id"] in seen: ndup_bets += 1
        seen.add(d["id"])
        sa, sb = d["score_a"], d["score_b"]
        if sa is None or sb is None: continue
        ha, hb = d["ht_score_a"], d["ht_score_b"]
        if ha is not None and hb is not None and (ha > sa or hb > sb): continue
        if d["goals_json"]:
            try:
                g = json.loads(d["goals_json"])
                if isinstance(g, list) and len(g) > 0 and len(g) != sa + sb: continue
            except Exception: pass
        oh, od, oa = d["odds_home"], d["odds_draw"], d["odds_away"]
        if not oh or not od or not oa or min(oh, od, oa) < 1.0 or oh == oa: continue
        if oh < oa:
            fo, win = oh, 1 if sa > sb else 0
        else:
            fo, win = oa, 1 if sb > sa else 0
        ovr = 1/oh + 1/od + 1/oa
        bets.append({"id": d["id"], "comp": d["competition"], "ts": str(d["expected_start"]),
                     "odds": fo, "win": win, "impl": (1/fo)/ovr, "ovr": ovr})
print(f"duplicated event rows in join: {ndup_bets} ; total bets: {len(bets)}")

def stat(sel, label):
    n = len(sel)
    if n == 0: return {"label": label, "n": 0}
    w = sum(b["win"] for b in sel)
    roi = 100*sum(b["odds"]*b["win"]-1 for b in sel)/n
    p_impl = sum(b["impl"] for b in sel)/n
    p_be = sum(1/b["odds"] for b in sel)/n
    return {"label": label, "n": n, "wins": w, "wr": round(w/n,4),
            "impl_wr": round(p_impl,4), "be_wr": round(p_be,4), "roi_pct": round(roi,2),
            "p_vs_impl": round(binomtest(w,n,p_impl,alternative="greater").pvalue,4),
            "p_vs_be": round(binomtest(w,n,p_be,alternative="greater").pvalue,4)}

out = {"sections": {}}

# ---- 2) buckets adjacents pooled-6 champ (la zone [1.15,1.20) etait-elle cherry-picked ?)
print("\n=== buckets fav odds, POOLED-6-CHAMP ===")
buckets = [(1.00,1.05),(1.05,1.10),(1.10,1.15),(1.15,1.20),(1.20,1.25),(1.25,1.30),(1.30,1.40),(1.40,1.60)]
sec = []
for lo,hi in buckets:
    s = stat([b for b in bets if b["comp"] in CHAMP6 and lo <= b["odds"] < hi], f"champ6 [{lo},{hi})")
    sec.append(s)
    if s["n"]: print(f"{s['label']:>22} n={s['n']:>5} wr={s['wr']:.4f} impl={s['impl_wr']:.4f} be={s['be_wr']:.4f} roi={s['roi_pct']:>7.2f} pI={s['p_vs_impl']:.4f} pBE={s['p_vs_be']:.4f}")
out["sections"]["buckets_champ6"] = sec

# ---- marge: overround moyen par bucket (la marge est-elle proportionnelle ?)
print("\n=== overround moyen par bucket champ6 ===")
for lo,hi in buckets:
    sel = [b for b in bets if b["comp"] in CHAMP6 and lo <= b["odds"] < hi]
    if sel:
        print(f"[{lo},{hi}) n={len(sel):>5} ovr_mean={sum(b['ovr'] for b in sel)/len(sel):.4f}")

# ---- 3) walk-forward strict 70/30 par expected_start
print("\n=== walk-forward 70/30 (train ignore, TEST only) zone [1.15,1.20) ===")
sec = []
for name, grp in [("8035", {"InstantLeague-8035"}), ("POOLED-6-CHAMP", CHAMP6), ("5-NEWCHAMP", NEWCHAMP)]:
    sub = sorted([b for b in bets if b["comp"] in grp], key=lambda b: b["ts"])
    cut = sub[int(0.7*len(sub))]["ts"]
    test = [b for b in sub if b["ts"] >= cut and 1.15 <= b["odds"] < 1.20]
    s = stat(test, f"WF-TEST {name} cut={cut[:16]}")
    sec.append(s)
    if s["n"]: print(f"{s['label']:>45} n={s['n']:>4} wr={s['wr']:.4f} impl={s['impl_wr']:.4f} roi={s['roi_pct']:>7.2f} pI={s['p_vs_impl']:.4f}")
out["sections"]["walkforward_test"] = sec

# ---- 4) increment depuis le finding: pooled-6 tri par ts, 354 premiers vs reste
sub = sorted([b for b in bets if b["comp"] in CHAMP6 and 1.15 <= b["odds"] < 1.20], key=lambda b: b["ts"])
first, rest = sub[:354], sub[354:]
comp_first = {}
for b in first: comp_first[b["comp"]] = comp_first.get(b["comp"],0)+1
print(f"\n=== reconstruction finding: 354 premiers bets pooled-6 [1.15,1.20) ===")
print("composition 354 premiers:", comp_first)
s1 = stat(first, "first354 (≈ sample du finding)")
s2 = stat(rest, f"INCREMENT post-finding (n={len(rest)})")
out["sections"]["increment"] = [s1, s2]
for s in (s1, s2):
    if s["n"]: print(f"{s['label']:>40} n={s['n']:>4} wr={s['wr']:.4f} impl={s['impl_wr']:.4f} roi={s['roi_pct']:>7.2f} pI={s['p_vs_impl']:.4f}")

# idem 5-newchamp: 110 premiers vs reste (la "replication independante")
subn = sorted([b for b in bets if b["comp"] in NEWCHAMP and 1.15 <= b["odds"] < 1.20], key=lambda b: b["ts"])
fn, rn = subn[:110], subn[110:]
s3 = stat(fn, "newchamp first110 (≈ replication du finding)")
s4 = stat(rn, f"newchamp INCREMENT (n={len(rn)})")
out["sections"]["increment_newchamp"] = [s3, s4]
for s in (s3, s4):
    if s["n"]: print(f"{s['label']:>48} n={s['n']:>4} wr={s['wr']:.4f} impl={s['impl_wr']:.4f} roi={s['roi_pct']:>7.2f} pI={s['p_vs_impl']:.4f}")

# ---- 5) bootstrap ROI pooled-6 courant
sel = sub
random.seed(42)
rois = []
for _ in range(10000):
    samp = [sel[random.randrange(len(sel))] for _ in range(len(sel))]
    rois.append(100*sum(b["odds"]*b["win"]-1 for b in samp)/len(samp))
rois.sort()
ci = (round(rois[250],2), round(rois[9750],2))
p_pos = sum(1 for x in rois if x > 0)/10000
print(f"\n=== bootstrap ROI pooled-6 [1.15,1.20) n={len(sel)} : CI95={ci}, P(ROI>0)={p_pos:.3f} ===")
out["sections"]["bootstrap"] = {"n": len(sel), "ci95": ci, "p_roi_pos": p_pos}

# Bonferroni sur le claim
out["sections"]["multiple_testing"] = {
    "claim_p": 0.0032, "n_tests_scanned": 505,
    "bonferroni": min(1.0, 0.0032*505),
    "note": "p ajuste = 1.0 -> claim non significatif apres correction"}
print(f"\nBonferroni: 0.0032 * 505 = {0.0032*505:.2f} (cap 1.0)")

with open("exports/wf4_favcalib5_adversarial.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
print("done")
