# WF4 - Validation ciblee de l'exces de calibration favoris [1.15,1.20) (et [1.13,1.20))
#  1) per-ligue + pooled 5 nouveaux championnats (replication independante de 8035/E2)
#  2) tests binomiaux EXACTS vs proba implicite moyenne (marge retiree) et vs break-even
#  3) stabilite temporelle 8035 (par tranche de ~25% du temps)
# Cote d'OUVERTURE = MIN(o.id). Lecture seule. Sortie: exports/wf4_favcalib4.json
import sys, json
sys.path.insert(0, ".")
from scraper.config import load_settings
from sqlalchemy import create_engine, text
from scipy.stats import binomtest

NEWCHAMP = {"InstantLeague-8036", "InstantLeague-8037", "InstantLeague-8042",
            "InstantLeague-8043", "InstantLeague-8044"}
CUP = {"InstantLeague-8056", "InstantLeague-8060", "InstantLeague-8065"}

corrupted = set(int(k) for k in json.load(open("exports/corrupted_events.json"))["events"].keys())
eng = create_engine(load_settings().db_url)
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
with eng.connect() as c:
    for r in c.execute(SQL):
        d = dict(r._mapping)
        if d["id"] in corrupted: continue
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
        bets.append({"comp": d["competition"], "ts": str(d["expected_start"]),
                     "odds": fo, "win": win, "impl": (1/fo)/ovr})


def stat(sel, label):
    n = len(sel)
    if n == 0:
        return {"label": label, "n": 0}
    w = sum(b["win"] for b in sel)
    avg_o = sum(b["odds"] for b in sel)/n
    roi = 100*sum(b["odds"]*b["win"]-1 for b in sel)/n
    p_impl_mean = sum(b["impl"] for b in sel)/n
    p_be_mean = sum(1/b["odds"] for b in sel)/n
    bt_impl = binomtest(w, n, p_impl_mean, alternative="greater").pvalue
    bt_be = binomtest(w, n, p_be_mean, alternative="greater").pvalue
    return {"label": label, "n": n, "wins": w, "wr": round(w/n, 4),
            "avg_odds": round(avg_o, 4), "impl_wr": round(p_impl_mean, 4),
            "be_wr": round(p_be_mean, 4), "roi_pct": round(roi, 2),
            "p_exact_vs_impl": float(f"{bt_impl:.3e}"),
            "p_exact_vs_breakeven": float(f"{bt_be:.3e}")}


def inzone(b, lo, hi):
    return lo <= b["odds"] < hi


out = {"tests": []}
print(f"total bets: {len(bets)}")

for lo, hi in [(1.15, 1.20), (1.13, 1.20)]:
    print(f"\n########## zone [{lo},{hi}) ##########")
    # per league
    for lg in sorted({b["comp"] for b in bets}):
        s = stat([b for b in bets if b["comp"] == lg and inzone(b, lo, hi)], f"{lg} [{lo},{hi})")
        out["tests"].append(s)
        if s["n"]:
            print(f"{s['label']:>38} n={s['n']:>4} wr={s['wr']:.4f} impl={s['impl_wr']:.4f} "
                  f"roi={s['roi_pct']:>7.2f} pIMPL={s['p_exact_vs_impl']:.4f} pBE={s['p_exact_vs_breakeven']:.4f}")
    # pooled groups
    for name, grp in [("POOLED-5-NEWCHAMP", NEWCHAMP), ("POOLED-CUP", CUP),
                      ("POOLED-6-CHAMP", NEWCHAMP | {"InstantLeague-8035"})]:
        s = stat([b for b in bets if b["comp"] in grp and inzone(b, lo, hi)], f"{name} [{lo},{hi})")
        out["tests"].append(s)
        print(f"{s['label']:>38} n={s['n']:>4} wr={s['wr']:.4f} impl={s['impl_wr']:.4f} "
              f"roi={s['roi_pct']:>7.2f} pIMPL={s['p_exact_vs_impl']:.4f} pBE={s['p_exact_vs_breakeven']:.4f}")

# stabilite temporelle 8035 dans [1.15,1.20): 4 quartiles temporels
b35 = sorted([b for b in bets if b["comp"] == "InstantLeague-8035" and inzone(b, 1.15, 1.20)],
             key=lambda b: b["ts"])
q = len(b35)//4
print(f"\n########## 8035 [1.15,1.20) par quartile temporel (n={len(b35)}) ##########")
for i in range(4):
    sel = b35[i*q:(i+1)*q if i < 3 else len(b35)]
    s = stat(sel, f"8035-Q{i+1} {sel[0]['ts'][:10]}->{sel[-1]['ts'][:10]}")
    out["tests"].append(s)
    print(f"{s['label']:>38} n={s['n']:>4} wr={s['wr']:.4f} impl={s['impl_wr']:.4f} "
          f"roi={s['roi_pct']:>7.2f} pIMPL={s['p_exact_vs_impl']:.4f}")

# zone 1.30-1.60 pooled-9 et champ-only: confirmation calibration parfaite (test bilateral)
for name, grp in [("POOLED-9", None), ("POOLED-6-CHAMP", NEWCHAMP | {"InstantLeague-8035"})]:
    sel = [b for b in bets if inzone(b, 1.30, 1.60) and (grp is None or b["comp"] in grp)]
    n = len(sel); w = sum(b["win"] for b in sel)
    p0 = sum(b["impl"] for b in sel)/n
    bt2 = binomtest(w, n, p0, alternative="two-sided").pvalue
    roi = 100*sum(b["odds"]*b["win"]-1 for b in sel)/n
    s = {"label": f"{name} [1.30,1.60) two-sided vs impl", "n": n, "wr": round(w/n, 4),
         "impl_wr": round(p0, 4), "roi_pct": round(roi, 2), "p_two_sided_vs_impl": float(f"{bt2:.3e}")}
    out["tests"].append(s)
    print(f"\n{s['label']}: n={n} wr={s['wr']} impl={s['impl_wr']} roi={s['roi_pct']} p2s={bt2:.3f}")

out["n_tests_this_script"] = len(out["tests"])
with open("exports/wf4_favcalib4.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
print(f"\nn_tests_this_script={len(out['tests'])}")
