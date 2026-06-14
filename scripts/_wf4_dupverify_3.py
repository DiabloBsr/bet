# -*- coding: utf-8 -*-
"""
WF4 - VERIF ADVERSARIALE 3: la regle 'back sur-regime rpts5' (walk-forward 8035
70/30 + pooled newleagues) sous 3 regimes de dedup:
  A) aucun dedup (doit reproduire le ROI fantome ~+14.8% test30)
  B) dedup du finding: <30min meme (comp,ta,tb), garde MIN(id)
  C) dedup chirurgical: uniquement les clones stricts (memes cotes ouverture
     + meme goals_json + meme FT, tout gap) -> garde MIN(id)
+ mecanisme: part des paris test30 (mode A) dont l'historique rpts5 contient
  le sibling clone du match lui-meme.
Replique la logique de scripts/_wf4_seq_1.py (rpts seulement, pas d'inversion
lambda necessaire). Sortie: exports/wf4_dupverify3.json. LECTURE SEULE.
"""
import sys, json, math
sys.path.insert(0, ".")
from datetime import datetime
from itertools import combinations
import numpy as np
from scraper.config import load_settings
from sqlalchemy import create_engine, text

RNG = np.random.default_rng(42)
LEAGUES = ["InstantLeague-8035", "InstantLeague-8036", "InstantLeague-8037",
           "InstantLeague-8042", "InstantLeague-8043", "InstantLeague-8044",
           "InstantLeague-8056", "InstantLeague-8060", "InstantLeague-8065"]
NEW = set(LEAGUES[1:])

def load_rows():
    eng = create_engine(load_settings().db_url)
    corrupted = set(int(k) for k in json.load(open("exports/corrupted_events.json"))["events"].keys())
    with eng.connect() as c:
        res = c.execute(text("""
            SELECT e.id, e.competition, e.team_a, e.team_b, e.expected_start,
                   r.score_a, r.score_b, r.ht_score_a, r.ht_score_b, r.goals_json,
                   o.odds_home, o.odds_draw, o.odds_away
            FROM events e
            JOIN results r ON r.event_id = e.id
            JOIN odds_snapshots o ON o.id = (
                SELECT MIN(o2.id) FROM odds_snapshots o2 WHERE o2.event_id = e.id)
            ORDER BY e.expected_start, e.id""")).fetchall()
    rows = []
    for r in res:
        (eid, comp, ta, tb, start, sa, sb, hta, htb, gj, oh, od, oa) = r
        if comp not in LEAGUES or eid in corrupted:
            continue
        if sa is None or oh is None or od is None or oa is None:
            continue
        if min(oh, od, oa) <= 1.0:
            continue
        if hta is not None and htb is not None and (hta > sa or htb > sb):
            continue
        rows.append(dict(id=eid, comp=comp, ta=ta, tb=tb, start=str(start),
                         ts=datetime.fromisoformat(str(start)).timestamp(),
                         sa=int(sa), sb=int(sb), gj=gj,
                         oh=float(oh), od=float(od), oa=float(oa)))
    rows.sort(key=lambda r: (r["ts"], r["id"]))
    return rows

def dedup_30m(rows):
    bykey = {}
    for r in rows:
        bykey.setdefault((r["comp"], r["ta"], r["tb"]), []).append(r)
    drop = set()
    for key, lst in bykey.items():
        lst.sort(key=lambda r: (r["ts"], r["id"]))
        for i in range(1, len(lst)):
            if lst[i]["ts"] - lst[i - 1]["ts"] < 1800 and lst[i - 1]["id"] not in drop:
                drop.add(lst[i]["id"])
    return drop

def dedup_clones(rows):
    bykey = {}
    for r in rows:
        bykey.setdefault((r["comp"], r["ta"], r["tb"]), []).append(r)
    drop, siblings = set(), {}
    for key, lst in bykey.items():
        for a, b in combinations(lst, 2):
            if (a["oh"], a["od"], a["oa"]) != (b["oh"], b["od"], b["oa"]):
                continue
            if (a["sa"], a["sb"]) != (b["sa"], b["sb"]):
                continue
            if a["gj"] is None or a["gj"] != b["gj"]:
                continue
            lo, hi = (a, b) if a["id"] < b["id"] else (b, a)
            drop.add(hi["id"])
            siblings.setdefault(lo["id"], set()).add(hi["id"])
            siblings.setdefault(hi["id"], set()).add(lo["id"])
    return drop, siblings

def build_rpts5(rows):
    oh = np.array([r["oh"] for r in rows]); od = np.array([r["od"] for r in rows])
    oa = np.array([r["oa"] for r in rows])
    inv = 1 / oh + 1 / od + 1 / oa
    ph, pd, pa = (1 / oh) / inv, (1 / od) / inv, (1 / oa) / inv
    hist = {}
    for i, r in enumerate(rows):
        r["ph"] = float(ph[i]); r["pd"] = float(pd[i]); r["pa"] = float(pa[i])
        feats, hist_ids = {}, {}
        for side, team in (("h", (r["comp"], r["ta"])), ("a", (r["comp"], r["tb"]))):
            hl = hist.get(team, [])
            if len(hl) >= 5:
                feats[f"{side}_rpts5"] = float(np.mean([x[0] for x in hl[-5:]]))
                hist_ids[side] = [x[1] for x in hl[-5:]]
            else:
                feats[f"{side}_rpts5"] = None
        r["feats"] = feats; r["hist_ids"] = hist_ids
        for team, pw, pdr, gf, ga in (((r["comp"], r["ta"]), r["ph"], r["pd"], r["sa"], r["sb"]),
                                      ((r["comp"], r["tb"]), r["pa"], r["pd"], r["sb"], r["sa"])):
            pts = 3.0 if gf > ga else (1.0 if gf == ga else 0.0)
            hist.setdefault(team, []).append((pts - (3 * pw + pdr), r["id"]))
    return rows

def eval_rule(pop, thr, siblings=None):
    pnls, odds_used, wins, tainted = [], [], 0, 0
    for r in pop:
        d = r["feats"]["h_rpts5"] - r["feats"]["a_rpts5"]
        side = "h" if d >= thr else ("a" if -d >= thr else None)
        if side is None:
            continue
        o = r["oh"] if side == "h" else r["oa"]
        won = (r["sa"] > r["sb"]) if side == "h" else (r["sb"] > r["sa"])
        odds_used.append(o); pnls.append((o - 1) if won else -1.0); wins += int(won)
        if siblings is not None:
            sibs = siblings.get(r["id"], set())
            allh = set(r["hist_ids"].get("h", [])) | set(r["hist_ids"].get("a", []))
            tainted += int(bool(sibs & allh))
    n = len(pnls)
    if n == 0:
        return dict(n=0)
    p = np.array(pnls)
    boot = RNG.choice(p, size=(4000, n), replace=True).mean(axis=1)
    return dict(n=n, wr=round(wins / n, 4), roi_pct=round(100 * p.mean(), 2),
                avg_odds=round(float(np.mean(odds_used)), 3),
                p_boot_roi_le_0=float((boot <= 0).mean()),
                n_bets_tainted_by_clone_sibling=tainted)

def run(rows, label, siblings=None):
    rows = build_rpts5(rows)
    sub8 = [r for r in rows if r["comp"] == "InstantLeague-8035"
            and r["feats"]["h_rpts5"] is not None and r["feats"]["a_rpts5"] is not None]
    cut = int(len(sub8) * 0.7)
    train, test = sub8[:cut], sub8[cut:]
    newl = [r for r in rows if r["comp"] in NEW
            and r["feats"]["h_rpts5"] is not None and r["feats"]["a_rpts5"] is not None]
    dtr = np.array([abs(r["feats"]["h_rpts5"] - r["feats"]["a_rpts5"]) for r in train])
    thr = float(np.quantile(dtr, 0.8))
    out = dict(mode=label, threshold=round(thr, 4), n_8035=len(sub8), n_new=len(newl),
               train70=eval_rule(train, thr, siblings),
               test30=eval_rule(test, thr, siblings),
               pooled_newleagues=eval_rule(newl, thr, siblings))
    return out

def main():
    base = load_rows()
    drop30, _ = None, None
    clones_drop, siblings = dedup_clones(base)
    drop30 = dedup_30m(base)
    print(f"rows={len(base)}  drop_30m={len(drop30)}  drop_clones_strict={len(clones_drop)}  "
          f"clones_aussi_dans_30m={len(clones_drop & drop30)}")
    results = []
    for label, keep in (("A_no_dedup", base),
                        ("B_dedup_30m", [r for r in base if r["id"] not in drop30]),
                        ("C_dedup_clones_only", [r for r in base if r["id"] not in clones_drop])):
        # deep-ish copy: build_rpts5 mute les dicts -> recharge propre
        rows = [dict(r) for r in keep]
        results.append(run(rows, label, siblings if label == "A_no_dedup" else None))
    with open("exports/wf4_dupverify3.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print(json.dumps(results, indent=1))

if __name__ == "__main__":
    main()
