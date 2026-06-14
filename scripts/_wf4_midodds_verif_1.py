# -*- coding: utf-8 -*-
"""WF4 calibration 1X2 mid-odds — passe 2 (segments de saison + priors pre-enregistres).

Ce script complete _wf4_calib1x2_1.py:
A) Re-test PRE-ENREGISTRE des claims STRATEGY_REPORT.md (DS/MS_mid x bucket x position)
   en walk-forward strict sur 8035 (decision sur TEST) + replication nouvelles ligues.
B) Re-test des 3 cellules triple-positives de la passe 1 (A[4.0,4.25), A[5.5,5.75), H[5.0,5.25))
   avec p combinee test+new (z-test exact sur profits sous H0 q_i=1/cote_i).
C) Scan grid position x bucket(0.5) x segment sur 8035-train -> candidats -> test -> new.
D) Table de calibration descriptive pooled-9 (livrable utilisateur).

Methodo: cote d'ouverture = MIN(o.id); WF 70/30 par expected_start sur 8035;
ROI = mise 1u a la cote offerte; exclusion corrupted_events.json + garde-fou
ht<=ft + len(goals_json)==score_a+score_b si present.

Sortie: exports/wf4_calib1x2_seg.json
"""
import sys, json, os, math
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from scipy.stats import norm, binomtest
from scraper.config import load_settings

OUT = "exports/wf4_midodds_verify_rerun.json"
L8035 = "InstantLeague-8035"
CHAMP_NEW = ["InstantLeague-8036", "InstantLeague-8037", "InstantLeague-8042",
             "InstantLeague-8043", "InstantLeague-8044"]
CUPS = ["InstantLeague-8056", "InstantLeague-8060", "InstantLeague-8065"]
NEW_LEAGUES = CHAMP_NEW + CUPS
MAX_ROUND = {"InstantLeague-8035": 38, "InstantLeague-8036": 38, "InstantLeague-8037": 38,
             "InstantLeague-8042": 34, "InstantLeague-8043": 34, "InstantLeague-8044": 34,
             "InstantLeague-8056": 70, "InstantLeague-8060": 46, "InstantLeague-8065": 94}

eng = create_engine(load_settings().db_url)
with eng.connect() as c:
    df = pd.read_sql(text("""
        SELECT e.id AS event_id, e.competition, e.expected_start, e.round_info,
               r.score_a, r.score_b, r.ht_score_a, r.ht_score_b, r.goals_json,
               o.odds_home, o.odds_draw, o.odds_away
        FROM events e
        JOIN results r ON r.event_id = e.id
        JOIN odds_snapshots o ON o.id = (
            SELECT MIN(o2.id) FROM odds_snapshots o2 WHERE o2.event_id = e.id)
    """), c)
print(f"raw rows: {len(df)}")

with open("exports/corrupted_events.json", encoding="utf-8") as f:
    corr = json.load(f)
corrupted_ids = set(int(k) for k in corr["events"].keys())
df = df[~df["event_id"].isin(corrupted_ids)].copy()

def goals_ok(row):
    if row.ht_score_a is not None and row.ht_score_b is not None:
        if row.ht_score_a > row.score_a or row.ht_score_b > row.score_b:
            return False
    gj = row.goals_json
    if gj:
        try:
            g = json.loads(gj)
            if isinstance(g, list) and len(g) > 0 and len(g) != row.score_a + row.score_b:
                return False
        except Exception:
            pass
    return True

mask = df.apply(goals_ok, axis=1)
print(f"guard removed: {(~mask).sum()}")
df = df[mask].copy()
df = df.dropna(subset=["odds_home", "odds_draw", "odds_away", "score_a", "score_b"])
df["outcome"] = np.where(df.score_a > df.score_b, "H",
                np.where(df.score_a < df.score_b, "A", "D"))
df["expected_start"] = pd.to_datetime(df["expected_start"])
df["round"] = pd.to_numeric(df["round_info"], errors="coerce").fillna(-1).astype(int)
df["max_round"] = df["competition"].map(MAX_ROUND)
df["rfrac"] = np.where(df["round"] >= 1, df["round"] / df["max_round"], np.nan)

# segments calques sur STRATEGY_REPORT (base 38 J) puis exprimes en fraction de saison
# DS=J1-3 (frac<=3/38=0.0789) ; MS_early=J4-12 ; MS_mid=J13-25 (0.329-0.658) ; late=J26+
def seg_of(rfrac):
    if np.isnan(rfrac):
        return "J0"
    if rfrac <= 3.0 / 38:
        return "DS"
    if rfrac <= 12.0 / 38:
        return "MS_early"
    if rfrac <= 25.0 / 38:
        return "MS_mid"
    return "late"

df["seg"] = df["rfrac"].apply(seg_of)
print(df.groupby("competition").size())
print(df[df.competition == L8035].groupby("seg").size())

POSCOL = {"H": "odds_home", "D": "odds_draw", "A": "odds_away"}

def eval_bets(sub, pos):
    """Mise 1u sur pos a la cote d'ouverture. H0: q_i = 1/cote_i (ROI espere = 0).
    z-test exact sur la somme des profits (gere les buckets a cotes heterogenes)."""
    n = len(sub)
    if n == 0:
        return dict(n=0)
    odds = sub[POSCOL[pos]].values.astype(float)
    win = (sub["outcome"] == pos).values
    k = int(win.sum())
    profit = np.where(win, odds - 1.0, -1.0)
    tot = float(profit.sum())
    q0 = 1.0 / odds
    var0 = q0 * (odds - 1.0) ** 2 + (1.0 - q0)  # E0[X^2], E0[X]=0
    sd = math.sqrt(float(var0.sum()))
    z = tot / sd if sd > 0 else 0.0
    pv = float(2 * (1 - norm.cdf(abs(z))))  # bilateral
    p0 = float(np.mean(q0))
    return dict(n=n, wins=k, wr=round(k / n, 4), roi_pct=round(tot / n * 100, 2),
                avg_odds=round(float(odds.mean()), 3), p0=round(p0, 4),
                z=round(z, 3), pvalue=round(pv, 6))

def pick(dframe, pos, lo, hi, seg=None, comp=None, comps=None):
    s = dframe
    if comp:
        s = s[s.competition == comp]
    if comps:
        s = s[s.competition.isin(comps)]
    if seg:
        s = s[s.seg == seg]
    col = POSCOL[pos]
    return s[(s[col] >= lo) & (s[col] < hi)]

# ---------- WF split 8035 ----------
d35 = df[df.competition == L8035].sort_values("expected_start").reset_index(drop=True)
cut = int(len(d35) * 0.70)
train, test = d35.iloc[:cut].copy(), d35.iloc[cut:].copy()
dnew = df[df.competition.isin(NEW_LEAGUES)].copy()
dchamp = df[df.competition.isin(CHAMP_NEW)].copy()
dcups = df[df.competition.isin(CUPS)].copy()
print(f"\n8035 clean={len(d35)} train={len(train)} test={len(test)} | new={len(dnew)} (champ={len(dchamp)} cups={len(dcups)})")

n_tests_scanned = 0
out = {"meta": dict(n_8035=len(d35), n_train=len(train), n_test=len(test),
                    n_new=len(dnew), n_champ=len(dchamp), n_cups=len(dcups),
                    corrupted_excluded=len(corrupted_ids), guard_removed=int((~mask).sum()),
                    train_end=str(train.expected_start.max()), test_start=str(test.expected_start.min()))}

def full_eval(name, pos, lo, hi, seg):
    """train / test / new(seg via rfrac) / champ / cups / per-league signs / combined test+new."""
    global n_tests_scanned
    n_tests_scanned += 1
    tr = eval_bets(pick(train, pos, lo, hi, seg=seg), pos)
    te = eval_bets(pick(test, pos, lo, hi, seg=seg), pos)
    nw = eval_bets(pick(dnew, pos, lo, hi, seg=seg), pos)
    ch = eval_bets(pick(dchamp, pos, lo, hi, seg=seg), pos)
    cu = eval_bets(pick(dcups, pos, lo, hi, seg=seg), pos)
    comb_df = pd.concat([pick(test, pos, lo, hi, seg=seg), pick(dnew, pos, lo, hi, seg=seg)])
    comb = eval_bets(comb_df, pos)
    signs = []
    for lg in NEW_LEAGUES:
        s = eval_bets(pick(dnew, pos, lo, hi, seg=seg, comp=lg), pos)
        if s.get("n", 0) >= 5:
            signs.append((lg.split("-")[1], s["n"], s["roi_pct"]))
    pos_leagues = sum(1 for _, _, r in signs if r > 0)
    return dict(name=name, pos=pos, lo=lo, hi=hi, seg=seg,
                train=tr, test=te, new_pooled=nw, champ=ch, cups=cu,
                combined_test_plus_new=comb,
                per_league=[dict(lg=a, n=b, roi=c) for a, b, c in signs],
                leagues_positive=f"{pos_leagues}/{len(signs)}")

# ========== A) PRIORS PRE-ENREGISTRES (STRATEGY_REPORT.md) ==========
PRIORS = [
    ("P1_MSmid_H_5plus",  "H", 5.0, 15.0, "MS_mid"),
    ("P2_MSmid_A_5plus",  "A", 5.0, 15.0, "MS_mid"),
    ("P3_DS_H_2.2-2.7",   "H", 2.2, 2.7,  "DS"),
    ("P4_DS_A_1.5-1.8",   "A", 1.5, 1.8,  "DS"),
    ("P5_DS_A_3.5-5.0",   "A", 3.5, 5.0,  "DS"),
]
print("\n===== A) PRIORS =====")
out["priors"] = []
for name, pos, lo, hi, seg in PRIORS:
    r = full_eval(name, pos, lo, hi, seg)
    out["priors"].append(r)
    print(f"{name}: train n={r['train'].get('n')} roi={r['train'].get('roi_pct')} | "
          f"TEST n={r['test'].get('n')} roi={r['test'].get('roi_pct')} p={r['test'].get('pvalue')} | "
          f"NEW n={r['new_pooled'].get('n')} roi={r['new_pooled'].get('roi_pct')} p={r['new_pooled'].get('pvalue')} | "
          f"comb n={r['combined_test_plus_new'].get('n')} roi={r['combined_test_plus_new'].get('roi_pct')} "
          f"p={r['combined_test_plus_new'].get('pvalue')} | lg+ {r['leagues_positive']}")

# ========== B) cellules triple-positives passe 1 (toutes saisons) ==========
print("\n===== B) CELLULES PASSE 1 =====")
CELLS = [("B1_A_4.00-4.25", "A", 4.00, 4.25, None),
         ("B2_A_5.50-5.75", "A", 5.50, 5.75, None),
         ("B3_H_5.00-5.25", "H", 5.00, 5.25, None),
         # zones agregees (pre-enregistrees ici, motivees par la passe 1)
         ("B4_zone_A_4-6",  "A", 4.0, 6.0, None),
         ("B5_zone_H_4-6",  "H", 4.0, 6.0, None),
         ("B6_zone_H_5-6",  "H", 5.0, 6.0, None)]
out["cells_pass1"] = []
for name, pos, lo, hi, seg in CELLS:
    r = full_eval(name, pos, lo, hi, seg)
    out["cells_pass1"].append(r)
    print(f"{name}: train n={r['train'].get('n')} roi={r['train'].get('roi_pct')} | "
          f"TEST n={r['test'].get('n')} roi={r['test'].get('roi_pct')} p={r['test'].get('pvalue')} | "
          f"NEW n={r['new_pooled'].get('n')} roi={r['new_pooled'].get('roi_pct')} p={r['new_pooled'].get('pvalue')} | "
          f"comb n={r['combined_test_plus_new'].get('n')} roi={r['combined_test_plus_new'].get('roi_pct')} "
          f"p={r['combined_test_plus_new'].get('pvalue')} | lg+ {r['leagues_positive']}")

# ========== C) GRID position x bucket(0.5) x segment, selection sur train ==========
print("\n===== C) GRID SEGMENTS (train -> candidats) =====")
EDGES = [1.5 + 0.5 * i for i in range(10)]  # 1.5..6.0
SEGS = ["DS", "MS_early", "MS_mid", "late", "J0"]
grid_rows = []
for pos in POSCOL:
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        for seg in SEGS:
            st = eval_bets(pick(train, pos, lo, hi, seg=seg), pos)
            n_tests_scanned += 1
            st.update(pos=pos, lo=lo, hi=hi, seg=seg)
            grid_rows.append(st)
out["grid_train"] = grid_rows
cands = [r for r in grid_rows if r.get("n", 0) >= 60 and r.get("roi_pct", -99) >= 5.0]
print(f"grid cells: {len(grid_rows)}, candidats train (n>=60, roi>=5%): {len(cands)}")
out["grid_candidates"] = []
for r in cands:
    fr = full_eval(f"G_{r['pos']}_{r['lo']}-{r['hi']}_{r['seg']}", r["pos"], r["lo"], r["hi"], r["seg"])
    n_tests_scanned -= 1  # deja compte dans la grille
    out["grid_candidates"].append(fr)
    print(f"  {fr['name']}: train roi={r['roi_pct']} (n={r['n']}) | TEST n={fr['test'].get('n')} "
          f"roi={fr['test'].get('roi_pct')} | NEW n={fr['new_pooled'].get('n')} roi={fr['new_pooled'].get('roi_pct')} "
          f"| comb p={fr['combined_test_plus_new'].get('pvalue')} | lg+ {fr['leagues_positive']}")

# ========== D) table calibration pooled-9 descriptive ==========
calib = []
EDGES_D = [round(1.8 + 0.2 * i, 1) for i in range(17)]  # 1.8..5.0
for pos in POSCOL:
    for lo, hi in zip(EDGES_D[:-1], EDGES_D[1:]):
        st = eval_bets(pick(df, pos, lo, hi), pos)
        st.update(pos=pos, lo=lo, hi=hi, scope="pooled-9-descriptif")
        calib.append(st)
out["calib_pooled9"] = calib

out["meta"]["n_tests_scanned"] = n_tests_scanned
print(f"\nn_tests_scanned (priors+cells+grid): {n_tests_scanned}")
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(f"written {OUT}")
