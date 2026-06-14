# -*- coding: utf-8 -*-
"""
WF4 ADVERSARIAL - refutation du finding NULL "sur/sous-regime recent ne predit pas".
Strategie: reutiliser EXACTEMENT le pipeline de scripts/_wf4_seq_1.py (import module)
puis attaquer:
  A. audit leakage independant (recalcul des features avec timestamps stricts)
  B. audit doublons residuels (paires same-teams 30min-6h, scores identiques ?)
  C. splits walk-forward alternatifs (50/50, 60/40, 80/20) sur 8035
  D. sous-periodes du test30 original (le +13% est-il une demi-periode chanceuse ?)
  E. replication championnats-seuls (8036/37/42/43/44) vs coupes-seules (8056/60/65)
  F. ROI OOS combine conservateur (test30 8035 + pooled-newleagues) + bootstrap
  G. coherence de signe des coefs "significatifs" 8035
Sortie: exports/wf4_seq_adv1.json. LECTURE SEULE DB.
"""
import sys, json, math, importlib.util
sys.path.insert(0, ".")
import numpy as np
from datetime import datetime
from scipy import stats

spec = importlib.util.spec_from_file_location("wf4seq", "scripts/_wf4_seq_1.py")
wf4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wf4)

RNG = np.random.default_rng(123)
OUT = {}

CHAMP_NEW = ["InstantLeague-8036", "InstantLeague-8037", "InstantLeague-8042",
             "InstantLeague-8043", "InstantLeague-8044"]
CUPS = ["InstantLeague-8056", "InstantLeague-8060", "InstantLeague-8065"]

rows = wf4.load_data()
rows = wf4.build(rows)


def ts(s):
    return datetime.fromisoformat(s).timestamp()

# ---------- A. audit leakage independant ----------
# recalcul rpts5 home pour un echantillon aleatoire, en n'utilisant QUE les matchs
# du meme (comp, team) avec timestamp STRICTEMENT < timestamp courant.
hist_idx = {}
for i, r in enumerate(rows):
    hist_idx.setdefault((r["comp"], r["ta"]), []).append(i)
    hist_idx.setdefault((r["comp"], r["tb"]), []).append(i)

mismatch, ties_same_ts, checked = 0, 0, 0
sample = RNG.choice(len(rows), size=min(3000, len(rows)), replace=False)
for i in sample:
    r = rows[i]
    if r["feats"].get("h_rpts5") is None:
        continue
    key = (r["comp"], r["ta"])
    prior = []
    for j in hist_idx[key]:
        rj = rows[j]
        if ts(rj["start"]) > ts(r["start"]) or (ts(rj["start"]) == ts(r["start"]) and rj["id"] >= r["id"]):
            continue
        if rj["ta"] == r["ta"]:
            pw, pdr, gf, ga = rj["ph"], rj["pd"], rj["sa"], rj["sb"]
        else:
            pw, pdr, gf, ga = rj["pa"], rj["pd"], rj["sb"], rj["sa"]
        pts = 3.0 if gf > ga else (1.0 if gf == ga else 0.0)
        prior.append((ts(rj["start"]), rj["id"], pts - (3 * pw + pdr)))
        if ts(rj["start"]) == ts(r["start"]):
            ties_same_ts += 1
    prior.sort()
    if len(prior) < 5:
        continue
    indep = float(np.mean([x[2] for x in prior[-5:]]))
    checked += 1
    if abs(indep - r["feats"]["h_rpts5"]) > 1e-9:
        mismatch += 1
OUT["A_leakage_audit"] = dict(checked=checked, mismatch=mismatch, ties_same_timestamp=ties_same_ts)

# ---------- B. doublons residuels apres dedup 30min ----------
bykey = {}
for r in rows:
    bykey.setdefault((r["comp"], r["ta"], r["tb"]), []).append(r)
n_pairs, n_same_score = 0, 0
for key, lst in bykey.items():
    lst.sort(key=lambda r: ts(r["start"]))
    for i in range(1, len(lst)):
        dt = ts(lst[i]["start"]) - ts(lst[i - 1]["start"])
        if 1800 <= dt <= 6 * 3600:
            n_pairs += 1
            if lst[i]["sa"] == lst[i - 1]["sa"] and lst[i]["sb"] == lst[i - 1]["sb"]:
                n_same_score += 1
OUT["B_residual_dup_audit"] = dict(pairs_30min_6h=n_pairs, same_score=n_same_score,
                                   same_score_rate=(n_same_score / n_pairs) if n_pairs else None)

# ---------- helpers ----------
def eval_rule(pop, sel, thr):
    pnls, odds_used, wins = [], [], 0
    for r in pop:
        d = r["feats"]["h_rpts5"] - r["feats"]["a_rpts5"]
        side = None
        if sel * d >= thr:
            side = "h"
        elif sel * (-d) >= thr:
            side = "a"
        if side is None:
            continue
        o = r["oh"] if side == "h" else r["oa"]
        won = (r["sa"] > r["sb"]) if side == "h" else (r["sb"] > r["sa"])
        odds_used.append(o)
        pnls.append((o - 1) if won else -1.0)
        wins += int(won)
    n = len(pnls)
    if n == 0:
        return dict(n=0)
    pnls = np.array(pnls)
    roi = float(pnls.mean())
    se = float(pnls.std(ddof=1) / math.sqrt(n))
    boot = RNG.choice(pnls, size=(4000, n), replace=True).mean(axis=1)
    return dict(n=n, wr=wins / n, roi_pct=100 * roi, se_roi_pct=100 * se,
                p_boot_roi_le_0=float((boot <= 0).mean()),
                avg_odds=float(np.mean(odds_used)), pnls=pnls)

def feats_ok(r):
    return r["feats"].get("h_rpts5") is not None and r["feats"].get("a_rpts5") is not None

sub8 = [r for r in rows if r["comp"] == "InstantLeague-8035" and feats_ok(r)]
sub8.sort(key=lambda r: (r["start"], r["id"]))
newl = [r for r in rows if r["comp"] in CHAMP_NEW + CUPS and feats_ok(r)]
champ = [r for r in rows if r["comp"] in CHAMP_NEW and feats_ok(r)]
cups = [r for r in rows if r["comp"] in CUPS and feats_ok(r)]

# ---------- C. splits alternatifs sur 8035 ----------
OUT["C_alt_splits_8035_back_overperf"] = {}
for frac in (0.5, 0.6, 0.7, 0.8):
    cut = int(len(sub8) * frac)
    train, test = sub8[:cut], sub8[cut:]
    dtr = np.array([r["feats"]["h_rpts5"] - r["feats"]["a_rpts5"] for r in train])
    thr = float(np.quantile(np.abs(dtr), 0.8))
    tr = eval_rule(train, 1, thr); te = eval_rule(test, 1, thr)
    tr.pop("pnls", None); te.pop("pnls", None)
    OUT["C_alt_splits_8035_back_overperf"][f"split{int(frac*100)}"] = dict(thr=thr, train=tr, test=te)

# ---------- D. sous-periodes du test30 original ----------
cut = int(len(sub8) * 0.7)
train, test = sub8[:cut], sub8[cut:]
dtr = np.array([r["feats"]["h_rpts5"] - r["feats"]["a_rpts5"] for r in train])
thr = float(np.quantile(np.abs(dtr), 0.8))
halves = {}
h = len(test) // 2
for name, pop in (("test30_first_half", test[:h]), ("test30_second_half", test[h:]),
                  ("test30_full", test)):
    e = eval_rule(pop, 1, thr); e.pop("pnls", None)
    halves[name] = e
OUT["D_test30_subperiods"] = dict(threshold=thr, **halves)

# ---------- E. replication championnats-seuls vs coupes-seules ----------
def lrt_scope(pop, fam="rpts", N=5):
    ok = [r for r in pop if r["feats"].get(f"h_{fam}{N}") is not None
          and r["feats"].get(f"a_{fam}{N}") is not None]
    y = np.array([1.0 if r["sa"] > r["sb"] else 0.0 for r in ok])
    Xb = wf4.logit(np.array([r["ph"] for r in ok]))[:, None]
    Xe = np.array([[r["feats"][f"h_{fam}{N}"], r["feats"][f"a_{fam}{N}"]] for r in ok])
    t = wf4.lrt_logistic(Xb, Xe, y)
    t["n"] = len(ok)
    return t

OUT["E_replication"] = {}
for nm, pop in (("champ_new_only", champ), ("cups_only", cups)):
    OUT["E_replication"][nm] = {}
    for fam in ("rpts", "rgd"):
        OUT["E_replication"][nm][f"{fam}5_homewin"] = lrt_scope(pop, fam)
    e1 = eval_rule(pop, 1, thr); e1.pop("pnls", None)
    OUT["E_replication"][nm]["rule_back_overperf_thr8035"] = e1

# ---------- F. ROI OOS combine conservateur ----------
e_test = eval_rule(test, 1, thr)
e_new = eval_rule(newl, 1, thr)
pnls_all = np.concatenate([e_test["pnls"], e_new["pnls"]])
n = len(pnls_all)
roi = float(pnls_all.mean())
boot = RNG.choice(pnls_all, size=(8000, n), replace=True).mean(axis=1)
OUT["F_combined_oos"] = dict(
    n=n, roi_pct=100 * roi,
    ci95=[100 * float(np.quantile(boot, 0.025)), 100 * float(np.quantile(boot, 0.975))],
    p_boot_roi_le_0=float((boot <= 0).mean()),
    parts=dict(test30=dict(n=e_test["n"], roi_pct=e_test["roi_pct"]),
               newleagues=dict(n=e_new["n"], roi_pct=e_new["roi_pct"])))

# ---------- G. coherence de signe ----------
# le "signal" 8035 (rpts5/rgd5 homewin p~0.005) a un coef POSITIF sur le residuel
# AWAY -> away en sur-regime rendrait la victoire HOME plus probable: incoherent
# avec toute histoire de forme. On le re-mesure + signe sur champ/cups.
g = {}
for nm, pop in (("8035", sub8), ("champ_new_only", champ), ("cups_only", cups)):
    t = lrt_scope(pop, "rpts", 5)
    g[nm] = dict(p=t["p"], coef_home_resid=t["coefs_extra"][0], coef_away_resid=t["coefs_extra"][1], n=t["n"])
OUT["G_sign_coherence_rpts5_homewin"] = g

with open("exports/wf4_seq_adv1.json", "w", encoding="utf-8") as f:
    json.dump(OUT, f, indent=1, ensure_ascii=False)
print(json.dumps(OUT, indent=1, ensure_ascii=False))
