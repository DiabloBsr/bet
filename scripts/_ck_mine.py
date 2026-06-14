"""Track B - minage de cles combinees sur exports/combokeys_features.parquet.

Pour chaque combinaison 1/2/3-cles (signaux binnes), mesure la distribution
conditionnelle du TOTAL de buts et du SCORE exact, le lift vs base, et surtout
l'EV vs la COTE OFFERTE. Protocole anti-overfit : split forward chrono 70/30
(decouverte TRAIN, decision TEST), seuils n, Bonferroni + BH FDR, bootstrap CI.
H1/H2 pre-enregistres testes a part.

Sorties : exports/combokeys_report.md + data/combokeys_registry.json
Usage: ./.venv/Scripts/python.exe scripts/_ck_mine.py
"""
import sys
sys.path.insert(0, ".")
import json
import itertools
import numpy as np
import pandas as pd
from scipy.stats import norm

ART = "exports/combokeys_features.parquet"
try:
    df = pd.read_parquet(ART)
except Exception:
    df = pd.read_csv("exports/combokeys_features.csv")
df["expected_start"] = pd.to_datetime(df.expected_start, utc=True, errors="coerce")
df = df.dropna(subset=["expected_start"]).sort_values("expected_start").reset_index(drop=True)
print(f"n={len(df)}  leagues={df.competition.nunique()}")

BINSPEC = json.load(open("exports/combokeys_binspec.json", encoding="utf-8"))
SIGNALS = ["fav", "dog", "odds_ratio", "od", "p_total_eq3", "p_btts",
           "lam_tot", "lam_diff", "residual", "dc_x2_cote"]
N_MIN_TR_TOT, N_MIN_TE_TOT = 200, 200
N_MIN_TR_SC, N_MIN_TE_SC = 400, 400
SCORES = ["1-1", "2-1", "1-2", "1-0", "0-1", "2-0", "0-2", "0-0", "2-2",
          "3-0", "0-3", "3-1", "1-3", "3-2", "2-3"]


def binned(col):
    edges = BINSPEC[col]
    return pd.cut(df[col], edges)


for s in SIGNALS:
    df[f"b_{s}"] = binned(s)

# split forward chrono 70/30 par ligue
df["is_test"] = False
for comp, g in df.groupby("competition"):
    k = int(len(g) * 0.70)
    df.loc[g.index[k:], "is_test"] = True
tr = df[~df.is_test]; te = df[df.is_test]
print(f"train={len(tr)} test={len(te)}")

# base rates globales (sur train)
base_total = tr.total_goals.value_counts(normalize=True).to_dict()
base_score = tr.exact_score.value_counts(normalize=True).to_dict()


def ev_and_ci(profit):
    """EV moyen + borne basse bootstrap 5% (par event)."""
    profit = np.asarray(profit, float)
    profit = profit[~np.isnan(profit)]
    if len(profit) < 20:
        return np.nan, np.nan, np.nan, 0
    ev = profit.mean()
    # z vs 0
    se = profit.std(ddof=1) / np.sqrt(len(profit))
    z = ev / se if se > 0 else 0.0
    # bootstrap CI lower (percentile, sans Random : sous-echantillons deterministes)
    n = len(profit)
    idx = np.arange(n)
    boots = []
    rng = np.random.default_rng(12345)  # graine fixe = reproductible
    for _ in range(2000):
        b = profit[rng.integers(0, n, n)]
        boots.append(b.mean())
    lo = float(np.percentile(boots, 5))
    return float(ev), lo, float(z), len(profit)


def eval_total(cell_tr, cell_te):
    """Predit le total modal du train ; renvoie metrics test."""
    if len(cell_tr) < N_MIN_TR_TOT or len(cell_te) < N_MIN_TE_TOT:
        return None
    T = int(cell_tr.total_goals.mode().iloc[0])
    cote = cell_te[f"off_t{T}"]
    hit = (cell_te.total_goals == T).astype(float)
    profit = cote.values * hit.values - 1.0
    profit = np.where(np.isnan(cote.values), np.nan, profit)
    hr = float(hit[cote.notna().values].mean()) if cote.notna().any() else np.nan
    ev, lo, z, n = ev_and_ci(profit)
    base = base_total.get(T, 1e-9)
    return dict(predict=f"Total={T}", n_tr=len(cell_tr), n_te=int(cote.notna().sum()),
                hit=round(100 * hr, 1), base=round(100 * base, 1),
                lift=round(hr / base, 2) if base else np.nan,
                ev=round(100 * ev, 1) if not np.isnan(ev) else np.nan,
                ev_lo=round(100 * lo, 1) if not np.isnan(lo) else np.nan,
                z=round(z, 2), cote=round(float(cell_te[f"off_t{T}"].median()), 2))


def eval_score(cell_tr, cell_te):
    if len(cell_tr) < N_MIN_TR_SC or len(cell_te) < N_MIN_TE_SC:
        return None
    vc = cell_tr.exact_score.value_counts()
    vc = vc[vc.index.isin(SCORES)]
    if vc.empty:
        return None
    S = vc.index[0]
    col = f"off_s_{S}"
    if col not in cell_te:
        return None
    cote = cell_te[col]
    hit = (cell_te.exact_score == S).astype(float)
    profit = np.where(np.isnan(cote.values), np.nan, cote.values * hit.values - 1.0)
    hr = float(hit[cote.notna().values].mean()) if cote.notna().any() else np.nan
    ev, lo, z, n = ev_and_ci(profit)
    base = base_score.get(S, 1e-9)
    return dict(predict=f"Score={S}", n_tr=len(cell_tr), n_te=int(cote.notna().sum()),
                hit=round(100 * hr, 1), base=round(100 * base, 1),
                lift=round(hr / base, 2) if base else np.nan,
                ev=round(100 * ev, 1) if not np.isnan(ev) else np.nan,
                ev_lo=round(100 * lo, 1) if not np.isnan(lo) else np.nan,
                z=round(z, 2), cote=round(float(cote.median()), 2) if cote.notna().any() else np.nan)


# ---- balayage systematique 1/2/3-cles ----
results = []
M = 0
for k in (1, 2, 3):
    for combo in itertools.combinations(SIGNALS, k):
        bcols = [f"b_{s}" for s in combo]
        gtr = tr.groupby(bcols, observed=True)
        gte_groups = {key: g for key, g in te.groupby(bcols, observed=True)}
        for key, cell_tr in gtr:
            cell_te = gte_groups.get(key)
            if cell_te is None:
                continue
            kv_tuple = key if isinstance(key, tuple) else (key,)
            keystr = " & ".join(f"{s}∈{kv}" for s, kv in zip(combo, kv_tuple))
            # definition structuree (signal -> [left, right]) pour le lookup live
            definition = {}
            for s, iv in zip(combo, kv_tuple):
                try:
                    definition[s] = [float(iv.left), float(iv.right)]
                except Exception:
                    definition[s] = None
            for fn, kind in ((eval_total, "TOTAL"), (eval_score, "SCORE")):
                r = fn(cell_tr, cell_te)
                if r is None:
                    continue
                M += 1
                r.update(kind=kind, key=keystr, ncombo=k, definition=definition)
                results.append(r)

print(f"\nM = {M} tests evalues")
res = pd.DataFrame(results)
if len(res):
    # Bonferroni z* et BH FDR sur p-values (test bilateral du z d'EV)
    alpha = 0.05
    zstar = norm.ppf(1 - alpha / (2 * max(M, 1)))
    res["p"] = 2 * (1 - norm.cdf(res.z.abs()))
    res["bonf"] = res.z.abs() >= zstar
    # BH
    res_sorted = res.sort_values("p").reset_index()
    res_sorted["bh_thr"] = 0.10 * (np.arange(1, len(res_sorted) + 1)) / len(res_sorted)
    res_sorted["bh_pass"] = res_sorted.p <= res_sorted.bh_thr
    # propagate bh_pass back
    passed_idx = set(res_sorted[res_sorted.bh_pass]["index"])
    res["bh"] = res.index.isin(passed_idx)
    print(f"z* Bonferroni = {zstar:.2f}")

    def status(r):
        if r.ev_lo is not None and not np.isnan(r.ev_lo) and r.ev_lo > 0 and r.bonf:
            return "CONFIRMED"
        if (r.ev is not None and not np.isnan(r.ev) and r.ev > 0) and (r.bh or r.bonf):
            return "WATCH"
        return "REJECTED"
    res["status"] = res.apply(status, axis=1)

    conf = res[res.status != "REJECTED"].sort_values(["ev_lo", "ev"], ascending=False)
    print("\n=== CLES non-rejetees (EV>0 & significatif) ===")
    cols = ["kind", "predict", "key", "n_tr", "n_te", "hit", "base", "lift", "cote", "ev", "ev_lo", "z", "status"]
    if len(conf):
        print(conf[cols].head(30).to_string(index=False))
    else:
        print("AUCUNE cle ne bat la cote offerte apres correction (attendu : marge mange l'edge).")
else:
    res = pd.DataFrame()
    print("aucun cell au-dessus des seuils n")


# ---- H1 / H2 pre-enregistres ----
def cell(mask_tr, mask_te):
    return tr[mask_tr], te[mask_te]

print("\n=== H1 (equilibre: fav∈[2.10,2.40) & dog∈[2.70,+) & od∈[3.6,4.8)) ===")
h1_tr = (tr.fav.between(2.10, 2.40, inclusive="left") & (tr.dog >= 2.70) & tr.od.between(3.6, 4.8, inclusive="left"))
h1_te = (te.fav.between(2.10, 2.40, inclusive="left") & (te.dog >= 2.70) & te.od.between(3.6, 4.8, inclusive="left"))
c_tr, c_te = tr[h1_tr], te[h1_te]
print(f"n_tr={len(c_tr)} n_te={len(c_te)}")
if len(c_te) > 30:
    p3 = (c_te.total_goals == 3).mean()
    p21_12 = c_te.exact_score.isin(["2-1", "1-2"]).mean()
    print(f"  P(total=3)={100*p3:.1f}% (base {100*base_total.get(3,0):.1f}%)  "
          f"P(score∈2-1/1-2)={100*p21_12:.1f}% (base {100*(base_score.get('2-1',0)+base_score.get('1-2',0)):.1f}%)")
    rT = eval_total(c_tr, c_te)
    if rT: print(f"  EV Total=3 : {rT}")

print("\n=== H2 (2.40/2.50 + X2~1.32: fav∈[2.40,2.70) & odds_ratio<1.15 & dc_x2_cote∈[1.25,1.35)) ===")
h2_tr = (tr.fav.between(2.40, 2.70, inclusive="left") & (tr.odds_ratio < 1.15) & tr.dc_x2_cote.between(1.25, 1.35, inclusive="left"))
h2_te = (te.fav.between(2.40, 2.70, inclusive="left") & (te.odds_ratio < 1.15) & te.dc_x2_cote.between(1.25, 1.35, inclusive="left"))
c_tr, c_te = tr[h2_tr], te[h2_te]
print(f"n_tr={len(c_tr)} n_te={len(c_te)}")
if len(c_te) > 30:
    p2 = (c_te.total_goals == 2).mean()
    print(f"  P(total=2)={100*p2:.1f}% (base {100*base_total.get(2,0):.1f}%)")

# ---- ecriture rapport + registre ----
lines = ["# Combined-Key Mining Report", "",
         f"- n events: {len(df)}  | leagues: {df.competition.nunique()}  | train {len(tr)} / test {len(te)}",
         f"- M tests: {M}  | Bonferroni z* = {zstar:.2f}" if len(res) else "- aucun test",
         f"- plafond score exact recalcule (corrige): Top1 11.6% / Top3 30.0%", "",
         "## Cles non-rejetees", ""]
if len(res) and len(conf):
    try:
        lines.append(conf[cols].head(40).to_markdown(index=False))
    except Exception:
        lines.append("```\n" + conf[cols].head(40).to_string(index=False) + "\n```")
else:
    lines.append("AUCUNE cle ne bat la cote offerte apres correction multiple "
                 "(coherent avec ENGINE_MODEL : grille Poisson pure + marge 12-24%).")
with open("exports/combokeys_report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\necrit exports/combokeys_report.md")

registry = {"frozen_at": str(df.expected_start.max()), "binspec_ref": "exports/combokeys_binspec.json", "rules": {}}
if len(res):
    for _, r in res[res.status != "REJECTED"].iterrows():
        registry["rules"][f"{r.kind}:{r.key}"] = dict(
            predict=r.predict, definition=r.definition, kind=r.kind,
            expected_cote=r.cote, expected_hit_rate=r.hit / 100,
            lift=r.lift, ev=r.ev, ev_lo=r.ev_lo, n_tr=int(r.n_tr), n_te=int(r.n_te),
            z=r.z, status=r.status)
with open("data/combokeys_registry.json", "w", encoding="utf-8") as f:
    json.dump(registry, f, indent=1, default=str)
print(f"ecrit data/combokeys_registry.json ({len(registry['rules'])} regles)")
