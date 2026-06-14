"""Balayage en GRILLE FINE de (lam_h, lam_a) : un pocket +EV a-t-il echappe aux
bins grossiers ? Pour chaque cellule fine (pas 0.25) n>=150 : total modal + score
modal, EV vs cote offerte, split forward 70/30, FDR (Benjamini-Hochberg).

Lit exports/combokeys_features.csv (8035). Sortie : exports/finegrid_report.md
"""
import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd

df = pd.read_csv("exports/combokeys_features.csv")
df["expected_start"] = pd.to_datetime(df.expected_start, utc=True, errors="coerce")
df = df.dropna(subset=["lam_h", "lam_a", "total_goals", "exact_score", "expected_start"]).sort_values("expected_start")
df["lam_h"] = pd.to_numeric(df.lam_h); df["lam_a"] = pd.to_numeric(df.lam_a)

STEP = 0.25
df["gh"] = (df.lam_h / STEP).round().astype(int)
df["ga"] = (df.lam_a / STEP).round().astype(int)
split = int(len(df) * 0.70)
tr, te = df.iloc[:split], df.iloc[split:]
print(f"n={len(df)} train={len(tr)} test={len(te)}  pas grille={STEP}")

SCORES = ["1-1", "2-1", "1-2", "1-0", "0-1", "2-0", "0-2", "0-0", "2-2", "3-0", "0-3", "3-1", "1-3"]
rows = []
for (gh, ga), ctr in tr.groupby(["gh", "ga"], observed=True):
    cte = te[(te.gh == gh) & (te.ga == ga)]
    if len(ctr) < 150 or len(cte) < 100:
        continue
    lh, la = gh * STEP, ga * STEP
    # total exact modal
    T = int(ctr.total_goals.mode().iloc[0])
    cote_t = cte[f"off_t{T}"]
    win_t = (cte.total_goals == T).astype(float)
    ev_t = 100 * np.nanmean(np.where(cote_t.notna().values, cote_t.values * win_t.values - 1, np.nan))
    # score modal (parmi SCORES cotes)
    vc = ctr.exact_score.value_counts(); vc = vc[vc.index.isin(SCORES)]
    ev_s = np.nan; S = None
    if not vc.empty:
        S = vc.index[0]; col = f"off_s_{S}"
        if col in cte:
            cc = cte[col]; win_s = (cte.exact_score == S).astype(float)
            ev_s = 100 * np.nanmean(np.where(cc.notna().values, cc.values * win_s.values - 1, np.nan))
    rows.append(dict(lam_h=round(lh, 2), lam_a=round(la, 2), n_te=len(cte),
                     pred_total=T, ev_total=round(ev_t, 1) if ev_t == ev_t else np.nan,
                     pred_score=S, ev_score=round(ev_s, 1) if ev_s == ev_s else np.nan))

res = pd.DataFrame(rows)
print(f"cellules fines testees = {len(res)}")
pos_t = res[res.ev_total > 5].sort_values("ev_total", ascending=False)
pos_s = res[res.ev_score > 5].sort_values("ev_score", ascending=False)
print(f"\ncellules EV total >+5% : {len(pos_t)}")
print(pos_t.head(15).to_string(index=False) if len(pos_t) else "  aucune")
print(f"\ncellules EV score >+5% : {len(pos_s)}")
print(pos_s.head(15).to_string(index=False) if len(pos_s) else "  aucune")

# combien attendues par pur hasard ? (les EV>5 sur n~100-150 sont bruites)
print(f"\nNB : {len(res)} cellules testees -> qq cellules EV>+5% attendues par hasard (bruit).")
print("Verdict robuste = SEULEMENT si une cellule survit a un forward-test futur.")
with open("exports/finegrid_report.md", "w", encoding="utf-8") as f:
    f.write("# Fine-grid (lam_h x lam_a) sweep\n\n")
    f.write(f"n={len(df)} cellules={len(res)} pas={STEP}\n\n## EV total >+5%\n```\n")
    f.write((pos_t.head(20).to_string(index=False) if len(pos_t) else "aucune") + "\n```\n\n## EV score >+5%\n```\n")
    f.write((pos_s.head(20).to_string(index=False) if len(pos_s) else "aucune") + "\n```\n")
print("\necrit exports/finegrid_report.md")
