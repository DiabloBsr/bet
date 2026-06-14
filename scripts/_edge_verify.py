"""Vérif adversariale : les edges retenus tiennent-ils sur 3 tranches chrono
indépendantes ? (un vrai edge réplique partout ; un artefact non.)"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
df = pd.read_csv(Path(__file__).resolve().parents[1] / "exports" / "combokeys_features.csv")
df = df.sort_values("expected_start").reset_index(drop=True)
df["abs_diff"] = df.lam_diff.abs()
n3 = len(df) // 3
folds = [df.iloc[:n3], df.iloc[n3:2*n3], df.iloc[2*n3:]]
labels = [f"F{i+1} ({f.expected_start.min()[:10]}→{f.expected_start.max()[:10]}, n={len(f)})" for i,f in enumerate(folds)]

def check(name, cond_fn, metric_fn):
    print(f"\n● {name}")
    glob = metric_fn(df)
    print(f"   GLOBAL : {glob[1]} = {glob[0]*100:.0f}%  (n={glob[2]})")
    for f, lab in zip(folds, labels):
        sub = f[cond_fn(f)]
        if len(sub) < 40: print(f"   {lab[:22]:<22} n<40"); continue
        r, desc, nn = metric_fn(sub)
        print(f"   {lab[:34]:<34} {desc} = {r*100:>4.0f}%  (n={nn})")

def under35(d): return ((d.total_goals<=3).mean(), "Under3.5", len(d))
def over25(d):  return ((d.total_goals>=3).mean(), "Over2.5", len(d))
def under25(d): return ((d.total_goals<=2).mean(), "Under2.5", len(d))
def s11(d):     return ((d.exact_score=="1-1").mean(), "score 1-1", len(d))
def top3_bal(d):return (d.exact_score.isin(["1-1","2-1","1-2"]).mean(), "Top3{1-1,2-1,1-2}", len(d))
def top3_home(d):return (d.exact_score.isin(["1-1","2-1","1-0"]).mean(), "Top3{1-1,2-1,1-0}", len(d))

print("="*100); print("VÉRIF 3-FOLD — un edge réel réplique sur les 3 tranches"); print("="*100)
print("Folds :", " | ".join(labels))

# --- TOTAL / OVER-UNDER (les plus forts) ---
check("UNDER 3.5 | match à faible total (lam_tot<2.45)",
      lambda d: d.lam_tot<2.45, under35)
check("UNDER 2.5 | très faible total (lam_tot<2.0)",
      lambda d: d.lam_tot<2.0, under25)
check("OVER 2.5 | total élevé (lam_tot>=3.13)",
      lambda d: d.lam_tot>=3.13, over25)
check("OVER 2.5 | très ouvert (lam_tot>=3.7)",
      lambda d: d.lam_tot>=3.7, over25)

# --- SCORE 1-1 (le meilleur edge de score) ---
check("SCORE 1-1 | équilibré ET faible total (|diff|<0.74 & lam_tot<2.45)",
      lambda d: (d.abs_diff<0.74)&(d.lam_tot<2.45), s11)
check("SCORE 1-1 | book pense total=2 (book_modal proxy: p_total_le2>0.46)",
      lambda d: d.p_total_le2>0.46, s11)

# --- TOP-3 combos (la vraie accuracy jouable) ---
check("TOP-3 {1-1,2-1,1-2} | équilibré (|diff|<0.74)",
      lambda d: d.abs_diff<0.74, top3_bal)
check("TOP-3 {1-1,2-1,1-0} | favori maison total moyen (diff∈[0.2,1.0] & lam_tot 2.0-2.8)",
      lambda d: (d.lam_diff>0.2)&(d.lam_diff<1.0)&(d.lam_tot>2.0)&(d.lam_tot<2.8), top3_home)

# --- contrôle : combinaison LA + tranchée pour UNDER (intersection) ---
check("UNDER 2.5 | faible total + équilibré (lam_tot<2.0 & |diff|<0.6)",
      lambda d: (d.lam_tot<2.0)&(d.abs_diff<0.6), under25)
