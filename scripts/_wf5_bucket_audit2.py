# -*- coding: utf-8 -*-
"""WF5 part 2 — Tests de replication globaux sur la grille bucket x segment.

S'appuie sur exports/wf5_bucket_audit.json (genere par _wf5_bucket_audit.py).

1. Correlation orig<->recent des ROI par cellule (n>=50 des 2 cotes) :
   si les edges etaient reels -> corr > 0 ; si bruit -> corr ~ 0.
2. Combien de cellules |z|>=2 en orig vs attendu par hasard (multiple testing).
3. Portefeuille declare : si on avait joue TOUTES les cellules edge+ de COTE_EDGES
   depuis le 2026-06-08, ROI ? Et les TRAPS (cellules a eviter) : leur ROI recent
   est-il pire que la baseline globale ?
4. Buckets pooles (sans dimension segment) : la dimension segment ajoute-t-elle
   quelque chose, ou tout est-il explique par le bucket seul ?

Sortie : exports/wf5_bucket_audit2.json
"""
import sys, json, math
sys.path.insert(0, ".")
import pandas as pd

data = json.load(open("exports/wf5_bucket_audit.json", encoding="utf-8"))
grid = data["full_grid"]
declared = data["declared_cells_audit"]

# ---------- 1) correlation orig <-> recent ----------
rows = []
for c in grid:
    o, r = c.get("orig"), c.get("recent")
    if o and r and o["n"] >= 50 and r["n"] >= 50:
        rows.append((c["segment"], c["side"], c["bucket"],
                     o["n"], o["roi_pct"], o["z"], r["n"], r["roi_pct"], r["z"]))
cf = pd.DataFrame(rows, columns=["segment", "side", "bucket",
                                 "n_o", "roi_o", "z_o", "n_r", "roi_r", "z_r"])
pearson = cf.roi_o.corr(cf.roi_r)
spearman = cf.roi_o.corr(cf.roi_r, method="spearman")
print(f"cellules n>=50 des 2 cotes : {len(cf)}")
print(f"corr(ROI_orig, ROI_recent)  pearson={pearson:.3f}  spearman={spearman:.3f}")
same_sign = int(((cf.roi_o * cf.roi_r) > 0).sum())
print(f"meme signe : {same_sign}/{len(cf)} (hasard ~ {len(cf)/2:.0f})")

# ---------- 2) significativite orig ----------
all_cells = [(c, c.get("orig")) for c in grid if c.get("orig")]
nz = [c for c, o in all_cells if o.get("z") is not None and abs(o["z"]) >= 2.0 and o["n"] >= 30]
print(f"\ncellules orig avec n>=30 : {sum(1 for c,o in all_cells if o['n']>=30)}")
print(f"dont |z|>=2 : {len(nz)} (attendu par hasard ~5% = "
      f"{0.05*sum(1 for c,o in all_cells if o['n']>=30):.1f})")
for c in nz:
    o, r = c["orig"], c.get("recent") or {}
    print(f"  {c['segment']:<9}{c['side']:<5}{c['bucket']:<24} orig n={o['n']:>4} roi={o['roi_pct']:>6.1f} z={o['z']:>5.2f}"
          f" | recent n={r.get('n','--'):>4} roi={r.get('roi_pct','--'):>6} z={r.get('z','--')}")

# ---------- 3) portefeuilles declares sur RECENT ----------
def pool(cells, period):
    n = w = 0; pnl = 0.0; pnl2 = 0.0
    for d in cells:
        c = d.get(period)
        if not c:
            continue
        n += c["n"]; w += c["wins"]
        pnl += c["roi_pct"] / 100.0 * c["n"]
    return n, w, (pnl / n * 100 if n else float("nan"))

edges_plus = [d for d in declared if not d["is_trap"]]
traps = [d for d in declared if d["is_trap"]]
for label, cells in (("EDGES+ declares", edges_plus), ("TRAPS declares", traps)):
    for per in ("orig", "recent"):
        n, w, roi = pool(cells, per)
        print(f"\n{label} [{per}] : n={n}, WR={w/n*100:.1f}%, ROI={roi:+.1f}%" if n else f"{label} [{per}] : n=0")

# baseline globale par periode (toutes cellules de la grille = tous les paris home+away)
for per in ("orig", "recent"):
    n = w = 0; pnl = 0.0
    for c in grid:
        cc = c.get(per)
        if cc:
            n += cc["n"]; w += cc["wins"]; pnl += cc["roi_pct"] / 100.0 * cc["n"]
    print(f"BASELINE tous paris [{per}] : n={n}, ROI={pnl/n*100:+.2f}%")

# z du portefeuille edges+ recent vs 0 (approx : sd du pnl ~ sqrt(mean(cote)*roi...) -> bootstrap impossible ici,
# on approxime sd par periode via la grille : on ne l'a pas conserve -> recalcul leger ci-dessous si besoin.

# ---------- 4) buckets pooles tous segments ----------
print("\n---------- BUCKETS POOLES (tous segments) ----------")
agg = {}
for c in grid:
    key = (c["side"], c["bucket"])
    for per in ("orig", "recent"):
        cc = c.get(per)
        if cc:
            a = agg.setdefault(key, {}).setdefault(per, [0, 0, 0.0])
            a[0] += cc["n"]; a[1] += cc["wins"]; a[2] += cc["roi_pct"] / 100 * cc["n"]
out_pool = []
hdr = f"{'side':<5}{'bucket':<24}{'n_o':>6}{'ROI_o':>8}{'n_r':>6}{'ROI_r':>8}"
print(hdr)
for key in sorted(agg):
    o = agg[key].get("orig", [0, 0, 0]); r = agg[key].get("recent", [0, 0, 0])
    roi_o = o[2] / o[0] * 100 if o[0] else float("nan")
    roi_r = r[2] / r[0] * 100 if r[0] else float("nan")
    print(f"{key[0]:<5}{key[1]:<24}{o[0]:>6}{roi_o:>8.1f}{r[0]:>6}{roi_r:>8.1f}")
    out_pool.append({"side": key[0], "bucket": key[1], "n_orig": o[0], "roi_orig": round(roi_o, 2),
                     "n_recent": r[0], "roi_recent": round(roi_r, 2)})

out = {
    "corr_cells_n50": {"n_cells": len(cf), "pearson": round(float(pearson), 3),
                       "spearman": round(float(spearman), 3),
                       "same_sign": same_sign},
    "orig_sig_cells_z2": [
        {"segment": c["segment"], "side": c["side"], "bucket": c["bucket"],
         "orig": c["orig"], "recent": c.get("recent")} for c in nz],
    "portfolios": {
        "edges_plus_orig": pool(edges_plus, "orig"),
        "edges_plus_recent": pool(edges_plus, "recent"),
        "traps_orig": pool(traps, "orig"),
        "traps_recent": pool(traps, "recent"),
    },
    "buckets_pooled": out_pool,
}
with open("exports/wf5_bucket_audit2.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("\nwritten exports/wf5_bucket_audit2.json")
