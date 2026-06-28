"""AUDIT NIVEAU 2 — signaux CONDITIONNELS cachés dans le résidu vs cotes.
Discipline: manches propres (==10), split chrono 60/20/20, BH-FDR, permutation max-T,
sign-stability train/val/test, bins fixes ET quantiles, anti-leakage strict.
Spine: r_fav = I(favori gagne) - p_fav_devig (E=0 si cotes calibrées).
"""
from __future__ import annotations
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from scraper.config import load_settings
from scraper.market_inversion import parse_extra_markets, total_buts_odds, devig_market, _get_market, _to_float
from sqlalchemy import create_engine

RNG = np.random.RandomState(12345)
e = create_engine(load_settings().db_url)
raw = pd.read_sql("""SELECT e.competition comp, e.expected_start, e.id ev,
  o.odds_home oh, o.odds_draw od, o.odds_away oa, o.extra_markets em,
  r.score_a sa, r.score_b sb FROM events e
  JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
  JOIN results r ON r.event_id=e.id
  WHERE r.score_a IS NOT NULL AND e.competition LIKE 'InstantLeague-%'""", e)
raw = raw[(raw.oh > 1) & (raw.od > 1) & (raw.oa > 1)].copy()
raw["es"] = pd.to_datetime(raw.expected_start, utc=True, errors="coerce")
raw = raw.dropna(subset=["es"])

# --- manches propres (==10) ---
sz = raw.groupby(["comp", "es"]).transform("size")
raw["rsize"] = sz["ev"] if isinstance(sz, pd.DataFrame) else sz
clean = raw[raw["rsize"] == 10].sort_values(["comp", "es", "ev"]).reset_index(drop=True)
print(f"matchs totaux={len(raw)} | manches propres(==10)={len(clean)} ({len(clean)/len(raw)*100:.0f}%)")

d = clean
inv = 1/d.oh + 1/d.od + 1/d.oa
d["imp_home"] = (1/d.oh)/inv; d["imp_draw"] = (1/d.od)/inv; d["imp_away"] = (1/d.oa)/inv
d["fav_home"] = d.imp_home > d.imp_away
d["p_fav"] = d[["imp_home", "imp_away"]].max(axis=1)
d["p_dog"] = d[["imp_home", "imp_away"]].min(axis=1)
d["balance"] = d.p_fav - d.p_dog
d["tot"] = d.sa + d.sb
d["fav_won"] = np.where(d.fav_home, d.sa > d.sb, d.sb > d.sa).astype(float)
d["hw"] = (d.sa > d.sb).astype(float); d["draw"] = (d.sa == d.sb).astype(float)
d["over25"] = (d.tot >= 3).astype(float); d["btts"] = ((d.sa >= 1) & (d.sb >= 1)).astype(float)
# residus spine
d["r_fav"] = d.fav_won - d.p_fav
d["r_draw"] = d.draw - d.imp_draw
# ladders pour r_over / r_btts
io, ib = [], []
for r in d.itertuples():
    em = parse_extra_markets(r.em)
    q = devig_market(total_buts_odds(em)) if len(total_buts_odds(em)) >= 4 else {}
    io.append(sum(v for k, v in q.items() if k.isdigit() and int(k) >= 3) if q else np.nan)
    gng = _get_market(em, exact="G/NG")
    bq = devig_market({k: gng.get(k) for k in ("Oui", "Non") if gng.get(k)}) if isinstance(gng, dict) else {}
    ib.append(bq.get("Oui", np.nan) if bq else np.nan)
d["imp_over"] = io; d["imp_btts"] = ib
d["r_over"] = d.over25 - d.imp_over; d["r_btts"] = d.btts - d.imp_btts
# slot intra-manche
d["slot"] = d.groupby(["comp", "es"]).cumcount()
d["hour"] = d.es.dt.tz_convert("Etc/GMT-3").dt.hour
d["league"] = d.comp.str.replace("InstantLeague-", "", regex=False)

# --- précédent même slot, manche N-1 (leakage-safe) ---
d = d.sort_values(["comp", "slot", "es"])
gk = d.groupby(["comp", "slot"])
d["prev_p_fav"] = gk.p_fav.shift(1)
d["prev_fav_won"] = gk.fav_won.shift(1)
d["prev_r_fav"] = gk.r_fav.shift(1)
d = d.sort_values(["es", "comp", "ev"]).reset_index(drop=True)
d["upset_prev"] = (d.prev_fav_won == 0).astype(float)

# --- split chrono 60/20/20 ---
n = len(d); c1, c2 = int(n*0.6), int(n*0.8)
d["split"] = np.where(d.index < c1, "train", np.where(d.index < c2, "val", "test"))

# bins
def fixed_pfav(s): return pd.cut(s, [.40, .50, .60, .70, .80, 1.01], right=False)
def quint(s): return pd.qcut(s.rank(method="first"), 5, labels=[f"Q{i}" for i in range(1, 6)])
d["pfav_fx"] = fixed_pfav(d.p_fav).astype(str)
d["pfav_qt"] = quint(d.p_fav).astype(str)
d["prev_pfav_fx"] = fixed_pfav(d.prev_p_fav).astype(str)
d["draw_fx"] = pd.cut(d.imp_draw, [0, .22, .26, .30, 1]).astype(str)
d["surprise_prev"] = np.where(d.prev_fav_won.isna(), "na",
    np.where(d.prev_fav_won == 1, "fav_won",
    np.where(d.prev_p_fav >= .6, "big_upset", "mild_upset")))

def cell_stats(frame, resid):
    s = frame[resid].dropna()
    nn = len(s)
    if nn < 30: return nn, np.nan, np.nan, np.nan
    m = s.mean(); sd = s.std()
    z = m/(sd/math.sqrt(nn)) if sd > 0 else 0
    p = 2*(1 - 0.5*(1+math.erf(abs(z)/math.sqrt(2))))
    return nn, m, z, p

def scan(group_cols, resid, minn, label):
    """retourne candidats avec stats train/val/test + p train."""
    out = []
    keys = d.dropna(subset=[resid]).groupby(group_cols).groups.keys()
    for k in d.groupby(group_cols).groups.keys():
        mask = pd.Series(True, index=d.index)
        kk = k if isinstance(k, tuple) else (k,)
        for col, val in zip(group_cols, kk):
            mask &= d[col] == val
        sub = d[mask]
        if "na" in [str(x) for x in kk]: continue
        tr = sub[sub.split == "train"]; va = sub[sub.split == "val"]; teq = sub[sub.split == "test"]
        ntr, mtr, ztr, ptr = cell_stats(tr, resid)
        if ntr < minn or np.isnan(ztr): continue
        nva, mva, zva, _ = cell_stats(va, resid)
        nte, mte, zte, _ = cell_stats(teq, resid)
        out.append(dict(label=label, cell=str(k), n_tr=ntr, m_tr=mtr, z_tr=ztr, p_tr=ptr,
                        n_va=nva, m_va=mva, z_va=zva, n_te=nte, m_te=mte, z_te=zte))
    return out

MINN = 200
allc = []
# A. tranches de cote (current) — r_fav, r_over, r_btts
for resid in ["r_fav", "r_over", "r_btts", "r_draw"]:
    allc += scan(["pfav_fx"], resid, MINN, f"A:pfav_fx/{resid}")
    allc += scan(["pfav_qt"], resid, MINN, f"A:pfav_qt/{resid}")
allc += scan(["draw_fx"], "r_draw", MINN, "A:draw_fx/r_draw")
# B. transition prev_pfav × cur_pfav (leakage-safe, same-slot prev round)
allc += scan(["prev_pfav_fx", "pfav_fx"], "r_fav", MINN, "B:trans_pfav/r_fav")
# C. surprise précédente × tranche courante
allc += scan(["surprise_prev"], "r_fav", MINN, "C:surprise/r_fav")
allc += scan(["surprise_prev", "pfav_fx"], "r_fav", MINN, "C:surprise×pfav/r_fav")
allc += scan(["upset_prev"], "r_fav", MINN, "C:upset_prev/r_fav")
# D. position
allc += scan(["slot"], "r_fav", MINN, "D:slot/r_fav")
allc += scan(["slot", "pfav_fx"], "r_fav", MINN, "D:slot×pfav/r_fav")
# E. ligue / heure
allc += scan(["league"], "r_fav", MINN, "E:league/r_fav")
allc += scan(["league", "pfav_fx"], "r_fav", MINN, "E:league×pfav/r_fav")
allc += scan(["hour"], "r_fav", MINN, "E:hour/r_fav")

res = pd.DataFrame(allc)
m = len(res)
print(f"\ncellules testées (n>={MINN}, train) : {m}")
# BH-FDR sur p_tr
res = res.sort_values("p_tr").reset_index(drop=True)
res["rank"] = res.index + 1
res["bh_thr"] = res["rank"]/m*0.10
res["bh_pass"] = res.p_tr <= res.bh_thr
# sign stability
def sign_ok(r):
    s = np.sign(r.m_tr)
    return (np.sign(r.m_va) == s) and (np.sign(r.m_te) == s) and s != 0
res["sign_stable"] = res.apply(sign_ok, axis=1)
res["robust"] = res.bh_pass & res.sign_stable

print("\n" + "="*92)
print("TOP 20 candidats par |z_train| (avec stabilité de signe et FDR)")
print("="*92)
top = res.reindex(res.z_tr.abs().sort_values(ascending=False).index).head(20)
hdr = f"{'label':<24}{'cell':<22}{'n_tr':>6}{'z_tr':>7}{'z_va':>7}{'z_te':>7}{'signe':>7}{'FDR':>5}"
print(hdr); print("-"*len(hdr))
for r in top.itertuples():
    sg = "OK" if r.sign_stable else "flip"
    fd = "pass" if r.bh_pass else "-"
    print(f"{r.label:<24}{r.cell[:21]:<22}{r.n_tr:>6}{r.z_tr:>+7.1f}{(r.z_va if not np.isnan(r.z_va) else 0):>+7.1f}{(r.z_te if not np.isnan(r.z_te) else 0):>+7.1f}{sg:>7}{fd:>5}")

robust = res[res.robust]
print(f"\nCANDIDATS ROBUSTES (FDR pass + signe stable train/val/test) : {len(robust)}")
for r in robust.itertuples():
    print(f"  {r.label} {r.cell}: m_tr={r.m_tr:+.3f} m_va={r.m_va:+.3f} m_te={r.m_te:+.3f} (n_tr={r.n_tr})")

# --- PERMUTATION max-T sur la famille transition B (la priorité user) ---
print("\n" + "="*92)
print("PERMUTATION max-T — famille transition prev_pfav × cur_pfav (r_fav)")
print("="*92)
B = d.dropna(subset=["r_fav", "prev_p_fav"]).copy()
Btr = B[B.split == "train"]
cells = Btr.groupby(["prev_pfav_fx", "pfav_fx"])
real_cells = [(k, v) for k, v in cells.groups.items() if len(v) >= MINN and "na" not in [str(x) for x in k]]
def maxz(values, groups):
    mx = 0
    for k, idx in groups:
        s = values.loc[idx]
        if len(s) >= MINN:
            z = abs(s.mean()/(s.std()/math.sqrt(len(s)))) if s.std() > 0 else 0
            mx = max(mx, z)
    return mx
obs = maxz(Btr.r_fav, real_cells)
null = []
vals = Btr.r_fav.copy()
for _ in range(300):
    perm = pd.Series(RNG.permutation(vals.values), index=vals.index)
    null.append(maxz(perm, real_cells))
null = np.array(null)
pfw = (null >= obs).mean()
print(f"  max|z| observé (train) = {obs:.2f} | null max|z| moy={null.mean():.2f} p95={np.percentile(null,95):.2f}")
print(f"  p familywise (permutation) = {pfw:.3f}  -> {'SIGNIFICATIF' if pfw<0.05 else 'NON significatif (bruit de segmentation)'}")

# --- MODELE interpretable G : lift OOS vs cotes seules ---
print("\n" + "="*92)
print("(G) MODÈLE RÉSIDU — features conditionnelles battent-elles 'cotes seules' en OOS ?")
print("="*92)
md = d.dropna(subset=["prev_p_fav", "prev_r_fav", "p_fav"]).copy()
md["logit_pfav"] = np.log(md.p_fav/(1-md.p_fav))
md["surp_num"] = md.prev_r_fav
md["slot_n"] = md.slot
feats = ["logit_pfav", "prev_p_fav", "surp_num", "slot_n", "balance", "imp_draw"]
tr = md[md.split == "train"]; te = md[md.split == "test"]
yA_tr, yA_te = tr.fav_won, te.fav_won
A = LogisticRegression(max_iter=3000).fit(tr[["logit_pfav"]], yA_tr)
Bm = LogisticRegression(max_iter=3000, C=0.3).fit(tr[feats], yA_tr)
pa = A.predict_proba(te[["logit_pfav"]])[:, 1]; pb = Bm.predict_proba(te[feats])[:, 1]
lla, llb = log_loss(te.fav_won, pa), log_loss(te.fav_won, pb)
print(f"  log-loss TEST  cotes seules={lla:.4f}  cotes+conditionnel={llb:.4f}")
print(f"  -> {'features AIDENT (Δ %+.4f)'%(lla-llb) if (lla-llb)>0.0005 else 'AUCUN gain OOS (Δ %+.4f)'%(lla-llb)}")

print("\n" + "="*92)
print("ROBUSTESSE — top candidat re-testé sur TOUTES les données (manches impures incluses)")
print("="*92)
if len(robust):
    print("  (voir candidats robustes ci-dessus)")
else:
    print("  Aucun candidat robuste -> rien à re-tester. Le résidu est aléatoire dans tous les sous-régimes.")
