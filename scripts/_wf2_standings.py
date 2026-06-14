# -*- coding: utf-8 -*-
"""WF2 - Le CLASSEMENT (position, points, dynamique) ajoute-t-il du pouvoir predictif
au-dela des cotes ? Walk-forward strict 70/30 temporel.

Tests:
 1. Logistic cote-only vs cote+classement (reconstruit ET snapshots) : delta acc/logloss/ROI OOS
 2. Desaccords classement vs cote (underdog mieux classe) : WR/ROI de suivre le classement
 3. Fin de saison (J34-38) : leader, top vs bottom, accuracy du favori
 4. Debut de saison (J1-3) : force historique all-time vs cote
"""
import sys, json, math
from collections import Counter, defaultdict
from datetime import datetime
sys.path.insert(0, '.')
import numpy as np
from scipy.optimize import minimize
from scraper.config import load_settings
from sqlalchemy import create_engine, text

SEP = "=" * 78
def parse_t(s):
    return datetime.fromisoformat(str(s).replace('Z', ''))

eng = create_engine(load_settings().db_url)

# ----------------------------------------------------------------- LOAD
with eng.connect() as c:
    evs = c.execute(text(
        "select e.id, cast(e.round_info as int) rd, e.team_a, e.team_b, e.expected_start, "
        "r.score_a, r.score_b "
        "from events e left join results r on r.event_id=e.id "
        "order by e.expected_start, e.id")).fetchall()
    # cotes d'ouverture = MIN(id) par event
    odds_rows = c.execute(text(
        "select o.event_id, o.odds_home, o.odds_draw, o.odds_away from odds_snapshots o "
        "join (select event_id, min(id) mid from odds_snapshots group by event_id) m "
        "on m.mid = o.id")).fetchall()
    rk = c.execute(text(
        "select captured_at, team_name, position, points, won, lost, draw, history "
        "from rankings_snapshots order by captured_at")).fetchall()

open_odds = {r[0]: (r[1], r[2], r[3]) for r in odds_rows
             if r[1] and r[2] and r[3] and r[1] > 1.0 and r[2] > 1.0 and r[3] > 1.0}

# snapshots par equipe : (t, played, points, position, form_avg)
snaps_by_team = defaultdict(list)
for r in rk:
    h = r[7]
    if isinstance(h, str):
        try: h = json.loads(h)
        except Exception: h = None
    form = None
    if isinstance(h, list) and h:
        m = {"Won": 3.0, "Draw": 1.0, "Lost": 0.0}
        vals = [m[v] for v in h if v in m]
        if vals:
            form = sum(vals) / len(vals)
    snaps_by_team[r[1]].append((parse_t(r[0]), r[4] + r[5] + r[6], r[3], r[2], form))

# ----------------------------------------------------------------- SAISONS ROBUSTES
# dedup (team_a, team_b, expected_start), garder celui avec result; drop round 0
seen = {}
for r in evs:
    if r[1] is None or r[1] == 0:
        continue
    k = (r[2], r[3], str(r[4]))
    if k not in seen or (seen[k][5] is None and r[5] is not None):
        seen[k] = r
evs2 = sorted(seen.values(), key=lambda r: (str(r[4]), r[0]))

seasons = []
cur, last_rd, last_t = [], None, None
for r in evs2:
    rd, t = r[1], parse_t(r[4])
    new = False
    if last_rd is not None:
        if rd < last_rd - 4: new = True
        if last_t is not None and (t - last_t).total_seconds() > 45 * 60: new = True
    if new and cur:
        seasons.append(cur); cur = []; last_rd = None
    cur.append(r)
    last_rd = rd if last_rd is None else max(last_rd, rd)
    last_t = t
if cur: seasons.append(cur)
print(f"events dedup={len(evs2)}  saisons reconstruites={len(seasons)}")

# ----------------------------------------------------------------- FEATURES PAR MATCH
# 1) classement RECONSTRUIT intra-segment : table avant la journee J (rounds < J uniquement)
# 2) classement SNAPSHOT : dernier snapshot avant le match, garde-fous played<=rd-1, age<=80min, played>=3
# 3) force historique ALL-TIME : ppg cumule avant expected_start (cross-saisons)

rows = []  # dicts par match avec result + odds
for sid, seg in enumerate(seasons):
    by_rd = defaultdict(list)
    for r in seg:
        by_rd[r[1]].append(r)
    table = defaultdict(lambda: {"pts": 0, "gf": 0, "ga": 0, "played": 0, "form": []})
    seg_start_rd = min(by_rd)
    for rd in sorted(by_rd):
        # classement fige AVANT la journee rd
        standing = sorted(table.items(),
                          key=lambda kv: (-kv[1]["pts"], -(kv[1]["gf"] - kv[1]["ga"]), -kv[1]["gf"]))
        pos_map = {t: i + 1 for i, (t, d) in enumerate(standing)}
        snap_table = {t: dict(d) for t, d in table.items()}
        for r in by_rd[rd]:
            eid, _, ta, tb, est, sa, sb = r
            if sa is None or eid not in open_odds:
                continue
            oh, od, oa = open_odds[eid]
            inv = 1/oh + 1/od + 1/oa
            ph, pd_, pa = (1/oh)/inv, (1/od)/inv, (1/oa)/inv
            t_match = parse_t(est)
            d = dict(eid=eid, sid=sid, rd=rd, ta=ta, tb=tb, t=t_match,
                     oh=oh, od=od, oa=oa, ph=ph, pd=pd_, pa=pa,
                     y=(0 if sa > sb else (1 if sa == sb else 2)),
                     seg_start_rd=seg_start_rd)
            # --- reconstruit
            da, db = snap_table.get(ta), snap_table.get(tb)
            if da and db and da["played"] >= 3 and db["played"] >= 3 and seg_start_rd <= 2:
                d["rec_pos_diff"] = pos_map[tb] - pos_map[ta]          # >0 = home mieux classe
                d["rec_ppg_diff"] = da["pts"]/da["played"] - db["pts"]/db["played"]
                d["rec_gd_diff"] = (da["gf"]-da["ga"])/da["played"] - (db["gf"]-db["ga"])/db["played"]
                fa = da["form"][-5:]; fb = db["form"][-5:]
                d["rec_form_diff"] = (sum(fa)/len(fa)) - (sum(fb)/len(fb))
                d["rec_pos_h"] = pos_map[ta]; d["rec_pos_a"] = pos_map[tb]
                d["rec_pts_diff"] = da["pts"] - db["pts"]
            # --- snapshot
            ok_sn, sn = True, {}
            for team, key in ((ta, "h"), (tb, "a")):
                best = None
                for s in snaps_by_team.get(team, []):
                    if s[0] < t_match: best = s
                    else: break
                if best is None: ok_sn = False; break
                age = (t_match - best[0]).total_seconds() / 60
                if age > 80 or best[1] > rd - 1 or best[1] < 3 or best[4] is None:
                    ok_sn = False; break
                sn[key] = best
            if ok_sn:
                h, a = sn["h"], sn["a"]
                d["sn_pos_diff"] = a[3] - h[3]
                d["sn_ppg_diff"] = h[2]/h[1] - a[2]/a[1]
                d["sn_form_diff"] = h[4] - a[4]
                d["sn_pos_h"] = h[3]; d["sn_pos_a"] = a[3]
                d["sn_pts_diff"] = h[2] - a[2]
            rows.append(d)
        # mise a jour table avec les resultats de la journee rd
        for r in by_rd[rd]:
            _, _, ta, tb, _, sa, sb = r
            if sa is None: continue
            table[ta]["gf"] += sa; table[ta]["ga"] += sb; table[ta]["played"] += 1
            table[tb]["gf"] += sb; table[tb]["ga"] += sa; table[tb]["played"] += 1
            if sa > sb:
                table[ta]["pts"] += 3; table[ta]["form"].append(3); table[tb]["form"].append(0)
            elif sa < sb:
                table[tb]["pts"] += 3; table[tb]["form"].append(3); table[ta]["form"].append(0)
            else:
                table[ta]["pts"] += 1; table[tb]["pts"] += 1
                table[ta]["form"].append(1); table[tb]["form"].append(1)

# --- force historique all-time (ppg, wr) accumulee en ordre temporel strict
rows.sort(key=lambda d: (d["t"], d["eid"]))
hist = defaultdict(lambda: {"pts": 0, "n": 0, "w": 0})
i = 0
while i < len(rows):
    j = i
    while j < len(rows) and rows[j]["t"] == rows[i]["t"]:
        j += 1
    for k in range(i, j):
        d = rows[k]
        ha, hb = hist[d["ta"]], hist[d["tb"]]
        if ha["n"] >= 20 and hb["n"] >= 20:
            d["hist_ppg_diff"] = ha["pts"]/ha["n"] - hb["pts"]/hb["n"]
            d["hist_wr_diff"] = ha["w"]/ha["n"] - hb["w"]/hb["n"]
            d["hist_ppg_h"] = ha["pts"]/ha["n"]; d["hist_ppg_a"] = hb["pts"]/hb["n"]
    for k in range(i, j):  # update APRES la fenetre simultanee
        d = rows[k]
        y = d["y"]
        hist[d["ta"]]["n"] += 1; hist[d["tb"]]["n"] += 1
        if y == 0:
            hist[d["ta"]]["pts"] += 3; hist[d["ta"]]["w"] += 1
        elif y == 2:
            hist[d["tb"]]["pts"] += 3; hist[d["tb"]]["w"] += 1
        else:
            hist[d["ta"]]["pts"] += 1; hist[d["tb"]]["pts"] += 1
    i = j

n_all = len(rows)
cut = int(n_all * 0.70)
t_cut = rows[cut]["t"]
train_all = [d for d in rows if d["t"] < t_cut]
oos_all = [d for d in rows if d["t"] >= t_cut]
print(f"matchs (result+odds)={n_all}  train={len(train_all)} (< {t_cut})  oos={len(oos_all)}")

# ----------------------------------------------------------------- OUTILS
def fit_mnlogit(X, y, l2=1.0):
    n, dd = X.shape
    Y = np.zeros((n, 3)); Y[np.arange(n), y] = 1.0
    def f(w):
        W = w.reshape(dd, 3)
        Z = X @ W; Z -= Z.max(axis=1, keepdims=True)
        E = np.exp(Z); P = E / E.sum(axis=1, keepdims=True)
        nll = -np.log(P[np.arange(n), y] + 1e-12).sum() + 0.5 * l2 * (W[1:] ** 2).sum()
        G = X.T @ (P - Y)
        G[1:] += l2 * W[1:]
        return nll / n, G.ravel() / n
    res = minimize(f, np.zeros(dd * 3), jac=True, method='L-BFGS-B',
                   options={'maxiter': 1000})
    return res.x.reshape(dd, 3)

def predict(W, X):
    Z = X @ W; Z -= Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)

def build_X(subset, feat_names, mu=None, sd=None):
    base = np.array([[1.0, math.log(d["ph"]), math.log(d["pd"]), math.log(d["pa"])]
                     for d in subset])
    if not feat_names:
        return base, mu, sd
    F = np.array([[d[f] for f in feat_names] for d in subset], dtype=float)
    if mu is None:
        mu, sd = F.mean(axis=0), F.std(axis=0) + 1e-9
    F = (F - mu) / sd
    return np.hstack([base, F]), mu, sd

def metrics(P, subset, label=""):
    y = np.array([d["y"] for d in subset])
    pick = P.argmax(axis=1)
    acc = float((pick == y).mean())
    ll = float(-np.log(P[np.arange(len(y)), y] + 1e-12).mean())
    odds = np.array([[d["oh"], d["od"], d["oa"]] for d in subset])
    o_pick = odds[np.arange(len(y)), pick]
    roi = float(np.where(pick == y, o_pick - 1.0, -1.0).mean())
    return acc, ll, roi

def devig_P(subset):
    return np.array([[d["ph"], d["pd"], d["pa"]] for d in subset])

def eval_pair(subset_tr, subset_oos, feat_names, tag):
    """Compare cote-only vs cote+features sur le MEME subset."""
    y_tr = np.array([d["y"] for d in subset_tr])
    X0_tr, _, _ = build_X(subset_tr, [])
    X0_oo, _, _ = build_X(subset_oos, [])
    W0 = fit_mnlogit(X0_tr, y_tr)
    P0 = predict(W0, X0_oo)
    a0, l0, r0 = metrics(P0, subset_oos)
    X1_tr, mu, sd = build_X(subset_tr, feat_names)
    X1_oo, _, _ = build_X(subset_oos, feat_names, mu, sd)
    W1 = fit_mnlogit(X1_tr, y_tr)
    P1 = predict(W1, X1_oo)
    a1, l1, r1 = metrics(P1, subset_oos)
    af, lf, rf = metrics(devig_P(subset_oos), subset_oos)
    print(f"\n[{tag}]  n_train={len(subset_tr)}  n_oos={len(subset_oos)}")
    print(f"  favori devig      : acc={af:.4f}  logloss={lf:.4f}  roi={rf:+.4f}")
    print(f"  logit cote-only   : acc={a0:.4f}  logloss={l0:.4f}  roi={r0:+.4f}")
    print(f"  logit cote+{','.join(feat_names)}")
    print(f"                    : acc={a1:.4f}  logloss={l1:.4f}  roi={r1:+.4f}")
    print(f"  DELTA (variante - cote-only): acc={a1-a0:+.4f}  logloss={l1-l0:+.4f}  roi={r1-r0:+.4f}")
    # poids appris (features standardisees) pour interpretation
    names = ["int", "ln_ph", "ln_pd", "ln_pa"] + list(feat_names)
    Wd = W1 - W1.mean(axis=1, keepdims=True)
    for i, nm in enumerate(names):
        if nm in feat_names:
            print(f"    poids {nm:<16} H/D/A = {Wd[i,0]:+.3f} {Wd[i,1]:+.3f} {Wd[i,2]:+.3f}")
    return dict(tag=tag, n_oos=len(subset_oos), acc0=a0, acc1=a1, ll0=l0, ll1=l1,
                roi0=r0, roi1=r1, acc_fav=af)

# ===================================================================
print("\n" + SEP); print("TEST 0 - BASELINE GLOBALE (tous matchs result+odds)"); print(SEP)
af, lf, rf = metrics(devig_P(oos_all), oos_all)
y_tr = np.array([d["y"] for d in train_all])
X0_tr, _, _ = build_X(train_all, [])
X0_oo, _, _ = build_X(oos_all, [])
W0 = fit_mnlogit(X0_tr, y_tr)
a0, l0, r0 = metrics(predict(W0, X0_oo), oos_all)
print(f"favori devig    : acc={af:.4f}  logloss={lf:.4f}  roi={rf:+.4f}   (n_oos={len(oos_all)})")
print(f"logit cote-only : acc={a0:.4f}  logloss={l0:.4f}  roi={r0:+.4f}")
BASELINE_ACC_OOS = af

# ===================================================================
print("\n" + SEP); print("TEST 1 - COTE + CLASSEMENT (logistic, walk-forward)"); print(SEP)

REC_F = ["rec_pos_diff", "rec_ppg_diff", "rec_gd_diff", "rec_form_diff"]
SN_F = ["sn_pos_diff", "sn_ppg_diff", "sn_form_diff"]

sub_rec = [d for d in rows if all(f in d for f in REC_F) and d["rd"] >= 6]
sub_sn = [d for d in rows if all(f in d for f in SN_F) and d["rd"] >= 6]
sub_both = [d for d in rows if all(f in d for f in REC_F + SN_F) and d["rd"] >= 6]
print(f"couverture: reconstruit={len(sub_rec)}  snapshot={len(sub_sn)}  both={len(sub_both)} (rd>=6)")

res1a = eval_pair([d for d in sub_rec if d["t"] < t_cut], [d for d in sub_rec if d["t"] >= t_cut],
                  REC_F, "1a. classement RECONSTRUIT (rd>=6)")
res1b = eval_pair([d for d in sub_sn if d["t"] < t_cut], [d for d in sub_sn if d["t"] >= t_cut],
                  SN_F, "1b. classement SNAPSHOT (rd>=6)")
res1c = eval_pair([d for d in sub_both if d["t"] < t_cut], [d for d in sub_both if d["t"] >= t_cut],
                  REC_F + SN_F, "1c. les deux (rd>=6)")

# interaction pos_diff x (ph-pa)
for d in rows:
    if "sn_pos_diff" in d:
        d["sn_pos_x_edge"] = d["sn_pos_diff"] * (d["ph"] - d["pa"])
res1d = eval_pair([d for d in sub_sn if d["t"] < t_cut], [d for d in sub_sn if d["t"] >= t_cut],
                  SN_F + ["sn_pos_x_edge"], "1d. snapshot + interaction pos*edge")

# pts_diff seuls (sans ppg)
res1e = eval_pair([d for d in sub_sn if d["t"] < t_cut], [d for d in sub_sn if d["t"] >= t_cut],
                  ["sn_pts_diff", "sn_pos_diff"], "1e. snapshot pts_diff+pos_diff bruts")

# ===================================================================
print("\n" + SEP); print("TEST 2 - DESACCORDS classement vs cote (OOS uniquement)"); print(SEP)

def side_metrics(subset, side_fn, tag):
    """side_fn(d) -> 0 (home) ou 2 (away) : equipe a backer. WR/ROI."""
    n = w = 0; pnl = 0.0; cotes = []
    for d in subset:
        s = side_fn(d)
        o = d["oh"] if s == 0 else d["oa"]
        cotes.append(o)
        n += 1
        if d["y"] == s:
            w += 1; pnl += o - 1.0
        else:
            pnl -= 1.0
    if n == 0:
        print(f"  {tag}: n=0"); return None
    print(f"  {tag:<46} n={n:>4}  WR={w/n:.4f}  ROI={pnl/n:+.4f}  cote_moy={sum(cotes)/n:.2f}")
    return dict(n=n, wr=w/n, roi=pnl/n, avg_odds=sum(cotes)/n)

def fav_side(d):  # favori marche H vs A (hors nul)
    return 0 if d["ph"] >= d["pa"] else 2

for src, posh, posa, ptsd, label in (
        ("sn", "sn_pos_h", "sn_pos_a", "sn_pts_diff", "SNAPSHOT"),
        ("rec", "rec_pos_h", "rec_pos_a", "rec_pts_diff", "RECONSTRUIT")):
    oos_v = [d for d in oos_all if posh in d and d["rd"] >= 6]
    print(f"\n--- source {label} (oos, rd>=6, n={len(oos_v)}) ---")
    # le favori cote n'est PAS le mieux classe
    def mismatch(d, gap):
        fs = fav_side(d)
        better = 0 if d[posh] < d[posa] else (2 if d[posa] < d[posh] else None)
        if better is None or better == fs: return None
        if abs(d[posh] - d[posa]) < gap: return None
        return better
    for gap in (1, 5, 8):
        sub = [d for d in oos_v if mismatch(d, gap) is not None]
        if not sub: continue
        print(f" desaccord pos (gap>={gap}):")
        side_metrics(sub, lambda d: mismatch(d, gap), f"  suivre CLASSEMENT (underdog mieux classe)")
        side_metrics(sub, fav_side, f"  suivre MARCHE (favori cote)")
    # points_diff >= +5 contre le favori
    def mismatch_pts(d, th):
        fs = fav_side(d)
        pd_ = d[ptsd]  # home - away
        if fs == 2 and pd_ >= th: return 0
        if fs == 0 and pd_ <= -th: return 2
        return None
    for th in (5, 8):
        sub = [d for d in oos_v if mismatch_pts(d, th) is not None]
        if not sub: continue
        print(f" desaccord points (|pts_diff|>={th} contre favori):")
        side_metrics(sub, lambda d: mismatch_pts(d, th), f"  suivre CLASSEMENT (pts)")
        side_metrics(sub, fav_side, f"  suivre MARCHE")

# ===================================================================
print("\n" + SEP); print("TEST 3 - FIN DE SAISON (J34-38) : enjeux, leader, top/bottom"); print(SEP)

def block_stats(subset, tag):
    if not subset:
        print(f"  {tag}: n=0"); return
    af, lf, rf = metrics(devig_P(subset), subset)
    yh = sum(1 for d in subset if d["y"] == 0); yd = sum(1 for d in subset if d["y"] == 1)
    n = len(subset)
    print(f"  {tag:<34} n={n:>4}  acc_favori={af:.4f}  logloss={lf:.4f}  roi_fav={rf:+.4f}  "
          f"home%={yh/n:.3f} draw%={yd/n:.3f}")

for scope, data in (("OOS", oos_all), ("FULL", rows)):
    print(f"\n--- {scope}: accuracy favori par segment de saison (matchs avec sn_pos) ---")
    v = [d for d in data if "sn_pos_h" in d]
    block_stats([d for d in v if 6 <= d["rd"] <= 15], "J6-15")
    block_stats([d for d in v if 16 <= d["rd"] <= 25], "J16-25")
    block_stats([d for d in v if 26 <= d["rd"] <= 33], "J26-33")
    block_stats([d for d in v if 34 <= d["rd"] <= 38], "J34-38 (FS)")

def leader_stats(data, lo, hi, tag):
    sub = []
    for d in data:
        if "sn_pos_h" not in d or not (lo <= d["rd"] <= hi): continue
        if d["sn_pos_h"] == 1: sub.append((d, 0))
        elif d["sn_pos_a"] == 1: sub.append((d, 2))
    if not sub:
        print(f"  {tag}: n=0"); return
    n = len(sub)
    w = sum(1 for d, s in sub if d["y"] == s)
    imp = sum((d["ph"] if s == 0 else d["pa"]) for d, s in sub) / n
    pnl = sum(((d["oh"] if s == 0 else d["oa"]) - 1.0) if d["y"] == s else -1.0 for d, s in sub)
    print(f"  {tag:<34} n={n:>4}  WR_leader={w/n:.4f}  p_implicite={imp:.4f}  "
          f"ecart={w/n-imp:+.4f}  ROI_back_leader={pnl/n:+.4f}")

for scope, data in (("OOS", oos_all), ("FULL", rows)):
    print(f"\n--- {scope}: le LEADER (pos snapshot=1) lache-t-il des matchs ? ---")
    leader_stats(data, 6, 25, "leader J6-25")
    leader_stats(data, 26, 33, "leader J26-33")
    leader_stats(data, 34, 38, "leader J34-38")

def topbot(data, tag):
    sub = []
    for d in data:
        if "sn_pos_h" not in d or not (34 <= d["rd"] <= 38): continue
        if d["sn_pos_h"] <= 5 and d["sn_pos_a"] >= 16: sub.append((d, 0))
        elif d["sn_pos_a"] <= 5 and d["sn_pos_h"] >= 16: sub.append((d, 2))
    if not sub:
        print(f"  {tag}: n=0"); return
    n = len(sub); w = sum(1 for d, s in sub if d["y"] == s)
    imp = sum((d["ph"] if s == 0 else d["pa"]) for d, s in sub) / n
    pnl = sum(((d["oh"] if s == 0 else d["oa"]) - 1.0) if d["y"] == s else -1.0 for d, s in sub)
    print(f"  {tag:<34} n={n:>4}  WR_top5={w/n:.4f}  p_implicite={imp:.4f}  "
          f"ecart={w/n-imp:+.4f}  ROI_back_top={pnl/n:+.4f}")

print("\n--- top5 vs bottom5 en J34-38 (back le top5) ---")
topbot(oos_all, "OOS")
topbot(rows, "FULL")

# meme comparaison J6-33 pour reference
def topbot_range(data, lo, hi, tag):
    sub = []
    for d in data:
        if "sn_pos_h" not in d or not (lo <= d["rd"] <= hi): continue
        if d["sn_pos_h"] <= 5 and d["sn_pos_a"] >= 16: sub.append((d, 0))
        elif d["sn_pos_a"] <= 5 and d["sn_pos_h"] >= 16: sub.append((d, 2))
    if not sub:
        print(f"  {tag}: n=0"); return
    n = len(sub); w = sum(1 for d, s in sub if d["y"] == s)
    imp = sum((d["ph"] if s == 0 else d["pa"]) for d, s in sub) / n
    pnl = sum(((d["oh"] if s == 0 else d["oa"]) - 1.0) if d["y"] == s else -1.0 for d, s in sub)
    print(f"  {tag:<34} n={n:>4}  WR_top5={w/n:.4f}  p_implicite={imp:.4f}  "
          f"ecart={w/n-imp:+.4f}  ROI_back_top={pnl/n:+.4f}")
topbot_range(rows, 6, 33, "FULL J6-33 (reference)")
topbot_range(oos_all, 6, 33, "OOS J6-33 (reference)")

# ===================================================================
print("\n" + SEP); print("TEST 4 - DEBUT DE SAISON (J1-3) : force historique all-time vs cote"); print(SEP)

HIST_F = ["hist_ppg_diff", "hist_wr_diff"]
sub_h = [d for d in rows if all(f in d for f in HIST_F)]
early_oos = [d for d in sub_h if d["t"] >= t_cut and d["rd"] <= 3]
early_tr = [d for d in sub_h if d["t"] < t_cut and d["rd"] <= 3]
print(f"matchs J1-3 avec hist (n>=20 par equipe): train={len(early_tr)}  oos={len(early_oos)}")

# pick naïf : equipe au meilleur hist_ppg (jamais nul) vs favori cote
def hist_side(d): return 0 if d["hist_ppg_diff"] >= 0 else 2
print("\n--- picks naïfs sur OOS J1-3 ---")
side_metrics(early_oos, fav_side, "favori cote (H vs A)")
side_metrics(early_oos, hist_side, "meilleur hist_ppg all-time")
agree = [d for d in early_oos if fav_side(d) == hist_side(d)]
disag = [d for d in early_oos if fav_side(d) != hist_side(d)]
print(f"  accord cote/hist: {len(agree)}  desaccord: {len(disag)}")
if disag:
    side_metrics(disag, hist_side, "desaccord: suivre HIST")
    side_metrics(disag, fav_side, "desaccord: suivre COTE")

# logistic cote vs cote+hist, entraine sur TOUS les rounds du train, evalue J1-3 OOS
y_tr = np.array([d["y"] for d in [x for x in sub_h if x["t"] < t_cut]])
tr_h = [x for x in sub_h if x["t"] < t_cut]
X0_tr, _, _ = build_X(tr_h, [])
W0h = fit_mnlogit(X0_tr, y_tr)
X1_tr, mu_h, sd_h = build_X(tr_h, HIST_F)
W1h = fit_mnlogit(X1_tr, y_tr)
for tag, subset in (("OOS J1-3", early_oos),
                    ("OOS J4-10", [d for d in sub_h if d["t"] >= t_cut and 4 <= d["rd"] <= 10]),
                    ("OOS tous rounds", [d for d in sub_h if d["t"] >= t_cut])):
    if not subset: continue
    X0_oo, _, _ = build_X(subset, [])
    X1_oo, _, _ = build_X(subset, HIST_F, mu_h, sd_h)
    a0, l0, r0 = metrics(predict(W0h, X0_oo), subset)
    a1, l1, r1 = metrics(predict(W1h, X1_oo), subset)
    af2, lf2, rf2 = metrics(devig_P(subset), subset)
    print(f"\n  [{tag}] n={len(subset)}")
    print(f"    favori devig    : acc={af2:.4f}  logloss={lf2:.4f}  roi={rf2:+.4f}")
    print(f"    logit cote-only : acc={a0:.4f}  logloss={l0:.4f}  roi={r0:+.4f}")
    print(f"    logit cote+hist : acc={a1:.4f}  logloss={l1:.4f}  roi={r1:+.4f}")
    print(f"    DELTA acc={a1-a0:+.4f}  logloss={l1-l0:+.4f}  roi={r1-r0:+.4f}")

print("\nFIN.")
