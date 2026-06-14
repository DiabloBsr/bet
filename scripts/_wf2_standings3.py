# -*- coding: utf-8 -*-
"""WF2 - Complement : le classement SEUL (sans cote) vs cote-only.
Montre si la cote subsume l'information du classement. Pipeline identique."""
import sys, json, math
from collections import defaultdict
from datetime import datetime
sys.path.insert(0, '.')
import numpy as np
from scipy.optimize import minimize
from scraper.config import load_settings
from sqlalchemy import create_engine, text

def parse_t(s):
    return datetime.fromisoformat(str(s).replace('Z', ''))

eng = create_engine(load_settings().db_url)
with eng.connect() as c:
    evs = c.execute(text(
        "select e.id, cast(e.round_info as int) rd, e.team_a, e.team_b, e.expected_start, "
        "r.score_a, r.score_b from events e left join results r on r.event_id=e.id "
        "order by e.expected_start, e.id")).fetchall()
    odds_rows = c.execute(text(
        "select o.event_id, o.odds_home, o.odds_draw, o.odds_away from odds_snapshots o "
        "join (select event_id, min(id) mid from odds_snapshots group by event_id) m "
        "on m.mid = o.id")).fetchall()
    rk = c.execute(text(
        "select captured_at, team_name, position, points, won, lost, draw, history "
        "from rankings_snapshots order by captured_at")).fetchall()

open_odds = {r[0]: (r[1], r[2], r[3]) for r in odds_rows
             if r[1] and r[2] and r[3] and r[1] > 1.0 and r[2] > 1.0 and r[3] > 1.0}
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
        if vals: form = sum(vals) / len(vals)
    snaps_by_team[r[1]].append((parse_t(r[0]), r[4] + r[5] + r[6], r[3], r[2], form))

seen = {}
for r in evs:
    if r[1] is None or r[1] == 0: continue
    k = (r[2], r[3], str(r[4]))
    if k not in seen or (seen[k][5] is None and r[5] is not None):
        seen[k] = r
evs2 = sorted(seen.values(), key=lambda r: (str(r[4]), r[0]))
seasons, cur, last_rd, last_t = [], [], None, None
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

rows = []
for sid, seg in enumerate(seasons):
    by_rd = defaultdict(list)
    for r in seg: by_rd[r[1]].append(r)
    for rd in sorted(by_rd):
        for r in by_rd[rd]:
            eid, _, ta, tb, est, sa, sb = r
            if sa is None or eid not in open_odds: continue
            oh, od, oa = open_odds[eid]
            inv = 1/oh + 1/od + 1/oa
            d = dict(eid=eid, rd=rd, t=parse_t(est),
                     oh=oh, od=od, oa=oa, ph=(1/oh)/inv, pd=(1/od)/inv, pa=(1/oa)/inv,
                     y=(0 if sa > sb else (1 if sa == sb else 2)))
            ok_sn, sn = True, {}
            for team, key in ((ta, "h"), (tb, "a")):
                best = None
                for s in snaps_by_team.get(team, []):
                    if s[0] < d["t"]: best = s
                    else: break
                if best is None: ok_sn = False; break
                age = (d["t"] - best[0]).total_seconds() / 60
                if age > 80 or best[1] > rd - 1 or best[1] < 3 or best[4] is None:
                    ok_sn = False; break
                sn[key] = best
            if ok_sn:
                h, a = sn["h"], sn["a"]
                d["sn_pos_diff"] = a[3] - h[3]
                d["sn_ppg_diff"] = h[2]/h[1] - a[2]/a[1]
                d["sn_form_diff"] = h[4] - a[4]
            rows.append(d)

rows.sort(key=lambda d: (d["t"], d["eid"]))
cut = int(len(rows) * 0.70)
t_cut = rows[cut]["t"]

def fit_mnlogit(X, y, l2=1.0):
    n, dd = X.shape
    Y = np.zeros((n, 3)); Y[np.arange(n), y] = 1.0
    def f(w):
        W = w.reshape(dd, 3)
        Z = X @ W; Z -= Z.max(axis=1, keepdims=True)
        E = np.exp(Z); P = E / E.sum(axis=1, keepdims=True)
        nll = -np.log(P[np.arange(n), y] + 1e-12).sum() + 0.5 * l2 * (W[1:] ** 2).sum()
        G = X.T @ (P - Y); G[1:] += l2 * W[1:]
        return nll / n, G.ravel() / n
    res = minimize(f, np.zeros(dd * 3), jac=True, method='L-BFGS-B', options={'maxiter': 1000})
    return res.x.reshape(dd, 3)

def predict(W, X):
    Z = X @ W; Z -= Z.max(axis=1, keepdims=True)
    E = np.exp(Z); return E / E.sum(axis=1, keepdims=True)

def metrics(P, subset):
    y = np.array([d["y"] for d in subset])
    pick = P.argmax(axis=1)
    acc = float((pick == y).mean())
    ll = float(-np.log(P[np.arange(len(y)), y] + 1e-12).mean())
    odds = np.array([[d["oh"], d["od"], d["oa"]] for d in subset])
    o_pick = odds[np.arange(len(y)), pick]
    roi = float(np.where(pick == y, o_pick - 1.0, -1.0).mean())
    return acc, ll, roi

SN_F = ["sn_pos_diff", "sn_ppg_diff", "sn_form_diff"]
sub = [d for d in rows if all(f in d for f in SN_F) and d["rd"] >= 6]
tr = [d for d in sub if d["t"] < t_cut]
oo = [d for d in sub if d["t"] >= t_cut]
y_tr = np.array([d["y"] for d in tr])
print(f"subset snapshot rd>=6 : train={len(tr)} oos={len(oo)}")

def X_odds(s): return np.array([[1.0, math.log(d["ph"]), math.log(d["pd"]), math.log(d["pa"])] for d in s])
F_tr = np.array([[d[f] for f in SN_F] for d in tr]); mu, sd = F_tr.mean(0), F_tr.std(0) + 1e-9
def X_stand(s):
    F = (np.array([[d[f] for f in SN_F] for d in s]) - mu) / sd
    return np.hstack([np.ones((len(s), 1)), F])

W_o = fit_mnlogit(X_odds(tr), y_tr)
W_s = fit_mnlogit(X_stand(tr), y_tr)
a_o, l_o, r_o = metrics(predict(W_o, X_odds(oo)), oo)
a_s, l_s, r_s = metrics(predict(W_s, X_stand(oo)), oo)
print(f"logit COTE-only       : acc={a_o:.4f}  logloss={l_o:.4f}  roi={r_o:+.4f}")
print(f"logit CLASSEMENT-only : acc={a_s:.4f}  logloss={l_s:.4f}  roi={r_s:+.4f}")
print(f"  -> deficit classement-seul vs cote: acc={a_s-a_o:+.4f}  logloss={l_s-l_o:+.4f}")

# correlations features <-> edge de cote
edge = np.array([d["ph"] - d["pa"] for d in sub])
for f in SN_F:
    v = np.array([d[f] for d in sub])
    print(f"corr({f}, ph-pa) = {np.corrcoef(v, edge)[0,1]:+.3f}")

# accord du pick: classement-only vs cote-only
P_o = predict(W_o, X_odds(oo)).argmax(1)
P_s = predict(W_s, X_stand(oo)).argmax(1)
print(f"accord des picks cote vs classement (OOS): {(P_o == P_s).mean():.3f}")
print("FIN.")
