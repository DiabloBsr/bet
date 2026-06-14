# -*- coding: utf-8 -*-
"""WF3 - SCRIPT DE SAISON, iteration 2.

A. Structure AR des series de snapshots (jitter blanc / AR(1) / marche aleatoire)
B. Head-to-head logloss 3-classes OOS : open vs last vs pair-mean(train)
C. Strategies avec p-values Monte Carlo propres (heterogeneite des cotes geree) :
   - fade deviation vs pair-mean (bet aux cotes d'ouverture / aux dernieres cotes)
   - follow drift intra-event (bet aux dernieres cotes)
   nulls : truth=devig(cotes du bet)  vs  truth=pair_mean
D. Q3 corrige : champion ppg J11+ (selection) -> perf J1-10 vs cotes (test) + MC dispersion
E. Overround + stabilite calendrier des probs de paire
"""
import sys, math
from collections import defaultdict
from datetime import datetime
sys.path.insert(0, '.')
import numpy as np
from scipy import stats
from scraper.config import load_settings
from sqlalchemy import create_engine, text

rng = np.random.default_rng(7)
SEP = "=" * 88

def parse_t(s):
    return datetime.fromisoformat(str(s).replace('Z', ''))

def devig(oh, od, oa):
    inv = 1/oh + 1/od + 1/oa
    return (1/oh)/inv, (1/od)/inv, (1/oa)/inv

def logit(p):
    p = min(max(p, 1e-6), 1-1e-6)
    return math.log(p/(1-p))

# ================================================================== LOAD
eng = create_engine(load_settings().db_url)
with eng.connect() as c:
    evs = c.execute(text(
        "SELECT e.id, CAST(e.round_info AS INT) rd, e.team_a, e.team_b, e.expected_start, "
        "r.score_a, r.score_b "
        "FROM events e LEFT JOIN results r ON r.event_id=e.id "
        "ORDER BY e.expected_start, e.id")).fetchall()
    snaps = c.execute(text(
        "SELECT event_id, id, odds_home, odds_draw, odds_away, captured_at "
        "FROM odds_snapshots ORDER BY event_id, id")).fetchall()

snaps_by_ev = defaultdict(list)
for r in snaps:
    if r[2] and r[3] and r[4] and r[2] > 1.0 and r[3] > 1.0 and r[4] > 1.0:
        snaps_by_ev[r[0]].append((r[1], r[2], r[3], r[4], parse_t(r[5])))
for lst in snaps_by_ev.values():
    lst.sort(key=lambda x: x[0])

seen = {}
for r in evs:
    if r[1] is None or r[1] == 0:
        continue
    k = (r[2], r[3], str(r[4]))
    if k not in seen or (seen[k][5] is None and r[5] is not None):
        seen[k] = r
evs2 = sorted(seen.values(), key=lambda r: (str(r[4]), r[0]))

# saisons
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

# rows : matchs finis avec open + last pre-match
rows = []
for sid, seg in enumerate(seasons):
    for r in seg:
        eid, rd, ta, tb, est, sa, sb = r
        if sa is None or eid not in snaps_by_ev:
            continue
        t_start = parse_t(est)
        lst = snaps_by_ev[eid]
        pre = [s for s in lst if s[4] <= t_start]
        open_s = lst[0]
        last_s = pre[-1] if pre else lst[0]
        po = devig(open_s[1], open_s[2], open_s[3])
        pl = devig(last_s[1], last_s[2], last_s[3])
        rows.append(dict(eid=eid, sid=sid, rd=rd, ta=ta, tb=tb, t=t_start,
                         o_open=(open_s[1], open_s[2], open_s[3]),
                         o_last=(last_s[1], last_s[2], last_s[3]),
                         p_open=po, p_last=pl,
                         overround=1/open_s[1] + 1/open_s[2] + 1/open_s[3] - 1,
                         gap=(last_s[4] - open_s[4]).total_seconds() / 60,
                         y=(0 if sa > sb else (1 if sa == sb else 2))))
rows.sort(key=lambda d: d["t"])
print(f"matchs finis+cotes = {len(rows)}")
ovr = np.array([d["overround"] for d in rows])
print(f"overround 1X2 : moy={ovr.mean()*100:.2f}%  std={ovr.std()*100:.2f}%  "
      f"min={ovr.min()*100:.2f}%  max={ovr.max()*100:.2f}%")

t_split = rows[int(len(rows) * 0.70)]["t"]
train = [d for d in rows if d["t"] < t_split]
test = [d for d in rows if d["t"] >= t_split]
print(f"split 70/30 @ {t_split} : train={len(train)} test={len(test)}")

# pair means (train, vecteurs complets, openings)
pair_acc = defaultdict(list)
for d in train:
    pair_acc[(d["ta"], d["tb"])].append(d["p_open"])
pair_mu = {k: tuple(np.mean(np.array(v), axis=0)) for k, v in pair_acc.items() if len(v) >= 3}

# ================================================================== A. STRUCTURE AR
print(f"\n{SEP}\nA. STRUCTURE TEMPORELLE DES SNAPSHOTS (logit p_home)\n{SEP}")
ac1_list, vr_list = [], []
for eid, lst in snaps_by_ev.items():
    if len(lst) < 12:
        continue
    x = np.array([logit(devig(s[1], s[2], s[3])[0]) for s in lst])
    dx = np.diff(x)
    if dx.std() < 1e-9:
        continue
    ac1 = np.corrcoef(dx[:-1], dx[1:])[0, 1]
    ac1_list.append(ac1)
    if len(x) >= 11:
        d5 = x[5:] - x[:-5]
        vr_list.append(d5.var() / (5 * dx.var()))
ac1_arr, vr_arr = np.array(ac1_list), np.array(vr_list)
tt_ac = stats.ttest_1samp(ac1_arr, -0.5)
tt_ac0 = stats.ttest_1samp(ac1_arr, 0.0)
print(f"events avec >=12 snapshots : {len(ac1_arr)}")
print(f"autocorr lag-1 des increments : moy={ac1_arr.mean():+.3f} (bruit blanc=-0.5, RW=0)")
print(f"  t-test vs -0.5 : t={tt_ac.statistic:+.2f} p={tt_ac.pvalue:.2e}   "
      f"vs 0 : t={tt_ac0.statistic:+.2f} p={tt_ac0.pvalue:.2e}")
print(f"variance ratio k=5 : moy={vr_arr.mean():.3f} mediane={np.median(vr_arr):.3f} "
      f"(RW=1.0, bruit stationnaire~0.2)")
tt_vr = stats.ttest_1samp(vr_arr, 1.0)
print(f"  t-test vs 1.0 : t={tt_vr.statistic:+.2f} p={tt_vr.pvalue:.2e}")

# ================================================================== B. HEAD-TO-HEAD LOGLOSS OOS
print(f"\n{SEP}\nB. LOGLOSS 3-CLASSES OOS : open vs last vs pair-mean(train)\n{SEP}")
sub = [d for d in test if (d["ta"], d["tb"]) in pair_mu]
print(f"n test avec pair-mean train (>=3 occ.) = {len(sub)}")
def ll3(d, p):
    return -math.log(max(p[d["y"]], 1e-9))
L = {"open": np.array([ll3(d, d["p_open"]) for d in sub]),
     "last": np.array([ll3(d, d["p_last"]) for d in sub]),
     "pair": np.array([ll3(d, pair_mu[(d["ta"], d["tb"])]) for d in sub])}
for k, v in L.items():
    print(f"  logloss {k:5s} = {v.mean():.4f}")
for a, b in (("open", "last"), ("open", "pair"), ("last", "pair")):
    tt = stats.ttest_rel(L[a], L[b])
    print(f"  paired t {a} vs {b} : delta={L[a].mean()-L[b].mean():+.5f}  "
          f"t={tt.statistic:+.2f} p={tt.pvalue:.3f}")
Lmix = np.array([ll3(d, tuple((np.array(d["p_last"]) + np.array(pair_mu[(d["ta"], d["tb"])]))/2))
                 for d in sub])
tt = stats.ttest_rel(Lmix, L["last"])
print(f"  mix 50/50 (last+pair) = {Lmix.mean():.4f}  vs last : t={tt.statistic:+.2f} p={tt.pvalue:.3f}")

# ================================================================== C. STRATEGIES + MC
print(f"\n{SEP}\nC. STRATEGIES, ROI OOS + p-values MONTE CARLO (B=20000)\n{SEP}")

def mc_pval(bets, roi_obs, B=20000):
    """bets = list (odds, null_prob). p = P(ROI_sim >= roi_obs) sous null."""
    if not bets:
        return float('nan'), float('nan')
    o = np.array([b[0] for b in bets]); q = np.array([b[1] for b in bets])
    wins = rng.random((B, len(o))) < q
    roi_sim = (wins * o).sum(axis=1) / len(o) - 1
    return float((roi_sim >= roi_obs).mean()), float(roi_sim.mean())

def run_strat(ds, signal_fn, label):
    bets_o, bets_q_self, bets_q_pair, rets, wins = [], [], [], 0.0, 0
    for d in ds:
        sig = signal_fn(d)
        if sig is None:
            continue
        side, odds_t, p_self = sig
        o = odds_t[0] if side == 0 else odds_t[2]
        q_self = p_self[0] if side == 0 else p_self[2]
        mu = pair_mu.get((d["ta"], d["tb"]))
        q_pair = (mu[0] if side == 0 else mu[2]) if mu else None
        bets_o.append(o); bets_q_self.append(q_self)
        bets_q_pair.append(q_pair if q_pair is not None else q_self)
        if d["y"] == side:
            rets += o; wins += 1
    n = len(bets_o)
    if n == 0:
        print(f"  {label:42s} n=0"); return
    roi = (rets - n) / n * 100
    wr = wins / n * 100
    ao = float(np.mean(bets_o))
    p_self, eroi_self = mc_pval(list(zip(bets_o, bets_q_self)), roi / 100)
    p_pair, eroi_pair = mc_pval(list(zip(bets_o, bets_q_pair)), roi / 100)
    print(f"  {label:42s} n={n:4d} roi={roi:+7.2f}% wr={wr:5.1f}% cote={ao:5.2f}")
    print(f"    null[truth=cotes du bet] : E[ROI]={eroi_self*100:+6.2f}%  p(>=obs)={p_self:.4f}")
    print(f"    null[truth=pair-mean]    : E[ROI]={eroi_pair*100:+6.2f}%  p(>=obs)={p_pair:.4f}")

THR = 0.05
def fade_open(d):
    mu = pair_mu.get((d["ta"], d["tb"]))
    if mu is None: return None
    dv = logit(d["p_open"][0]) - logit(mu[0])
    if dv < -THR: return (0, d["o_open"], d["p_open"])   # home sous-cote vs paire -> bet home
    if dv > THR: return (2, d["o_open"], d["p_open"])    # home sur-cote -> bet away
    return None

def fade_open_away_only(d):
    s = fade_open(d)
    return s if s and s[0] == 2 else None

def fade_last(d):
    mu = pair_mu.get((d["ta"], d["tb"]))
    if mu is None: return None
    dv = logit(d["p_last"][0]) - logit(mu[0])
    if dv < -THR: return (0, d["o_last"], d["p_last"])
    if dv > THR: return (2, d["o_last"], d["p_last"])
    return None

DTHR = 0.03
def follow_drift(d):
    if d["gap"] < 5: return None
    dv = logit(d["p_last"][0]) - logit(d["p_open"][0])
    if dv > DTHR: return (0, d["o_last"], d["p_last"])
    if dv < -DTHR: return (2, d["o_last"], d["p_last"])
    return None

print("\n[TRAIN] (reference, signaux fixes a priori)")
for fn, lab in ((fade_open, "fade-open 2 cotes (thr .05)"),
                (fade_open_away_only, "fade-open away only"),
                (fade_last, "fade-last 2 cotes"),
                (follow_drift, "follow-drift (thr .03, bet last)")):
    run_strat(train, fn, lab)
print("\n[OOS TEST]")
for fn, lab in ((fade_open, "fade-open 2 cotes (thr .05)"),
                (fade_open_away_only, "fade-open away only"),
                (fade_last, "fade-last 2 cotes"),
                (follow_drift, "follow-drift (thr .03, bet last)")):
    run_strat(test, fn, lab)

half = test[len(test)//2]["t"]
print("\n[OOS par moitie] fade-open 2 cotes")
for part, lab in ((([d for d in test if d["t"] < half]), "OOS-1ere moitie"),
                  (([d for d in test if d["t"] >= half]), "OOS-2eme moitie")):
    run_strat(part, fade_open, lab)

# ================================================================== D. CHAMPION SCRIPTE
print(f"\n{SEP}\nD. CHAMPION SCRIPTE (selection ppg J11+, test J1-10) + MC DISPERSION\n{SEP}")
by_sid = defaultdict(list)
for d in rows:
    by_sid[d["sid"]].append(d)
qual = []
for sid, ds in by_sid.items():
    rds = {d["rd"] for d in ds}
    late = [d for d in ds if d["rd"] >= 11]
    early = [d for d in ds if d["rd"] <= 10]
    if len(rds) >= 25 and len(late) >= 60 and len(early) >= 20:
        qual.append((sid, ds, early, late))
print(f"saisons qualifiees (>=25 journees, >=60 matchs J11+, >=20 matchs J1-10) : {len(qual)}")

def ppg_table(ds):
    pts, gp = defaultdict(float), defaultdict(int)
    for d in ds:
        ta, tb, y = d["ta"], d["tb"], d["y"]
        gp[ta] += 1; gp[tb] += 1
        if y == 0: pts[ta] += 3
        elif y == 1: pts[ta] += 1; pts[tb] += 1
        else: pts[tb] += 3
    return {t: pts[t]/gp[t] for t in gp if gp[t] >= 6}, gp

agg = dict(dp=0.0, vp=0.0, nm=0, wa=0, we=0.0)
aggb = dict(dp=0.0, vp=0.0, nm=0)
per_season = []
for sid, ds, early, late in qual:
    tbl, gp = ppg_table(late)
    if len(tbl) < 15: continue
    champ = max(tbl, key=tbl.get)
    bottom = min(tbl, key=tbl.get)
    for team, a in ((champ, agg), (bottom, aggb)):
        dp, vp, nm, wa, we = 0.0, 0.0, 0, 0, 0.0
        for d in early:
            if team not in (d["ta"], d["tb"]): continue
            p = d["p_open"]
            if d["ta"] == team:
                pw, pdr = p[0], p[1]; apts = 3 if d["y"] == 0 else (1 if d["y"] == 1 else 0)
            else:
                pw, pdr = p[2], p[1]; apts = 3 if d["y"] == 2 else (1 if d["y"] == 1 else 0)
            epts = 3*pw + pdr
            dp += apts - epts; vp += 9*pw + pdr - epts**2
            nm += 1; wa += 1 if apts == 3 else 0; we += pw
        a["dp"] += dp; a["vp"] += vp; a["nm"] += nm
        if team == champ:
            a["wa"] += wa; a["we"] += we
            per_season.append((sid, champ, nm, dp))
z = agg["dp"]/math.sqrt(agg["vp"]) if agg["vp"] > 0 else float('nan')
zb = aggb["dp"]/math.sqrt(aggb["vp"]) if aggb["vp"] > 0 else float('nan')
print(f"\nCHAMPION : J1-10 n={agg['nm']} matchs  delta_pts(obs-attendu)={agg['dp']:+.1f}  "
      f"z={z:+.2f}  p(2s)={2*stats.norm.sf(abs(z)):.3f}")
print(f"  victoires {agg['wa']} vs {agg['we']:.1f} attendues")
print(f"LANTERNE  : J1-10 n={aggb['nm']} matchs  delta_pts={aggb['dp']:+.1f}  "
      f"z={zb:+.2f}  p={2*stats.norm.sf(abs(zb)):.3f}")
for sid, ch, nm, dp in per_season:
    print(f"   sid={sid:3d} champ={ch:18s} nJ1-10={nm:2d} delta={dp:+5.1f}")

print("\n-- MC dispersion (2000 sims, matchs observes, probas=open devig) --")
NSIM = 2000
champ_pctl, std_pctl = [], []
for sid, ds, early, late in qual:
    probs = np.array([d["p_open"] for d in ds])
    teams = sorted({d["ta"] for d in ds} | {d["tb"] for d in ds})
    tidx = {t: i for i, t in enumerate(teams)}
    hi = np.array([tidx[d["ta"]] for d in ds]); ai = np.array([tidx[d["tb"]] for d in ds])
    act = np.zeros(len(teams))
    for d in ds:
        if d["y"] == 0: act[tidx[d["ta"]]] += 3
        elif d["y"] == 1: act[tidx[d["ta"]]] += 1; act[tidx[d["tb"]]] += 1
        else: act[tidx[d["tb"]]] += 3
    u = rng.random((NSIM, len(ds)))
    out = np.where(u < probs[:, 0], 0, np.where(u < probs[:, 0] + probs[:, 1], 1, 2))
    sim_pts = np.zeros((NSIM, len(teams)))
    for k in range(len(ds)):
        o = out[:, k]
        sim_pts[:, hi[k]] += np.where(o == 0, 3, np.where(o == 1, 1, 0))
        sim_pts[:, ai[k]] += np.where(o == 2, 3, np.where(o == 1, 1, 0))
    champ_pctl.append(((sim_pts.max(axis=1) < act.max()).mean()
                       + 0.5 * (sim_pts.max(axis=1) == act.max()).mean()))
    std_pctl.append(((sim_pts.std(axis=1) < act.std()).mean()
                     + 0.5 * (sim_pts.std(axis=1) == act.std()).mean()))
champ_pctl, std_pctl = np.array(champ_pctl), np.array(std_pctl)
print(f"n saisons MC = {len(champ_pctl)}")
print(f"percentile points champion reel : moy={champ_pctl.mean():.3f} (0.5 si i.i.d.)")
print(f"  valeurs={np.round(champ_pctl,2)}")
print(f"percentile std points finaux    : moy={std_pctl.mean():.3f}")
print(f"  valeurs={np.round(std_pctl,2)}")
if len(champ_pctl) >= 5:
    ks1 = stats.kstest(champ_pctl, 'uniform'); ks2 = stats.kstest(std_pctl, 'uniform')
    tt1 = stats.ttest_1samp(champ_pctl, 0.5); tt2 = stats.ttest_1samp(std_pctl, 0.5)
    print(f"KS unif : champ D={ks1.statistic:.3f} p={ks1.pvalue:.3f} | "
          f"std D={ks2.statistic:.3f} p={ks2.pvalue:.3f}")
    print(f"t vs .5 : champ t={tt1.statistic:+.2f} p={tt1.pvalue:.3f} | "
          f"std t={tt2.statistic:+.2f} p={tt2.pvalue:.3f}")

# ================================================================== E. STABILITE CALENDRIER
print(f"\n{SEP}\nE. DERIVE CALENDAIRE DES PROBS DE PAIRE (demean par paire)\n{SEP}")
t0 = rows[0]["t"]
grp = defaultdict(list)
for d in rows:
    grp[(d["ta"], d["tb"])].append(d)
xs, ys = [], []
for pair, lst in grp.items():
    if len(lst) < 4: continue
    days = np.array([(d["t"] - t0).total_seconds()/86400 for d in lst])
    ph = np.array([d["p_open"][0] for d in lst])
    xs.extend(days - days.mean()); ys.extend(ph - ph.mean())
xs, ys = np.array(xs), np.array(ys)
slope, icpt, r, p, se = stats.linregress(xs, ys)
print(f"dev p(home) ~ jours calendaires (within-pair) : slope={slope:+.2e}/jour  "
      f"r={r:+.4f}  p={p:.3f}  n={len(xs)}")

print(f"\n{SEP}\nFIN\n{SEP}")
