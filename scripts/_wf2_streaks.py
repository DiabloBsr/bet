# -*- coding: utf-8 -*-
"""WF2 - SERIES & RETOUR A LA MOYENNE (streaks / mean reversion), walk-forward strict.

Tests :
 1. P(win | serie de N victoires/defaites) vs proba marche (devig) - controle force
 2. ROI fade (parier contre serie de victoires quand favorite) + rebond (parier serie
    de defaites a cote elevee) - walk-forward (seuil choisi sur train, evalue OOS)
 3. Sur-regime / sous-regime : WR saison courante vs WR historique expanding (+/-15pp)
 4. Series de buts : 3+ buts sur chacun des 2-3 derniers matchs -> continue ?
 5. Logistic cote-only vs cote+streaks : delta accuracy / log-loss OOS

Aucun leakage : features avant journee J = journees J-1 et avant uniquement
(update des etats par round complet) ; WR historique = matchs strictement anterieurs.
Split temporel : train = premiers 70% (par expected_start), OOS = derniers 30%.
"""
import sys, json
from collections import Counter, defaultdict
from datetime import datetime
sys.path.insert(0, '.')
from scraper.config import load_settings
from sqlalchemy import create_engine, text
import numpy as np
from scipy.optimize import minimize

SEP = "=" * 78


def parse_ts(s):
    return datetime.fromisoformat(str(s).replace('Z', ''))


# ---------------------------------------------------------------- chargement
eng = create_engine(load_settings().db_url)
with eng.connect() as c:
    evs = c.execute(text(
        "select e.id, cast(e.round_info as int) rd, e.team_a, e.team_b, e.expected_start, "
        "r.score_a, r.score_b "
        "from events e left join results r on r.event_id=e.id "
        "order by e.expected_start, e.id")).fetchall()
    odds_rows = c.execute(text(
        "select o.event_id, o.odds_home, o.odds_draw, o.odds_away from odds_snapshots o "
        "join (select event_id, min(id) mid from odds_snapshots group by event_id) m "
        "on m.mid = o.id")).fetchall()

open_odds = {r[0]: (r[1], r[2], r[3]) for r in odds_rows
             if r[1] and r[2] and r[3] and r[1] > 1.0 and r[2] > 1.0 and r[3] > 1.0}

# dedup (team_a, team_b, expected_start) en gardant celui AVEC result, drop round 0
seen = {}
for r in evs:
    if r[1] is None or r[1] == 0:
        continue
    k = (r[2], r[3], str(r[4]))
    if k not in seen or (seen[k][5] is None and r[5] is not None):
        seen[k] = r
evs2 = sorted(seen.values(), key=lambda r: (str(r[4]), r[0]))
print(f"events apres dedup+drop round0 : {len(evs2)}")

# ------------------------------------------------- saisons robustes (cf. _wf2_explore2)
seasons = []
cur, last_rd, last_t = [], None, None
for r in evs2:
    rd, t = r[1], parse_ts(r[4])
    new = False
    if last_rd is not None:
        if rd < last_rd - 4:
            new = True
        if last_t is not None and (t - last_t).total_seconds() > 45 * 60:
            new = True
    if new and cur:
        seasons.append(cur); cur = []
    cur.append(r)
    last_rd = rd if (last_rd is None or new) else max(rd, last_rd)
    last_t = t
if cur:
    seasons.append(cur)
print(f"saisons reconstruites : {len(seasons)}")

# ------------------------------------------------- pass 1 : features intra-saison
# etat par equipe, remis a zero a chaque saison ; update PAR ROUND COMPLET
# (les matchs d'un meme round sont simultanes -> aucune info intra-round)
matches = []          # dicts match-niveau
for sid, s in enumerate(seasons):
    by_rd = defaultdict(list)
    for x in s:
        by_rd[x[1]].append(x)
    state = defaultdict(lambda: {'res': [], 'gf': [], 'wins': 0, 'played': 0})
    for rd in sorted(by_rd):
        # 1) features avant le round
        for x in by_rd[rd]:
            eid, _, ta, tb, ts, sa, sb = x
            if sa is None:
                continue
            feats = {}
            for side, team in (('h', ta), ('a', tb)):
                st = state[team]
                res = st['res']
                ws = 0
                for v in reversed(res):
                    if v == 'W': ws += 1
                    else: break
                ls = 0
                for v in reversed(res):
                    if v == 'L': ls += 1
                    else: break
                feats[side + '_ws'] = ws
                feats[side + '_ls'] = ls
                feats[side + '_played'] = st['played']
                feats[side + '_season_wr'] = (st['wins'] / st['played']) if st['played'] else None
                gf = st['gf']
                feats[side + '_g3_last2'] = (len(gf) >= 2 and gf[-1] >= 3 and gf[-2] >= 3)
                feats[side + '_g3_last3'] = (len(gf) >= 3 and gf[-1] >= 3 and gf[-2] >= 3 and gf[-3] >= 3)
            out = 'H' if sa > sb else ('A' if sa < sb else 'D')
            oh = open_odds.get(eid)
            m = dict(eid=eid, sid=sid, rd=rd, ts=parse_ts(ts), home=ta, away=tb,
                     sa=sa, sb=sb, out=out, odds=oh, **feats)
            if oh:
                inv = np.array([1 / oh[0], 1 / oh[1], 1 / oh[2]])
                p = inv / inv.sum()
                m['ph'], m['pd'], m['pa'] = float(p[0]), float(p[1]), float(p[2])
            matches.append(m)
        # 2) update etat APRES le round (tous les matchs du round, meme sans cote)
        for x in by_rd[rd]:
            eid, _, ta, tb, ts, sa, sb = x
            if sa is None:
                continue
            for team, g_for, g_ag in ((ta, sa, sb), (tb, sb, sa)):
                st = state[team]
                v = 'W' if g_for > g_ag else ('L' if g_for < g_ag else 'D')
                st['res'].append(v)
                st['gf'].append(g_for)
                st['played'] += 1
                if v == 'W':
                    st['wins'] += 1

matches.sort(key=lambda m: (m['ts'], m['eid']))
print(f"matchs avec result (features intra-saison) : {len(matches)}")

# ------------------------------------------------- pass 2 : WR historique expanding
hist = defaultdict(lambda: [0, 0])   # team -> [wins, played]  (strictement anterieur)
for m in matches:
    for side, team in (('h', m['home']), ('a', m['away'])):
        w, n = hist[team]
        m[side + '_hist_wr'] = (w / n) if n else None
        m[side + '_hist_n'] = n
    for team, g_for, g_ag in ((m['home'], m['sa'], m['sb']), (m['away'], m['sb'], m['sa'])):
        hist[team][1] += 1
        if g_for > g_ag:
            hist[team][0] += 1

# ------------------------------------------------- split temporel 70/30
usable = [m for m in matches if m['odds']]
print(f"matchs avec result + cotes ouverture : {len(usable)}")
n70 = int(0.7 * len(usable))
train, oos = usable[:n70], usable[n70:]
print(f"train={len(train)} ({train[0]['ts']} -> {train[-1]['ts']})")
print(f"oos  ={len(oos)} ({oos[0]['ts']} -> {oos[-1]['ts']})")

# ------------------------------------------------- baseline cote-only (favori devig)
def fav_pick(m):
    return 'HDA'[int(np.argmax([m['ph'], m['pd'], m['pa']]))]

def acc(ms):
    return sum(1 for m in ms if fav_pick(m) == m['out']) / len(ms)

def roi_bet(ms, sel):
    """sel(m) -> ('H'|'D'|'A') ou None ; mise 1, cote d'ouverture."""
    stake = payout = 0
    nb = 0
    for m in ms:
        pick = sel(m)
        if pick is None:
            continue
        nb += 1
        stake += 1
        if pick == m['out']:
            payout += m['odds']['HDA'.index(pick)]
    return (payout - stake) / stake if stake else None, nb

base_acc_tr, base_acc_oos = acc(train), acc(oos)
roi_fav_oos, n_fav = roi_bet(oos, fav_pick)
print(SEP)
print(f"BASELINE favori devig : acc train={base_acc_tr:.4f}  acc OOS={base_acc_oos:.4f} "
      f"(n_oos={len(oos)})  ROI favori OOS={roi_fav_oos:+.4f}")

# log-loss des probas devig brutes OOS
def logloss_devig(ms):
    ll = 0.0
    for m in ms:
        p = {'H': m['ph'], 'D': m['pd'], 'A': m['pa']}[m['out']]
        ll -= np.log(max(p, 1e-12))
    return ll / len(ms)
ll_devig_oos = logloss_devig(oos)
print(f"log-loss devig brut OOS = {ll_devig_oos:.4f}")

# ================================================================ TEST 1
print(); print(SEP)
print("TEST 1 - P(win | serie) vs proba marche (devig), team-obs")
print(SEP)

def team_obs(ms):
    """une obs par (match, equipe) avec p_market = proba devig que CETTE equipe gagne."""
    out = []
    for m in ms:
        out.append(dict(m=m, side='h', win=m['out'] == 'H', lose=m['out'] == 'A',
                        p=m['ph'], odds=m['odds'][0], ws=m['h_ws'], ls=m['h_ls'],
                        played=m['h_played'], swr=m['h_season_wr'],
                        hwr=m['h_hist_wr'], hn=m['h_hist_n'],
                        g2=m['h_g3_last2'], g3=m['h_g3_last3'], gf=m['sa']))
        out.append(dict(m=m, side='a', win=m['out'] == 'A', lose=m['out'] == 'H',
                        p=m['pa'], odds=m['odds'][2], ws=m['a_ws'], ls=m['a_ls'],
                        played=m['a_played'], swr=m['a_season_wr'],
                        hwr=m['a_hist_wr'], hn=m['a_hist_n'],
                        g2=m['a_g3_last2'], g3=m['a_g3_last3'], gf=m['sb']))
    return out

obs_tr, obs_oos, obs_all = team_obs(train), team_obs(oos), team_obs(usable)

def streak_table(obs, key, label):
    print(f"\n--- {label} ---")
    print(f"{'bin':>6} {'n':>5} {'WR reel':>8} {'WR marche':>9} {'delta':>7} {'z':>6}   "
          f"{'P(lose) reel':>12} {'P(lose) attendu*':>16}")
    rows = {}
    bins = [(0, '=0'), (1, '=1'), (2, '=2'), (3, '=3'), (4, '=4'), (5, '>=5'),
            (3, '>=3'), (4, '>=4'), (2, '>=2')]
    done = set()
    for n_, lab in bins:
        if lab in done:
            continue
        done.add(lab)
        if lab.startswith('>='):
            sub = [o for o in obs if o[key] >= n_]
        else:
            sub = [o for o in obs if o[key] == n_]
        if len(sub) < 10:
            print(f"{lab:>6} {len(sub):>5}   (trop peu)")
            continue
        wr = np.mean([o['win'] for o in sub])
        mk = np.mean([o['p'] for o in sub])
        se = np.sqrt(max(mk * (1 - mk), 1e-9) / len(sub))
        z = (wr - mk) / se
        pl = np.mean([o['lose'] for o in sub])
        # P(lose) attendu = moyenne devig de la proba que l'ADVERSAIRE gagne
        pl_mk = np.mean([(o['m']['pa'] if o['side'] == 'h' else o['m']['ph']) for o in sub])
        print(f"{lab:>6} {len(sub):>5} {wr:>8.4f} {mk:>9.4f} {wr-mk:>+7.4f} {z:>6.2f}   "
              f"{pl:>12.4f} {pl_mk:>16.4f}")
        rows[lab] = dict(n=len(sub), wr=float(wr), mkt=float(mk), z=float(z),
                         plose=float(pl), plose_mkt=float(pl_mk))
    return rows

print("\n###### ECHANTILLON COMPLET (descriptif) ######")
t1_full_ws = streak_table(obs_all, 'ws', 'serie de VICTOIRES (ws)')
t1_full_ls = streak_table(obs_all, 'ls', 'serie de DEFAITES (ls)')
print("\n###### OOS UNIQUEMENT ######")
t1_oos_ws = streak_table(obs_oos, 'ws', 'serie de VICTOIRES (ws) - OOS')
t1_oos_ls = streak_table(obs_oos, 'ls', 'serie de DEFAITES (ls) - OOS')

# sanity : calibration globale du marche sur team-obs OOS
wr_all = np.mean([o['win'] for o in obs_oos]); mk_all = np.mean([o['p'] for o in obs_oos])
print(f"\nsanity calibration OOS toutes team-obs : WR reel={wr_all:.4f} vs marche={mk_all:.4f}")

# ================================================================ TEST 2
print(); print(SEP)
print("TEST 2 - ROI walk-forward : FADE serie de victoires / REBOND serie de defaites")
print(SEP)

def fade_roi(obs, nmin):
    """equipe favorite (p > p adversaire ET p > p nul) avec ws>=nmin :
       on parie l'ADVERSAIRE (roi_opp) et le NUL (roi_draw)."""
    stake_o = pay_o = stake_d = pay_d = 0.0
    nb = 0
    for o in obs:
        m = o['m']
        p_opp = m['pa'] if o['side'] == 'h' else m['ph']
        if o['ws'] >= nmin and o['p'] > p_opp and o['p'] > m['pd']:
            nb += 1
            opp_odds = m['odds'][2] if o['side'] == 'h' else m['odds'][0]
            stake_o += 1; stake_d += 1
            if o['lose']:
                pay_o += opp_odds
            if m['out'] == 'D':
                pay_d += m['odds'][1]
    roi_o = (pay_o - stake_o) / stake_o if stake_o else None
    roi_d = (pay_d - stake_d) / stake_d if stake_d else None
    return roi_o, roi_d, nb

def fade_baseline(obs):
    """meme pari (adversaire du favori / nul) sur TOUS les matchs ou l'equipe est favorite,
       sans condition de serie -> baseline appariee."""
    return fade_roi(obs, 0)

def rebound_roi(obs, nmin, omin):
    stake = pay = 0.0
    nb = 0
    for o in obs:
        if o['ls'] >= nmin and o['odds'] >= omin:
            nb += 1; stake += 1
            if o['win']:
                pay += o['odds']
    return (pay - stake) / stake if stake else None, nb

def rebound_baseline(obs, omin):
    stake = pay = 0.0
    nb = 0
    for o in obs:
        if o['odds'] >= omin:
            nb += 1; stake += 1
            if o['win']:
                pay += o['odds']
    return (pay - stake) / stake if stake else None, nb

print("\n--- FADE : grille sur TRAIN (choix du seuil) ---")
print(f"{'N>=':>4} {'n':>5} {'ROI opp':>9} {'ROI nul':>9}")
best_fade = None
for nmin in (2, 3, 4, 5):
    ro, rd_, nb = fade_roi(obs_tr, nmin)
    print(f"{nmin:>4} {nb:>5} {ro if ro is None else format(ro,'+.4f'):>9} "
          f"{rd_ if rd_ is None else format(rd_,'+.4f'):>9}")
    if nb >= 80:
        for tag, roi in (('opp', ro), ('draw', rd_)):
            if best_fade is None or roi > best_fade[2]:
                best_fade = (nmin, tag, roi)
bo, bd, nb0 = fade_baseline(obs_tr)
print(f"baseline train (tous favoris, sans serie): n={nb0} ROI opp={bo:+.4f} ROI nul={bd:+.4f}")
print(f"-> seuil retenu sur train : N>={best_fade[0]}, pari={best_fade[1]} (ROI train={best_fade[2]:+.4f})")

print("\n--- FADE : evaluation OOS ---")
print(f"{'N>=':>4} {'n':>5} {'ROI opp':>9} {'ROI nul':>9}")
fade_oos = {}
for nmin in (2, 3, 4, 5):
    ro, rd_, nb = fade_roi(obs_oos, nmin)
    fade_oos[nmin] = (ro, rd_, nb)
    print(f"{nmin:>4} {nb:>5} {ro if ro is None else format(ro,'+.4f'):>9} "
          f"{rd_ if rd_ is None else format(rd_,'+.4f'):>9}")
bo_o, bd_o, nb_o = fade_baseline(obs_oos)
print(f"baseline OOS (tous favoris): n={nb_o} ROI opp={bo_o:+.4f} ROI nul={bd_o:+.4f}")
ro_sel, rd_sel, nb_sel = fade_oos[best_fade[0]]
roi_sel = ro_sel if best_fade[1] == 'opp' else rd_sel
base_sel = bo_o if best_fade[1] == 'opp' else bd_o
print(f"WALK-FORWARD fade OOS (N>={best_fade[0]}, {best_fade[1]}): n={nb_sel} "
      f"ROI={roi_sel:+.4f} vs baseline appariee {base_sel:+.4f} -> delta={roi_sel-base_sel:+.4f}")

print("\n--- REBOND : grille sur TRAIN ---")
print(f"{'N>=':>4} {'cote>=':>7} {'n':>5} {'ROI':>9} {'base n':>7} {'base ROI':>9}")
best_reb = None
for nmin in (2, 3, 4):
    for omin in (2.0, 2.5, 3.0):
        r, nb = rebound_roi(obs_tr, nmin, omin)
        rb, nbb = rebound_baseline(obs_tr, omin)
        print(f"{nmin:>4} {omin:>7} {nb:>5} {r if r is None else format(r,'+.4f'):>9} "
              f"{nbb:>7} {rb:+.4f}")
        if nb >= 80 and (best_reb is None or r > best_reb[2]):
            best_reb = (nmin, omin, r)
print(f"-> retenu sur train : ls>={best_reb[0]} & cote>={best_reb[1]} (ROI train={best_reb[2]:+.4f})")

print("\n--- REBOND : evaluation OOS ---")
print(f"{'N>=':>4} {'cote>=':>7} {'n':>5} {'ROI':>9} {'base n':>7} {'base ROI':>9}")
reb_oos = {}
for nmin in (2, 3, 4):
    for omin in (2.0, 2.5, 3.0):
        r, nb = rebound_roi(obs_oos, nmin, omin)
        rb, nbb = rebound_baseline(obs_oos, omin)
        reb_oos[(nmin, omin)] = (r, nb, rb, nbb)
        print(f"{nmin:>4} {omin:>7} {nb:>5} {r if r is None else format(r,'+.4f'):>9} "
              f"{nbb:>7} {rb:+.4f}")
r_sel, nb_rsel, rb_sel, _ = reb_oos[(best_reb[0], best_reb[1])]
print(f"WALK-FORWARD rebond OOS (ls>={best_reb[0]}, cote>={best_reb[1]}): n={nb_rsel} "
      f"ROI={r_sel:+.4f} vs baseline appariee {rb_sel:+.4f} -> delta={r_sel-rb_sel:+.4f}")

# ================================================================ TEST 3
print(); print(SEP)
print("TEST 3 - SUR-REGIME / SOUS-REGIME (WR saison avant J vs WR historique expanding)")
print(SEP)

def regime_table(obs, label):
    print(f"\n--- {label} ---")
    print(f"{'bucket':>22} {'n':>5} {'WR reel':>8} {'WR marche':>9} {'delta':>7} {'z':>6} "
          f"{'ROI back':>9}")
    out = {}
    elig = [o for o in obs if o['played'] >= 5 and o['hn'] >= 50 and o['swr'] is not None
            and o['hwr'] is not None]
    buckets = [('sur-regime >=+15pp', lambda d: d >= 0.15),
               ('sur-regime +15..+25', lambda d: 0.15 <= d < 0.25),
               ('sur-regime >=+25pp', lambda d: d >= 0.25),
               ('sous-regime <=-15pp', lambda d: d <= -0.15),
               ('sous-regime <=-25pp', lambda d: d <= -0.25),
               ('neutre |d|<15pp', lambda d: abs(d) < 0.15)]
    for lab, f in buckets:
        sub = [o for o in elig if f(o['swr'] - o['hwr'])]
        if len(sub) < 10:
            print(f"{lab:>22} {len(sub):>5}   (trop peu)")
            continue
        wr = np.mean([o['win'] for o in sub])
        mk = np.mean([o['p'] for o in sub])
        se = np.sqrt(max(mk * (1 - mk), 1e-9) / len(sub))
        z = (wr - mk) / se
        stake = len(sub)
        pay = sum(o['odds'] for o in sub if o['win'])
        roi = (pay - stake) / stake
        print(f"{lab:>22} {len(sub):>5} {wr:>8.4f} {mk:>9.4f} {wr-mk:>+7.4f} {z:>6.2f} {roi:>+9.4f}")
        out[lab] = dict(n=len(sub), wr=float(wr), mkt=float(mk), z=float(z), roi=float(roi))
    print(f"(eligibles: {len(elig)} team-obs)")
    return out

t3_tr = regime_table(obs_tr, 'TRAIN')
t3_oos = regime_table(obs_oos, 'OOS')

# ================================================================ TEST 4
print(); print(SEP)
print("TEST 4 - SERIES DE BUTS (3+ buts sur chacun des 2-3 derniers matchs)")
print(SEP)

def goals_table(obs, label):
    print(f"\n--- {label} ---")
    print(f"{'cond':>14} {'n':>5} {'P(3+ ce match)':>15} {'buts moyens':>12} {'WR reel':>8} "
          f"{'WR marche':>9}")
    out = {}
    base = [o for o in obs if o['played'] >= 2]
    for lab, sub in (('toutes (>=2 j)', base),
                     ('g3_last2', [o for o in base if o['g2']]),
                     ('g3_last3', [o for o in base if o['g3']])):
        if len(sub) < 10:
            print(f"{lab:>14} {len(sub):>5}   (trop peu)")
            continue
        p3 = np.mean([o['gf'] >= 3 for o in sub])
        gavg = np.mean([o['gf'] for o in sub])
        wr = np.mean([o['win'] for o in sub])
        mk = np.mean([o['p'] for o in sub])
        print(f"{lab:>14} {len(sub):>5} {p3:>15.4f} {gavg:>12.3f} {wr:>8.4f} {mk:>9.4f}")
        out[lab] = dict(n=len(sub), p3=float(p3), gavg=float(gavg), wr=float(wr), mkt=float(mk))
    return out

t4_tr = goals_table(obs_tr, 'TRAIN')
t4_oos = goals_table(obs_oos, 'OOS')

# ---- TEST 4b : controle par bande de cote (force de l'equipe) -------------
# courbe P(3+ | decile de proba devig) apprise sur TRAIN uniquement,
# puis attendu vs reel pour le sous-ensemble g3_last2 / g3_last3 en OOS.
print("\n--- TEST 4b : controle force (P(3+|decile p devig) appris sur train) ---")
ps_tr = np.array([o['p'] for o in obs_tr])
g3_tr = np.array([o['gf'] >= 3 for o in obs_tr], float)
qs = np.quantile(ps_tr, np.linspace(0, 1, 11))
qs[0], qs[-1] = -1, 2


def bin_idx(p):
    return int(np.clip(np.searchsorted(qs, p, side='right') - 1, 0, 9))


bin_rate = np.zeros(10)
for b in range(10):
    mask = (np.digitize(ps_tr, qs) - 1 == b)
    bin_rate[b] = g3_tr[mask].mean() if mask.sum() else g3_tr.mean()
print("P(3+|decile p) train :", np.round(bin_rate, 3).tolist())

t4b = {}
for lab, cond in (('g3_last2', 'g2'), ('g3_last3', 'g3')):
    sub = [o for o in obs_oos if o['played'] >= 2 and o[cond]]
    if len(sub) < 10:
        print(f"{lab}: n={len(sub)} (trop peu)")
        continue
    actual = np.mean([o['gf'] >= 3 for o in sub])
    expected = np.mean([bin_rate[bin_idx(o['p'])] for o in sub])
    se = np.sqrt(max(expected * (1 - expected), 1e-9) / len(sub))
    z = (actual - expected) / se
    print(f"{lab}: n={len(sub)} P(3+) reel={actual:.4f} attendu(controle force)={expected:.4f} "
          f"delta={actual-expected:+.4f} z={z:.2f}")
    t4b[lab] = dict(n=len(sub), actual=float(actual), expected=float(expected), z=float(z))

# ================================================================ TEST 5
print(); print(SEP)
print("TEST 5 - LOGISTIC cote-only vs cote+streaks (multinomial, scipy L-BFGS)")
print(SEP)

def build_xy(ms, with_streaks):
    X, y = [], []
    for m in ms:
        row = [np.log(m['ph'] / m['pd']), np.log(m['pa'] / m['pd'])]
        if with_streaks == 'mini':
            row += [1.0 if m['h_ws'] >= 5 else 0.0, 1.0 if m['a_ws'] >= 5 else 0.0,
                    1.0 if m['h_ls'] >= 3 else 0.0, 1.0 if m['a_ls'] >= 3 else 0.0,
                    1.0 if m['h_g3_last2'] else 0.0, 1.0 if m['a_g3_last2'] else 0.0]
        elif with_streaks:
            dwr_h = (m['h_season_wr'] - m['h_hist_wr']) if (
                m['h_played'] >= 5 and m['h_hist_n'] >= 50 and m['h_season_wr'] is not None
                and m['h_hist_wr'] is not None) else 0.0
            dwr_a = (m['a_season_wr'] - m['a_hist_wr']) if (
                m['a_played'] >= 5 and m['a_hist_n'] >= 50 and m['a_season_wr'] is not None
                and m['a_hist_wr'] is not None) else 0.0
            row += [min(m['h_ws'], 5), min(m['a_ws'], 5),
                    min(m['h_ls'], 5), min(m['a_ls'], 5),
                    dwr_h, dwr_a,
                    1.0 if m['h_g3_last2'] else 0.0, 1.0 if m['a_g3_last2'] else 0.0]
        X.append(row)
        y.append('HDA'.index(m['out']))
    return np.array(X, float), np.array(y, int)

def fit_softmax(X, y, lam=1e-3):
    n, d = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])
    K = 3

    def nll(w):
        W = w.reshape(K - 1, d + 1)
        Z = Xb @ W.T                       # n x 2
        Zfull = np.hstack([Z, np.zeros((n, 1))])
        Zfull -= Zfull.max(axis=1, keepdims=True)
        P = np.exp(Zfull)
        P /= P.sum(axis=1, keepdims=True)
        ll = -np.log(np.maximum(P[np.arange(n), y], 1e-12)).mean()
        return ll + lam * (w ** 2).sum()

    w0 = np.zeros((K - 1) * (d + 1))
    res = minimize(nll, w0, method='L-BFGS-B', options=dict(maxiter=2000))
    return res.x.reshape(K - 1, d + 1)

def predict_softmax(W, X):
    n = X.shape[0]
    Xb = np.hstack([np.ones((n, 1)), X])
    Z = Xb @ W.T
    Zfull = np.hstack([Z, np.zeros((n, 1))])
    Zfull -= Zfull.max(axis=1, keepdims=True)
    P = np.exp(Zfull)
    P /= P.sum(axis=1, keepdims=True)
    return P

def eval_model(with_streaks):
    Xtr, ytr = build_xy(train, with_streaks)
    Xte, yte = build_xy(oos, with_streaks)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    W = fit_softmax(Xtr, ytr)
    P = predict_softmax(W, Xte)
    acc_ = (P.argmax(1) == yte).mean()
    ll_ = -np.log(np.maximum(P[np.arange(len(yte)), yte], 1e-12)).mean()
    return float(acc_), float(ll_)

acc_base, ll_base = eval_model(False)
acc_full, ll_full = eval_model(True)
acc_mini, ll_mini = eval_model('mini')
print(f"logistic cote-only        : acc OOS={acc_base:.4f}  logloss OOS={ll_base:.4f}")
print(f"logistic cote+streaks(8f) : acc OOS={acc_full:.4f}  logloss OOS={ll_full:.4f}  "
      f"d_acc={acc_full-acc_base:+.4f} d_ll={ll_full-ll_base:+.4f}")
print(f"logistic cote+mini(6flags): acc OOS={acc_mini:.4f}  logloss OOS={ll_mini:.4f}  "
      f"d_acc={acc_mini-acc_base:+.4f} d_ll={ll_mini-ll_base:+.4f}")
print("(delta logloss negatif = mieux)")
print(f"(rappel favori devig OOS acc={base_acc_oos:.4f} ; logloss devig brut={ll_devig_oos:.4f})")

# dump resume json pour le rapport
summary = dict(
    n_train=len(train), n_oos=len(oos),
    base_acc_oos=base_acc_oos, roi_fav_oos=roi_fav_oos, ll_devig_oos=ll_devig_oos,
    t1_oos_ws=t1_oos_ws, t1_oos_ls=t1_oos_ls, t1_full_ws=t1_full_ws, t1_full_ls=t1_full_ls,
    fade_best=dict(n=best_fade[0], bet=best_fade[1], roi_train=best_fade[2],
                   roi_oos=roi_sel, base_oos=base_sel, n_oos=nb_sel),
    reb_best=dict(ls=best_reb[0], omin=best_reb[1], roi_train=best_reb[2],
                  roi_oos=r_sel, base_oos=rb_sel, n_oos=nb_rsel),
    t3_oos=t3_oos, t4_oos=t4_oos, t4b=t4b,
    logit=dict(acc_base=acc_base, ll_base=ll_base, acc_full=acc_full, ll_full=ll_full,
               acc_mini=acc_mini, ll_mini=ll_mini),
)
print("\nJSON_SUMMARY=" + json.dumps(summary, default=float))
print("FIN.")
