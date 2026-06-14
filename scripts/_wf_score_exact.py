# -*- coding: utf-8 -*-
"""
Walk-forward analysis: exact score (FT) + 2nd-half CS market.
Split temporel strict: 70% train / 30% OOS sur expected_start.
Tous les signaux (distributions, selections de scores, combos) sont
calcules sur le train uniquement, puis evalues sur l'OOS.

Sections:
  S1. Calibration marche 'Score exact' (devig implied vs freq reelle) + selection ROI train -> OOS
  S2. Calibration marche '2eme mi-tps - CS' (score 2e MT = FT - HT) -> OOS
  S3. Prediction top1/top3: pair_dist lisse bayesien vs profile_dist vs marche
  S4. Combos 2-3 scores par profil de cotes -> hit rate + ROI OOS
  S5. (info) P(FT | HT, profil) - live only
"""
import sys, json, math, collections
sys.path.insert(0, '.')
from scraper.config import load_settings
from sqlalchemy import create_engine, text

SEGMENTS = [(1, 3, 'DS'), (4, 12, 'MS_early'), (13, 25, 'MS_mid'), (26, 33, 'MS_late'), (34, 38, 'FS')]

def seg_of(rnd):
    for lo, hi, name in SEGMENTS:
        if lo <= rnd <= hi:
            return name
    return None

FAV_BUCKETS = [(0.0, 1.40, 'fav<=1.40'), (1.40, 1.70, 'fav1.40-1.70'), (1.70, 2.10, 'fav1.70-2.10'),
               (2.10, 2.60, 'fav2.10-2.60'), (2.60, 99.0, 'balanced/no-fav')]

def profile_of(oh, oa):
    fav_side = 'H' if oh <= oa else 'A'
    fav = min(oh, oa)
    for lo, hi, name in FAV_BUCKETS:
        if lo < fav <= hi:
            return f'{fav_side}|{name}' if name != 'balanced/no-fav' else 'BAL'
    return 'BAL'

def load_data():
    eng = create_engine(load_settings().db_url)
    with eng.connect() as c:
        rows = c.execute(text('''
        SELECT e.id, e.round_info, e.team_a, e.team_b, e.expected_start,
               o.odds_home, o.odds_draw, o.odds_away, o.extra_markets,
               r.score_a, r.score_b, r.ht_score_a, r.ht_score_b
        FROM events e
        JOIN results r ON r.event_id = e.id
        JOIN odds_snapshots o ON o.id = (SELECT MIN(id) FROM odds_snapshots WHERE event_id = e.id)
        WHERE e.round_info != '0' AND r.score_a IS NOT NULL AND r.ht_score_a IS NOT NULL
          AND o.odds_home IS NOT NULL AND o.extra_markets IS NOT NULL
        ORDER BY e.expected_start
        ''')).fetchall()
    data = []
    dropped = 0
    for (eid, rnd, ta, tb, start, oh, od, oa, em, sa, sb, ha, hb) in rows:
        try:
            rnd = int(rnd)
        except Exception:
            continue
        seg = seg_of(rnd)
        if seg is None:
            continue
        if sa < ha or sb < hb:           # FT corrompu (94 cas verifies vs goals_json)
            dropped += 1
            continue
        if isinstance(em, str):
            try:
                em = json.loads(em)
            except Exception:
                continue
        se = em.get('Score exact') or {}
        h2 = em.get('2ème mi-tps - CS') or {}
        if not se:
            continue
        data.append(dict(eid=eid, seg=seg, ta=ta, tb=tb, start=start,
                         oh=float(oh), od=float(od), oa=float(oa),
                         se={k: float(v) for k, v in se.items()},
                         h2={k: float(v) for k, v in h2.items()},
                         ft=f'{sa}-{sb}', h2s=f'{sa-ha}-{sb-hb}',
                         profile=profile_of(float(oh), float(oa))))
    print(f'[load] usable={len(data)}  dropped_bad_ft={dropped}')
    return data

def roi_line(bets):
    """bets: list of (won:0/1, cote). Returns n, wr, avg_cote, roi."""
    n = len(bets)
    if n == 0:
        return 0, 0.0, 0.0, 0.0
    wins = sum(w for w, _ in bets)
    wr = wins / n
    avg_c = sum(c for _, c in bets) / n
    roi = sum((c - 1) if w else -1 for w, c in bets) / n
    return n, wr, avg_c, roi

# ---------------------------------------------------------------- S1
def s1_market_calibration(train, oos):
    print('\n================ S1. Marche "Score exact" : devig vs reel ================')
    # per score key: train implied (devig) vs train freq, ROI train -> select -> OOS
    def collect(ds, cond=lambda m: True):
        imp = collections.defaultdict(list)   # score -> implied probs
        bets = collections.defaultdict(list)  # score -> (won, cote)
        for m in ds:
            if not cond(m):
                continue
            se = m['se']
            inv = {s: 1.0 / o for s, o in se.items() if o > 1.0}
            tot = sum(inv.values())
            for s, o in se.items():
                if o >= 100.0:        # cap operateur, jamais selectionne
                    continue
                imp[s].append(inv[s] / tot)
                bets[s].append((1 if m['ft'] == s else 0, o))
        return imp, bets

    imp_tr, bets_tr = collect(train)
    _, bets_oos = collect(oos)

    print(f'{"score":>6} {"n_tr":>5} {"impl%":>6} {"reel%":>6} {"ratio":>5} {"roi_tr":>7} | {"n_oos":>5} {"wr_oos":>6} {"cote":>5} {"roi_oos":>8}')
    rows = []
    for s in sorted(bets_tr, key=lambda s: -len(bets_tr[s])):
        n_tr, wr_tr, c_tr, r_tr = roi_line(bets_tr[s])
        if n_tr < 200:
            continue
        ip = sum(imp_tr[s]) / len(imp_tr[s])
        n_o, wr_o, c_o, r_o = roi_line(bets_oos.get(s, []))
        rows.append((s, n_tr, ip, wr_tr, r_tr, n_o, wr_o, c_o, r_o))
        print(f'{s:>6} {n_tr:>5} {ip*100:>6.2f} {wr_tr*100:>6.2f} {wr_tr/ip if ip else 0:>5.2f} {r_tr*100:>+6.1f}% | {n_o:>5} {wr_o*100:>6.2f} {c_o:>5.1f} {r_o*100:>+7.1f}%')

    # selection sur train: roi_tr >= +5% et >= 20 hits train
    sel = [r for r in rows if r[4] >= 0.05 and r[1] * r[3] >= 20]
    print('\n-- Selection train (roi_tr>=+5%, >=20 hits train) evaluee OOS :')
    agg = []
    for s, n_tr, ip, wr_tr, r_tr, n_o, wr_o, c_o, r_o in sel:
        print(f'   {s}: train roi {r_tr*100:+.1f}% (n={n_tr}) -> OOS n={n_o} wr={wr_o*100:.2f}% cote~{c_o:.1f} roi={r_o*100:+.1f}%')
        agg += bets_oos.get(s, [])
    if agg:
        n, wr, c, r = roi_line(agg)
        print(f'   PORTFOLIO OOS: n={n} wr={wr*100:.2f}% cote~{c:.1f} roi={r*100:+.1f}%')

    # conditionne par fav side (les scores asymetriques dependent du cote favori)
    print('\n-- Conditionne fav_side=H (oh<oa) :')
    imp_h, bets_h_tr = collect(train, lambda m: m['oh'] < m['oa'])
    _, bets_h_oos = collect(oos, lambda m: m['oh'] < m['oa'])
    sel_h = []
    for s in bets_h_tr:
        n_tr, wr_tr, c_tr, r_tr = roi_line(bets_h_tr[s])
        if n_tr >= 150 and r_tr >= 0.08 and n_tr * wr_tr >= 15:
            sel_h.append(s)
    aggh = []
    for s in sorted(sel_h):
        n_tr, wr_tr, _, r_tr = roi_line(bets_h_tr[s])
        n_o, wr_o, c_o, r_o = roi_line(bets_h_oos.get(s, []))
        print(f'   {s}: train roi {r_tr*100:+.1f}% (n={n_tr}) -> OOS n={n_o} wr={wr_o*100:.2f}% cote~{c_o:.1f} roi={r_o*100:+.1f}%')
        aggh += bets_h_oos.get(s, [])
    if aggh:
        n, wr, c, r = roi_line(aggh)
        print(f'   PORTFOLIO OOS (H fav): n={n} wr={wr*100:.2f}% cote~{c:.1f} roi={r*100:+.1f}%')

    print('\n-- Conditionne fav_side=A (oa<oh) :')
    _, bets_a_tr = collect(train, lambda m: m['oa'] < m['oh'])
    _, bets_a_oos = collect(oos, lambda m: m['oa'] < m['oh'])
    sel_a = []
    for s in bets_a_tr:
        n_tr, wr_tr, c_tr, r_tr = roi_line(bets_a_tr[s])
        if n_tr >= 150 and r_tr >= 0.08 and n_tr * wr_tr >= 15:
            sel_a.append(s)
    agga = []
    for s in sorted(sel_a):
        n_tr, wr_tr, _, r_tr = roi_line(bets_a_tr[s])
        n_o, wr_o, c_o, r_o = roi_line(bets_a_oos.get(s, []))
        print(f'   {s}: train roi {r_tr*100:+.1f}% (n={n_tr}) -> OOS n={n_o} wr={wr_o*100:.2f}% cote~{c_o:.1f} roi={r_o*100:+.1f}%')
        agga += bets_a_oos.get(s, [])
    if agga:
        n, wr, c, r = roi_line(agga)
        print(f'   PORTFOLIO OOS (A fav): n={n} wr={wr*100:.2f}% cote~{c:.1f} roi={r*100:+.1f}%')

# ---------------------------------------------------------------- S1b
def s1b_fav_oriented(train, oos):
    """Calibration en espace oriente favori: score 'f-d' = buts favori - buts outsider.
    Teste si l'operateur sous-cote systematiquement certains scores du favori."""
    print('\n================ S1b. Marche "Score exact" oriente FAVORI ================')
    def orient(m, s):
        a, b = s.split('-')
        if m['oh'] <= m['oa']:
            return f'{a}-{b}'
        return f'{b}-{a}'
    def collect(ds, max_fav=2.60):
        imp = collections.defaultdict(list)
        bets = collections.defaultdict(list)
        for m in ds:
            if min(m['oh'], m['oa']) > max_fav:
                continue
            se = m['se']
            inv = {s: 1.0 / o for s, o in se.items() if o > 1.0}
            tot = sum(inv.values())
            fto = orient(m, m['ft'])
            for s, o in se.items():
                if o >= 100.0:
                    continue
                so = orient(m, s)
                imp[so].append(inv[s] / tot)
                bets[so].append((1 if fto == so else 0, o))
        return imp, bets
    imp_tr, bets_tr = collect(train)
    _, bets_oos = collect(oos)
    print(f'{"sc_fav":>6} {"n_tr":>5} {"impl%":>6} {"reel%":>6} {"ratio":>5} {"roi_tr":>7} | {"n_oos":>5} {"wr_oos":>6} {"cote":>5} {"roi_oos":>8}')
    sel = []
    for s in sorted(bets_tr, key=lambda s: -len(bets_tr[s])):
        n_tr, wr_tr, c_tr, r_tr = roi_line(bets_tr[s])
        if n_tr < 250:
            continue
        ip = sum(imp_tr[s]) / len(imp_tr[s])
        n_o, wr_o, c_o, r_o = roi_line(bets_oos.get(s, []))
        flag = ''
        if r_tr >= 0.05 and n_tr * wr_tr >= 25:
            sel.append(s)
            flag = '  <== SELECT'
        print(f'{s:>6} {n_tr:>5} {ip*100:>6.2f} {wr_tr*100:>6.2f} {wr_tr/ip if ip else 0:>5.2f} {r_tr*100:>+6.1f}% | {n_o:>5} {wr_o*100:>6.2f} {c_o:>5.1f} {r_o*100:>+7.1f}%{flag}')
    agg = []
    for s in sel:
        agg += bets_oos.get(s, [])
    if agg:
        n, wr, c, r = roi_line(agg)
        hits = sum(w for w, _ in agg)
        print(f'  PORTFOLIO OOS oriente-fav {sel}: n={n} hits={hits} wr={wr*100:.2f}% cote~{c:.1f} roi={r*100:+.1f}%')

# ---------------------------------------------------------------- S2
def s2_secondhalf_market(train, oos):
    print('\n================ S2. Marche "2eme mi-tps - CS" : devig vs reel ================')
    def collect(ds):
        imp = collections.defaultdict(list)
        bets = collections.defaultdict(list)
        for m in ds:
            h2 = m['h2']
            if not h2:
                continue
            inv = {s: 1.0 / o for s, o in h2.items() if o > 1.0}
            tot = sum(inv.values())
            for s, o in h2.items():
                if o >= 100.0:
                    continue
                imp[s].append(inv[s] / tot)
                bets[s].append((1 if m['h2s'] == s else 0, o))
        return imp, bets
    imp_tr, bets_tr = collect(train)
    _, bets_oos = collect(oos)
    # part des scores 2MT hors grille du marche (tout le monde perd)
    keys = set()
    for m in train:
        keys |= set(m['h2'].keys())
    out_tr = sum(1 for m in train if m['h2'] and m['h2s'] not in m['h2']) / max(1, sum(1 for m in train if m['h2']))
    print(f'  grille marche 2MT: {sorted(keys)}  | part scores 2MT hors grille (train): {out_tr*100:.1f}%')
    print(f'{"score":>6} {"n_tr":>5} {"impl%":>6} {"reel%":>6} {"ratio":>5} {"roi_tr":>7} | {"n_oos":>5} {"wr_oos":>6} {"cote":>5} {"roi_oos":>8}')
    sel = []
    for s in sorted(bets_tr, key=lambda s: -len(bets_tr[s])):
        n_tr, wr_tr, c_tr, r_tr = roi_line(bets_tr[s])
        if n_tr < 200:
            continue
        ip = sum(imp_tr[s]) / len(imp_tr[s])
        n_o, wr_o, c_o, r_o = roi_line(bets_oos.get(s, []))
        print(f'{s:>6} {n_tr:>5} {ip*100:>6.2f} {wr_tr*100:>6.2f} {wr_tr/ip if ip else 0:>5.2f} {r_tr*100:>+6.1f}% | {n_o:>5} {wr_o*100:>6.2f} {c_o:>5.1f} {r_o*100:>+7.1f}%')
        if r_tr >= 0.05 and n_tr * wr_tr >= 20:
            sel.append(s)
    agg = []
    for s in sel:
        agg += bets_oos.get(s, [])
    if agg:
        n, wr, c, r = roi_line(agg)
        print(f'  PORTFOLIO OOS (sel train roi>=+5%): scores={sel} n={n} wr={wr*100:.2f}% cote~{c:.1f} roi={r*100:+.1f}%')
    else:
        print('  aucun score 2MT selectionne sur train.')

# ---------------------------------------------------------------- S3
def s3_bayes_pair(train, oos):
    print('\n================ S3. Top1/Top3 : pair lisse bayesien vs profil vs marche ================')
    # distributions train
    prof_dist = collections.defaultdict(collections.Counter)
    pair_dist = collections.defaultdict(collections.Counter)
    glob = collections.Counter()
    for m in train:
        prof_dist[m['profile']][m['ft']] += 1
        pair_dist[(m['ta'], m['tb'])][m['ft']] += 1
        glob[m['ft']] += 1
    glob_n = sum(glob.values())

    def dist_to_p(cnt):
        n = sum(cnt.values())
        return {s: c / n for s, c in cnt.items()}, n

    def blend(m, prior):
        pp, _ = dist_to_p(prof_dist[m['profile']]) if prof_dist[m['profile']] else dist_to_p(glob)
        pc = pair_dist.get((m['ta'], m['tb']))
        if not pc:
            return pp
        pd, npair = dist_to_p(pc)
        w = npair / (npair + prior)
        keys = set(pp) | set(pd)
        return {s: w * pd.get(s, 0.0) + (1 - w) * pp.get(s, 0.0) for s in keys}

    def evaluate(name, top_fn):
        hit1 = hit3 = n = 0
        cotes1, cotes3 = [], []
        for m in oos:
            tops = top_fn(m)
            if not tops:
                continue
            n += 1
            if m['ft'] == tops[0]:
                hit1 += 1
            if m['ft'] in tops[:3]:
                hit3 += 1
            cotes1.append(m['se'].get(tops[0], 100.0))
            cotes3.append(sum(m['se'].get(t, 100.0) for t in tops[:3]) / len(tops[:3]))
        print(f'  {name:<34} n_oos={n}  top1={hit1/n*100:.2f}% (cote~{sum(cotes1)/n:.1f})  top3={hit3/n*100:.2f}% (cote moy~{sum(cotes3)/n:.1f})')
        return hit1 / n, hit3 / n, n

    res = {}
    res['market'] = evaluate('Marche (cotes les + basses)',
                             lambda m: [s for s, _ in sorted(m['se'].items(), key=lambda kv: kv[1])[:3]])
    res['global'] = evaluate('Global train (sans condition)',
                             lambda m: [s for s, _ in glob.most_common(3)])
    res['profile'] = evaluate('Profil cotes (fav side+bucket)',
                              lambda m: [s for s, _ in (prof_dist[m['profile']] or glob).most_common(3)])
    res['pair_raw'] = evaluate('Paire brute (fallback profil)',
                               lambda m: [s for s, _ in (pair_dist.get((m['ta'], m['tb'])) or prof_dist[m['profile']] or glob).most_common(3)])
    for prior in (4, 8, 16):
        res[f'blend{prior}'] = evaluate(f'Blend bayesien prior={prior}',
                                        lambda m, pr=prior: [s for s, _ in sorted(blend(m, pr).items(), key=lambda kv: -kv[1])[:3]])

    # --- S3b: modeles ancres marche ---
    print('\n-- S3b. Modeles ancres marche (correction ratio train + blend profil) :')
    # ratio train par score x fav_side
    def fav_side(m):
        return 'H' if m['oh'] <= m['oa'] else 'A'
    imp_sum = collections.defaultdict(float); imp_n = collections.defaultdict(int)
    act_n = collections.defaultdict(int); tot_n = collections.defaultdict(int)
    for m in train:
        fs = fav_side(m)
        inv = {s: 1.0 / o for s, o in m['se'].items() if o > 1.0}
        tot = sum(inv.values())
        for s in m['se']:
            imp_sum[(fs, s)] += inv[s] / tot
            imp_n[(fs, s)] += 1
        act_n[(fs, m['ft'])] += 1
        tot_n[fs] += 1
    ratio = {}
    for k in imp_n:
        fs, s = k
        ip = imp_sum[k] / imp_n[k]
        freq = act_n.get(k, 0) / tot_n[fs]
        # lissage du ratio: shrink vers 1 selon nb de hits
        nh = act_n.get(k, 0)
        raw = freq / ip if ip > 0 else 1.0
        ratio[k] = (nh * raw + 30 * 1.0) / (nh + 30)

    def p_market(m):
        inv = {s: 1.0 / o for s, o in m['se'].items() if o > 1.0}
        tot = sum(inv.values())
        return {s: v / tot for s, v in inv.items()}

    def top_ratio(m):
        fs = fav_side(m)
        pm = p_market(m)
        pc = {s: p * ratio.get((fs, s), 1.0) for s, p in pm.items()}
        return [s for s, _ in sorted(pc.items(), key=lambda kv: -kv[1])[:3]]
    evaluate('Marche x ratio train (shrink 30)', top_ratio)

    # S3c: ratio en espace oriente favori (plus de donnees par cle de score)
    def orient(m, s):
        a, b = s.split('-')
        return f'{a}-{b}' if m['oh'] <= m['oa'] else f'{b}-{a}'
    imp_sum_o = collections.defaultdict(float); imp_n_o = collections.defaultdict(int)
    act_n_o = collections.defaultdict(int)
    n_tr_tot = 0
    for m in train:
        inv = {s: 1.0 / o for s, o in m['se'].items() if o > 1.0}
        tot = sum(inv.values())
        for s in m['se']:
            so = orient(m, s)
            imp_sum_o[so] += inv[s] / tot
            imp_n_o[so] += 1
        act_n_o[orient(m, m['ft'])] += 1
        n_tr_tot += 1
    ratio_o = {}
    for so in imp_n_o:
        ip_tot = imp_sum_o[so]          # somme des probas implicites = "hits attendus"
        nh = act_n_o.get(so, 0)
        raw = nh / ip_tot if ip_tot > 0 else 1.0
        ratio_o[so] = (nh * raw + 30 * 1.0) / (nh + 30)

    def top_ratio_o(m):
        pm = p_market(m)
        pc = {s: p * ratio_o.get(orient(m, s), 1.0) for s, p in pm.items()}
        return [s for s, _ in sorted(pc.items(), key=lambda kv: -kv[1])[:3]]
    evaluate('Marche x ratio ORIENTE-FAV', top_ratio_o)

    # S3d: ratio par profil (10 cellules), shrink 30, fallback ratio fav_side
    imp_sum_p = collections.defaultdict(float)
    act_n_p = collections.defaultdict(int)
    for m in train:
        inv = {s: 1.0 / o for s, o in m['se'].items() if o > 1.0}
        tot = sum(inv.values())
        for s in m['se']:
            imp_sum_p[(m['profile'], s)] += inv[s] / tot
        act_n_p[(m['profile'], m['ft'])] += 1
    ratio_p = {}
    for k in imp_sum_p:
        nh = act_n_p.get(k, 0)
        raw = nh / imp_sum_p[k] if imp_sum_p[k] > 0 else 1.0
        ratio_p[k] = (nh * raw + 30 * 1.0) / (nh + 30)

    def top_ratio_p(m):
        pm = p_market(m)
        fs = fav_side(m)
        pc = {s: p * ratio_p.get((m['profile'], s), ratio.get((fs, s), 1.0)) for s, p in pm.items()}
        return [s for s, _ in sorted(pc.items(), key=lambda kv: -kv[1])[:3]]
    evaluate('Marche x ratio PAR PROFIL', top_ratio_p)

    # blend marche + profil, poids a choisi sur split interne du train
    cut2 = int(len(train) * 0.8)
    tr_in2, tr_val2 = train[:cut2], train[cut2:]
    prof_in = collections.defaultdict(collections.Counter)
    for m in tr_in2:
        prof_in[m['profile']][m['ft']] += 1
    def mk_topblend(prof_d, a):
        def f(m):
            pm = p_market(m)
            base = prof_d[m['profile']]
            nb = sum(base.values())
            pp = {s: c / nb for s, c in base.items()} if nb else {}
            keys = set(pm) | set(pp)
            pb = {s: a * pm.get(s, 0.0) + (1 - a) * pp.get(s, 0.0) for s in keys}
            return [s for s, _ in sorted(pb.items(), key=lambda kv: -kv[1])[:3]]
        return f
    best_a, best_top1 = None, -1
    for a in (0.3, 0.5, 0.7, 0.85):
        fn = mk_topblend(prof_in, a)
        h1 = sum(1 for m in tr_val2 if m['ft'] == fn(m)[0]) / len(tr_val2)
        h3 = sum(1 for m in tr_val2 if m['ft'] in fn(m)) / len(tr_val2)
        print(f'   [train-val] a={a}: top1={h1*100:.2f}% top3={h3*100:.2f}%')
        if h1 > best_top1:
            best_a, best_top1 = a, h1
    evaluate(f'Blend marche({best_a})+profil OOS', mk_topblend(prof_dist, best_a))

    # value bets EV: p_blend * cote >= seuil, choisi sur train (split interne 80/20 du train)
    print('\n-- Value bets EV = p_blend(prior=8) x cote_SE, seuil choisi sur train interne :')
    cut = int(len(train) * 0.8)
    tr_in, tr_val = train[:cut], train[cut:]
    pd_in = collections.defaultdict(collections.Counter)
    pr_in = collections.defaultdict(collections.Counter)
    gl_in = collections.Counter()
    for m in tr_in:
        pd_in[(m['ta'], m['tb'])][m['ft']] += 1
        pr_in[m['profile']][m['ft']] += 1
        gl_in[m['ft']] += 1

    def blend_with(m, pair_d, prof_d, glob_c, prior=8):
        base = prof_d[m['profile']] or glob_c
        nb = sum(base.values())
        pp = {s: c / nb for s, c in base.items()}
        pc = pair_d.get((m['ta'], m['tb']))
        if not pc:
            return pp
        npair = sum(pc.values())
        pdd = {s: c / npair for s, c in pc.items()}
        w = npair / (npair + prior)
        return {s: w * pdd.get(s, 0.0) + (1 - w) * pp.get(s, 0.0) for s in set(pp) | set(pdd)}

    def ev_bets(ds, pair_d, prof_d, glob_c, thr, max_odds=30.0):
        bets = []
        for m in ds:
            p = blend_with(m, pair_d, prof_d, glob_c)
            for s, o in m['se'].items():
                if o >= max_odds or o >= 100.0:
                    continue
                if p.get(s, 0.0) * o >= thr:
                    bets.append((1 if m['ft'] == s else 0, o))
        return bets

    best_thr, best_roi = None, -9
    for thr in (1.1, 1.2, 1.3, 1.4, 1.5):
        b = ev_bets(tr_val, pd_in, pr_in, gl_in, thr)
        n, wr, c, r = roi_line(b)
        print(f'   [train-val] thr={thr}: n={n} wr={wr*100:.1f}% cote~{c:.1f} roi={r*100:+.1f}%')
        if n >= 100 and r > best_roi:
            best_thr, best_roi = thr, r
    if best_thr is not None:
        b = ev_bets(oos, pair_dist, prof_dist, glob, best_thr)
        n, wr, c, r = roi_line(b)
        print(f'   => OOS thr={best_thr}: n={n} wr={wr*100:.2f}% cote~{c:.1f} roi={r*100:+.1f}%')
    return res

# ---------------------------------------------------------------- S4
def s4_combos(train, oos):
    print('\n================ S4. Combos 2-3 scores par profil ================')
    prof_dist = collections.defaultdict(collections.Counter)
    for m in train:
        prof_dist[m['profile']][m['ft']] += 1

    def eval_combo(ds, prof, combo):
        k = len(combo)
        profits, hits, cotes, n = [], 0, [], 0
        for m in ds:
            if m['profile'] != prof:
                continue
            n += 1
            cs = [m['se'].get(s, 100.0) for s in combo]
            cotes.append(sum(cs) / k)
            if m['ft'] in combo:
                hits += 1
                profits.append(m['se'].get(m['ft'], 100.0) - k)
            else:
                profits.append(-k)
        if n == 0:
            return 0, 0.0, 0.0, 0.0
        return n, hits / n, sum(cotes) / n, sum(profits) / (k * n)

    print(f'{"profil":<18} {"k":>2} {"combo":<18} {"n_tr":>5} {"hit_tr":>6} {"roi_tr":>7} | {"n_oos":>5} {"hit_oos":>7} {"cote_moy":>8} {"roi_oos":>8}')
    out = []
    sel_bets_freq = []   # portefeuille des combos freq selectionnes par roi_tr>=0
    for prof in sorted(prof_dist):
        cnt = prof_dist[prof]
        n_tr = sum(cnt.values())
        if n_tr < 150:
            continue
        for k in (2, 3):
            combo = [s for s, _ in cnt.most_common(k)]
            hit_tr = sum(cnt[s] for s in combo) / n_tr
            n_t, h_t, c_t, r_t = eval_combo(train, prof, combo)
            n_o, hit_o, avg_c, roi_o = eval_combo(oos, prof, combo)
            if n_o == 0:
                continue
            flag = '  <== SELECT' if r_t >= 0.0 else ''
            print(f'{prof:<18} {k:>2} {"/".join(combo):<18} {n_tr:>5} {hit_tr*100:>5.1f}% {r_t*100:>+6.1f}% | {n_o:>5} {hit_o*100:>6.1f}% {avg_c:>8.1f} {roi_o*100:>+7.1f}%{flag}')
            out.append((prof, k, combo, n_o, hit_o, avg_c, roi_o, r_t))

    # combos optimises ROI train (au lieu de freq max): top-k par freq*cote moyenne train
    print('\n-- Combos optimises EV train (freq_train x cote moy train) :')
    se_avg = collections.defaultdict(lambda: collections.defaultdict(list))
    for m in train:
        for s, o in m['se'].items():
            if o < 100:
                se_avg[m['profile']][s].append(o)
    for prof in sorted(prof_dist):
        cnt = prof_dist[prof]
        n_tr = sum(cnt.values())
        if n_tr < 150:
            continue
        evs = []
        for s, c in cnt.items():
            os_ = se_avg[prof].get(s)
            if not os_ or len(os_) < 50:
                continue
            evs.append((s, (c / n_tr) * (sum(os_) / len(os_))))
        evs.sort(key=lambda kv: -kv[1])
        for k in (2, 3):
            combo = [s for s, _ in evs[:k]]
            if len(combo) < k:
                continue
            n_t, h_t, c_t, r_t = eval_combo(train, prof, combo)
            n_o, hit_o, avg_c, roi_o = eval_combo(oos, prof, combo)
            if n_o == 0:
                continue
            flag = '  <== SELECT' if r_t >= 0.10 else ''
            print(f'{prof:<18} {k:>2} {"/".join(combo):<18} {n_tr:>5} {h_t*100:>5.1f}% {r_t*100:>+6.1f}% | {n_o:>5} {hit_o*100:>6.1f}% {avg_c:>8.1f} {roi_o*100:>+7.1f}%{flag}')
    return out

# ---------------------------------------------------------------- S5
def s5_ht_conditional(train, oos):
    print('\n================ S5. (info live) P(FT | HT, fav_side) ================')
    tbl = collections.defaultdict(collections.Counter)
    for m in train:
        ht = m['ft']  # placeholder replaced below
    tbl = collections.defaultdict(collections.Counter)
    for m in train:
        fa, fb = map(int, m['ft'].split('-'))
        h2a, h2b = map(int, m['h2s'].split('-'))
        ha, hb = fa - h2a, fb - h2b
        fav = 'H' if m['oh'] <= m['oa'] else 'A'
        tbl[(f'{ha}-{hb}', fav)][m['ft']] += 1
    hit1 = hit3 = n = 0
    by_ht = collections.defaultdict(lambda: [0, 0])
    for m in oos:
        fa, fb = map(int, m['ft'].split('-'))
        h2a, h2b = map(int, m['h2s'].split('-'))
        ht = f'{fa-h2a}-{fb-h2b}'
        fav = 'H' if m['oh'] <= m['oa'] else 'A'
        cnt = tbl.get((ht, fav))
        if not cnt or sum(cnt.values()) < 30:
            continue
        n += 1
        tops = [s for s, _ in cnt.most_common(3)]
        if m['ft'] == tops[0]:
            hit1 += 1
            by_ht[ht][0] += 1
        if m['ft'] in tops:
            hit3 += 1
        by_ht[ht][1] += 1
    if n:
        print(f'  OOS (HT connus, n_train(ht,fav)>=30): n={n} top1={hit1/n*100:.2f}% top3={hit3/n*100:.2f}%')
        for ht, (h, tot) in sorted(by_ht.items(), key=lambda kv: -kv[1][1])[:8]:
            print(f'    HT {ht}: top1 {h}/{tot} = {h/tot*100:.1f}%')

# ---------------------------------------------------------------- S6
def s6_stability(train, oos):
    """Stabilite des 3 signaux positifs: split OOS en 2 moities + hits bruts."""
    print('\n================ S6. Stabilite des signaux positifs (OOS coupe en 2) ================')
    def orient(m, s):
        a, b = s.split('-')
        return f'{a}-{b}' if m['oh'] <= m['oa'] else f'{b}-{a}'

    def bets_for(ds, mode):
        out = []
        for m in ds:
            for s, o in m['se'].items():
                if o >= 100.0:
                    continue
                if mode == 'global_42_04' and s in ('4-2', '0-4'):
                    out.append((1 if m['ft'] == s else 0, o, m['start']))
                elif mode == 'fav42' and min(m['oh'], m['oa']) <= 2.60 and orient(m, s) == '4-2':
                    out.append((1 if m['ft'] == s else 0, o, m['start']))
                elif mode == 'afav_04' and m['oa'] < m['oh'] and s == '0-4':
                    out.append((1 if m['ft'] == s else 0, o, m['start']))
        return out

    for mode, label in (('global_42_04', 'S1 global {4-2, 0-4}'),
                        ('fav42', 'S1b oriente-fav 4-2 (fav<=2.60)'),
                        ('afav_04', '0-4 quand A favori')):
        b_tr = [(w, c) for w, c, _ in bets_for(train, mode)]
        b_oos = bets_for(oos, mode)
        b_oos_sorted = sorted(b_oos, key=lambda x: x[2])
        half = len(b_oos_sorted) // 2
        h1 = [(w, c) for w, c, _ in b_oos_sorted[:half]]
        h2 = [(w, c) for w, c, _ in b_oos_sorted[half:]]
        n_t, wr_t, c_t, r_t = roi_line(b_tr)
        n1, wr1, c1, r1 = roi_line(h1)
        n2, wr2, c2, r2 = roi_line(h2)
        n_o, wr_o, c_o, r_o = roi_line([(w, c) for w, c, _ in b_oos])
        hits_o = sum(w for w, c, _ in b_oos)
        # ecart-type approx du ROI OOS
        import statistics
        prof = [(c - 1) if w else -1.0 for w, c, _ in b_oos]
        se = statistics.pstdev(prof) / max(1, len(prof)) ** 0.5
        print(f'  {label}:')
        print(f'    train: n={n_t} roi={r_t*100:+.1f}% | OOS total: n={n_o} hits={hits_o} roi={r_o*100:+.1f}% (se~{se*100:.0f}pp)')
        print(f'    OOS 1ere moitie: n={n1} roi={r1*100:+.1f}%  | OOS 2eme moitie: n={n2} roi={r2*100:+.1f}%')

# ----------------------------------------------------------------
def main():
    data = load_data()
    cut = int(len(data) * 0.7)
    train, oos = data[:cut], data[cut:]
    print(f'[split] train={len(train)} ({train[0]["start"]} -> {train[-1]["start"]})')
    print(f'        oos  ={len(oos)} ({oos[0]["start"]} -> {oos[-1]["start"]})')
    s1_market_calibration(train, oos)
    s1b_fav_oriented(train, oos)
    s2_secondhalf_market(train, oos)
    s3_bayes_pair(train, oos)
    s4_combos(train, oos)
    s5_ht_conditional(train, oos)
    s6_stability(train, oos)

if __name__ == '__main__':
    main()
