# -*- coding: utf-8 -*-
"""WF3 - MICROSTRUCTURE DU PRICING : inverser cote = f(proba).

S1. Data prep (dedupe, opening odds, resolution des marchés)
S2. Grille des cotes par marché (pas, décimales, valeurs interdites, min/max)
S3. Stabilité des snapshots (les cotes bougent-elles ?)
S4. Calibration 1X2 : fit proportionnel / power / Shin / additif (MLE + chi2)
S5. Fonction de marge universelle cross-marchés g(1/cote)
S6. CAPS : scores jamais vus, cote 100.0 (placeholders ?), bornes par marché
S7. Extrêmes : upsets cote>=8, scores 5-x/6-x
S8. Walk-forward des candidats edge
"""
import sys, json, math
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from scipy import stats
from scipy.optimize import brentq, minimize_scalar
from sqlalchemy import create_engine, text
from scraper.config import load_settings

pd.set_option('display.width', 200)
eng = create_engine(load_settings().db_url)

# ---------------------------------------------------------------- S1 DATA
with eng.connect() as c:
    ev = pd.read_sql(text("""
        SELECT e.id, e.round_info, e.team_a, e.team_b, e.expected_start,
               r.score_a, r.score_b, r.ht_score_a, r.ht_score_b, r.goals_json
        FROM events e JOIN results r ON r.event_id = e.id
        WHERE e.round_info != '0' AND r.score_a IS NOT NULL
    """), c)
    op = pd.read_sql(text("""
        SELECT o.event_id, o.odds_home, o.odds_draw, o.odds_away, o.extra_markets
        FROM odds_snapshots o
        JOIN (SELECT event_id, MIN(id) mid FROM odds_snapshots GROUP BY event_id) m
          ON m.mid = o.id
    """), c)
    snap_var = pd.read_sql(text("""
        SELECT event_id, COUNT(*) n, COUNT(DISTINCT odds_home) dh,
               COUNT(DISTINCT odds_draw) dd, COUNT(DISTINCT odds_away) da
        FROM odds_snapshots GROUP BY event_id HAVING COUNT(*) > 1
    """), c)

ev = ev.sort_values('id').drop_duplicates(['team_a', 'team_b', 'expected_start'], keep='first')
df = ev.merge(op, left_on='id', right_on='event_id', how='inner')
df = df.dropna(subset=['odds_home', 'odds_draw', 'odds_away'])
df = df.sort_values('expected_start').reset_index(drop=True)
print(f"S1: matchs finis dédupliqués avec cotes d'ouverture: {len(df)}")

def parse_em(x):
    if x is None: return {}
    return json.loads(x) if isinstance(x, str) else x
df['em'] = df['extra_markets'].apply(parse_em)

# verif mapping minute -> mi-temps
ok, tot = 0, 0
for _, r in df.head(800).iterrows():
    if pd.isna(r['ht_score_a']) or not r['goals_json']: continue
    g = json.loads(r['goals_json']) if isinstance(r['goals_json'], str) else r['goals_json']
    if not isinstance(g, list): continue
    ht_count = sum(1 for x in g if x['minute'] <= 45)
    tot += 1
    if ht_count == r['ht_score_a'] + r['ht_score_b']: ok += 1
print(f"S1: goals_json minute<=45 == ht total: {ok}/{tot}")
mins = []
for _, r in df.iterrows():
    if r['goals_json'] is None or (isinstance(r['goals_json'], float)): continue
    g = json.loads(r['goals_json']) if isinstance(r['goals_json'], str) else r['goals_json']
    if not isinstance(g, list): continue
    mins += [x['minute'] for x in g]
print(f"S1: minutes des buts: min={min(mins)} max={max(mins)} n={len(mins)}")

# ------------------------------------------------------- S2 GRILLE COTES
print("\n" + "="*80 + "\nS2: GRILLE DES COTES PAR MARCHE")
market_vals = defaultdict(list)
for _, r in df.iterrows():
    market_vals['1X2'] += [r['odds_home'], r['odds_draw'], r['odds_away']]
    for mk, sels in r['em'].items():
        if isinstance(sels, dict):
            market_vals[mk] += [v for v in sels.values() if isinstance(v, (int, float))]

allv = []
for mk, vals in sorted(market_vals.items(), key=lambda kv: -len(kv[1])):
    v = np.array(vals, dtype=float)
    allv.append(v)
    dv = np.unique(np.round(v, 2))
    n100 = (v == 100.0).mean()
    print(f"  {mk[:34]:36s} n={len(v):7d} distinct={len(dv):4d} min={v.min():6.2f} "
          f"max={v.max():7.2f} %@100={100*n100:5.1f}%")
allv = np.concatenate(allv)
allv = allv[allv < 100.0]
cents = np.round(allv * 100).astype(int) % 10
cc = np.bincount(cents, minlength=10)
chi2, p = stats.chisquare(cc)
print(f"  Dernier digit (centimes) des cotes <100: {list(cc)}  chi2={chi2:.1f} p={p:.2e}")
c5 = np.round(allv * 100).astype(int) % 5
cc5 = np.bincount(c5, minlength=5)
chi2b, pb = stats.chisquare(cc5)
print(f"  Modulo 5 centimes: {list(cc5)}  chi2={chi2b:.1f} p={pb:.2e}")
# couverture grille dans [1.00, 3.00] (cotes 1X2 uniquement)
v1 = np.unique(np.round(np.array(market_vals['1X2']), 2))
band = v1[(v1 >= 1.0) & (v1 < 3.0)]
print(f"  1X2 valeurs distinctes dans [1,3): {len(band)} / 200 possibles au pas 0.01")
missing = sorted(set(np.round(np.arange(1.01, 3.0, 0.01), 2)) - set(band))
print(f"  valeurs absentes [1.01,3.00): {missing[:25]}{'...' if len(missing)>25 else ''}")
gmin = allv.min()
print(f"  Cote minimale globale observée: {gmin}")

# égalité cross-marchés du même p sous-jacent : 0-0 CS == 'Pas de but' (minute/FTTS)
eq, neq, nn = 0, 0, 0
for _, r in df.iterrows():
    em = r['em']
    try:
        cs00 = em.get('Score exact', {}).get('0-0')
        m_nb = em.get('Minute du premier but', {}).get('Pas de but')
        f_nb = em.get('FTTS', {}).get('Pas de but')
        if cs00 and m_nb and f_nb:
            nn += 1
            if cs00 == m_nb == f_nb: eq += 1
            else: neq += 1
    except AttributeError:
        pass
print(f"  Identité P(0-0): CS '0-0' == Minute 'Pas de but' == FTTS 'Pas de but' : {eq}/{nn} égaux")

# ------------------------------------------------------ S3 SNAPSHOTS
print("\n" + "="*80 + "\nS3: STABILITE DES SNAPSHOTS")
if len(snap_var):
    moved = ((snap_var['dh'] > 1) | (snap_var['dd'] > 1) | (snap_var['da'] > 1)).mean()
    print(f"  events multi-snapshots: {len(snap_var)}; % avec cotes 1X2 qui bougent: {100*moved:.2f}%")
else:
    print("  aucun event multi-snapshot")

# ------------------------------------------------- S4 CALIBRATION 1X2
print("\n" + "="*80 + "\nS4: CALIBRATION 1X2 — INVERSION cote -> proba")
O = df[['odds_home', 'odds_draw', 'odds_away']].values.astype(float)
inv = 1.0 / O
over = inv.sum(axis=1)
print(f"  Overround 1X2: mean={over.mean():.4f} std={over.std():.4f} "
      f"min={over.min():.4f} max={over.max():.4f}")
# overround vs force du favori
fav = inv.max(axis=1)
for lo, hi in [(0.33, 0.45), (0.45, 0.55), (0.55, 0.65), (0.65, 0.75), (0.75, 0.90)]:
    m = (fav >= lo) & (fav < hi)
    if m.sum() > 30:
        print(f"    fav_implied [{lo:.2f},{hi:.2f}): n={m.sum():5d} overround={over[m].mean():.4f}")

y = np.zeros((len(df), 3))
res_idx = np.where(df['score_a'] > df['score_b'], 0,
                   np.where(df['score_a'] == df['score_b'], 1, 2))
y[np.arange(len(df)), res_idx] = 1

def ll(P):
    P = np.clip(P, 1e-9, 1)
    return -np.mean(np.log(P[np.arange(len(P)), res_idx]))

P_prop = inv / over[:, None]

def power_norm(row):
    f = lambda k: (row ** k).sum() - 1.0
    k = brentq(f, 0.5, 3.0)
    return row ** k
P_pow = np.array([power_norm(r) for r in inv])

def shin_norm(row):
    B = row.sum()
    def f(z):
        p = (np.sqrt(z*z + 4*(1-z)*row*row/B) - z) / (2*(1-z))
        return p.sum() - 1.0
    try:
        z = brentq(f, 0.0, 0.2)
    except ValueError:
        z = 0.0
    return (np.sqrt(z*z + 4*(1-z)*row*row/B) - z) / (2*(1-z))
P_shin = np.array([shin_norm(r) for r in inv])
P_add = inv - (over[:, None] - 1) / 3.0

print(f"  Log-loss  proportionnel={ll(P_prop):.5f}  power={ll(P_pow):.5f}  "
      f"shin={ll(P_shin):.5f}  additif={ll(P_add):.5f}")

# calibration binned: freq réelle vs proba modèle (proportionnel)
print("  Calibration (proportionnel), bins de proba:")
flat_p = P_prop.ravel(); flat_y = y.ravel(); flat_inv = inv.ravel()
bins = [0, .05, .10, .15, .20, .25, .30, .35, .40, .45, .50, .60, .70, .85]
chi2_tot, dfree = 0.0, 0
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (flat_p >= lo) & (flat_p < hi)
    n = m.sum()
    if n < 50: continue
    obs = flat_y[m].sum(); exp = flat_p[m].sum()
    freq = obs / n; pm = flat_p[m].mean()
    z = (obs - exp) / math.sqrt(max(exp * (1 - exp / n), 1e-9))
    chi2_tot += z*z; dfree += 1
    print(f"    p[{lo:.2f},{hi:.2f}): n={n:5d} p_model={pm:.4f} freq={freq:.4f} z={z:+.2f}")
print(f"  Chi2 calibration prop: {chi2_tot:.1f} (df~{dfree}) p={1-stats.chi2.cdf(chi2_tot, dfree):.3f}")

# fit direct g: freq = c * (1/cote) -> WLS through origin sur bins
xb, yb, nb = [], [], []
qs = np.quantile(flat_inv, np.linspace(0, 1, 26))
for lo, hi in zip(qs[:-1], qs[1:]):
    m = (flat_inv >= lo) & (flat_inv < hi)
    if m.sum() < 30: continue
    xb.append(flat_inv[m].mean()); yb.append(flat_y[m].mean()); nb.append(m.sum())
xb, yb, nb = np.array(xb), np.array(yb), np.array(nb)
c_hat = np.sum(nb * xb * yb) / np.sum(nb * xb * xb)
# power fit: log(freq) = g*log(inv) + b
lf = np.log(np.clip(yb, 1e-6, 1)); li = np.log(xb)
A = np.vstack([li, np.ones_like(li)]).T
W = np.diag(nb)
coef = np.linalg.solve(A.T @ W @ A, A.T @ W @ lf)
print(f"  Fit freq = c*(1/cote):       c = {c_hat:.4f}  (1/c = marge {100*(1/c_hat-1):.2f}%)")
print(f"  Fit freq = exp(b)*(1/cote)^g: g = {coef[0]:.4f}, exp(b) = {math.exp(coef[1]):.4f}")
resid_lin = yb - c_hat * xb
print(f"  Résidus fit linéaire par bin (favori->longshot):")
for i in range(0, len(xb), 5):
    print(f"    inv={xb[i]:.3f} freq={yb[i]:.4f} pred={c_hat*xb[i]:.4f} resid={resid_lin[i]:+.4f} n={nb[i]}")

# --------------------------------------- S5 FONCTION UNIVERSELLE CROSS-MARCHES
print("\n" + "="*80 + "\nS5: g(1/cote) UNIVERSELLE — toutes sélections résolues")

def resolve_rows(r):
    """yield (market, selection, odds, won) pour les marchés résolubles."""
    a, b = int(r['score_a']), int(r['score_b'])
    ha = r['ht_score_a']; hb = r['ht_score_b']
    ht_ok = not (pd.isna(ha) or pd.isna(hb))
    if ht_ok: ha, hb = int(ha), int(hb)
    tot = a + b
    em = r['em']
    g = []
    gj = r['goals_json']
    if gj is not None and not isinstance(gj, float):
        g = json.loads(gj) if isinstance(gj, str) else gj
        if not isinstance(g, list): g = []
        g = sorted(g, key=lambda x: x['minute'])
    out = []
    def add(mk, key, won):
        sels = em.get(mk)
        if isinstance(sels, dict):
            for sel, o in sels.items():
                if isinstance(o, (int, float)):
                    out.append((mk, sel, float(o), 1 if sel == key else 0))
    res = '1' if a > b else ('X' if a == b else '2')
    add('1X2_em' if '1X2' in em else None, None, None) if False else None
    out.append(('1X2', '1', float(r['odds_home']), 1 if res == '1' else 0))
    out.append(('1X2', 'X', float(r['odds_draw']), 1 if res == 'X' else 0))
    out.append(('1X2', '2', float(r['odds_away']), 1 if res == '2' else 0))
    add('Score exact', f"{a}-{b}", None) or None
    # add() marque won par comparaison sel==key :
    # (réécrit proprement ci-dessous)
    return out

# version propre du resolver
def resolve(r):
    a, b = int(r['score_a']), int(r['score_b'])
    ha, hb = r['ht_score_a'], r['ht_score_b']
    ht_ok = not (pd.isna(ha) or pd.isna(hb))
    if ht_ok: ha, hb = int(ha), int(hb)
    tot = a + b
    em = r['em']
    g = []
    gj = r['goals_json']
    if gj is not None and not isinstance(gj, float):
        g = json.loads(gj) if isinstance(gj, str) else gj
        if not isinstance(g, list): g = []
        g = sorted(g, key=lambda x: x['minute'])
    out = []
    def market(mk, winner_keys):
        sels = em.get(mk)
        if isinstance(sels, dict):
            for sel, o in sels.items():
                if isinstance(o, (int, float)):
                    out.append((mk, sel, float(o), 1 if sel in winner_keys else 0))
    res = '1' if a > b else ('X' if a == b else '2')
    out.append(('1X2', '1', float(r['odds_home']), int(res == '1')))
    out.append(('1X2', 'X', float(r['odds_draw']), int(res == 'X')))
    out.append(('1X2', '2', float(r['odds_away']), int(res == '2')))
    market('Score exact', {f"{a}-{b}"})
    market('Double Chance', {k for k in ['1X', 'X2', '12'] if res in k})
    market('+/-', {'> 3.5' if tot > 3.5 else '< 3.5'})
    market('Total de buts', {str(tot)})
    market('G/NG', {'Oui' if (a > 0 and b > 0) else 'Non'})
    market('Pair/Impair', {'Pair' if tot % 2 == 0 else 'Impair'})
    market('Total equipe domicile', {'> 3.5' if a > 3.5 else '< 3.5'})
    market('Total equipe extérieur', {'> 3.5' if b > 3.5 else '< 3.5'})
    market('G/NG equipe domicile', {'Oui' if a > 0 else 'Non'})
    market('G/NG equipe extérieur', {'Oui' if b > 0 else 'Non'})
    market('1X2 & Total', {f"{res} / {'> 3.5' if tot > 3.5 else '< 3.5'}"})
    mb = set()
    if tot in (0, 1, 2): mb.add('Le total de buts est de 0, 1 ou 2')
    if tot in (1, 2, 3): mb.add('Le total de buts est de 1, 2 ou 3')
    if tot in (2, 3, 4): mb.add('Le total de buts est de 2, 3 ou 4')
    if tot > 4: mb.add('Le total de buts est supérieur à 4')
    market('Multi-Buts', mb)
    if g:
        fm = g[0]['minute']
        buck = ('1-15' if fm <= 15 else '16-30' if fm <= 30 else '31-45' if fm <= 45
                else '46-60' if fm <= 60 else '61-75' if fm <= 75 else '76-90')
        market('Minute du premier but', {buck})
        market('FTTS', {'1' if g[0]['team'] == 'Home' else '2'})
    elif tot == 0:
        market('Minute du premier but', {'Pas de but'})
        market('FTTS', {'Pas de but'})
    if ht_ok:
        hres = '1' if ha > hb else ('X' if ha == hb else '2')
        market('Mi-tps 1X2', {hres})
        market('Mi-tps DC', {k for k in ['1X', 'X2', '12'] if hres in k})
        market('Mi-tps CS', {f"{ha}-{hb}"})
        market('HT/FT', {f"{hres}/{res}"})
        market('2ème mi-tps - CS', {f"{a-ha}-{b-hb}"})
        market('Les deux équipes marquent / 1ère mi temps', {'Oui' if (ha > 0 and hb > 0) else 'Non'})
    return out

rows = []
for _, r in df.iterrows():
    rows += [(r['id'], r['expected_start']) + t for t in resolve(r)]
S = pd.DataFrame(rows, columns=['eid', 'start', 'market', 'sel', 'odds', 'won'])
print(f"  sélections résolues: {len(S)} sur {S['eid'].nunique()} matchs, {S['market'].nunique()} marchés")

# sanity: chaque marché a-t-il ~1 gagnant par match ?
chk = S.groupby(['market', 'eid'])['won'].sum().groupby('market').mean()
print("  gagnants/match par marché (doit être ~1, DC/Multi ~2, CS<1 si score hors grille):")
for mk, v in chk.items():
    print(f"    {mk[:40]:42s} {v:.3f}")

Snc = S[S['odds'] < 100.0]  # exclure les caps du fit
qs = np.quantile(Snc['odds'].apply(lambda o: 1/o), np.linspace(0, 1, 31))
xinv = 1 / Snc['odds'].values
ywon = Snc['won'].values
print("  Bins globaux (toutes sélections <100):")
xb2, yb2, nb2 = [], [], []
for lo, hi in zip(qs[:-1], qs[1:]):
    m = (xinv >= lo) & (xinv < hi)
    if m.sum() < 100: continue
    xb2.append(xinv[m].mean()); yb2.append(ywon[m].mean()); nb2.append(m.sum())
xb2, yb2, nb2 = np.array(xb2), np.array(yb2), np.array(nb2)
c2 = np.sum(nb2 * xb2 * yb2) / np.sum(nb2 * xb2 * xb2)
print(f"  Fit global freq = c*(1/cote): c = {c2:.4f} (marge {100*(1/c2-1):.2f}%)")
for i in range(len(xb2)):
    pred = c2 * xb2[i]
    se = math.sqrt(max(yb2[i]*(1-yb2[i])/nb2[i], 1e-9))
    flag = ' ***' if abs(yb2[i]-pred) > 3*se else ''
    print(f"    1/o={xb2[i]:.4f} (o~{1/xb2[i]:6.2f}) freq={yb2[i]:.4f} pred={pred:.4f} "
          f"n={nb2[i]:6d}{flag}")

# c par marché
print("  c estimé par marché (freq moyenne / implied moyenne):")
for mk, grp in Snc.groupby('market'):
    if len(grp) < 500: continue
    iv = 1/grp['odds'].values
    cm = grp['won'].mean() / iv.mean()
    print(f"    {mk[:40]:42s} n={len(grp):6d} E[1/o]={iv.mean():.4f} freq={grp['won'].mean():.4f} c={cm:.4f}")

# ---------------------------------------------------------- S6 CAPS
print("\n" + "="*80 + "\nS6: CAPS DU MOTEUR")
sc = Counter((int(a), int(b)) for a, b in zip(df['score_a'], df['score_b']))
print(f"  Max buts home={df['score_a'].max()} away={df['score_b'].max()} "
      f"total={int((df['score_a']+df['score_b']).max())}")
print("  Distribution des totaux:", dict(sorted(Counter((df['score_a']+df['score_b']).astype(int)).items())))
priced_lines = set()
for _, r in df.iterrows():
    se = r['em'].get('Score exact')
    if isinstance(se, dict): priced_lines |= set(se.keys())
print(f"  Lignes pricées dans Score exact: {sorted(priced_lines)}")
never = [l for l in sorted(priced_lines) if tuple(map(int, l.split('-'))) not in sc]
print(f"  Lignes pricées JAMAIS réalisées: {never}")
unpriced = [s for s in sc if f'{s[0]}-{s[1]}' not in priced_lines]
print(f"  Scores réalisés JAMAIS pricés: {unpriced}")

cap = S[(S['market'] == 'Score exact') & (S['odds'] == 100.0)]
n, w = len(cap), cap['won'].sum()
lo_ci, hi_ci = stats.beta.ppf([0.025, 0.975], w + .5, n - w + .5) if n else (0, 0)
print(f"  Score exact @100.0: n={n} wins={w} freq={w/max(n,1):.5f} "
      f"CI95=[{lo_ci:.5f},{hi_ci:.5f}]  EV flat = {100*w/max(n,1)-1:+.3f} u")
print(f"    breakeven=0.01 ; binom test p={stats.binomtest(int(w), n, 0.01).pvalue:.3g}" if n else "")
print("  Détail par ligne @100 (n>=200):")
for line, grp in cap.groupby('sel'):
    if len(grp) < 200: continue
    wl = grp['won'].sum()
    print(f"    {line}: n={len(grp):5d} wins={wl:3d} freq={wl/len(grp):.5f} EV={100*wl/len(grp)-1:+.2f}")

cap_any = S[S['odds'] == 100.0]
print("  Sélections @100.0 par marché:")
for mk, grp in cap_any.groupby('market'):
    wl = grp['won'].sum()
    print(f"    {mk[:40]:42s} n={len(grp):6d} wins={wl:4d} freq={wl/len(grp):.5f} EV={100*wl/len(grp)-1:+.2f}")

# cote max non-cap par marché + bornes
print("  Bornes de cotes hors cap par marché:")
for mk, grp in Snc.groupby('market'):
    print(f"    {mk[:40]:42s} min={grp['odds'].min():5.2f} max={grp['odds'].max():6.2f}")

# ---------------------------------------------------------- S7 EXTREMES
print("\n" + "="*80 + "\nS7: EVENEMENTS EXTREMES")
x12 = S[S['market'] == '1X2']
for lo, hi in [(8, 12), (12, 20), (20, 50), (8, 999)]:
    m = (x12['odds'] >= lo) & (x12['odds'] < hi)
    g2 = x12[m]
    if not len(g2): continue
    w2 = g2['won'].sum(); iv = (1/g2['odds']).mean()
    roi = (g2['won'] * g2['odds']).sum() / len(g2) - 1
    pv = stats.binomtest(int(w2), len(g2), iv).pvalue
    print(f"  1X2 cote[{lo},{hi}): n={len(g2):5d} wins={w2:4d} freq={w2/len(g2):.4f} "
          f"implied={iv:.4f} ROI={100*roi:+.1f}% binom p={pv:.3f}")
big = S[(S['market'] == 'Score exact') & (S['sel'].str.match(r'^[56]-|-[56]$')) & (S['odds'] < 100)]
print(f"  Lignes 5-x/6-x pricées <100: n={len(big)} wins={big['won'].sum()} "
      f"odds moy={big['odds'].mean():.1f}" if len(big) else "  Aucune ligne 5-x/6-x pricée <100")
n5 = sum(v for k, v in sc.items() if max(k) >= 5)
print(f"  Matchs avec une équipe à 5+ buts: {n5}/{len(df)} = {n5/len(df):.5f}")
# la plus grosse cote jamais gagnante par marché
print("  Cote max gagnante par marché:")
for mk, grp in S[S['won'] == 1].groupby('market'):
    print(f"    {mk[:40]:42s} {grp['odds'].max():7.2f}")

# ---------------------------------------------------------- S8 WALK-FORWARD
print("\n" + "="*80 + "\nS8: WALK-FORWARD DES CANDIDATS")
S = S.sort_values('start')
S['start'] = pd.to_datetime(S['start'])
cut = S['start'].quantile(0.7)
train, test = S[S['start'] <= cut], S[S['start'] > cut]
print(f"  split: train {train['eid'].nunique()} matchs / OOS {test['eid'].nunique()} matchs (cut={cut})")

def wf(name, mask_fn):
    tr, te = train[mask_fn(train)], test[mask_fn(test)]
    if not len(tr) or not len(te):
        print(f"  {name}: vide"); return None
    roi_tr = (tr['won'] * tr['odds']).sum() / len(tr) - 1
    roi_te = (te['won'] * te['odds']).sum() / len(te) - 1
    wr = te['won'].mean()
    print(f"  {name}: train n={len(tr)} ROI={100*roi_tr:+.1f}% | OOS n={len(te)} "
          f"ROI={100*roi_te:+.1f}% WR={100*wr:.2f}% cote moy={te['odds'].mean():.2f}")
    return dict(n=len(te), roi=roi_te, wr=wr, avg=te['odds'].mean())

r1 = wf("CAP100 Score exact", lambda d: (d['market'] == 'Score exact') & (d['odds'] == 100.0))
r2 = wf("CAP100 tous marchés", lambda d: d['odds'] == 100.0)
r3 = wf("1X2 longshot >=8", lambda d: (d['market'] == '1X2') & (d['odds'] >= 8))
# bins où freq>pred dans S5 -> candidat générique : à remplir après lecture
print("\nDONE")
