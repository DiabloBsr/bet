# Strict temporal 70/30 walk-forward re-verification of the 10 "CONFIRMED" mined findings.
# Opening odds = MIN(id) snapshot per event. No signal fitted here (rules are fixed),
# so the OOS 30% acts as a clean holdout vs. the (suspected full-history) verification.
import sys, json, sqlite3
sys.path.insert(0, '.')

con = sqlite3.connect('data/virtual_sports.db')
cur = con.cursor()

rows = cur.execute("""
SELECT e.id, e.round_info, e.expected_start,
       o.odds_home, o.odds_draw, o.odds_away, o.extra_markets,
       r.score_a, r.score_b, r.ht_score_a, r.ht_score_b
FROM events e
JOIN results r ON r.event_id = e.id
JOIN odds_snapshots o ON o.id = (SELECT MIN(id) FROM odds_snapshots WHERE event_id = e.id)
WHERE e.round_info IS NOT NULL AND e.round_info != '0'
  AND r.score_a IS NOT NULL AND r.ht_score_a IS NOT NULL
  AND o.odds_home IS NOT NULL
ORDER BY e.expected_start
""").fetchall()

matches = []
for (eid, rnd, ts, oh, od, oa, em, sa, sb, ha, hb) in rows:
    try:
        j = int(rnd)
    except Exception:
        continue
    if not (1 <= j <= 38):
        continue
    if isinstance(em, str):
        try:
            em = json.loads(em)
        except Exception:
            em = {}
    if not isinstance(em, dict):
        em = {}
    matches.append(dict(eid=eid, j=j, ts=ts, oh=oh, od=od, oa=oa, em=em,
                        sa=sa, sb=sb, ha=ha, hb=hb))

n = len(matches)
cut = int(n * 0.7)
print(f"total matches usable: {n} | train: {cut} | oos: {n-cut}")
print(f"train period: {matches[0]['ts']} -> {matches[cut-1]['ts']}")
print(f"oos   period: {matches[cut]['ts']} -> {matches[-1]['ts']}")

def seg(j):
    if j <= 3: return 'DS'
    if j <= 12: return 'MS_early'
    if j <= 25: return 'MS_mid'
    if j <= 33: return 'MS_late'
    return 'FS'

def home_slight(m): return 1.6 <= m['oh'] < 2.2 and m['oa'] >= 2.5
def away_slight(m): return 1.6 <= m['oa'] < 2.2 and m['oh'] >= 2.5

def get_odd(m, market, sel):
    mk = m['em'].get(market)
    if isinstance(mk, dict):
        v = mk.get(sel)
        if isinstance(v, (int, float)) and v > 1.0:
            return float(v)
    return None

# settlement helpers
def w_htft(m, combo):
    ht = '1' if m['ha'] > m['hb'] else ('2' if m['hb'] > m['ha'] else 'X')
    ft = '1' if m['sa'] > m['sb'] else ('2' if m['sb'] > m['sa'] else 'X')
    return f"{ht}/{ft}" == combo

FINDINGS = [
    ("HT/FT 1/X home longshot MS_early",
     lambda m: seg(m['j']) == 'MS_early' and m['oh'] >= 3.5,
     'HT/FT', '1/X', lambda m: w_htft(m, '1/X')),
    ("F1 away team total >3.5 away_slight MS_mid",
     lambda m: seg(m['j']) == 'MS_mid' and away_slight(m),
     'Total equipe extérieur', '> 3.5', lambda m: m['sb'] >= 4),
    ("HT/FT 2/1 favori home MS_mid",
     lambda m: seg(m['j']) == 'MS_mid' and m['oh'] < 2.0,
     'HT/FT', '2/1', lambda m: w_htft(m, '2/1')),
    ("F4 Total de buts = 6 away_slight",
     lambda m: away_slight(m),
     'Total de buts', '6', lambda m: (m['sa'] + m['sb']) == 6),
    ("1X2&Total 2/>3.5 FS away non-favori",
     lambda m: seg(m['j']) == 'FS' and m['oa'] >= m['oh'],
     '1X2 & Total', '2 / > 3.5', lambda m: m['sb'] > m['sa'] and (m['sa'] + m['sb']) >= 4),
    ("F2 home team total >3.5 home_slight cote 5-8",
     lambda m: home_slight(m) and (lambda c: c is not None and 5.0 <= c < 8.0)(get_odd(m, 'Total equipe domicile', '> 3.5')),
     'Total equipe domicile', '> 3.5', lambda m: m['sa'] >= 4),
    ("HT/FT X/2 favori home MS_mid",
     lambda m: seg(m['j']) == 'MS_mid' and 1.25 <= m['oh'] < 1.70,
     'HT/FT', 'X/2', lambda m: w_htft(m, 'X/2')),
    ("MT-1X2 1 home longshot MS_early",
     lambda m: seg(m['j']) == 'MS_early' and m['oh'] >= 3.5,
     'Mi-tps 1X2', '1', lambda m: m['ha'] > m['hb']),
    ("F3 Total de buts = 1 home_slight MS_early",
     lambda m: seg(m['j']) == 'MS_early' and home_slight(m),
     'Total de buts', '1', lambda m: (m['sa'] + m['sb']) == 1),
]

def evaluate(sub, name, filt, market, sel, winfn):
    picks = []
    for m in sub:
        if not filt(m):
            continue
        c = get_odd(m, market, sel)
        if c is None:
            continue
        won = 1 if winfn(m) else 0
        picks.append((c, won))
    if not picks:
        return dict(name=name, n=0)
    n_ = len(picks)
    wr = sum(w for _, w in picks) / n_
    roi = sum((w * (c - 1) - (1 - w)) for c, w in picks) / n_
    avg_c = sum(c for c, _ in picks) / n_
    n100 = sum(1 for c, w in picks if c >= 99 and w)
    return dict(name=name, n=n_, wr=wr, roi=roi, avg_c=avg_c, wins=sum(w for _, w in picks), wins_at_100=n100)

train, oos = matches[:cut], matches[cut:]

print(f"\n{'finding':50s} | {'split':5s} | {'n':>5s} {'wins':>4s} {'wr%':>6s} {'roi%':>7s} {'avgC':>6s} {'w@100':>5s}")
results_table = []
for (name, filt, market, sel, winfn) in FINDINGS:
    for label, sub in (('train', train), ('OOS', oos)):
        r = evaluate(sub, name, filt, market, sel, winfn)
        if r['n'] == 0:
            print(f"{name:50.50s} | {label:5s} |     0")
            continue
        print(f"{name:50.50s} | {label:5s} | {r['n']:5d} {r['wins']:4d} {100*r['wr']:6.1f} {100*r['roi']:7.1f} {r['avg_c']:6.2f} {r['wins_at_100']:5d}")
        results_table.append((name, label, r))

# PORTFOLIO P1 = A (MT-1X2 '1' oh>=3.5, all segs) + E (HT/FT 2/1, oh<2.0 MS_mid) + F (HT/FT X/2, 1.25<=oh<1.70 MS_mid)
def p1_picks(sub):
    picks = []
    overlap_matches = set()
    for m in sub:
        legs = []
        if m['oh'] >= 3.5:
            legs.append(('Mi-tps 1X2', '1', m['ha'] > m['hb']))
        if seg(m['j']) == 'MS_mid' and m['oh'] < 2.0:
            legs.append(('HT/FT', '2/1', w_htft(m, '2/1')))
        if seg(m['j']) == 'MS_mid' and 1.25 <= m['oh'] < 1.70:
            legs.append(('HT/FT', 'X/2', w_htft(m, 'X/2')))
        if len(legs) > 1:
            overlap_matches.add(m['eid'])
        for mk, sl, won in legs:
            c = get_odd(m, mk, sl)
            if c is not None:
                picks.append((c, 1 if won else 0))
    return picks, overlap_matches

for label, sub in (('train', train), ('OOS', oos)):
    picks, ovl = p1_picks(sub)
    n_ = len(picks)
    wr = sum(w for _, w in picks) / n_
    roi = sum((w * (c - 1) - (1 - w)) for c, w in picks) / n_
    print(f"{'PORTFOLIO P1 (A+E+F)':50s} | {label:5s} | {n_:5d} {sum(w for _,w in picks):4d} {100*wr:6.1f} {100*roi:7.1f} {sum(c for c,_ in picks)/n_:6.2f}   matches w/ 2 legs (mutually exclusive): {len(ovl)}")
