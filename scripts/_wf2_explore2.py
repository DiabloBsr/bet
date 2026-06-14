# -*- coding: utf-8 -*-
"""WF2 - inventaire approfondi: doublons, round_info=0, saisons robustes, join rankings<->events."""
import sys, json
from collections import Counter, defaultdict
from datetime import datetime
sys.path.insert(0, '.')
from scraper.config import load_settings
from sqlalchemy import create_engine, text

eng = create_engine(load_settings().db_url)
SEP = "=" * 70

def parse(s):
    return datetime.fromisoformat(str(s).replace('Z', ''))

with eng.connect() as c:
    print(SEP); print("A. ROUND_INFO + DOUBLONS EVENTS"); print(SEP)
    rd_dist = c.execute(text("select round_info, count(*) from events group by 1 order by cast(round_info as int)")).fetchall()
    print("round_info dist:", {r[0]: r[1] for r in rd_dist})

    dups = c.execute(text(
        "select team_a, team_b, expected_start, count(*) c, group_concat(id) ids, group_concat(round_info) rds "
        "from events group by 1,2,3 having c>1")).fetchall()
    print(f"\ndoublons exacts (team_a,team_b,expected_start): {len(dups)} groupes")
    for r in dups[:5]:
        print(f"  {r[0]} vs {r[1]} @ {r[2]}  ids={r[4]} rounds={r[5]}")
    # doublons par paire + start proche meme round ?
    dups2 = c.execute(text(
        "select team_a, team_b, round_info, count(*) c from events group by 1,2,3 having c>1")).fetchall()
    print(f"doublons (team_a,team_b,round_info) toutes saisons confondues: {len(dups2)} (normal si saisons cyclent)")

    # events sans result
    nores = c.execute(text(
        "select count(*) from events e left join results r on r.event_id=e.id where r.event_id is null")).scalar()
    print(f"events sans result: {nores}")

    print(); print(SEP); print("B. SAISONS ROBUSTES (dedup + split sur drop>=5 de round)"); print(SEP)
    evs = c.execute(text(
        "select e.id, cast(e.round_info as int) rd, e.team_a, e.team_b, e.expected_start, "
        "r.score_a, r.score_b "
        "from events e left join results r on r.event_id=e.id "
        "order by e.expected_start, e.id")).fetchall()

    # dedup par (team_a, team_b, expected_start) en gardant celui AVEC result sinon min id
    seen = {}
    for r in evs:
        k = (r[2], r[3], str(r[4]))
        if k not in seen or (seen[k][5] is None and r[5] is not None):
            seen[k] = r
    evs2 = sorted(seen.values(), key=lambda r: (str(r[4]), r[0]))
    print(f"events apres dedup: {len(evs2)} (avant {len(evs)})")

    # split: nouvelle saison si round < last_round - 4 (tolere desordre local) OU gap temps > 30 min
    seasons = []
    cur = []
    last_rd = None
    last_t = None
    for r in evs2:
        rd = r[1]
        t = parse(r[4])
        new = False
        if last_rd is not None:
            if rd < last_rd - 4:
                new = True
            if last_t is not None and (t - last_t).total_seconds() > 45 * 60:
                new = True
        if new and cur:
            seasons.append(cur); cur = []
        cur.append(r)
        last_rd = max(rd, last_rd if (last_rd is not None and not new) else rd)
        last_t = t
    if cur:
        seasons.append(cur)

    info = []
    for i, s in enumerate(seasons):
        rds = sorted(set(x[1] for x in s))
        nres = sum(1 for x in s if x[5] is not None)
        pairs = Counter((x[2], x[3]) for x in s)
        ndup = sum(v - 1 for v in pairs.values() if v > 1)
        info.append((i, len(s), len(rds), rds[0], rds[-1], nres, ndup,
                     str(s[0][4])[:16], str(s[-1][4])[:16]))
    print(f"saisons detectees: {len(seasons)}")
    dist = Counter(x[1] for x in info)
    print("dist nb matchs:", dict(sorted(dist.items())))
    big = [x for x in info if x[1] >= 100]
    print(f"\nsaisons >=100 matchs: {len(big)}, total matchs avec result dedans: {sum(x[5] for x in big)}")
    print(f"{'sid':>4} {'n':>4} {'nrds':>4} {'r0':>3} {'r1':>3} {'nres':>4} {'ndup':>4}  debut             fin")
    for x in info:
        if x[1] >= 60:
            print(f"{x[0]:>4} {x[1]:>4} {x[2]:>4} {x[3]:>3} {x[4]:>3} {x[5]:>4} {x[6]:>4}  {x[7]}  {x[8]}")

    # combien de matchs ont TOUTES les journees precedentes completes (10 matchs avec result) ?
    usable = 0
    total_with_res = 0
    for s in seasons:
        by_rd = defaultdict(list)
        for x in s:
            by_rd[x[1]].append(x)
        rds = sorted(by_rd)
        complete_until = 0
        for rd in range(1, 39):
            if len(by_rd.get(rd, [])) == 10 and all(x[5] is not None for x in by_rd[rd]):
                complete_until = rd
            else:
                break
        for x in s:
            if x[5] is None: continue
            total_with_res += 1
            if 2 <= x[1] <= complete_until + 1:
                usable += 1
    print(f"\nmatchs avec result: {total_with_res}; dont round J avec J1..J-1 COMPLETES (classement exact possible): {usable}")

    print(); print(SEP); print("C. JOIN RANKINGS_SNAPSHOTS <-> EVENTS"); print(SEP)
    rk = c.execute(text(
        "select captured_at, team_name, position, points, won, lost, draw, history "
        "from rankings_snapshots order by captured_at")).fetchall()
    # index par team -> liste (t, played, points, pos, history)
    by_team = defaultdict(list)
    for r in rk:
        by_team[r[1]].append((parse(r[0]), r[4] + r[5] + r[6], r[3], r[2], r[7]))

    # pour un echantillon de matchs avec result et round>=6 : dernier snapshot AVANT expected_start
    # coherence: played du snapshot doit etre <= round-1 et idealement == round-1 ou round-2
    import random
    random.seed(42)
    sample = [x for s in seasons for x in s if x[5] is not None and x[1] >= 6]
    random.shuffle(sample)
    sample = sample[:800]
    stats = Counter()
    lags = []
    for x in sample:
        t_match = parse(x[4])
        for team in (x[2], x[3]):
            snaps = by_team.get(team, [])
            # dernier snapshot strictement avant le match
            best = None
            for sn in snaps:
                if sn[0] < t_match:
                    best = sn
                else:
                    break
            if best is None:
                stats['no_snapshot'] += 1
                continue
            age_min = (t_match - best[0]).total_seconds() / 60
            diff = (x[1] - 1) - best[1]  # round-1 - played
            if age_min > 80:
                stats['snapshot_trop_vieux(>80min)'] += 1
            elif diff < 0:
                stats['played>round-1 (saison differente ou leak)'] += 1
            elif diff == 0:
                stats['exact (played==round-1)'] += 1
                lags.append(age_min)
            elif diff <= 3:
                stats['stale 1-3 rounds'] += 1
            else:
                stats['stale >3 rounds'] += 1
    tot = sum(stats.values())
    print(f"join sur {len(sample)} matchs x2 equipes = {tot} lookups:")
    for k, v in stats.most_common():
        print(f"  {k}: {v} ({100*v/tot:.1f}%)")
    if lags:
        lags.sort()
        print(f"age median des snapshots 'exact': {lags[len(lags)//2]:.1f} min")

    # history: longueur distribution + valeurs
    print("\n--- history field ---")
    hlen = Counter(); hval = Counter()
    for r in rk[:5000]:
        h = r[7]
        if isinstance(h, str):
            try: h = json.loads(h)
            except Exception: h = None
        if isinstance(h, list):
            hlen[len(h)] += 1
            for v in h: hval[v] += 1
        else:
            hlen['non-list'] += 1
    print(f"longueur history: {dict(hlen)}")
    print(f"valeurs: {dict(hval)}")

print("\nFIN.")
