# -*- coding: utf-8 -*-
"""WF2 - Inventaire des donnees fondamentales (rankings, saisons, features intra-saison)."""
import sys, json
from collections import Counter, defaultdict
sys.path.insert(0, '.')
from scraper.config import load_settings
from sqlalchemy import create_engine, text

eng = create_engine(load_settings().db_url)

SEP = "=" * 70

with eng.connect() as c:
    # ------------------------------------------------------------------ 1. RANKINGS
    print(SEP); print("1. RANKINGS_SNAPSHOTS"); print(SEP)
    n = c.execute(text("select count(*) from rankings_snapshots")).scalar()
    ncap = c.execute(text("select count(distinct captured_at) from rankings_snapshots")).scalar()
    ncomp = c.execute(text("select count(distinct competition) from rankings_snapshots")).scalar()
    span = c.execute(text("select min(captured_at), max(captured_at) from rankings_snapshots")).fetchone()
    print(f"rows={n}  distinct captured_at={ncap}  competitions={ncomp}  span={span[0]} -> {span[1]}")

    comps = c.execute(text(
        "select competition, count(*), count(distinct captured_at) from rankings_snapshots group by 1")).fetchall()
    for r in comps:
        print(f"  comp={r[0]!r}  rows={r[1]}  snapshots={r[2]}")

    # frequence : ecarts entre captured_at consecutifs
    caps = [r[0] for r in c.execute(text(
        "select distinct captured_at from rankings_snapshots order by 1")).fetchall()]
    from datetime import datetime
    def parse(s):
        return datetime.fromisoformat(str(s).replace('Z', ''))
    gaps = []
    for a, b in zip(caps, caps[1:]):
        gaps.append((parse(b) - parse(a)).total_seconds() / 60.0)
    if gaps:
        gaps_sorted = sorted(gaps)
        med = gaps_sorted[len(gaps_sorted)//2]
        print(f"gap entre snapshots (min): median={med:.1f}  p10={gaps_sorted[len(gaps_sorted)//10]:.1f}  "
              f"p90={gaps_sorted[9*len(gaps_sorted)//10]:.1f}  max={gaps_sorted[-1]:.0f}")

    # teams per snapshot
    tps = c.execute(text(
        "select captured_at, count(*) cnt from rankings_snapshots group by 1")).fetchall()
    cnt_dist = Counter(r[1] for r in tps)
    print(f"equipes par snapshot: {dict(sorted(cnt_dist.items()))}")

    # 3 exemples bruts de history
    print("\n--- 3 exemples bruts (position, points, W/L/D, history) ---")
    rows = c.execute(text(
        "select captured_at, team_name, position, points, won, lost, draw, history "
        "from rankings_snapshots order by id limit 200")).fetchall()
    # prendre 3 a des positions variees
    for r in [rows[0], rows[len(rows)//2], rows[-1]]:
        print(f"\ncaptured_at={r[0]} team={r[1]} pos={r[2]} pts={r[3]} W/L/D={r[4]}/{r[5]}/{r[6]}")
        h = r[7]
        if isinstance(h, str):
            try: h = json.loads(h)
            except Exception: pass
        print(f"history type={type(h).__name__} -> {json.dumps(h, ensure_ascii=False)[:600]}")

    # W/L/D cumulatif ou par saison ? regarder evolution de matches joues pour 1 equipe dans le temps
    print("\n--- evolution W+L+D (matchs joues) pour 1 equipe (40 snapshots espaces) ---")
    team = c.execute(text("select team_name from rankings_snapshots limit 1")).scalar()
    ev = c.execute(text(
        "select captured_at, points, won+lost+draw as played, position from rankings_snapshots "
        "where team_name=:t order by captured_at"), {"t": team}).fetchall()
    step = max(1, len(ev)//40)
    for r in ev[::step]:
        print(f"  {r[0]}  pts={r[1]:>3}  played={r[2]:>3}  pos={r[3]}")
    played = [r[2] for r in ev]
    resets = sum(1 for a, b in zip(played, played[1:]) if b < a)
    print(f"team={team}: snapshots={len(ev)}, resets de 'played' (b<a) = {resets} -> "
          f"{'PAR SAISON (reset)' if resets > 0 else 'CUMULATIF'}")

    # ------------------------------------------------------------------ 2. SAISONS sur events
    print(); print(SEP); print("2. SEASON_ID RECONSTRUIT SUR EVENTS"); print(SEP)
    evs = c.execute(text(
        "select e.id, e.round_info, e.team_a, e.team_b, e.expected_start, e.competition, "
        "r.score_a, r.score_b "
        "from events e left join results r on r.event_id = e.id "
        "order by e.expected_start, e.id")).fetchall()
    print(f"events total={len(evs)}  avec result={sum(1 for r in evs if r[6] is not None)}")

    comps_ev = Counter(r[5] for r in evs)
    print(f"competitions events: {dict(comps_ev)}")

    # construire season_id par competition : nouvelle saison quand round redescend
    season_id = {}
    seasons = defaultdict(lambda: defaultdict(list))  # comp -> sid -> list rows
    last_round = {}
    sid_counter = defaultdict(int)
    for r in evs:
        comp = r[5]
        try:
            rd = int(r[1])
        except (TypeError, ValueError):
            rd = None
        if rd is None:
            continue
        if comp in last_round and rd < last_round[comp]:
            sid_counter[comp] += 1
        last_round[comp] = rd
        sid = sid_counter[comp]
        season_id[r[0]] = (comp, sid)
        seasons[comp][sid].append((rd, r))

    print("\n--- distribution longueurs de saisons (par competition) ---")
    complete = []
    for comp, sdict in seasons.items():
        lens = []
        for sid, lst in sorted(sdict.items()):
            rounds = sorted(set(x[0] for x in lst))
            nmatch = len(lst)
            lens.append((sid, nmatch, len(rounds), rounds[0], rounds[-1]))
        dist = Counter(x[1] for x in lens)
        n_full = sum(1 for x in lens if x[2] == 38 and x[1] == 380)
        print(f"comp={comp!r}: saisons={len(lens)}  completes(38 rounds & 380 matchs)={n_full}")
        print(f"  dist nb_matchs/saison: {dict(sorted(dist.items()))}")
        for sid, nm, nr, r0, r1 in lens:
            tag = "OK" if (nr == 38 and nm == 380) else ("partielle" if nm < 380 else "ANOMALIE")
            if tag != "OK":
                print(f"    sid={sid}: matchs={nm} rounds_distincts={nr} [{r0}..{r1}] -> {tag}")
            else:
                complete.append((comp, sid))
    print(f"\nTOTAL saisons completes = {len(complete)}")

    # ------------------------------------------------------------------ 4. INCOHERENCES
    print(); print(SEP); print("4. INCOHERENCES"); print(SEP)
    dup_total = 0
    for comp, sdict in seasons.items():
        for sid, lst in sorted(sdict.items()):
            per_round = Counter(x[0] for x in lst)
            bad_rounds = {k: v for k, v in per_round.items() if v != 10}
            # doublons paire equipes dans une saison
            pairs = Counter((x[1][2], x[1][3]) for x in lst)
            dups = {k: v for k, v in pairs.items() if v > 1}
            missing = [rd for rd in range(1, 39) if rd not in per_round] if len(per_round) > 30 else []
            if bad_rounds or dups or missing:
                dup_total += 1
                print(f"comp={comp} sid={sid}: rounds!=10matchs={dict(sorted(bad_rounds.items()))} "
                      f"rounds_manquants={missing} paires_dupliquees={len(dups)}")
                for k, v in list(dups.items())[:3]:
                    print(f"    dup pair {k} x{v}")
    if dup_total == 0:
        print("aucune incoherence (toutes saisons: 38 rounds x 10 matchs, paires uniques)")

    # equipes par saison
    comp0 = max(seasons, key=lambda cmp: len(seasons[cmp]))
    teams_per_season = []
    for sid, lst in seasons[comp0].items():
        ts = set()
        for _, r in lst:
            ts.add(r[2]); ts.add(r[3])
        teams_per_season.append(len(ts))
    print(f"equipes par saison ({comp0}): {dict(Counter(teams_per_season))}")

    # ------------------------------------------------------------------ 3. FEATURES INTRA-SAISON (saison exemple)
    print(); print(SEP); print("3. DEMO FEATURES AVANT-MATCH (saison exemple, comp la plus fournie)"); print(SEP)
    # choisir une saison complete avec results
    demo = None
    for comp, sid in complete:
        lst = seasons[comp][sid]
        with_res = sum(1 for _, r in lst if r[6] is not None)
        if with_res >= 370:
            demo = (comp, sid, lst, with_res)
            break
    if demo is None:
        # fallback : la saison la plus couverte
        best = max(((comp, sid, lst, sum(1 for _, r in lst if r[6] is not None))
                    for comp, sdict in seasons.items() for sid, lst in sdict.items()),
                   key=lambda x: x[3])
        demo = best
    comp, sid, lst, with_res = demo
    print(f"saison demo: comp={comp} sid={sid}  matchs={len(lst)}  avec result={with_res}")

    # table virtuelle journee par journee (uniquement J-1 et avant)
    table = defaultdict(lambda: {"pts": 0, "gf": 0, "ga": 0, "form": []})
    lst_sorted = sorted(lst, key=lambda x: (x[0], x[1][4]))
    rounds_in = sorted(set(x[0] for x in lst_sorted))
    target_round = rounds_in[min(9, len(rounds_in)-1)]  # journee 10 si dispo
    for rd, r in lst_sorted:
        if rd >= target_round:
            break
        sa, sb = r[6], r[7]
        if sa is None:
            continue
        ta, tb = r[2], r[3]
        table[ta]["gf"] += sa; table[ta]["ga"] += sb
        table[tb]["gf"] += sb; table[tb]["ga"] += sa
        if sa > sb:
            table[ta]["pts"] += 3; table[ta]["form"].append("W"); table[tb]["form"].append("L")
        elif sa < sb:
            table[tb]["pts"] += 3; table[tb]["form"].append("W"); table[ta]["form"].append("L")
        else:
            table[ta]["pts"] += 1; table[tb]["pts"] += 1
            table[ta]["form"].append("D"); table[tb]["form"].append("D")
    standing = sorted(table.items(), key=lambda kv: (-kv[1]["pts"], -(kv[1]["gf"]-kv[1]["ga"]), -kv[1]["gf"]))
    print(f"\nclassement virtuel AVANT la journee {target_round} (base J1..J{target_round-1}):")
    print(f"{'pos':>3} {'equipe':<22} {'pts':>3} {'gf':>3} {'ga':>3} {'diff':>4}  forme(5 derniers)")
    for i, (t, d) in enumerate(standing, 1):
        print(f"{i:>3} {t:<22} {d['pts']:>3} {d['gf']:>3} {d['ga']:>3} {d['gf']-d['ga']:>4}  {''.join(d['form'][-5:])}")

    # croiser avec rankings_snapshots : trouver un snapshot dont points matchent ?
    print("\n--- croisement avec rankings_snapshots (meme comp ?) ---")
    rk_comps = [r[0] for r in c.execute(text("select distinct competition from rankings_snapshots")).fetchall()]
    print(f"competitions rankings: {rk_comps}")
    # chercher snapshot ou le leader a les memes points
    if standing:
        leader, d0 = standing[0]
        hit = c.execute(text(
            "select captured_at, position, points, won, lost, draw from rankings_snapshots "
            "where team_name=:t and points=:p limit 3"), {"t": leader, "p": d0["pts"]}).fetchall()
        print(f"snapshots ou {leader} a pts={d0['pts']}: {[tuple(r) for r in hit]}")

print("\nFIN.")
