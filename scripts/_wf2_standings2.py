# -*- coding: utf-8 -*-
"""WF2 - Robustesse du TEST 2 (desaccords classement vs cote) :
train vs OOS, erreurs-types, back du nul, stabilite intra-OOS.
Pipeline identique a _wf2_standings.py."""
import sys, json, math
from collections import defaultdict
from datetime import datetime
sys.path.insert(0, '.')
import numpy as np
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
    table = defaultdict(lambda: {"pts": 0, "gf": 0, "ga": 0, "played": 0})
    seg_start_rd = min(by_rd)
    for rd in sorted(by_rd):
        standing = sorted(table.items(),
                          key=lambda kv: (-kv[1]["pts"], -(kv[1]["gf"]-kv[1]["ga"]), -kv[1]["gf"]))
        pos_map = {t: i + 1 for i, (t, d) in enumerate(standing)}
        snap_table = {t: dict(d) for t, d in table.items()}
        for r in by_rd[rd]:
            eid, _, ta, tb, est, sa, sb = r
            if sa is None or eid not in open_odds: continue
            oh, od, oa = open_odds[eid]
            inv = 1/oh + 1/od + 1/oa
            d = dict(eid=eid, sid=sid, rd=rd, ta=ta, tb=tb, t=parse_t(est),
                     oh=oh, od=od, oa=oa, ph=(1/oh)/inv, pd=(1/od)/inv, pa=(1/oa)/inv,
                     y=(0 if sa > sb else (1 if sa == sb else 2)))
            da, db = snap_table.get(ta), snap_table.get(tb)
            if da and db and da["played"] >= 3 and db["played"] >= 3 and seg_start_rd <= 2:
                d["rec_pos_h"], d["rec_pos_a"] = pos_map[ta], pos_map[tb]
                d["rec_pts_diff"] = da["pts"] - db["pts"]
            ok_sn, sn = True, {}
            for team, key in ((ta, "h"), (tb, "a")):
                best = None
                for s in snaps_by_team.get(team, []):
                    if s[0] < d["t"]: best = s
                    else: break
                if best is None: ok_sn = False; break
                age = (d["t"] - best[0]).total_seconds() / 60
                if age > 80 or best[1] > rd - 1 or best[1] < 3:
                    ok_sn = False; break
                sn[key] = best
            if ok_sn:
                d["sn_pos_h"], d["sn_pos_a"] = sn["h"][3], sn["a"][3]
                d["sn_pts_diff"] = sn["h"][2] - sn["a"][2]
            rows.append(d)
        for r in by_rd[rd]:
            _, _, ta, tb, _, sa, sb = r
            if sa is None: continue
            table[ta]["gf"] += sa; table[ta]["ga"] += sb; table[ta]["played"] += 1
            table[tb]["gf"] += sb; table[tb]["ga"] += sa; table[tb]["played"] += 1
            if sa > sb: table[ta]["pts"] += 3
            elif sa < sb: table[tb]["pts"] += 3
            else: table[ta]["pts"] += 1; table[tb]["pts"] += 1

rows.sort(key=lambda d: (d["t"], d["eid"]))
cut = int(len(rows) * 0.70)
t_cut = rows[cut]["t"]
print(f"matchs={len(rows)}  t_cut={t_cut}")

def fav_side(d): return 0 if d["ph"] >= d["pa"] else 2

def filt_sn_gap(d, gap):
    if "sn_pos_h" not in d or d["rd"] < 6: return None
    fs = fav_side(d)
    better = 0 if d["sn_pos_h"] < d["sn_pos_a"] else (2 if d["sn_pos_a"] < d["sn_pos_h"] else None)
    if better is None or better == fs: return None
    if abs(d["sn_pos_h"] - d["sn_pos_a"]) < gap: return None
    return better

def filt_pts(d, th, src):
    key = f"{src}_pts_diff"
    if key not in d or d["rd"] < 6: return None
    fs = fav_side(d)
    pd_ = d[key]
    if fs == 2 and pd_ >= th: return 0
    if fs == 0 and pd_ <= -th: return 2
    return None

def settle(subset_sides, mode):
    """mode: 'cls' = back mieux classe, 'fav' = back favori cote, 'draw' = back nul."""
    n = len(subset_sides)
    if n == 0: return None
    rets = []
    wins = 0
    cotes = []
    for d, s in subset_sides:
        if mode == "cls": side, o = s, (d["oh"] if s == 0 else d["oa"])
        elif mode == "fav":
            side = fav_side(d); o = d["oh"] if side == 0 else d["oa"]
        else:
            side, o = 1, d["od"]
        cotes.append(o)
        if d["y"] == side:
            wins += 1; rets.append(o - 1.0)
        else:
            rets.append(-1.0)
    rets = np.array(rets)
    roi = rets.mean(); se = rets.std(ddof=1) / math.sqrt(n)
    return dict(n=n, wr=wins/n, roi=float(roi), se=float(se), z=float(roi/se) if se > 0 else 0.0,
                avg_o=sum(cotes)/n)

FILTERS = [
    ("SN pos gap>=1", lambda d: filt_sn_gap(d, 1)),
    ("SN pos gap>=5", lambda d: filt_sn_gap(d, 5)),
    ("SN pts_diff>=5", lambda d: filt_pts(d, 5, "sn")),
    ("SN pts_diff>=8", lambda d: filt_pts(d, 8, "sn")),
    ("REC pts_diff>=5", lambda d: filt_pts(d, 5, "rec")),
    ("REC pts_diff>=8", lambda d: filt_pts(d, 8, "rec")),
]

oos = [d for d in rows if d["t"] >= t_cut]
half = oos[len(oos)//2]["t"]
WINDOWS = [("TRAIN", [d for d in rows if d["t"] < t_cut]),
           ("OOS", oos),
           ("OOS-1ere moitie", [d for d in oos if d["t"] < half]),
           ("OOS-2eme moitie", [d for d in oos if d["t"] >= half])]

for fname, ffn in FILTERS:
    print("\n" + "=" * 78)
    print(f"FILTRE: {fname}  (favori cote != mieux classe)")
    for wname, wdata in WINDOWS:
        sub = [(d, ffn(d)) for d in wdata]
        sub = [(d, s) for d, s in sub if s is not None]
        if not sub:
            print(f"  {wname:<16} n=0"); continue
        out = []
        for mode, lbl in (("cls", "back CLASSEMENT"), ("fav", "back FAVORI"), ("draw", "back NUL")):
            m = settle(sub, mode)
            out.append(f"{lbl}: WR={m['wr']:.3f} ROI={m['roi']:+.4f} (z={m['z']:+.2f})")
        # distribution des issues
        from collections import Counter
        cnt = Counter(("cls" if d["y"] == s else ("draw" if d["y"] == 1 else "fav")) for d, s in sub)
        n = len(sub)
        print(f"  {wname:<16} n={n:>4} | " + " | ".join(out))
        print(f"  {'':<16} issues: classe={cnt.get('cls',0)/n:.3f} nul={cnt.get('draw',0)/n:.3f} favori={cnt.get('fav',0)/n:.3f}")

print("\nFIN.")
