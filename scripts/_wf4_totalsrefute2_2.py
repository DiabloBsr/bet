# follow-up: (a) timezone/offset of captured_at vs expected_start (look-ahead audit),
# (b) odds>=10 bucket walk-forward on 8035 + same bucket pooled-newleagues (multiple-testing audit)
import sys, json, math
sys.path.insert(0, ".")
import numpy as np
from scipy.stats import norm
from scraper.config import load_settings
from sqlalchemy import create_engine, text
from datetime import datetime

e = create_engine(load_settings().db_url)
with open("exports/corrupted_events.json", "r", encoding="utf-8") as f:
    CORRUPT = set(int(k) for k in json.load(f)["events"].keys())

LEAGUES = ["InstantLeague-8035", "InstantLeague-8036", "InstantLeague-8037",
           "InstantLeague-8042", "InstantLeague-8043", "InstantLeague-8044",
           "InstantLeague-8056", "InstantLeague-8060", "InstantLeague-8065"]
with e.connect() as conn:
    rows = conn.execute(text("""
        SELECT ev.id, ev.competition, ev.expected_start, ev.first_seen_at,
               r.score_a, r.score_b, o.extra_markets, o.captured_at
        FROM events ev
        JOIN results r ON r.event_id = ev.id
        JOIN odds_snapshots o ON o.event_id = ev.id
        WHERE o.id = (SELECT MIN(o2.id) FROM odds_snapshots o2 WHERE o2.event_id = ev.id)
          AND ev.competition IN ('%s')
    """ % "','".join(LEAGUES))).fetchall()

def parse(s):
    s = str(s)
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s[:26], fmt)
        except ValueError:
            pass
    return None

deltas = []
recs = []
seen = set()
for (eid, comp, start, fseen, sa, sb, xm, cap) in rows:
    if eid in CORRUPT or eid in seen or sa is None or sb is None:
        continue
    seen.add(eid)
    ds, dc = parse(start), parse(cap)
    if ds and dc:
        deltas.append((dc - ds).total_seconds() / 60.0)
    try:
        totx = (json.loads(xm) or {}).get("Total de buts") or {}
    except Exception:
        totx = {}
    v = totx.get("1")
    if v is None:
        continue
    o = float(v)
    if not (1 < o < 100):
        continue
    recs.append(dict(comp=comp, start=str(start), tot=sa + sb, o=o))

d = np.array(deltas)
print(f"captured_at - expected_start (min): median={np.median(d):.1f} p5={np.percentile(d,5):.1f} "
      f"p95={np.percentile(d,95):.1f} frac>0={np.mean(d>0):.3f} frac>+10min={np.mean(d>10):.3f} "
      f"frac>+60min={np.mean(d>60):.3f}")

def roi_stats(bets):
    if not bets:
        return dict(n=0)
    r = np.array([(o - 1) if w else -1.0 for w, o in bets])
    n = len(r); roi = float(r.mean())
    se = r.std(ddof=1) / math.sqrt(n) if n > 1 else 1e9
    p = 2 * (1 - norm.cdf(abs(roi) / se)) if se > 0 else 1.0
    return dict(n=n, wr=round(float(np.mean([w for w, _ in bets])), 4),
                roi_pct=round(roi * 100, 2), odds=round(float(np.mean([o for _, o in bets])), 3),
                p=round(float(p), 5))

# bucket odds>=10, 8035 walk-forward 70/30
L35 = sorted([r for r in recs if r["comp"] == "InstantLeague-8035"], key=lambda r: r["start"])
cut = L35[int(0.7 * len(L35))]["start"]
hi_train = [(r["tot"] == 1, r["o"]) for r in L35 if r["o"] >= 10 and r["start"] < cut]
hi_test = [(r["tot"] == 1, r["o"]) for r in L35 if r["o"] >= 10 and r["start"] >= cut]
print("8035 odds>=10 TRAIN:", roi_stats(hi_train))
print("8035 odds>=10 TEST :", roi_stats(hi_test))
# same bucket on the 8 new leagues (independent confirmation set)
hi_new = [(r["tot"] == 1, r["o"]) for r in recs if r["comp"] != "InstantLeague-8035" and r["o"] >= 10]
print("newleagues odds>=10:", roi_stats(hi_new))
# quartiles of the >=10 bucket on 8035 (stability)
hi_all = [(r["tot"] == 1, r["o"]) for r in L35 if r["o"] >= 10]
n = len(hi_all)
for i in range(4):
    print(f"  8035 hi Q{i+1}:", roi_stats(hi_all[i * n // 4:(i + 1) * n // 4]))
