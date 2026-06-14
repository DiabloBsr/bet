# WF4 - BTTS ADVERSARIAL - check 2: ROI zone restreint aux snapshots captures AVANT le coup d'envoi
import sys
import json
import math
from datetime import datetime

sys.path.insert(0, ".")
import numpy as np
from scraper.config import load_settings
from sqlalchemy import create_engine, text

rows = json.load(open("exports/wf4_btts_data.json", encoding="utf-8"))["rows"]
lz = np.load("exports/wf4_btts_lambdas.npz")
lh, la = lz["lh"], lz["la"]
ids = np.array([r["id"] for r in rows])
o_no = np.array([r["o_no"] for r in rows])
o_yes = np.array([r["o_yes"] for r in rows])
sa = np.array([r["sa"] for r in rows])
sb = np.array([r["sb"] for r in rows])
win = (sa > 0) & (sb > 0)
ret_n = o_no * (~win) - 1
gap = np.abs(lh - la)
zone = (gap >= 0.6) & (gap < 1.3)
p_imp = (1 / o_yes) / (1 / o_yes + 1 / o_no)

eng = create_engine(load_settings().db_url)
lg = ",".join("'InstantLeague-80%s'" % x for x in ["35", "36", "37", "42", "43", "44", "56", "60", "65"])
with eng.connect() as c:
    db = c.execute(text("""
        SELECT e.id, e.expected_start, o.captured_at, r.finished_at
        FROM events e JOIN results r ON r.event_id=e.id
        JOIN odds_snapshots o ON o.event_id=e.id
        WHERE o.id=(SELECT MIN(o2.id) FROM odds_snapshots o2 WHERE o2.event_id=e.id)
          AND e.competition IN (%s)""" % lg)).fetchall()


def pd_(s):
    s = str(s).replace("T", " ").split("+")[0].strip()
    for f in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, f)
        except ValueError:
            pass
    return None


m = {r[0]: r for r in db}
pre = np.array([pd_(m[int(i)][2]) < pd_(m[int(i)][1]) for i in ids])
notafterfin = np.array([pd_(m[int(i)][2]) < pd_(m[int(i)][3]) for i in ids])


def st(mask, label):
    mm = zone & mask
    r = ret_n[mm]
    se = r.std(ddof=1) / math.sqrt(len(r))
    real = float(win[mm].mean())
    imp = float(p_imp[mm].mean())
    zd = (real - imp) / math.sqrt(real * (1 - real) / len(r))
    print("%-30s n=%5d ROIno=%+6.2f%% (se %.2f) dev=%+.4f z=%+.2f" % (
        label, len(r), r.mean() * 100, se * 100, real - imp, zd))


st(np.ones(len(ids), bool), "zone all")
st(pre, "zone captured BEFORE start")
st(~pre, "zone captured after start")
st(notafterfin, "zone excl. captured>finished")
