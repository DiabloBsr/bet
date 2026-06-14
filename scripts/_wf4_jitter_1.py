# WF4 jitter/drift campaign - step 1: extract per-event open/close snapshots + result
# Read-only on DB. Output: scripts/_wf4_jitter_data.pkl
import sys, json, pickle
sys.path.insert(0, ".")
from scraper.config import load_settings
from sqlalchemy import create_engine, text
import pandas as pd

e = create_engine(load_settings().db_url)

# corrupted ids (8035 only audited)
with open("exports/corrupted_events.json", "r", encoding="utf-8") as f:
    corrupted = set(int(k) for k in json.load(f)["events"].keys())

with e.connect() as c:
    snaps = pd.read_sql(text("""
        SELECT o.id snap_id, o.event_id, o.captured_at, o.odds_home, o.odds_draw, o.odds_away,
               o.scrape_run_id
        FROM odds_snapshots o
        JOIN results r ON r.event_id = o.event_id
        ORDER BY o.event_id, o.id
    """), c)
    ev = pd.read_sql(text("""
        SELECT e.id event_id, e.competition, e.team_a, e.team_b, e.round_info,
               e.expected_start, r.score_a, r.score_b
        FROM events e JOIN results r ON r.event_id = e.id
    """), c)

snaps = snaps[~snaps.event_id.isin(corrupted)]
ev = ev[~ev.event_id.isin(corrupted)]

snaps["captured_at"] = pd.to_datetime(snaps["captured_at"])
ev["expected_start"] = pd.to_datetime(ev["expected_start"])

df = snaps.merge(ev, on="event_id", how="inner")
# guard: drop snapshots captured >2 min after expected_start (stale/rescheduled events)
df["lag_min"] = (df["captured_at"] - df["expected_start"]).dt.total_seconds() / 60.0
n_before = len(df)
df = df[df["lag_min"] <= 2.0]
print(f"snapshots kept {len(df)}/{n_before} (lag<=2min)")

# valid odds only
for col in ["odds_home", "odds_draw", "odds_away"]:
    df = df[df[col].notna() & (df[col] > 1.0)]

# open = first snap_id, close = last snap_id (within lag guard)
g = df.sort_values("snap_id").groupby("event_id")
first = g.first()
last = g.last()
nsnaps = g.size().rename("n_snaps")

out = first[["competition", "team_a", "team_b", "round_info", "expected_start",
             "score_a", "score_b"]].copy()
out["n_snaps"] = nsnaps
for col in ["odds_home", "odds_draw", "odds_away", "captured_at", "scrape_run_id", "snap_id"]:
    out["open_" + col.replace("odds_", "")] = first[col]
    out["close_" + col.replace("odds_", "")] = last[col]
out = out.reset_index()
print(out.groupby("competition").agg(n=("event_id", "count"),
                                     multi=("n_snaps", lambda s: (s >= 2).sum())))
with open("scripts/_wf4_jitter_data.pkl", "wb") as f:
    pickle.dump(out, f)
print("saved", len(out), "events")
