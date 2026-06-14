import sys, json
sys.path.insert(0, ".")
from sqlalchemy import create_engine
from scraper.config import load_settings
from scraper.analysis_utils import load_corrupted_ids
from datetime import datetime, timezone, timedelta
import pandas as pd

e = create_engine(load_settings().db_url)
MG = timezone(timedelta(hours=3))
now = datetime.now(timezone.utc)
corrupted = load_corrupted_ids()  # 473 ids sous ["events"] (fix bug set(json.load(...)))

hist = pd.read_sql("""
    SELECT e.id, e.team_a, e.team_b, r.score_a, r.score_b
    FROM events e JOIN results r ON r.event_id=e.id
    WHERE r.score_a IS NOT NULL AND e.competition='InstantLeague-8035'
""", e)
hist = hist[~hist.id.isin(corrupted)]
hist["score"] = hist.score_a.astype(int).astype(str) + "-" + hist.score_b.astype(int).astype(str)

fut = pd.read_sql("""
    SELECT e.id, e.team_a, e.team_b, e.expected_start, o.odds_home, o.odds_draw, o.odds_away, o.extra_markets
    FROM events e
    JOIN odds_snapshots o ON o.id=(SELECT MAX(id) FROM odds_snapshots WHERE event_id=e.id)
    LEFT JOIN results r ON r.event_id=e.id
    WHERE r.id IS NULL AND e.competition='InstantLeague-8035' AND e.expected_start IS NOT NULL
""", e)
fut["expected_start"] = pd.to_datetime(fut.expected_start, utc=True)
fut = fut[fut.expected_start > now]
fut["local"] = fut.expected_start.dt.tz_convert(MG).dt.strftime("%H:%M")
fut = fut.sort_values("expected_start").drop_duplicates(["team_a","team_b","local"])
print(f"{len(fut)} matchs futurs, rounds: {sorted(fut.local.unique())}")

# inspecter format des selections Score exact
em0 = json.loads(fut.iloc[0].extra_markets)
print("Format Score exact:", dict(list(em0.get("Score exact", {}).items())[:6]) if isinstance(em0.get("Score exact"), dict) else em0.get("Score exact"))

def cs_odds(em_raw):
    try: em = json.loads(em_raw) if isinstance(em_raw, str) else em_raw
    except Exception: return {}
    cs = em.get("Score exact") if isinstance(em, dict) else None
    out = {}
    if isinstance(cs, dict):
        for k, v in cs.items():
            kk = k.strip().replace(":", "-").replace(" ", "")
            try: out[kk] = float(v)
            except Exception: pass
    elif isinstance(cs, list):
        for it in cs:
            if isinstance(it, dict):
                k = str(it.get("name", it.get("selection",""))).strip().replace(":", "-").replace(" ","")
                try: out[k] = float(it.get("odds", it.get("price")))
                except Exception: pass
    return out

rows = []
for _, m in fut.iterrows():
    h = hist[(hist.team_a==m.team_a) & (hist.team_b==m.team_b)]
    n = len(h)
    if n < 8: continue
    odds_map = cs_odds(m.extra_markets)
    vc = h.score.value_counts()
    for score, cnt in vc.items():
        if cnt < 3: continue
        freq = cnt / n
        cote = odds_map.get(score)
        if not cote: continue
        ev = (freq * cote - 1) * 100
        rows.append(dict(local=m.local, match=f"{m.team_a[:14]} vs {m.team_b[:14]}",
                         score=score, freq=round(freq*100), n=n, cnt=int(cnt),
                         cote=cote, ev_pct=round(ev,1),
                         oh=round(m.odds_home,2)))

df = pd.DataFrame(rows)
if len(df):
    df = df.sort_values("ev_pct", ascending=False)
    print("\n=== TOP EV (freq paire x cote offerte) ===")
    print(df.head(20).to_string(index=False))
else:
    print("rien")
