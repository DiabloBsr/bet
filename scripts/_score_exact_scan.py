"""Scan des rounds futurs : scores exacts notables (freq H2H + cote offerte -> EV)."""
import sys, json
sys.path.insert(0, ".")
from sqlalchemy import create_engine
from scraper.config import load_settings
from datetime import datetime, timezone, timedelta
import pandas as pd

e = create_engine(load_settings().db_url)
MG = timezone(timedelta(hours=3))
now = datetime.now(timezone.utc)

corrupted = set(json.load(open("exports/corrupted_events.json")))

# Historique H2H (même orientation) 8035
hist = pd.read_sql("""
    SELECT e.id, e.team_a, e.team_b, r.score_a, r.score_b
    FROM events e JOIN results r ON r.event_id=e.id
    WHERE r.score_a IS NOT NULL AND e.competition='InstantLeague-8035'
""", e)
hist = hist[~hist.id.isin(corrupted)]
hist = hist.drop_duplicates(["team_a","team_b","score_a","score_b"], keep="last")
hist["score"] = hist.score_a.astype(int).astype(str) + "-" + hist.score_b.astype(int).astype(str)

# Futurs avec cotes + extra_markets
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

def score_odds(em_raw, score):
    """Cherche la cote du score exact dans extra_markets."""
    try:
        em = json.loads(em_raw) if isinstance(em_raw, str) else em_raw
    except Exception:
        return None
    if not em: return None
    for mname, sels in (em.items() if isinstance(em, dict) else []):
        ml = mname.lower()
        if "score" in ml and "exact" in ml or ml == "score correct" or "correct" in ml:
            if isinstance(sels, dict):
                for k, v in sels.items():
                    if k.replace(" ","").replace(":","-") == score:
                        try: return float(v)
                        except Exception: pass
    return None

rows = []
for _, m in fut.iterrows():
    h = hist[(hist.team_a==m.team_a) & (hist.team_b==m.team_b)]
    n = len(h)
    if n < 8: continue
    vc = h.score.value_counts()
    top_score, top_n = vc.index[0], int(vc.iloc[0])
    freq = top_n / n
    if freq < 0.30: continue
    cote = score_odds(m.extra_markets, top_score)
    ev = (freq * cote - 1) * 100 if cote else None
    rows.append(dict(local=m.local, match=f"{m.team_a} vs {m.team_b}",
                     score=top_score, freq=f"{freq*100:.0f}%", n=n,
                     cote=cote, ev_pct=round(ev,1) if ev is not None else None,
                     o1x2=f"{m.odds_home:.2f}/{m.odds_draw:.2f}/{m.odds_away:.2f}"))

df = pd.DataFrame(rows)
if len(df):
    print(df.sort_values(["ev_pct"], ascending=False, na_position="last").to_string(index=False))
else:
    print("Aucune paire avec score dominant >=30% dans les rounds futurs.")
# bonus : structure extra_markets d'un event pour vérifier le nom du marché score exact
sample = fut.iloc[0].extra_markets if len(fut) else None
if sample:
    try:
        em = json.loads(sample)
        print("\nMarchés dispo:", list(em.keys())[:30])
    except Exception as ex:
        print("extra_markets parse err:", ex)
