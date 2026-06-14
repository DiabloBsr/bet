"""WF5 — Verifications de l'audit du jour : entonnoir, timing des snapshots,
spot-checks, baselines historiques. Lecture seule.
Sortie : exports/wf5_today_verify.json
"""
import sys, json, math
sys.path.insert(0, ".")
from sqlalchemy import create_engine, text
from scraper.config import load_settings

DAY_START = "2026-06-12 06:00:00"
COMP = "InstantLeague-8035"
eng = create_engine(load_settings().db_url)

with open("exports/corrupted_events.json", encoding="utf-8") as f:
    corrupted = set(int(k) for k in json.load(f)["events"].keys())

out = {}

with eng.connect() as c:
    # --- 1. entonnoir ---
    total = c.execute(text("""SELECT COUNT(*) FROM events ev JOIN results r ON r.event_id=ev.id
        WHERE ev.competition=:comp AND ev.expected_start>=:s"""), {"comp": COMP, "s": DAY_START}).scalar()
    with_snap = c.execute(text("""SELECT COUNT(*) FROM events ev JOIN results r ON r.event_id=ev.id
        WHERE ev.competition=:comp AND ev.expected_start>=:s
        AND EXISTS (SELECT 1 FROM odds_snapshots s2 WHERE s2.event_id=ev.id)"""),
        {"comp": COMP, "s": DAY_START}).scalar()
    today_ids = [r[0] for r in c.execute(text("""SELECT ev.id FROM events ev JOIN results r ON r.event_id=ev.id
        WHERE ev.competition=:comp AND ev.expected_start>=:s"""), {"comp": COMP, "s": DAY_START})]
    n_corr_today = sum(1 for i in today_ids if i in corrupted)
    out["funnel"] = {"finished_today": total, "with_snapshot": with_snap,
                     "corrupted_today": n_corr_today}

    # --- 2. timing du snapshot d'ouverture vs kickoff ---
    rows = c.execute(text("""
        SELECT ev.id, ev.expected_start, s.captured_at
        FROM events ev
        JOIN results r ON r.event_id=ev.id
        JOIN odds_snapshots s ON s.id=(SELECT MIN(s2.id) FROM odds_snapshots s2 WHERE s2.event_id=ev.id)
        WHERE ev.competition=:comp AND ev.expected_start>=:s"""),
        {"comp": COMP, "s": DAY_START}).fetchall()
    from datetime import datetime
    def pdt(x):
        x = str(x)[:26]
        for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
            try: return datetime.strptime(x, fmt)
            except ValueError: pass
        return None
    deltas = []
    post_ko = 0
    for eid, es, fa in rows:
        a, b = pdt(es), pdt(fa)
        if a and b:
            d = (b - a).total_seconds()
            deltas.append(d)
            if d > 0: post_ko += 1
    deltas.sort()
    out["open_snapshot_timing"] = {
        "n": len(deltas), "post_kickoff_count": post_ko,
        "min_s": deltas[0] if deltas else None,
        "p50_s": deltas[len(deltas)//2] if deltas else None,
        "max_s": deltas[-1] if deltas else None}

    # --- 3. spot-checks E1 (3 matchs) + SWEET wins ---
    sc = []
    q = c.execute(text("""
        SELECT ev.id, ev.team_a, ev.team_b, s.odds_home, s.extra_markets, r.score_a, r.score_b, r.goals_json
        FROM events ev JOIN results r ON r.event_id=ev.id
        JOIN odds_snapshots s ON s.id=(SELECT MIN(s2.id) FROM odds_snapshots s2 WHERE s2.event_id=ev.id)
        WHERE ev.competition=:comp AND ev.expected_start>=:s AND s.odds_home<=1.5
        ORDER BY ev.id LIMIT 3"""), {"comp": COMP, "s": DAY_START})
    for eid, ta, tb, oh, em_raw, sa, sb, gj in q:
        em = json.loads(em_raw) if em_raw else {}
        goals = json.loads(gj) if gj else []
        fg = min(goals, key=lambda g: g.get("minute", 999))["team"] if goals else "None"
        sc.append({"eid": eid, "match": f"{ta} v {tb}", "oh": oh,
                   "ftts1": (em.get("FTTS") or {}).get("1"),
                   "score": f"{sa}-{sb}", "first_goal": fg,
                   "e1_win": fg == "Home"})
    out["spot_E1"] = sc

# --- 4. baselines historiques (avant aujourd'hui) sur les MEMES regles ---
# fenetre : tout l'historique 8035 avec snapshot, expected_start < DAY_START
hist = []
with eng.connect() as c:
    q = c.execute(text("""
        SELECT ev.id, r.score_a, r.score_b, s.id, s.odds_home, s.odds_away
        FROM events ev
        JOIN results r ON r.event_id=ev.id
        JOIN odds_snapshots s ON s.id=(SELECT MIN(s2.id) FROM odds_snapshots s2 WHERE s2.event_id=ev.id)
        WHERE ev.competition=:comp AND ev.expected_start < :s"""),
        {"comp": COMP, "s": DAY_START})
    for eid, sa, sb, sid, oh, oa in q:
        if eid in corrupted or not oh or not oa or oh < 1.01 or oa < 1.01:
            continue
        hist.append((eid, int(sa), int(sb), sid, float(oh), float(oa)))
print(f"hist matches: {len(hist)}", file=sys.stderr)

def agg(bets):
    n = len(bets)
    if n == 0:
        return dict(n=0)
    wins = sum(1 for w, _ in bets if w)
    pnl = sum((c - 1) if w else -1.0 for w, c in bets)
    return dict(n=n, wins=wins, wr=round(wins / n, 4), roi=round(pnl / n, 4))

# TIER1_approx + E2 hist
t1, e2 = [], []
for eid, sa, sb, sid, oh, oa in hist:
    fav_home = oh <= oa
    fo = oh if fav_home else oa
    won = (sa > sb) if fav_home else (sb > sa)
    if fo <= 1.30:
        t1.append((won, fo))
    if 1.10 <= fo <= 1.20:
        e2.append((won, fo))
out["hist_TIER1_approx"] = agg(t1)
out["hist_E2"] = agg(e2)

# BTTS_NON hist : besoin cote G/NG Non -> fetch extra_markets en chunks pour oh<=1.30
bn_sids = [(eid, sa, sb, sid) for eid, sa, sb, sid, oh, oa in hist if oh <= 1.30]
bn_bets = []
with eng.connect() as c:
    sids = [x[3] for x in bn_sids]
    meta = {x[3]: x for x in bn_sids}
    for i in range(0, len(sids), 200):
        chunk = sids[i:i+200]
        q = text(f"SELECT id, extra_markets FROM odds_snapshots WHERE id IN ({','.join(map(str, chunk))})")
        for sid, em_raw in c.execute(q):
            try:
                em = json.loads(em_raw) if em_raw else {}
            except Exception:
                continue
            v = (em.get("G/NG") or {}).get("Non")
            if not isinstance(v, (int, float)) or not (1.01 <= v <= 50):
                continue
            eid, sa, sb, _ = meta[sid]
            bn_bets.append(((sa == 0 or sb == 0), float(v)))
out["hist_BTTS_NON"] = agg(bn_bets)
print("hist BTTS_NON done", file=sys.stderr)

# E1 hist : oh<=1.50 -> FTTS '1' ; settle goals_json. Chunks.
e1_rows = [(eid, sa, sb, sid) for eid, sa, sb, sid, oh, oa in hist if oh <= 1.50]
ftts_odds = {}
with eng.connect() as c:
    sids = [x[3] for x in e1_rows]
    for i in range(0, len(sids), 200):
        chunk = sids[i:i+200]
        q = text(f"SELECT id, extra_markets FROM odds_snapshots WHERE id IN ({','.join(map(str, chunk))})")
        for sid, em_raw in c.execute(q):
            try:
                em = json.loads(em_raw) if em_raw else {}
            except Exception:
                continue
            v = (em.get("FTTS") or {}).get("1")
            if isinstance(v, (int, float)) and 1.01 <= v <= 50:
                ftts_odds[sid] = float(v)
    fg_map = {}
    eids = [x[0] for x in e1_rows]
    for i in range(0, len(eids), 300):
        chunk = eids[i:i+300]
        q = text(f"SELECT event_id, goals_json, score_a, score_b FROM results WHERE event_id IN ({','.join(map(str, chunk))})")
        for eid, gj, sa, sb in c.execute(q):
            try:
                goals = json.loads(gj) if gj else []
            except Exception:
                fg_map[eid] = None
                continue
            if not goals:
                fg_map[eid] = "None" if (int(sa) + int(sb)) == 0 else None
            else:
                fg_map[eid] = min(goals, key=lambda g: g.get("minute", 999)).get("team")
e1_bets = []
for eid, sa, sb, sid in e1_rows:
    fg = fg_map.get(eid)
    cote = ftts_odds.get(sid)
    if fg is None or cote is None:
        continue
    e1_bets.append((fg == "Home", cote))
out["hist_E1"] = agg(e1_bets)

with open("exports/wf5_today_verify.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
