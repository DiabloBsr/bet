# -*- coding: utf-8 -*-
"""WF4 inventory: data cartography for 13 downstream miners. READ-ONLY."""
import sys, json, collections
sys.path.insert(0, ".")
from scraper.config import load_settings
from sqlalchemy import create_engine, text

e = create_engine(load_settings().db_url)

_corr = json.load(open("exports/corrupted_events.json", encoding="utf-8"))
corrupted = set(int(k) for k in _corr["events"].keys())
print(f"corrupted ids loaded: {len(corrupted)}", file=sys.stderr)

out = {}

with e.connect() as c:
    # ---------- (1) finished matches WITH opening odds, per league ----------
    rows = c.execute(text("""
        SELECT e.competition, e.id
        FROM events e
        JOIN results r ON r.event_id = e.id
        WHERE EXISTS (SELECT 1 FROM odds_snapshots o WHERE o.event_id = e.id)
    """)).fetchall()
    per_league = collections.Counter()
    per_league_excl = collections.Counter()
    for comp, eid in rows:
        per_league[comp] += 1
        if eid not in corrupted:
            per_league_excl[comp] += 1
    out["finished_with_opening_odds"] = {
        k: {"raw": per_league[k], "clean_excl_corrupted": per_league_excl[k]}
        for k in sorted(per_league)
    }

    # also: total events & total with odds (not necessarily finished)
    rows = c.execute(text("""
        SELECT e.competition,
               COUNT(*) AS n_events,
               SUM(CASE WHEN EXISTS (SELECT 1 FROM odds_snapshots o WHERE o.event_id=e.id) THEN 1 ELSE 0 END) AS n_with_odds,
               SUM(CASE WHEN EXISTS (SELECT 1 FROM results r WHERE r.event_id=e.id) THEN 1 ELSE 0 END) AS n_finished,
               MIN(e.expected_start), MAX(e.expected_start)
        FROM events e GROUP BY e.competition
    """)).fetchall()
    out["events_overview"] = {
        r[0]: {"n_events": r[1], "n_with_odds": r[2], "n_finished": r[3],
               "start_min": str(r[4]), "start_max": str(r[5])}
        for r in rows
    }

    # ---------- (2) extra_markets sampling on 8035, 8037, 8065 ----------
    em_report = {}
    for lg in ["InstantLeague-8035", "InstantLeague-8037", "InstantLeague-8065"]:
        rows = c.execute(text("""
            SELECT o.extra_markets FROM odds_snapshots o
            JOIN events e ON e.id = o.event_id
            WHERE e.competition = :lg AND o.extra_markets IS NOT NULL
            ORDER BY o.id DESC LIMIT 40
        """), {"lg": lg}).fetchall()
        markets = {}  # name -> {"selections": set, "n_seen": int, "sample_struct": ...}
        raw_type = None
        for (em,) in rows:
            if not em:
                continue
            try:
                data = json.loads(em)
            except Exception:
                continue
            if raw_type is None:
                raw_type = type(data).__name__
            # handle dict-of-markets or list-of-markets
            if isinstance(data, dict):
                items = data.items()
            elif isinstance(data, list):
                items = [(m.get("name", m.get("marketName", "?")), m) for m in data if isinstance(m, dict)]
            else:
                continue
            for name, val in items:
                rec = markets.setdefault(name, {"selections": set(), "n_seen": 0, "sample": None})
                rec["n_seen"] += 1
                if rec["sample"] is None:
                    rec["sample"] = val
                if isinstance(val, dict):
                    for k2, v2 in val.items():
                        if isinstance(v2, (int, float, str)):
                            rec["selections"].add(str(k2))
                        elif isinstance(v2, dict):
                            for k3 in v2:
                                rec["selections"].add(f"{k2}/{k3}")
                elif isinstance(val, list):
                    for s in val:
                        if isinstance(s, dict):
                            rec["selections"].add(str(s.get("name", s.get("selection", "?"))))
        em_report[lg] = {
            "raw_json_type": raw_type,
            "n_snapshots_sampled": len(rows),
            "markets": {
                name: {
                    "n_seen": rec["n_seen"],
                    "selections": sorted(rec["selections"]),
                    "sample": rec["sample"],
                }
                for name, rec in sorted(markets.items())
            },
        }
    out["extra_markets"] = em_report

    # ---------- (3) goals_json availability/format per league ----------
    gj_report = {}
    leagues = sorted(per_league)
    for lg in leagues:
        rows = c.execute(text("""
            SELECT r.goals_json, r.score_a, r.score_b FROM results r
            JOIN events e ON e.id = r.event_id
            WHERE e.competition = :lg
            ORDER BY r.id DESC LIMIT 20
        """), {"lg": lg}).fetchall()
        n = len(rows)
        n_null = sum(1 for g, _, _ in rows if g is None or g == "" or g == "null")
        n_parse_ok = 0
        n_count_match = 0
        sample = None
        for g, sa, sb in rows:
            if not g or g == "null":
                continue
            try:
                gj = json.loads(g)
                n_parse_ok += 1
                if sample is None:
                    sample = gj if not isinstance(gj, list) else gj[:4]
                if isinstance(gj, list) and len(gj) == (sa or 0) + (sb or 0):
                    n_count_match += 1
            except Exception:
                pass
        gj_report[lg] = {
            "n_sampled": n, "n_null_or_empty": n_null, "n_parse_ok": n_parse_ok,
            "n_len_equals_total_goals": n_count_match, "sample": sample,
        }
    out["goals_json"] = gj_report

    # ---------- (4) ht_score availability per league ----------
    rows = c.execute(text("""
        SELECT e.competition,
               COUNT(*),
               SUM(CASE WHEN r.ht_score_a IS NOT NULL AND r.ht_score_b IS NOT NULL THEN 1 ELSE 0 END)
        FROM results r JOIN events e ON e.id = r.event_id
        GROUP BY e.competition
    """)).fetchall()
    out["ht_score"] = {r[0]: {"n_results": r[1], "n_ht_available": r[2]} for r in rows}

    # ---------- (5) round_info distribution per league ----------
    rows = c.execute(text("""
        SELECT e.competition, e.round_info, COUNT(*)
        FROM events e GROUP BY e.competition, e.round_info
    """)).fetchall()
    ri = collections.defaultdict(dict)
    for comp, rinfo, n in rows:
        ri[comp][str(rinfo)] = n
    # compact: per league -> n_distinct rounds, min, max, count of J0-like, top few
    ri_report = {}
    for comp, dist in ri.items():
        keys = list(dist.keys())
        # try numeric extraction
        nums = []
        for k in keys:
            try:
                nums.append(int("".join(ch for ch in k if ch.isdigit())))
            except Exception:
                pass
        j0 = sum(v for k, v in dist.items() if k.strip() in ("0", "J0", "Journée 0", "Journee 0", "None", "null", ""))
        ri_report[comp] = {
            "n_distinct_rounds": len(keys),
            "round_min": min(nums) if nums else None,
            "round_max": max(nums) if nums else None,
            "n_events_round0_or_null": j0,
            "sample_keys": sorted(keys)[:8],
            "full_dist": dist if len(dist) <= 100 else None,
        }
    out["round_info"] = ri_report


def default(o):
    if isinstance(o, set):
        return sorted(o)
    return str(o)

with open("exports/wf4_inventory.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1, default=default)

# compact stdout report
print(json.dumps(out, ensure_ascii=False, indent=1, default=default)[:200])
print("WROTE exports/wf4_inventory.json")
