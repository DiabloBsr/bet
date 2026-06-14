# WF4 TOTALS - step 1: extraction data (opening odds, lambda inversion, settlement)
# READ-ONLY DB. Cache -> exports/wf4_totals_data.pkl
import sys, json, pickle, math
sys.path.insert(0, ".")
from scraper.config import load_settings
from sqlalchemy import create_engine, text
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import poisson

e = create_engine(load_settings().db_url)

# corrupted ids (dict, ids = keys of d["events"])
with open("exports/corrupted_events.json", "r", encoding="utf-8") as f:
    corr = json.load(f)
CORRUPT = set(int(k) for k in corr["events"].keys())

LEAGUES = ["InstantLeague-8035", "InstantLeague-8036", "InstantLeague-8037",
           "InstantLeague-8042", "InstantLeague-8043", "InstantLeague-8044",
           "InstantLeague-8056", "InstantLeague-8060", "InstantLeague-8065"]

SQL = text("""
SELECT ev.id, ev.competition, ev.expected_start, ev.round_info,
       r.score_a, r.score_b, r.ht_score_a, r.ht_score_b, r.goals_json,
       o.odds_home, o.odds_draw, o.odds_away, o.extra_markets
FROM events ev
JOIN results r ON r.event_id = ev.id
JOIN odds_snapshots o ON o.event_id = ev.id
WHERE o.id = (SELECT MIN(o2.id) FROM odds_snapshots o2 WHERE o2.event_id = ev.id)
  AND ev.competition IN :comps
""").bindparams()

GMAX = 13  # goal grid 0..GMAX

def grid_probs(lh, la):
    ph = poisson.pmf(np.arange(GMAX + 1), lh)
    pa = poisson.pmf(np.arange(GMAX + 1), la)
    return np.outer(ph, pa)

def invert_lambdas(oh, od, oa):
    """fair probs (multiplicative norm) -> (lh, la) on independent Poisson grid"""
    imp = np.array([1 / oh, 1 / od, 1 / oa])
    fair = imp / imp.sum()
    H = np.arange(GMAX + 1)

    def resid(x):
        lh, la = np.exp(x)
        g = grid_probs(lh, la)
        phome = np.tril(g, -1).sum()
        pdraw = np.trace(g)
        return [phome - fair[0], pdraw - fair[1]]

    # init from rough heuristic
    diff0 = math.log(max(fair[0], 1e-6) / max(fair[2], 1e-6)) * 0.55
    tot0 = 2.8
    x0 = [math.log(max(0.2, tot0 / 2 + diff0 / 2)), math.log(max(0.2, tot0 / 2 - diff0 / 2))]
    sol = least_squares(resid, x0, xtol=1e-12, ftol=1e-12)
    lh, la = np.exp(sol.x)
    err = max(abs(v) for v in resid(sol.x))
    return float(lh), float(la), float(err)

def main():
    with e.connect() as conn:
        rows = conn.execute(
            text(SQL.text.replace(":comps", "(" + ",".join("'" + c + "'" for c in LEAGUES) + ")"))
        ).fetchall()
    print(f"raw rows: {len(rows)}")

    out = []
    n_corrupt = n_guard = n_badodds = n_badinv = 0
    for row in rows:
        (eid, comp, exp_start, rnd, sa, sb, hta, htb, gj,
         oh, od, oa, xm) = row
        if eid in CORRUPT:
            n_corrupt += 1
            continue
        if sa is None or sb is None:
            continue
        # guard: HT > FT inconsistency
        if hta is not None and htb is not None and (hta > sa or htb > sb):
            n_guard += 1
            continue
        # guard: goals_json length mismatch (only when parseable & non-empty expected)
        gj_ok = None
        if gj:
            try:
                gl = json.loads(gj)
                if isinstance(gl, list):
                    gj_ok = (len(gl) == sa + sb)
            except Exception:
                pass
        if gj_ok is False:
            n_guard += 1
            continue
        if not oh or not od or not oa or oh <= 1 or od <= 1 or oa <= 1:
            n_badodds += 1
            continue
        try:
            xmd = json.loads(xm) if xm else {}
        except Exception:
            xmd = {}
        lh, la, err = invert_lambdas(oh, od, oa)
        if err > 1e-5:
            n_badinv += 1
            continue
        rec = dict(eid=eid, comp=comp, start=str(exp_start), rnd=rnd,
                   sa=sa, sb=sb, tot=sa + sb,
                   oh=oh, od=od, oa=oa, lh=lh, la=la,
                   ou_u=None, ou_o=None,  # +/- 3.5
                   th_u=None, th_o=None, ta_u=None, ta_o=None,  # team totals 3.5
                   totx={},  # Total de buts exact
                   x2t={})   # 1X2 & Total
        m = xmd.get("+/-") or {}
        rec["ou_u"] = m.get("< 3.5"); rec["ou_o"] = m.get("> 3.5")
        m = xmd.get("Total equipe domicile") or {}
        rec["th_u"] = m.get("< 3.5"); rec["th_o"] = m.get("> 3.5")
        m = xmd.get("Total equipe extérieur") or {}
        rec["ta_u"] = m.get("< 3.5"); rec["ta_o"] = m.get("> 3.5")
        rec["totx"] = xmd.get("Total de buts") or {}
        rec["x2t"] = xmd.get("1X2 & Total") or {}
        out.append(rec)

    print(f"kept {len(out)} | corrupt {n_corrupt} | guard {n_guard} | badodds {n_badodds} | badinv {n_badinv}")
    from collections import Counter
    print(Counter(r["comp"] for r in out))
    with open("exports/wf4_totals_data.pkl", "wb") as f:
        pickle.dump(out, f)
    print("cached -> exports/wf4_totals_data.pkl")

if __name__ == "__main__":
    main()
