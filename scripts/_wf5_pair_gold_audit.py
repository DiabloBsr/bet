# -*- coding: utf-8 -*-
"""WF5 — Audit hors-echantillon des signaux de PAIRE (team_gold_data.py).

Les tables GOLD ont ete calculees sur la "BDD CLEAN (3225 matchs, cleanup 2026-06-05)".
On reconstruit la frontiere in-sample = les 3225 premiers matchs propres (ordre finished_at),
on VALIDE la frontiere en recalculant les stats in-sample de chaque paire (doivent matcher
les valeurs hardcodees), puis on mesure WR/ROI OOS sur tout ce qui a ete joue APRES.

Sortie: exports/wf5_pair_gold_audit.json + resume stdout.
LECTURE SEULE sur la DB. Requetes ciblees (colonnes minimales, chunks pour extra_markets).
"""
import sys, json, math
from collections import defaultdict, Counter

sys.path.insert(0, ".")
from scraper.config import load_settings
from sqlalchemy import create_engine, text

from scraper.team_gold_data import (
    PAIR_HOME_GOLD, PAIR_AWAY_GOLD, PAIR_TRAP_HOME,
    BRACKET_GOLD_HOME, BRACKET_GOLD_AWAY, BRACKET_TRAP_HOME,
    OVER_GOLD, UNDER_GOLD, BTTS_OUI_GOLD, BTTS_NON_GOLD,
    SCORE_COMBO_GOLD, SCORE_DOMINANT_GOLD,
)

LEAGUE = "InstantLeague-8035"
IS_N = 3225  # taille de la BDD clean au moment du calcul des tables GOLD

engine = create_engine(load_settings().db_url)

corrupted = set(int(k) for k in json.load(open("exports/corrupted_events.json"))["events"].keys())

# ---------------------------------------------------------------- load matches
SQL = """
SELECT e.id, e.team_a, e.team_b, r.score_a, r.score_b, r.finished_at,
       os.odds_home, os.odds_draw, os.odds_away
FROM events e
JOIN results r ON r.event_id = e.id
JOIN (SELECT event_id, MIN(id) AS sid FROM odds_snapshots GROUP BY event_id) f
     ON f.event_id = e.id
JOIN odds_snapshots os ON os.id = f.sid
WHERE e.competition = :lg
ORDER BY r.finished_at
"""
with engine.connect() as c:
    rows = c.execute(text(SQL), {"lg": LEAGUE}).fetchall()

matches = [dict(id=r[0], ta=r[1], tb=r[2], sa=r[3], sb=r[4], fin=r[5],
                oh=r[6], od=r[7], oa=r[8])
           for r in rows if r[0] not in corrupted
           and r[3] is not None and r[6] and r[7] and r[8]]
print(f"matchs propres avec cotes d'ouverture: {len(matches)}")

IS = matches[:IS_N]
# Frontiere exacte irreproductible (la liste corrupted_events.json date du 2026-06-11,
# posterieure aux tables GOLD du 2026-06-05). Cutoff CONSERVATEUR: tout match fini
# >= 2026-06-06 00:00 est garanti hors-echantillon (tables generees le 2026-06-05).
CUTOFF = "2026-06-06"
OOS = [m for m in matches if str(m["fin"]) >= CUTOFF]
cutoff_ts = CUTOFF
print(f"cutoff conservateur: {CUTOFF} (3225e match propre fini le {IS[-1]['fin']})")
print(f"in-sample~={len(IS)}  OOS={len(OOS)}  (OOS: {OOS[0]['fin']} -> {OOS[-1]['fin']})")

# ------------------------------------------------------- helper index by pair
def by_pair(ms):
    d = defaultdict(list)
    for m in ms:
        d[(m["ta"], m["tb"])].append(m)
    return d

IS_pair = by_pair(IS)
OOS_pair = by_pair(OOS)

def wilson_z(p_obs, p0, n):
    if n == 0 or p0 <= 0 or p0 >= 1:
        return None
    return (p_obs - p0) / math.sqrt(p0 * (1 - p0) / n)

def roi_stats(pnls):
    n = len(pnls)
    if n == 0:
        return dict(n=0, roi=None, z=None)
    mean = sum(pnls) / n
    var = sum((x - mean) ** 2 for x in pnls) / max(n - 1, 1)
    sd = math.sqrt(var)
    z = mean / (sd / math.sqrt(n)) if sd > 0 else None
    return dict(n=n, roi=round(mean, 4), z=round(z, 2) if z else None)

# ------------------------------------------------- 0. VALIDATION de la frontiere
print("\n=== VALIDATION frontiere in-sample (recalcul vs hardcode) ===")
val_ok, val_tot = 0, 0
for (ta, tb), d in list(PAIR_HOME_GOLD.items()):
    ms = IS_pair.get((ta, tb), [])
    n = len(ms)
    w = sum(1 for m in ms if m["sa"] > m["sb"])
    wr = w / n if n else None
    ok = (n == d["n"] and wr is not None and abs(wr - d["win"]) < 0.01)
    val_tot += 1
    val_ok += ok
    if not ok:
        print(f"  MISMATCH {ta} v {tb}: hard n={d['n']} win={d['win']:.3f} | recalc n={n} win={wr}")
print(f"PAIR_HOME_GOLD frontiere: {val_ok}/{val_tot} paires matchent exactement")

# baselines globaux OOS
n_oos = len(OOS)
base_home_wr = sum(1 for m in OOS if m["sa"] > m["sb"]) / n_oos
base_away_wr = sum(1 for m in OOS if m["sb"] > m["sa"]) / n_oos
base_over = sum(1 for m in OOS if m["sa"] + m["sb"] >= 3) / n_oos
base_btts = sum(1 for m in OOS if m["sa"] >= 1 and m["sb"] >= 1) / n_oos
score_freq = Counter(f"{m['sa']}-{m['sb']}" for m in OOS)
print(f"\nBaselines OOS (n={n_oos}): home WR={base_home_wr:.3f} away WR={base_away_wr:.3f} "
      f"over2.5={base_over:.3f} btts={base_btts:.3f}")

report = {"cutoff": str(cutoff_ts), "n_is": len(IS), "n_oos": n_oos,
          "baselines_oos": dict(home_wr=round(base_home_wr, 4), away_wr=round(base_away_wr, 4),
                                over25=round(base_over, 4), btts=round(base_btts, 4)),
          "frontier_validation": f"{val_ok}/{val_tot}", "families": {}}

# ---------------------------------------------------------- 1. PAIR_HOME_GOLD
def audit_pair_1x2(table, side, label, trap=False):
    per, pnls, wins, tot, implied_sum = [], [], 0, 0, 0.0
    is_wr_w = 0.0
    for key, d in (table.items() if isinstance(table, dict) else [(k, None) for k in table]):
        ms = OOS_pair.get(key, [])
        n = len(ms)
        if side == "home":
            w = sum(1 for m in ms if m["sa"] > m["sb"])
            pl = [(m["oh"] - 1) if m["sa"] > m["sb"] else -1.0 for m in ms]
            imp = [1 / m["oh"] for m in ms]
        else:
            w = sum(1 for m in ms if m["sb"] > m["sa"])
            pl = [(m["oa"] - 1) if m["sb"] > m["sa"] else -1.0 for m in ms]
            imp = [1 / m["oa"] for m in ms]
        wins += w; tot += n; pnls += pl; implied_sum += sum(imp)
        if d:
            is_wr_w += d["win"] * n
        per.append(dict(pair=f"{key[0]} v {key[1]}", n_oos=n,
                        wr_oos=round(w / n, 3) if n else None,
                        wr_is=(d["win"] if d else None),
                        roi_oos=round(sum(pl) / n, 3) if n else None,
                        roi_is=(d.get("roi") if d else None),
                        cote_is=(d.get("cote") if d else None),
                        avg_cote_oos=round(sum((m["oh"] if side == "home" else m["oa"]) for m in ms) / n, 2) if n else None))
    rs = roi_stats(pnls)
    wr = wins / tot if tot else None
    imp_wr = implied_sum / tot if tot else None  # proba impliquee brute (avec marge)
    out = dict(label=label, n_pairs=len(per), n_oos=tot, wins=wins,
               wr_oos=round(wr, 4) if wr is not None else None,
               wr_is_weighted=round(is_wr_w / tot, 4) if (tot and is_wr_w) else None,
               implied_wr_avg=round(imp_wr, 4) if imp_wr else None,
               roi_oos=rs["roi"], roi_z=rs["z"],
               z_vs_implied=round(wilson_z(wr, imp_wr, tot), 2) if (wr and imp_wr) else None,
               per_pair=per)
    return out

fam = audit_pair_1x2(PAIR_HOME_GOLD, "home", "PAIR_HOME_GOLD (back 1)")
report["families"]["PAIR_HOME_GOLD"] = fam
fam = audit_pair_1x2(PAIR_AWAY_GOLD, "away", "PAIR_AWAY_GOLD (back 2)")
report["families"]["PAIR_AWAY_GOLD"] = fam

# filtre max_cote_factor (sensibilite) : cote publiee <= cote_hist * 1.05
pnls_f, w_f, t_f = [], 0, 0
for key, d in PAIR_AWAY_GOLD.items():
    for m in OOS_pair.get(key, []):
        if m["oa"] <= d["cote"] * d.get("max_cote_factor", 1.05):
            t_f += 1
            win = m["sb"] > m["sa"]
            w_f += win
            pnls_f.append((m["oa"] - 1) if win else -1.0)
rs = roi_stats(pnls_f)
report["families"]["PAIR_AWAY_GOLD"]["with_max_cote_filter"] = dict(
    n=t_f, wr=round(w_f / t_f, 4) if t_f else None, roi=rs["roi"], z=rs["z"])

# PAIR_TRAP_HOME : home WR OOS (claim historique ~0%)
trap_rows = []
tw, tn, t_imp = 0, 0, 0.0
for key in PAIR_TRAP_HOME:
    ms = OOS_pair.get(key, [])
    w = sum(1 for m in ms if m["sa"] > m["sb"])
    tw += w; tn += len(ms); t_imp += sum(1 / m["oh"] for m in ms)
    trap_rows.append(dict(pair=f"{key[0]} v {key[1]}", n_oos=len(ms),
                          home_wr_oos=round(w / len(ms), 3) if ms else None))
report["families"]["PAIR_TRAP_HOME"] = dict(
    label="PAIR_TRAP_HOME (home censé perdre)", n_oos=tn, home_wins=tw,
    home_wr_oos=round(tw / tn, 4) if tn else None,
    implied_home_wr=round(t_imp / tn, 4) if tn else None, per_pair=trap_rows)

# ---------------------------------------------------------- 2. BRACKETS (team x cote)
def audit_bracket(table, side, label):
    per, pnls = [], []
    for (team, (lo, hi)), roi_is in table.items():
        pl = []
        for m in OOS:
            cote = m["oh"] if side == "home" else m["oa"]
            tm = m["ta"] if side == "home" else m["tb"]
            if tm == team and lo <= cote < hi:
                win = (m["sa"] > m["sb"]) if side == "home" else (m["sb"] > m["sa"])
                pl.append((cote - 1) if win else -1.0)
        pnls += pl
        per.append(dict(key=f"{team} {lo}-{hi}", n_oos=len(pl),
                        roi_oos=round(sum(pl) / len(pl), 3) if pl else None,
                        roi_is=roi_is))
    rs = roi_stats(pnls)
    return dict(label=label, n_oos=rs["n"], roi_oos=rs["roi"], roi_z=rs["z"], per_bracket=per)

report["families"]["BRACKET_GOLD_HOME"] = audit_bracket(BRACKET_GOLD_HOME, "home", "BRACKET_GOLD_HOME (back 1)")
report["families"]["BRACKET_GOLD_AWAY"] = audit_bracket(BRACKET_GOLD_AWAY, "away", "BRACKET_GOLD_AWAY (back 2)")
report["families"]["BRACKET_TRAP_HOME"] = audit_bracket(BRACKET_TRAP_HOME, "home", "BRACKET_TRAP_HOME (eviter 1)")

# ------------------------------------- 3. extra_markets pour paires flaggees (chunks)
need_ids = set()
flagged_pairs = set(OVER_GOLD) | set(UNDER_GOLD) | set(BTTS_OUI_GOLD) | set(BTTS_NON_GOLD) \
    | set(SCORE_COMBO_GOLD) | set(SCORE_DOMINANT_GOLD)
for key in flagged_pairs:
    for m in OOS_pair.get(key, []):
        need_ids.add(m["id"])
print(f"\nextra_markets a charger pour {len(need_ids)} events OOS (paires flaggees)")

em_data = {}  # event_id -> {"ou25": (over_cote, under_cote)|None, "gng": (oui,non)|None, "cs": {score: cote}}
ids = sorted(need_ids)
CH = 400
with engine.connect() as c:
    for i in range(0, len(ids), CH):
        chunk = ids[i:i + CH]
        q = text("""SELECT f.event_id, os.extra_markets
                    FROM (SELECT event_id, MIN(id) sid FROM odds_snapshots
                          WHERE event_id IN :ids GROUP BY event_id) f
                    JOIN odds_snapshots os ON os.id = f.sid""").bindparams()
        rs_ = c.execute(text(
            "SELECT f.event_id, os.extra_markets FROM (SELECT event_id, MIN(id) sid "
            "FROM odds_snapshots WHERE event_id IN ({}) GROUP BY event_id) f "
            "JOIN odds_snapshots os ON os.id = f.sid".format(",".join(str(x) for x in chunk))
        )).fetchall()
        for ev, raw in rs_:
            d = {"ou25": None, "gng": None, "cs": {}}
            if raw:
                try:
                    em = json.loads(raw)
                except Exception:
                    em = {}
                ou = em.get("+/-") or {}
                if "> 2.5" in ou and "< 2.5" in ou:
                    d["ou25"] = (ou["> 2.5"], ou["< 2.5"])
                g = em.get("G/NG") or {}
                if "Oui" in g and "Non" in g:
                    d["gng"] = (g["Oui"], g["Non"])
                d["cs"] = em.get("Score exact") or {}
            em_data[ev] = d

# ---------------------------------------------------------- 4. OVER / UNDER GOLD
def audit_total(table, want_over, label):
    per, hits, tot, pnls = [], 0, 0, []
    is_rate_w = 0.0
    for key, d in table.items():
        ms = OOS_pair.get(key, [])
        n = len(ms)
        h = sum(1 for m in ms if (m["sa"] + m["sb"] >= 3) == want_over)
        hits += h; tot += n
        rate_is = d.get("rate", 1 - d.get("over_rate", 0))  # rate = taux de succes du pari
        is_rate_w += rate_is * n
        for m in ms:
            emd = em_data.get(m["id"])
            if emd and emd["ou25"]:
                cote = emd["ou25"][0] if want_over else emd["ou25"][1]
                ok = (m["sa"] + m["sb"] >= 3) == want_over
                pnls.append((cote - 1) if ok else -1.0)
        per.append(dict(pair=f"{key[0]} v {key[1]}", n_oos=n,
                        hit_oos=round(h / n, 3) if n else None, hit_is=round(rate_is, 3)))
    rs = roi_stats(pnls)
    base = base_over if want_over else (1 - base_over)
    wr = hits / tot if tot else None
    return dict(label=label, n_oos=tot, hit_rate_oos=round(wr, 4) if wr else None,
                hit_rate_is_weighted=round(is_rate_w / tot, 4) if tot else None,
                baseline_oos=round(base, 4),
                z_vs_baseline=round(wilson_z(wr, base, tot), 2) if wr else None,
                roi_oos_with_25line=rs["roi"], roi_n=rs["n"], roi_z=rs["z"], per_pair=per)

report["families"]["OVER_GOLD"] = audit_total(OVER_GOLD, True, "OVER_GOLD (back Over 2.5)")
report["families"]["UNDER_GOLD"] = audit_total(UNDER_GOLD, False, "UNDER_GOLD (back Under 2.5)")

# ---------------------------------------------------------- 5. BTTS GOLD
def audit_btts(table, want_yes, label, use_min_cote=False):
    per, hits, tot, pnls = [], 0, 0, []
    is_rate_w = 0.0
    for key, d in table.items():
        ms = OOS_pair.get(key, [])
        if use_min_cote and "min_cote_h" in d:
            ms = [m for m in ms if m["oh"] >= d["min_cote_h"]]
        n = len(ms)
        btts = lambda m: m["sa"] >= 1 and m["sb"] >= 1
        h = sum(1 for m in ms if btts(m) == want_yes)
        hits += h; tot += n
        rate_is = d.get("rate", 1 - d.get("bts_rate", 0))
        is_rate_w += rate_is * n
        for m in ms:
            emd = em_data.get(m["id"])
            if emd and emd["gng"]:
                cote = emd["gng"][0] if want_yes else emd["gng"][1]
                pnls.append((cote - 1) if btts(m) == want_yes else -1.0)
        per.append(dict(pair=f"{key[0]} v {key[1]}", n_oos=n,
                        hit_oos=round(h / n, 3) if n else None, hit_is=round(rate_is, 3)))
    rs = roi_stats(pnls)
    base = base_btts if want_yes else (1 - base_btts)
    wr = hits / tot if tot else None
    return dict(label=label, n_oos=tot, hit_rate_oos=round(wr, 4) if wr else None,
                hit_rate_is_weighted=round(is_rate_w / tot, 4) if tot else None,
                baseline_oos=round(base, 4),
                z_vs_baseline=round(wilson_z(wr, base, tot), 2) if wr else None,
                roi_oos=rs["roi"], roi_n=rs["n"], roi_z=rs["z"], per_pair=per)

report["families"]["BTTS_OUI_GOLD"] = audit_btts(BTTS_OUI_GOLD, True, "BTTS_OUI_GOLD (back G/NG Oui, filtre min_cote_h)", use_min_cote=True)
report["families"]["BTTS_OUI_GOLD_nofilter"] = audit_btts(BTTS_OUI_GOLD, True, "BTTS_OUI_GOLD sans filtre", use_min_cote=False)
report["families"]["BTTS_NON_GOLD"] = audit_btts(BTTS_NON_GOLD, False, "BTTS_NON_GOLD (back G/NG Non)")

# ---------------------------------------------------------- 6. SCORE DOMINANT / COMBO
def audit_scores(table, label, combo=False):
    per, hits, tot, pnls = [], 0, 0, []
    is_rate_w, base_w = 0.0, 0.0
    mkt_imp_sum, mkt_imp_n = 0.0, 0
    for key, d in table.items():
        ms = OOS_pair.get(key, [])
        n = len(ms)
        targets = [d["top1"], d["top2"]] if combo else [d["score"]]
        h = sum(1 for m in ms if f"{m['sa']}-{m['sb']}" in targets)
        hits += h; tot += n
        rate_is = d["combo"] if combo else d["rate"]
        is_rate_w += rate_is * n
        base_pair = sum(score_freq.get(t, 0) for t in targets) / n_oos
        base_w += base_pair * n
        for m in ms:
            emd = em_data.get(m["id"])
            if emd and emd["cs"]:
                cotes = [emd["cs"].get(t) for t in targets]
                if all(cotes):
                    mkt_imp_sum += sum(1 / c for c in cotes); mkt_imp_n += 1
                    # pari 1u sur chaque score cible
                    res = f"{m['sa']}-{m['sb']}"
                    pnl = sum((c - 1) if res == t else -1.0 for t, c in zip(targets, cotes))
                    pnls.append(pnl / len(targets))  # normalise a 1u total
        per.append(dict(pair=f"{key[0]} v {key[1]}", target="+".join(targets), n_oos=n,
                        hit_oos=round(h / n, 3) if n else None, hit_is=round(rate_is, 3),
                        baseline_global=round(base_pair, 3)))
    rs = roi_stats(pnls)
    wr = hits / tot if tot else None
    base = base_w / tot if tot else None
    return dict(label=label, n_oos=tot, hit_rate_oos=round(wr, 4) if wr is not None else None,
                hit_rate_is_weighted=round(is_rate_w / tot, 4) if tot else None,
                baseline_global_weighted=round(base, 4) if base else None,
                market_implied_avg=round(mkt_imp_sum / mkt_imp_n, 4) if mkt_imp_n else None,
                z_vs_baseline=round(wilson_z(wr, base, tot), 2) if (wr is not None and base) else None,
                roi_oos=rs["roi"], roi_n=rs["n"], roi_z=rs["z"], per_pair=per)

report["families"]["SCORE_DOMINANT_GOLD"] = audit_scores(SCORE_DOMINANT_GOLD, "SCORE_DOMINANT_GOLD (back score exact)")
report["families"]["SCORE_COMBO_GOLD"] = audit_scores(SCORE_COMBO_GOLD, "SCORE_COMBO_GOLD (back top1+top2)", combo=True)

# ---------------------------------------------------------- dump + resume
with open("exports/wf5_pair_gold_audit.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=1)

print("\n================ RESUME ================")
for name, fam in report["families"].items():
    if name == "BTTS_OUI_GOLD_nofilter":
        continue
    keys = ["n_oos", "wr_oos", "wr_is_weighted", "implied_wr_avg", "hit_rate_oos",
            "hit_rate_is_weighted", "baseline_oos", "baseline_global_weighted",
            "market_implied_avg", "home_wr_oos", "implied_home_wr",
            "roi_oos", "roi_oos_with_25line", "roi_z", "z_vs_implied", "z_vs_baseline"]
    s = "  ".join(f"{k}={fam[k]}" for k in keys if fam.get(k) is not None)
    print(f"{name}: {s}")
print("\nJSON: exports/wf5_pair_gold_audit.json")
