"""FACETTE chains : dynamique d'état in-match (catch-up mechanics).

1. Taux de but par minute selon le différentiel courant (-2..+2), home/away séparés
2. Catch-up : l'équipe menée marque-t-elle plus que son taux de base ?
3. Comebacks (mené à HT → gagne) : fréquence réelle vs HT/FT implicite
4. Drama : taux de but 80-90' selon score serré vs plié
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
from sqlalchemy import create_engine
from scraper.config import load_settings

engine = create_engine(load_settings().db_url)
df = pd.read_sql("""
    SELECT e.id, e.team_a, e.team_b, e.expected_start,
           o.odds_home, o.odds_draw, o.odds_away, o.extra_markets,
           r.score_a, r.score_b, r.ht_score_a, r.ht_score_b, r.goals_json
    FROM events e
    JOIN odds_snapshots o ON o.id = (SELECT MIN(id) FROM odds_snapshots WHERE event_id = e.id)
    JOIN results r ON r.event_id = e.id
    WHERE r.score_a IS NOT NULL AND e.round_info IS NOT NULL AND e.round_info != '0'
    ORDER BY e.expected_start
""", engine)
df = df.drop_duplicates(["team_a", "team_b", "expected_start"]).reset_index(drop=True)
# exclure corrompus
try:
    corrupted = set(json.load(open("exports/corrupted_events.json", encoding="utf-8"))["hard"])
except Exception:
    try:
        corrupted = set(json.load(open("exports/corrupted_events.json", encoding="utf-8")))
    except Exception:
        corrupted = set()
df = df[~df.id.isin(corrupted)].reset_index(drop=True)
print(f"n={len(df)} (apres exclusion corrompus)")

# ============ 1+2 : hazard par différentiel ============
# Pour chaque match : reconstituer la timeline minute par minute
# exposure[diff][side] = minutes passées dans cet état ; goals[diff][side] = buts marqués
from collections import defaultdict
exposure = defaultdict(float)   # (diff_clamped) -> minutes
goals_h = defaultdict(int)      # diff -> buts home
goals_a = defaultdict(int)
n_matches_used = 0

for _, r in df.iterrows():
    gj = r.goals_json
    if gj is None or isinstance(gj, float): continue
    g = json.loads(gj) if isinstance(gj, str) else gj
    if not isinstance(g, list): continue
    # vérifier cohérence
    if len(g) != int(r.score_a) + int(r.score_b): continue
    g = sorted(g, key=lambda x: x["minute"])
    n_matches_used += 1
    cur_h = cur_a = 0
    prev_min = 0
    for ev in g:
        m = min(int(ev["minute"]), 90)
        d = max(-2, min(2, cur_h - cur_a))
        exposure[d] += (m - prev_min)
        if ev["team"] == "Home":
            goals_h[d] += 1; cur_h += 1
        else:
            goals_a[d] += 1; cur_a += 1
        prev_min = m
    d = max(-2, min(2, cur_h - cur_a))
    exposure[d] += (90 - prev_min)

print(f"matchs exploitables timeline: {n_matches_used}")
print("\n=== TAUX DE BUT PAR 90min SELON DIFFÉRENTIEL (home - away) ===")
print(f"{'diff':<6} {'expo(min)':<12} {'rate_home/90':<14} {'rate_away/90':<14} {'ratio H/A'}")
for d in [-2, -1, 0, 1, 2]:
    e_min = exposure[d]
    if e_min < 1000: continue
    rh = goals_h[d] / e_min * 90
    ra = goals_a[d] / e_min * 90
    print(f"{d:<6} {e_min:<12.0f} {rh:<14.3f} {ra:<14.3f} {rh/ra:.3f}")

# Catch-up test : le rate de l'équipe MENÉE vs son rate à 0-0
rh_0 = goals_h[0] / exposure[0] * 90
ra_0 = goals_a[0] / exposure[0] * 90
rh_down = goals_h[-1] / exposure[-1] * 90   # home mené de 1
ra_down = goals_a[1] / exposure[1] * 90     # away mené de 1
rh_up = goals_h[1] / exposure[1] * 90       # home mène de 1
ra_up = goals_a[-1] / exposure[-1] * 90     # away mène de 1
print(f"\nCATCH-UP : home rate quand mené -1 : {rh_down:.3f} vs à égalité : {rh_0:.3f} ({(rh_down/rh_0-1)*100:+.1f}%)")
print(f"           away rate quand mené -1 : {ra_down:.3f} vs à égalité : {ra_0:.3f} ({(ra_down/ra_0-1)*100:+.1f}%)")
print(f"LEADER   : home rate quand mène +1 : {rh_up:.3f} ({(rh_up/rh_0-1)*100:+.1f}%)")
print(f"           away rate quand mène +1 : {ra_up:.3f} ({(ra_up/ra_0-1)*100:+.1f}%)")

# ============ 3 : comebacks vs HT/FT marché ============
def parse_em(x):
    if isinstance(x, str):
        try: return json.loads(x)
        except Exception: return {}
    return x if isinstance(x, dict) else {}

df["em"] = df.extra_markets.apply(parse_em)
mask_ht = df.ht_score_a.notna()
d2 = df[mask_ht].copy()
d2["ht_o"] = np.where(d2.ht_score_a > d2.ht_score_b, "1", np.where(d2.ht_score_a == d2.ht_score_b, "X", "2"))
d2["ft_o"] = np.where(d2.score_a > d2.score_b, "1", np.where(d2.score_a == d2.score_b, "X", "2"))

# fréquence réelle des 9 combos vs implicite HT/FT
key_map = {("1","1"):["1/1"],("1","X"):["1/X"],("1","2"):["1/2"],
           ("X","1"):["X/1"],("X","X"):["X/X"],("X","2"):["X/2"],
           ("2","1"):["2/1"],("2","X"):["2/X"],("2","2"):["2/2"]}
print("\n=== COMBOS HT/FT : réel vs implicite (devig global marché) ===")
stats_combo = []
for (h, f), keys in key_map.items():
    obs = ((d2.ht_o == h) & (d2.ft_o == f)).mean()
    # implicite : moyenne devig des cotes du combo
    imps = []
    for _, r in d2.iterrows():
        htft = r.em.get("HT/FT")
        if not isinstance(htft, dict): continue
        # retrouver la clé (formats possibles '1/1', '1 / 1', etc.)
        for k_try in keys + [f"{h} / {f}", f"{h}-{f}", f"{h}/{f}"]:
            if k_try in htft and isinstance(htft[k_try], (int, float)) and htft[k_try] > 1:
                # devig avec overround 12%
                imps.append(1 / htft[k_try] / 1.12)
                break
    imp = np.mean(imps) if imps else float("nan")
    cote_juste = 1/obs if obs > 0 else float("inf")
    stats_combo.append((f"{h}/{f}", obs, imp, obs/imp if imp and imp > 0 else float("nan")))
    print(f"  {h}/{f} : réel={obs*100:5.2f}%  implicite={imp*100 if imp==imp else float('nan'):5.2f}%  ratio={obs/imp if imp and imp==imp else float('nan'):.3f}")

# Comeback global
comeback = (((d2.ht_o == "1") & (d2.ft_o == "2")) | ((d2.ht_o == "2") & (d2.ft_o == "1"))).mean()
print(f"\nP(comeback total) = {comeback*100:.2f}%")

# ============ 4 : drama fin de match ============
# taux de but 80-90 selon différentiel à la 80e
expo_late = defaultdict(float); goals_late = defaultdict(int)
for _, r in df.iterrows():
    gj = r.goals_json
    if gj is None or isinstance(gj, float): continue
    g = json.loads(gj) if isinstance(gj, str) else gj
    if not isinstance(g, list) or len(g) != int(r.score_a) + int(r.score_b): continue
    g = sorted(g, key=lambda x: x["minute"])
    h80 = sum(1 for x in g if x["minute"] <= 80 and x["team"] == "Home")
    a80 = sum(1 for x in g if x["minute"] <= 80 and x["team"] == "Away")
    d = max(-1, min(1, h80 - a80))
    state = "serré(0)" if d == 0 else ("écart")
    late_goals = sum(1 for x in g if x["minute"] > 80)
    expo_late[state] += 10
    goals_late[state] += late_goals

print("\n=== DRAMA 80-90' ===")
for state in ["serré(0)", "écart"]:
    if expo_late[state] > 0:
        rate = goals_late[state] / expo_late[state] * 90
        print(f"  état {state:<10} : taux de but/90 dans les 10 dernières min = {rate:.3f}")

print("\nDONE")
