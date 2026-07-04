# -*- coding: utf-8 -*-
# Tableau final 3x3 : p1_result domicile (V/N/D) x p1_result exterieur (V/N/D)
# resid = mean(home_win - imp) — pari DOMICILE ; TRAIN / TEST / FULL
import pandas as pd, numpy as np

CSV = "d:/AGENTOVA/SAMY/virtual-sports-scraper/data/vfoot_ml/trajectory.csv"
d = pd.read_csv(CSV)
d['ts'] = pd.to_datetime(d.ts)

h = d[d.venue == 'H'].copy()
a = d[d.venue == 'A'][['ts', 'team', 'p1_result', 'd_odds', 'odds', 'imp']].copy()
a = a.rename(columns={'team': 'away_team', 'p1_result': 'away_p1_result',
                      'd_odds': 'away_d_odds', 'odds': 'away_odds', 'imp': 'away_imp'})
h = h.drop_duplicates(subset=['ts', 'team'], keep='first')
a = a.drop_duplicates(subset=['ts', 'away_team'], keep='first')
m = h.merge(a, left_on=['opp', 'ts'], right_on=['away_team', 'ts'], how='inner')
m['home_win'] = (m.gf > m.ga).astype(int)
m['away_win'] = (m.gf < m.ga).astype(int)
med = m.ts.median()
tr, te = m[m.ts <= med], m[m.ts > med]
print("matches:", len(m), "| train:", len(tr), "| test:", len(te), "| mediane:", med)
print("calib globale resid home FULL:", round(float((m.home_win - m.imp).mean()), 5))

order = ['W', 'D', 'L']
print("\n=== TABLEAU 3x3 — resid pari DOMICILE (home_win - imp_home) ===")
rows = []
for ph in order:
    for pa in order:
        sub_f = m[(m.p1_result == ph) & (m.away_p1_result == pa)]
        sub_tr = tr[(tr.p1_result == ph) & (tr.away_p1_result == pa)]
        sub_te = te[(te.p1_result == ph) & (te.away_p1_result == pa)]
        rows.append(dict(
            homePrev=ph, awayPrev=pa,
            n_full=len(sub_f),
            resid_train=round(float((sub_tr.home_win - sub_tr.imp).mean()), 4),
            resid_test=round(float((sub_te.home_win - sub_te.imp).mean()), 4),
            resid_full=round(float((sub_f.home_win - sub_f.imp).mean()), 4),
            roi_full_pct=round(float((sub_f.home_win * sub_f.odds - 1).mean()) * 100, 2),
            resid_away_full=round(float((sub_f.away_win - sub_f.away_imp).mean()), 4),
            roi_away_full_pct=round(float((sub_f.away_win * sub_f.away_odds - 1).mean()) * 100, 2),
        ))
t = pd.DataFrame(rows)
pd.set_option('display.width', 220)
print(t.to_string(index=False))

# matrices compactes
print("\n--- matrice resid_full (pari DOMICILE), lignes=prev domicile, cols=prev exterieur ---")
print(t.pivot(index='homePrev', columns='awayPrev', values='resid_full').loc[order, order].to_string())
print("\n--- matrice resid_train ---")
print(t.pivot(index='homePrev', columns='awayPrev', values='resid_train').loc[order, order].to_string())
print("\n--- matrice resid_test ---")
print(t.pivot(index='homePrev', columns='awayPrev', values='resid_test').loc[order, order].to_string())
