import pandas as pd
d = pd.read_csv(r"d:/AGENTOVA/SAMY/virtual-sports-scraper/data/vfoot_ml/trajectory.csv")
print('dup (ts,team,venue) rows:', d.duplicated(['ts', 'team', 'venue']).sum())
print('dup full rows:', d.duplicated().sum())
h = d[d.venue == 'H']
a = d[d.venue == 'A']
print('h dup key (ts,team):', h.duplicated(['ts', 'team']).sum())
print('a dup key (ts,team):', a.duplicated(['ts', 'team']).sum())
