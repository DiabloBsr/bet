import pandas as pd, numpy as np
from scipy import stats

CSV='D:/AGENTOVA/SAMY/virtual-sports-scraper/data/vfoot_ml/trajectory.csv'
d=pd.read_csv(CSV)
n0=len(d)
d=d.drop_duplicates(['ts','team'],keep='first')
print('rows after dedup ts/team:',len(d),'(removed',n0-len(d),')')

h=d[d.venue=='H'].copy()
a=d[d.venue=='A'][['ts','team','p1_result','p1_margin','d_odds','odds','p2_result']].copy()
a.columns=['ts','away_team','away_p1_result','away_p1_margin','away_d_odds','away_odds','away_p2_result']
m=h.merge(a,left_on=['opp','ts'],right_on=['away_team','ts'],how='inner')
print('matches after merge:',len(m),'| unique (ts,team):',m.duplicated(['ts','team']).sum(),'dup rows')
m=m.drop_duplicates(['ts','team'],keep='first')
print('final matches:',len(m))

m['ts']=pd.to_datetime(m['ts'])
med=m['ts'].median()
print('median ts:',med)
m['is_test']=m['ts']>med
tr=m[~m.is_test]; te=m[m.is_test]
print('train:',len(tr),'test:',len(te))
m.to_pickle('D:/AGENTOVA/SAMY/virtual-sports-scraper/tmp_refute/faceoff.pkl')

# The claimed cell: p1_result home == W, away_p1_result == D, d_odds home < 0 -> bet HOME
def cellstats(df,label):
    c=df[(df.p1_result=='W')&(df.away_p1_result=='D')&(df.d_odds<0)]
    c=c.dropna(subset=['odds','imp','win'])
    n=len(c); w=int(c.win.sum()); p0=c.imp.mean()
    resid=c.win.mean()-p0
    roi=(c.win*c.odds-1).mean()
    pv=stats.binomtest(w,n,p0,alternative='greater').pvalue if n>0 else np.nan
    print(f'{label}: n={n} wins={w} winrate={c.win.mean():.4f} mean_imp={p0:.4f} resid={resid:+.4f} ROI={roi*100:+.2f}% binom_p={pv:.5f} mean_odds={c.odds.mean():.3f}')
    return c
ctr=cellstats(tr,'TRAIN')
cte=cellstats(te,'TEST ')
cellstats(m,'FULL ')
