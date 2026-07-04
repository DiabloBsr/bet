import pandas as pd, numpy as np
from scipy import stats

m=pd.read_pickle('D:/AGENTOVA/SAMY/virtual-sports-scraper/tmp_refute/faceoff.pkl')
med=m['ts'].median()

def cell(df):
    return df[(df.p1_result=='W')&(df.away_p1_result=='D')&(df.d_odds<0)].dropna(subset=['odds','imp','win'])

early=cell(m[m.ts<=med]); late=cell(m[m.ts>med]); full=cell(m)

print('=== REVERSE SPLIT: select on LATE (train), validate on EARLY (test) ===')
w=int(early.win.sum()); n=len(early); p0=early.imp.mean()
pv=stats.binomtest(w,n,p0,alternative='greater').pvalue
roi=(early.win*early.odds-1).mean()*100
print(f'EARLY as TEST: n={n} resid={w/n-p0:+.4f} binom_p={pv:.4f} ROI={roi:+.2f}%')
print(f'-> Passes method (p<~0.05 pre-FDR AND ROI>0)? {"YES" if (pv<0.05 and roi>0) else "NO"}')

print()
print('=== FULL-SAMPLE ROI bootstrap CI ===')
rng=np.random.default_rng(7)
vals=(full.win.values*full.odds.values-1)
boots=np.array([rng.choice(vals,len(vals),replace=True).mean() for _ in range(5000)])
print(f'ROI_full={vals.mean()*100:+.2f}%  CI95=[{np.percentile(boots,2.5)*100:+.2f}%, {np.percentile(boots,97.5)*100:+.2f}%]  P(ROI<=0)={np.mean(boots<=0):.3f}')

print()
print('=== Margin structure: is imp normalized (imp < 1/odds)? ===')
full2=full.copy(); full2['raw']=1/full2.odds
print(f'mean imp={full2.imp.mean():.4f}  mean 1/odds (break-even wr)={full2.raw.mean():.4f}  ratio={ (full2.raw/full2.imp).mean():.4f}')
print(f'-> resid=win-imp beats the MARGIN-FREE prob; break-even resid needed = {full2.raw.mean()-full2.imp.mean():+.4f}')

print()
print('=== Campaign-level multiplicity ===')
# within this lot: 21 candidate one-sided tests, min p = 0.0029
for ntests in [21, 21*8, 124*8]:
    print(f'P(min p <= 0.0029 among {ntests} null tests) = {1-(1-0.0029)**ntests:.3f}')

print()
print('=== How unusual is the survivor among its 18-cell family? cross-half swing distribution ===')
m2=m.dropna(subset=['odds','imp','win','p1_result','away_p1_result','d_odds']).copy()
m2['dir']=np.where(m2.d_odds<0,'DOWN','UP')
tr=m2[m2.ts<=med]; te=m2[m2.ts>med]
sw=[]
for ph in ['W','D','L']:
    for pa in ['W','D','L']:
        for dr in ['DOWN','UP']:
            a=tr[(tr.p1_result==ph)&(tr.away_p1_result==pa)&(tr['dir']==dr)]
            b=te[(te.p1_result==ph)&(te.away_p1_result==pa)&(te['dir']==dr)]
            ra=a.win.mean()-a.imp.mean(); rb=b.win.mean()-b.imp.mean()
            sw.append(abs(rb-ra))
print(f'mean |resid swing| across 18 cells = {np.mean(sw):.4f}; survivor swing = 0.0206; survivor TEST resid = 0.0503')
print(f'cells with |swing| >= 0.0206: {sum(s>=0.0206 for s in sw)}/18')
