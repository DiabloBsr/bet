import pandas as pd, numpy as np
from scipy import stats

m=pd.read_pickle('D:/AGENTOVA/SAMY/virtual-sports-scraper/tmp_refute/faceoff.pkl')

def cell(df):
    return df[(df.p1_result=='W')&(df.away_p1_result=='D')&(df.d_odds<0)].dropna(subset=['odds','imp','win'])

def st(c):
    n=len(c)
    if n==0: return None
    w=int(c.win.sum()); p0=c.imp.mean(); roi=(c.win*c.odds-1).mean()
    pv=stats.binomtest(w,n,p0,alternative='greater').pvalue
    return n,w/n,p0,(w/n)-p0,roi*100,pv

print('=== 1) SPLIT SENSITIVITY: TEST = data after quantile q of ts ===')
for q in [0.3,0.4,0.45,0.5,0.55,0.6,0.7]:
    cut=m['ts'].quantile(q)
    tr=cell(m[m.ts<=cut]); te=cell(m[m.ts>cut])
    a=st(tr); b=st(te)
    print(f'q={q:.2f} cut={cut.date()} | TRAIN n={a[0]} resid={a[3]:+.4f} ROI={a[4]:+.2f}% | TEST n={b[0]} resid={b[3]:+.4f} ROI={b[4]:+.2f}% p={b[5]:.4f}')

print()
print('=== 2) WEEKLY breakdown of the cell (full sample) ===')
c=cell(m).copy()
c['week']=c['ts'].dt.to_period('W')
g=c.groupby('week').apply(lambda x: pd.Series({'n':len(x),'resid':x.win.mean()-x.imp.mean(),'ROI%':((x.win*x.odds-1).mean())*100}))
print(g.to_string())

print()
print('=== 3) TEST split into quarters (chrono) ===')
med=m['ts'].median()
cte=cell(m[m.ts>med]).sort_values('ts').reset_index(drop=True)
idx=np.array_split(np.arange(len(cte)),4)
for i,ix in enumerate(idx):
    part=cte.iloc[ix]
    a=st(part)
    print(f'Q{i+1} [{part.ts.min().date()}..{part.ts.max().date()}] n={a[0]} resid={a[3]:+.4f} ROI={a[4]:+.2f}% p={a[5]:.4f}')

print()
print('=== 4) MECHANISM: 3x3x2 grid (p1H x p1A x dirH), HOME bet, TRAIN vs TEST resid/ROI ===')
m2=m.dropna(subset=['odds','imp','win','p1_result','away_p1_result','d_odds']).copy()
m2['dir']=np.where(m2.d_odds<0,'DOWN','UP')
tr=m2[m2.ts<=med]; te=m2[m2.ts>med]
rows=[]
for ph in ['W','D','L']:
    for pa in ['W','D','L']:
        for dr in ['DOWN','UP']:
            ctr=tr[(tr.p1_result==ph)&(tr.away_p1_result==pa)&(tr['dir']==dr)]
            cte2=te[(te.p1_result==ph)&(te.away_p1_result==pa)&(te['dir']==dr)]
            rows.append({'cell':f'({ph},{pa},{dr})','n_tr':len(ctr),
                'resid_tr':ctr.win.mean()-ctr.imp.mean(),'roi_tr':((ctr.win*ctr.odds-1).mean())*100,
                'n_te':len(cte2),'resid_te':cte2.win.mean()-cte2.imp.mean(),'roi_te':((cte2.win*cte2.odds-1).mean())*100})
print(pd.DataFrame(rows).to_string(index=False,float_format=lambda x:f'{x:+.4f}'))

print()
print('=== 5) TRAIN vs TEST winrate difference in the cell (is TEST regime distinguishable?) ===')
ctr=cell(m[m.ts<=med]); cte=cell(m[m.ts>med])
n1,n2=len(ctr),len(cte); w1,w2=ctr.win.sum(),cte.win.sum()
p_pool=(w1+w2)/(n1+n2)
se=np.sqrt(p_pool*(1-p_pool)*(1/n1+1/n2))
z=(w2/n2-w1/n1)/se
print(f'TRAIN wr={w1/n1:.4f} TEST wr={w2/n2:.4f} diff z={z:.3f} p(two-sided)={2*(1-stats.norm.cdf(abs(z))):.3f}')
print(f'-> pooled wr={p_pool:.4f}; TRAIN ROI={((ctr.win*ctr.odds-1).mean())*100:+.2f}% ; if true wr = TRAIN wr, ROI at TEST odds = {(w1/n1*cte.odds.mean()-1)*100:+.2f}%')

print()
print('=== 6) Bootstrap 95% CI of ROI_test ===')
rng=np.random.default_rng(42)
vals=(cte.win.values*cte.odds.values-1)
boots=[rng.choice(vals,len(vals),replace=True).mean() for _ in range(5000)]
print(f'ROI_test={vals.mean()*100:+.2f}%  CI95=[{np.percentile(boots,2.5)*100:+.2f}%, {np.percentile(boots,97.5)*100:+.2f}%]')
