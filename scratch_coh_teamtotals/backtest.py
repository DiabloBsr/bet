# coh_teamtotals — step 2: backtest 'bet the underpriced side when |gap|>threshold'
# proxy odds = 1/p_devig (no o| columns for these markets) -> upper bound on real ROI (real odds carry margin).
# Discovery on TRAIN (n>=100 & ROI_train>0), OOS validation on TEST, z-test vs breakeven + bootstrap CI, BH-FDR.
import numpy as np, pandas as pd
from scipy import stats

rng = np.random.default_rng(42)
df = pd.read_csv(r"D:\AGENTOVA\SAMY\virtual-sports-scraper\scratch_coh_teamtotals\gaps.csv")

# outcomes
y = {'TTdom_gt': (df.sa>=4), 'TText_gt': (df.sb>=4), 'GNGdom_oui': (df.sa>=1), 'GNGext_oui': (df.sb>=1)}
# (market, direction): direction '+' = bet reference side (>3.5/Oui) when gap>thr; '-' = bet complement (<3.5/Non) when gap<-thr
opp = {'TTdom_gt':'q_TTdom_lt','TText_gt':'q_TText_lt','GNGdom_oui':'q_GNGdom_non','GNGext_oui':'q_GNGext_non'}
THR = [0.005, 0.01, 0.02, 0.03, 0.05, 0.07]

rows = []
for mkt in y:
    gap = df['gap_'+mkt]
    for direc in ['+','-']:
        if direc == '+':
            qside = df['q_'+mkt]; win = y[mkt].astype(float)
        else:
            qside = df[opp[mkt]]; win = 1.0 - y[mkt].astype(float)
        for thr in THR:
            sel = (gap > thr) if direc=='+' else (gap < -thr)
            sel = sel & qside.notna() & gap.notna()
            for ph in ['train','test']:
                m = sel & (df.phase==ph)
                nb = int(m.sum())
                if nb == 0:
                    rows.append(dict(mkt=mkt, direc=direc, thr=thr, phase=ph, n=0, roi=np.nan, wr=np.nan, be=np.nan, pval=np.nan)); continue
                q = qside[m].to_numpy(); w = win[m].to_numpy()
                odds = 1.0/q
                pnl = w*odds - 1.0
                roi = pnl.mean()
                # H0: P(win_i)=q_i (fair proxy odds) -> one-sided z on wins
                mu0 = q.sum(); var0 = (q*(1-q)).sum()
                z = (w.sum()-mu0)/np.sqrt(var0) if var0>0 else np.nan
                pv = 1-stats.norm.cdf(z) if var0>0 else np.nan
                # bootstrap 95% CI on ROI (test only, done later for survivors) — store arrays via index
                rows.append(dict(mkt=mkt, direc=direc, thr=thr, phase=ph, n=nb, roi=roi, wr=w.mean(), be=q.mean(), pval=pv))
res = pd.DataFrame(rows)
piv = res.pivot_table(index=['mkt','direc','thr'], columns='phase', values=['n','roi','wr','be','pval'], aggfunc='first')
piv.columns = [f"{a}_{b}" for a,b in piv.columns]
piv = piv.reset_index()

n_cells = len(piv)
selected = piv[(piv.n_train>=100) & (piv.roi_train>0)].copy()
print(f"cells total={n_cells} | selected on train (n>=100 & ROI_train>0): {len(selected)}")

# BH-FDR on test p-values of selected cells
if len(selected):
    selected = selected.sort_values('pval_test').reset_index(drop=True)
    mtest = len(selected)
    selected['rank'] = np.arange(1, mtest+1)
    selected['bh_crit'] = 0.05*selected['rank']/mtest
    passed = selected[selected.pval_test <= selected.bh_crit]
    kmax = passed['rank'].max() if len(passed) else 0
    selected['fdr_pass'] = selected['rank'] <= kmax
    selected['survivor'] = selected.fdr_pass & (selected.roi_test>0)
else:
    selected['survivor'] = []

pd.set_option('display.width', 250)
print("\n=== ALL CELLS (train/test) ===")
print(piv.round(4).to_string(index=False))
print("\n=== SELECTED ON TRAIN -> OOS TEST + BH-FDR ===")
if len(selected):
    print(selected[['mkt','direc','thr','n_train','roi_train','n_test','roi_test','wr_test','be_test','pval_test','bh_crit','fdr_pass','survivor']].round(4).to_string(index=False))
    print(f"\nsurvivors (FDR q<0.05 & ROI_test>0): {int(selected.survivor.sum())}")
    # bootstrap CI for best test-ROI selected cell
    best = selected.sort_values('roi_test', ascending=False).iloc[0]
    mkt, direc, thr = best.mkt, best.direc, best.thr
    gap = df['gap_'+mkt]
    if direc=='+': qside = df['q_'+mkt]; win = y[mkt].astype(float)
    else: qside = df[opp[mkt]]; win = 1.0 - y[mkt].astype(float)
    m = ((gap>thr) if direc=='+' else (gap<-thr)) & qside.notna() & (df.phase=='test')
    q = qside[m].to_numpy(); w = win[m].to_numpy(); pnl = w/q - 1.0
    boots = np.array([pnl[rng.integers(0,len(pnl),len(pnl))].mean() for _ in range(10000)])
    print(f"best test cell: {mkt} {direc} thr={thr} n={len(pnl)} ROI_test={pnl.mean()*100:.2f}% bootstrap95%CI=[{np.percentile(boots,2.5)*100:.2f}%, {np.percentile(boots,97.5)*100:.2f}%]")
else:
    print("none selected -> nothing to validate OOS")
