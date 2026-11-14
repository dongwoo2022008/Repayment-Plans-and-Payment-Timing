"""Sibling-loan alignment (Section 5.1, Online Appendix H).
Do borrowers who state a repayment day in one loan pay their other loans, whose plans
give no date, near that day? Also prints the due-day column of Table 3, Panel C."""
import numpy as np, pandas as pd, sys
d=pd.read_csv('data/analysis_v3_public.csv'); p=pd.read_csv('data/panel_long.csv'); p=p[p.loan_id.isin(d.loan_id)].copy()
p['sd']=pd.to_datetime(p.sched_date,errors='coerce'); p['ad']=pd.to_datetime(p.actual_date,errors='coerce')
start=p.groupby('loan_id').sd.min(); end=pd.concat([p.groupby('loan_id').sd.max(),p.groupby('loan_id').ad.max()],axis=1).max(axis=1)
due=p.groupby('loan_id').sd.agg(lambda s:s.dt.day.mode().iloc[0] if s.notna().any() else np.nan)
pp=p[p.ad.notna()]; pay={l:g.ad.dt.day.to_numpy(float) for l,g in pp.groupby('loan_id')}
circ=lambda a,b: np.minimum(np.abs(a-b),30-np.abs(a-b))
sgn=lambda a,b: (lambda x: np.where(x>15,x-30,x))((np.asarray(a,float)-np.asarray(b,float))%30)
wrap=lambda x: int(((x-1)%30)+1)
d['list_date']=pd.to_datetime('20'+d.loan_id.str[1:7],format='%Y%m%d',errors='coerce'); d=d.set_index('loan_id',drop=False)
nodate=d.day_v3.isna()&d.salday.isna()&(d.pay_sched_rx==0)&d.loan_id.isin(pay)
def table(W):  # share of payments within +-W of each reference day 1..31
    return {l:np.array([np.nan]+[(circ(v,x)<=W).mean() for x in range(1,32)]) for l,v in pay.items()}
def pairs(base_ids,col):
    rows=[]
    for a in base_ids:
        bg=d.bgroup[a]; s=d.loc[a,col]
        for b in d.index[(d.bgroup==bg)&(d.index!=a)&nodate]:
            rows.append(dict(A=a,B=b,bg=bg,s=float(s),dueB=float(due[b]),g=float(sgn(min(s,30),min(due[b],30))),overlap=not(end[b]<start[a] or end[a]<start[b]),Bfirst=d.list_date[b]<d.list_date[a],gapm=abs((d.list_date[b]-d.list_date[a]).days)/30.4))
    r=pd.DataFrame(rows); r['dist']=r.g.abs(); return r
def agg(r,vals): return pd.Series(vals,index=r.index).groupby(r.bg).mean()   # borrower-level means
donors=d[nodate&np.array([len(pay.get(l,[]))>=3 for l in d.index])]
don_due=donors.loan_id.map(due).to_numpy(float); don_bg=donors.bgroup.to_numpy(); don_id=donors.loan_id.to_numpy()
def evaluate(r,S,rng,B=2000,verbose=True,boot=5000):
    obs_pair=np.array([S[b][int(s)] for b,s in zip(r.B,r.s)]); obs_b=agg(r,obs_pair); obs=obs_b.mean()
    g=r.g.to_numpy(); dueB=r.dueB.to_numpy(); Bs=r.B.to_numpy()
    # benchmark 1: own contract fixed, gaps reassigned across pairs
    n1=np.empty(B)
    for i in range(B):
        gp=rng.permutation(g); n1[i]=agg(r,[S[b][wrap(db+x)] for b,db,x in zip(Bs,dueB,gp)]).mean()
    # benchmark 2: same calendar day on other borrowers' no-date loans with due day within one day
    pools=[don_id[(circ(np.minimum(don_due,30),min(db,30))<=1)&(don_bg!=bg)] for db,bg in zip(dueB,r.bg)]
    n2=np.empty(B)
    for i in range(B):
        n2[i]=agg(r,[S[rng.choice(pl)][int(s)] for pl,s in zip(pools,r.s)]).mean()
    p1=(1+(n1>=obs).sum())/(B+1); p2=(1+(n2>=obs).sum())/(B+1)
    if not verbose: return obs,n1.mean(),p1,n2.mean(),p2
    # per-pair expected benchmark-1 share (exact over the gap pool) for the transfer ratio
    e1=np.array([np.mean([S[b][wrap(db+x)] for x in g]) for b,db in zip(Bs,dueB)]); sib_ex=agg(r,obs_pair-e1)
    a=r.drop_duplicates('A'); ga=sgn(np.minimum(a.s,30),np.minimum(a.A.map(due),30)); dueA=a.A.map(due).to_numpy(float)
    own_pair=np.array([S[x][int(s)] for x,s in zip(a.A,a.s)]); eo=np.array([np.mean([S[x][wrap(da+y)] for y in ga]) for x,da in zip(a.A,dueA)])
    own_ex=pd.Series(own_pair-eo,index=a.index).groupby(a.bg).mean(); own_sh=pd.Series(own_pair,index=a.index).groupby(a.bg).mean()
    bgs=sib_ex.index.to_numpy(); se=sib_ex.to_numpy(); oe=own_ex[bgs].to_numpy(); rr=np.random.default_rng(7); ratio=[]; diff=[]; sx=[]
    for _ in range(boot):
        k=rr.integers(0,len(bgs),len(bgs)); sx.append(se[k].mean()); diff.append(oe[k].mean()-se[k].mean()); ratio.append(se[k].mean()/oe[k].mean() if oe[k].mean()!=0 else np.nan)
    q=lambda v: tuple(np.nanpercentile(v,[2.5,97.5]))
    return dict(borrowers=r.bg.nunique(),pairs=len(r),sib_loans=r.B.nunique(),payments=int(sum(len(pay[b]) for b in r.B.unique())),sibling=100*obs,b1=100*n1.mean(),p1=p1,b2=100*n2.mean(),p2=p2,own_share=100*own_sh.mean(),own_excess=100*oe.mean(),sib_excess=100*se.mean(),sib_excess_ci=tuple(100*x for x in q(sx)),ratio=se.mean()/oe.mean(),ratio_ci=q(ratio),diff=100*(oe.mean()-se.mean()),diff_ci=tuple(100*x for x in q(diff)))
def show(lab,o):
    print('%-34s %2d borrowers / %3d pairs / %3d loans / %4d payments | sibling %.1f | B1 %.1f (p=%.4f) | B2 %.1f (p=%.4f) | own share %.1f, own excess %.1f | sibling excess %.1f [%.1f, %.1f] | transfer ratio %.2f [%.2f, %.2f] | own-sibling %.1f [%.1f, %.1f]'%(lab,o['borrowers'],o['pairs'],o['sib_loans'],o['payments'],o['sibling'],o['b1'],o['p1'],o['b2'],o['p2'],o['own_share'],o['own_excess'],o['sib_excess'],*o['sib_excess_ci'],o['ratio'],*o['ratio_ci'],o['diff'],*o['diff_ci']))
if __name__=='__main__':
    S3=table(3); base=d.index[d.day_v3.notna()&d.loan_id.isin(pay)]; R=pairs(base,'day_v3'); rng=np.random.default_rng(20261113)
    M=R[R.dist>=4]
    for lab,r in [('gap >=4 days (main)',M),('gap >=4, excluding overlapping pair',M[~M.overlap]),('gap >=4, sibling loan earlier',M[M.Bfirst]),('gap >=4, sibling loan later',M[~M.Bfirst]),('gap >=4, within 24 months',M[M.gapm<=24]),('gap >=7 days',R[R.dist>=7]),('gap <=3 days',R[R.dist<=3])]:
        show(lab,evaluate(r.reset_index(drop=True),S3,rng))
    for W in (1,5): show('gap >=4, window +-%d'%W,evaluate(M.reset_index(drop=True),table(W),rng))
    pb=d.index[(d.P3v3<4)&d.salday.notna()&d.loan_id.isin(pay)]; RP=pairs(pb,'salday'); 
    print('payday-only base: pairs',len(RP),'borrowers',RP.bg.nunique(),'| gap>=4:',int((RP.dist>=4).sum()),RP[RP.dist>=4].bg.nunique())
    if RP[RP.dist>=4].bg.nunique()>=5: show('payday-only base, gap >=4',evaluate(RP[RP.dist>=4].reset_index(drop=True),S3,rng))
    # Table 3, Panel C, last column: share of each sibling loan's payments within +-3 days of its own scheduled day
    own={l:(circ(g.ad.dt.day.to_numpy(float),g.sd.dt.day.to_numpy(float))<=3).mean() for l,g in pp.groupby('loan_id')}
    for lab,r in (('gap >=4 days',M),('gap >=7 days',R[R.dist>=7])): print('%-14s due-day share in sibling loans %.1f'%(lab,100*agg(r,[own[b] for b in r.B]).mean()))
