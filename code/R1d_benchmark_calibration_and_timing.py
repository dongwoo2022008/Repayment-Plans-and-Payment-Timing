"""Calibration of the alignment benchmarks under a true null, and timing of payments near stated days that follow the due day.
Placebo design: loans that state neither a repayment day nor a payday receive pseudo stated days whose gaps to the due day are
drawn from the gaps of the real date-stating loans in the matching subsample. Run from the folder that contains data/."""
import numpy as np, pandas as pd
d=pd.read_csv('data/analysis_v3_public.csv'); p=pd.read_csv('data/panel_long.csv'); p=p[p.loan_id.isin(d.loan_id)&p.actual_date.notna()].copy()
p['ad']=pd.to_datetime(p.actual_date); p['sd']=pd.to_datetime(p.sched_date); p['aday']=p.ad.dt.day; p['sday']=p.sd.dt.day; p['off']=(p.ad-p.sd).dt.days
circ=lambda a,b: np.minimum(np.abs(a-b),30-np.abs(a-b)); wrap=lambda x: ((x-1)%30)+1
def sgn(a,b):
    x=(a-b)%30; return np.where(x>15,x-30,x)
pay={l:g.aday.to_numpy(float) for l,g in p.groupby('loan_id')}; due=p.groupby('loan_id').sday.agg(lambda s:s.mode().iloc[0])
dd=d.set_index('loan_id'); A=dd[dd.day_v3.notna()&dd.index.isin(pay)].copy(); A['due']=due[A.index]; A['sg']=sgn(A.day_v3.to_numpy(),A.due.to_numpy())
don=dd[dd.day_v3.isna()&dd.salday.isna()&dd.index.isin(pay)]; don=don[[len(pay[l])>=3 for l in don.index]]; ids=don.index.to_numpy(); n=len(ids); dued=due[ids].to_numpy(int)
N=np.array([[ (circ(pay[l],day)<=3).mean() for day in range(1,31)] for l in ids])      # share near calendar day
pools=[np.where((circ(dued,dued[i])<=1)&(np.arange(n)!=i))[0] for i in range(n)]
rng=np.random.default_rng(2026)
plen=np.array([len(x) for x in pools]); Ppad=np.zeros((n,plen.max()),int)
for i_,x in enumerate(pools): Ppad[i_,:len(x)]=x
def simulate(gaps,m,R=2000,Bi=499,label=''):
    res={k:[] for k in ('plain','B1','B2')}; rej={k:0 for k in res}
    for r in range(R):
        s=rng.choice(n,m,replace=False); g=rng.choice(gaps,m,replace=True); day=wrap(dued[s]+g).astype(int)
        obs=N[s,day-1].mean()
        perm=np.argsort(rng.random((Bi,m)),axis=1)
        nul_p=N[s[None,:],day[perm]-1].mean(1)
        nul_1=N[s[None,:],wrap(dued[s][None,:]+g[perm]).astype(int)-1].mean(1)
        idx=(rng.random((Bi,m))*plen[s][None,:]).astype(int); donors=Ppad[s[None,:],idx]
        nul_2=N[donors,(day-1)[None,:]].mean(1)
        for k,nul in (('plain',nul_p),('B1',nul_1),('B2',nul_2)):
            res[k].append(obs-nul.mean()); rej[k]+= ((nul>=obs).sum()+1)/(Bi+1)<=0.05
    print(f'{label:34s} m={m:2d} R={R} | '+' | '.join(f'{k}: mean excess {100*np.mean(res[k]):+5.2f}pp, rejection at 5% {100*rej[k]/R:4.1f}%' for k in res),flush=True)
simulate(A.sg[np.abs(A.sg)>=4].to_numpy(),76,label='gap >=4 (as 76 staters)')
simulate(A.sg[np.abs(A.sg)>=7].to_numpy(),59,label='gap >=7 (as 59 staters)')
simulate(A.sg[A.sg<=-4].to_numpy(),40,label='stated before due (40)')
simulate(A.sg[A.sg>=4].to_numpy(),36,label='stated after due (36)')
simulate(A.sg.to_numpy(),91,label='all gaps (as 91 staters)')
# ---- timing of payments near the stated day, by side
q=p[p.loan_id.isin(A.index)].join(A[['day_v3','sg']],on='loan_id'); q['near']=circ(q.aday.to_numpy(float),q.day_v3.to_numpy())<=3
don_p=p[p.loan_id.isin(ids)]
def prof(x,lab):
    o=x.off; print(f'{lab:44s} n={len(x):4d} | paid >=8 d early {100*(o<=-8).mean():5.1f}% | 4-7 d early {100*((o>=-7)&(o<=-4)).mean():5.1f}% | within ±3 of due {100*(o.abs()<=3).mean():5.1f}% | 4-7 d late {100*((o>=4)&(o<=7)).mean():5.1f}% | >=8 d late {100*(o>=8).mean():5.1f}% | late >=1 d {100*(o>=1).mean():5.1f}% | median offset {o.median():.0f}')
for lab,c in (('after-due staters: payments near stated day',(q.sg>=4)&q.near),('after-due staters: other payments',(q.sg>=4)&~q.near),('after-due staters: all payments',q.sg>=4),('before-due staters: payments near stated day',(q.sg<=-4)&q.near),('before-due staters: all payments',q.sg<=-4)): prof(q[c],lab)
prof(don_p,'non-stating loans: all payments')
L=q.groupby('loan_id').agg(sg=('sg','first'),late1=('off',lambda s:(s>=1).mean()),late7=('off',lambda s:(s>=7).mean()))
for lab,c in (('after (>=4)',L.sg>=4),('near (<=3)',L.sg.abs()<=3),('before (<=-4)',L.sg<=-4)): print(f'loan-level mean share of installments late: {lab:14s} >=1 day {100*L[c].late1.mean():.1f}%  >=7 days {100*L[c].late7.mean():.1f}%  (loans {c.sum()})')
