"""Alignment in the 87 date-stating loans whose plan text contains the stated day as a numeral (Online Appendix H).
Requires the raw plan texts (private/plans.pkl, column plan), which are not public; not runnable from the public files alone."""
import numpy as np, pandas as pd, re
m=pd.read_pickle('private/plans.pkl'); st=m[m.day_v3.notna()]  # loan_id, plan (raw text), day_v3; not distributed
lit=st.apply(lambda x: bool(re.search(r'(?<!\d)%d\s*일'%int(x.day_v3),x.plan)),axis=1); L=set(st.loan_id[lit]); print('literal',len(L),'of',len(st))
d=pd.read_csv('data/analysis_v3_public.csv'); p=pd.read_csv('data/panel_long.csv'); p=p[p.loan_id.isin(d.loan_id)&p.actual_date.notna()].copy()
p['aday']=pd.to_datetime(p.actual_date).dt.day; p['sday']=pd.to_datetime(p.sched_date).dt.day
circ=lambda a,b: np.minimum(np.abs(a-b),30-np.abs(a-b))
def sgn(a,b):
    x=(a-b)%30; return np.where(x>15,x-30,x)
wrap=lambda x: ((x-1)%30)+1
pay={l:g.aday.to_numpy(float) for l,g in p.groupby('loan_id')}; due=p.groupby('loan_id').sday.agg(lambda s:s.mode().iloc[0])
dd=d.set_index('loan_id'); A=dd[dd.day_v3.notna()&dd.index.isin(pay)]; A=A.assign(due=due[A.index]); A['sg']=sgn(A.day_v3.to_numpy(),A.due.to_numpy())
don=dd[dd.day_v3.isna()&dd.salday.isna()&dd.index.isin(pay)]; don=don[[len(pay[l])>=3 for l in don.index]]; don_ids=don.index.to_numpy(); don_due=due[don_ids].to_numpy(float)
sh=lambda l,day:(circ(pay[l],day)<=3).mean(); B=10000
for seed in (1,2):
  rng=np.random.default_rng(seed)
  for lab,G in (('all 91',A),('literal 87',A[A.index.isin(L)]),('non-literal 4',A[~A.index.isin(L)])):
    for sub,ids in (('all',G.index),('gap>=4',G.index[np.abs(G.sg)>=4])):
        if len(ids)<5: print(lab,sub,'n',len(ids),'obs %.1f'%(100*np.mean([sh(l,G.day_v3[l]) for l in ids])) if len(ids) else ''); continue
        o=np.mean([sh(l,G.day_v3[l]) for l in ids]); r=G.day_v3[ids].to_numpy(); g=G.sg[ids].to_numpy()
        n0=np.array([np.mean([sh(l,x) for l,x in zip(ids,rng.permutation(r))]) for _ in range(B)])
        n1=np.array([np.mean([sh(l,wrap(G.due[l]+x)) for l,x in zip(ids,rng.permutation(g))]) for _ in range(B)])
        pools={l:don_ids[circ(don_due,G.due[l])<=1] for l in ids}
        n2=np.array([np.mean([sh(rng.choice(pools[l]),G.day_v3[l]) for l in ids]) for _ in range(2000)])
        pv=lambda n:(1+(n>=o).sum())/(len(n)+1)
        print('%-14s %-7s loans %2d inst %3d | obs %.1f | plain %.1f (p=%.4f) | B1 %.1f (p=%.4f) | B2 %.1f (p=%.4f) | due %.1f'%(lab,sub,len(ids),sum(len(pay[l]) for l in ids),100*o,100*n0.mean(),pv(n0),100*n1.mean(),pv(n1),100*n2.mean(),pv(n2),100*np.mean([sh(l,G.due[l]) for l in ids])))
