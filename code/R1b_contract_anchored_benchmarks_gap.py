"""Contract-anchored benchmarks that keep each loan's gap between stated day and due day
(Table 3, Benchmark 1; Online Appendix H). Run from the folder that contains data/."""
import numpy as np, pandas as pd
d=pd.read_csv('data/analysis_v3_public.csv'); p=pd.read_csv('data/panel_long.csv'); p=p[p.loan_id.isin(d.loan_id)&p.actual_date.notna()].copy()
p['aday']=pd.to_datetime(p.actual_date).dt.day; p['sday']=pd.to_datetime(p.sched_date).dt.day
p['off']=(pd.to_datetime(p.actual_date)-pd.to_datetime(p.sched_date)).dt.days
circ=lambda a,b: np.minimum(np.abs(a-b),30-np.abs(a-b))
def sgn(a,b):
    x=(a-b)%30; return np.where(x>15,x-30,x)
pay={l:g.aday.to_numpy(float) for l,g in p.groupby('loan_id')}
due=p.groupby('loan_id').sday.agg(lambda s:s.mode().iloc[0])
A=d[d.day_v3.notna()&d.loan_id.isin(pay)].set_index('loan_id'); A=A.assign(due=due[A.index],); A['sg']=sgn(A.day_v3.to_numpy(),A.due.to_numpy())
share=lambda l,day: (circ(pay[l],day)<=3).mean()
obs_i=pd.Series({l:share(l,A.day_v3[l]) for l in A.index})
wrap=lambda x: ((x-1)%30)+1
don=d[d.day_v3.isna()&d.salday.isna()&d.loan_id.isin(pay)].set_index('loan_id'); don=don[[len(pay[l])>=3 for l in don.index]]
print('date-stating loans',len(A),'| donor loans (no stated day, no payday, >=3 payments)',len(don))
print('share of payments made before due date: staters %.1f%%, donors %.1f%%'%(100*(p[p.loan_id.isin(A.index)].off<0).mean(),100*(p[p.loan_id.isin(don.index)].off<0).mean()))
print('offset distribution (days from due date), donors, pct in [-15..-8],[-7..-4],[-3..-1],0,[1..3],[4..15]:',[round(100*((p[p.loan_id.isin(don.index)].off>=a)&(p[p.loan_id.isin(don.index)].off<=b)).mean(),1) for a,b in ((-15,-8),(-7,-4),(-3,-1),(0,0),(1,3),(4,15))])
print('same, staters:',[round(100*((p[p.loan_id.isin(A.index)].off>=a)&(p[p.loan_id.isin(A.index)].off<=b)).mean(),1) for a,b in ((-15,-8),(-7,-4),(-3,-1),(0,0),(1,3),(4,15))])
rng=np.random.default_rng(20261113)
don_ids=don.index.to_numpy(); don_due=due[don_ids].to_numpy(float)
q=pd.qcut(d.set_index('loan_id').on_time_rate.rank(method='first'),4,labels=False)
subs={'all 91':A.index,'gap<=3 (15)':A.index[np.abs(A.sg)<=3],'gap>=4 (76)':A.index[np.abs(A.sg)>=4],'gap>=7 (59)':A.index[np.abs(A.sg)>=7],'stated before due, >=4 (40)':A.index[A.sg<=-4],'stated after due, >=4 (36)':A.index[A.sg>=4]}
B=2000
def donor_null(ids,matched):
    out=np.empty(B); 
    pools={k:don_ids[(q[don_ids]==k).to_numpy()] for k in range(4)} if matched else None
    for b in range(B):
        tot=0
        for l in ids:
            j=rng.choice(pools[q[l]]) if matched else rng.choice(don_ids)
            tot+=share(j,wrap(due[j]+A.sg[l]))
        out[b]=tot/len(ids)
    return out
def gap_perm(ids):
    ids=np.array(ids); g=A.sg[ids].to_numpy(); out=np.empty(B)
    for b in range(B):
        gp=rng.permutation(g); out[b]=np.mean([share(l,wrap(A.due[l]+x)) for l,x in zip(ids,gp)])
    return out
print('\nsample | observed | A: comparison loans, same gap | B: comparison loans matched on on-time quartile, same gap | C: Benchmark 1, own due day + another stater\'s gap')
for k,ids in subs.items():
    o=obs_i[ids].mean(); r=[f'{k:30s} obs {100*o:5.1f}']
    for nm,nul in (('A',donor_null(ids,False)),('B',donor_null(ids,True)),('C',gap_perm(ids))):
        r.append(f'{nm}: {100*nul.mean():5.1f} (excess {100*(o-nul.mean()):+5.1f}, p={(int((nul>=o).sum())+1)/(B+1):.3f})')
    print(' | '.join(r))
