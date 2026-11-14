"""Benchmark 2 of Table 3: the same calendar day evaluated on loans without a stated day or payday day
whose due day is within one day of the stater's due day (output label D; D-matched adds on-time quartile matching).
Output label C is Benchmark 1. Also: payday-only loans and window sensitivity. Run from the folder that contains data/."""
import numpy as np, pandas as pd
d=pd.read_csv('data/analysis_v3_public.csv'); p=pd.read_csv('data/panel_long.csv'); p=p[p.loan_id.isin(d.loan_id)&p.actual_date.notna()].copy()
p['aday']=pd.to_datetime(p.actual_date).dt.day; p['sday']=pd.to_datetime(p.sched_date).dt.day
circ=lambda a,b: np.minimum(np.abs(a-b),30-np.abs(a-b))
def sgn(a,b):
    x=(a-b)%30; return np.where(x>15,x-30,x)
wrap=lambda x: ((x-1)%30)+1
pay={l:g.aday.to_numpy(float) for l,g in p.groupby('loan_id')}
due=p.groupby('loan_id').sday.agg(lambda s:s.mode().iloc[0])
dd=d.set_index('loan_id')
don=dd[dd.day_v3.isna()&dd.salday.isna()&dd.index.isin(pay)]; don=don[[len(pay[l])>=3 for l in don.index]]; don_ids=don.index.to_numpy(); don_due=due[don_ids].to_numpy(float)
q=pd.qcut(dd.on_time_rate.rank(method='first'),4,labels=False)
rng=np.random.default_rng(7); B=2000
def run(G,col,label):
    G=G.assign(due=due[G.index]); G['sg']=sgn(G[col].to_numpy(),G.due.to_numpy())
    for w in (3,1,5):
        share=lambda l,day: (circ(pay[l],day)<=w).mean()
        obs=pd.Series({l:share(l,G[col][l]) for l in G.index})
        subs={'all':G.index,'gap<=3':G.index[np.abs(G.sg)<=3],'gap>=4':G.index[np.abs(G.sg)>=4],'gap>=7':G.index[np.abs(G.sg)>=7],'before (<=-4)':G.index[G.sg<=-4],'after (>=4)':G.index[G.sg>=4]}
        for k,ids in subs.items():
            if len(ids)<5: continue
            pools={l:don_ids[circ(don_due,G.due[l])<=1] for l in ids}
            poolsM={l:pools[l][(q[pools[l]]==q[l]).to_numpy()] for l in ids}
            o=obs[ids].mean(); res=[]
            for nm,P in (('D',pools),('D-matched',poolsM)):
                ok=[l for l in ids if len(P[l])>0]
                exp=np.mean([np.mean([share(j,G[col][l]) for j in P[l]]) for l in ok])
                nul=np.empty(B)
                for b in range(B): nul[b]=np.mean([share(rng.choice(P[l]),G[col][l]) for l in ok])
                oo=obs[ok].mean(); res.append(f'{nm}: {100*exp:5.1f} (excess {100*(oo-exp):+5.1f}, p={(int((nul>=oo).sum())+1)/(B+1):.4f}; donors/loan median {int(np.median([len(P[l]) for l in ok]))}, loans used {len(ok)})')
            # C: own due day + another loan's gap within subgroup
            g=G.sg[ids].to_numpy(); nul=np.empty(B)
            for b in range(B): nul[b]=np.mean([share(l,wrap(G.due[l]+x)) for l,x in zip(ids,rng.permutation(g))])
            res.append(f'C: {100*nul.mean():5.1f} (excess {100*(o-nul.mean()):+5.1f}, p={(int((nul>=o).sum())+1)/(B+1):.4f})')
            print(f'{label} ±{w} | {k:14s} n={len(ids):2d} obs {100*o:5.1f} | '+' | '.join(res))
        print()
run(dd[dd.day_v3.notna()&dd.index.isin(pay)],'day_v3','STATED')
run(dd[(dd.P3v3<4)&dd.salday.notna()&dd.index.isin(pay)],'salday','PAYDAY')
# calendar-day popularity among donors
pd_=p[p.loan_id.isin(don_ids)]; print('donor payments by calendar day (top):',(100*pd_.aday.value_counts(normalize=True)).round(1).head(6).to_dict(),'| on 25th: %.1f%%'%(100*(pd_.aday==25).mean()))
