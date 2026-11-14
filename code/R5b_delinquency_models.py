"""Tables 5 and 6, Section 5.3 and Online Appendix G/H: transition hazards."""
import numpy as np, pandas as pd
from scipy.stats import norm, chi2
from R5_lib import load, cloglog, cloglog_if
d,p=load(); d=d.merge(pd.read_csv('data/listing_history.csv'),on='loan_id')
d=d.sort_values(['bgroup','list_date']); d['ord']=d.groupby('bgroup').cumcount(); d['late_period']=(d.year>=2014).astype(float); d['nb']=d.groupby('bgroup').loan_id.transform('size')
P=p.merge(d,on='loan_id',suffixes=('','_l'))
hist=['h_nloans','h_ontime','h_late','h_overdue','h_none']; base=['lnamt','dur_m','rate','male','lnlen']; blk=['pay_sched_rx','P2','P3b']; K=len(blk)
def initiation(thr):
    q=P.copy()
    if thr==1: q['fk']=q.first_delinq_k
    else:
        un=(q.unpaid==1)&(q.status!='중도상환'); q['e']=((q.delay>=thr)|un).astype(int); q=q.join(q[q.e==1].groupby('loan_id').k.min().rename('fk'),on='loan_id')
    rs=q[(q.fk.isna())|(q.k<=q.fk)].copy(); rs['y']=(rs.k==rs.fk).astype(float); return rs
def X_(rs,ctrl,useblk=True,extra=None):
    t=rs.k/rs.n_inst; parts=[pd.Series(1.0,index=rs.index,name='c')]
    if useblk: parts.append(rs[blk].astype(float))
    if extra is not None: parts.append(extra)
    parts+=[t.rename('t'),(t**2).rename('t2'),(t**3).rename('t3'),pd.get_dummies(rs.year,drop_first=True).astype(float),rs[ctrl].astype(float)]
    X=pd.concat(parts,axis=1); return X.loc[:,(X.std()>0)|(X.columns=='c')]
def hr(rs,ctrl,cl):
    X=X_(rs,ctrl); b,se,ll=cloglog(X.values,rs.y.values,cl=rs[cl].values)
    return '%.2f [%.2f, %.2f] p=%.4f'%(np.exp(b[1]),np.exp(b[1]-1.96*se[1]),np.exp(b[1]+1.96*se[1]),2*norm.sf(abs(b[1]/se[1])))
A=initiation(1)
B=P[(P.first_delinq_k.notna())&(P.k>P.first_delinq_k)&((P.return_to_current_k.isna())|(P.k<=P.return_to_current_k))].copy(); B['y']=(B.k==B.return_to_current_k).astype(float)
C=P[(P.return_to_current_k.notna())&(P.k>P.return_to_current_k)&((P.re_delinq_k.isna())|(P.k<=P.re_delinq_k))].copy(); C['y']=(C.k==C.re_delinq_k).astype(float)
print('== Table 5'); ps=[]
for nm,rs in [('initiation',A),('return',B),('re-delinquency',C)]:
    X=X_(rs,base+hist); b,se,ll=cloglog(X.values,rs.y.values,cl=rs.loan_id.values); X0=X_(rs,base+hist,useblk=False); b0,se0,ll0=cloglog(X0.values,rs.y.values,cl=rs.loan_id.values)
    pv=chi2.sf(2*(ll-ll0),K); ps.append(pv); print(nm,len(rs),int(rs.y.sum()),'payday disclosure HR',hr(rs,base+hist,'loan_id'),'block LR p=%.4f'%pv)
o=np.argsort(ps); adj=[0]*3; run=0
for r_,i in enumerate(o): run=max(run,ps[i]*(3-r_)); adj[i]=min(1,run)
print('Holm p:',np.round(adj,3))
X=X_(B,base+hist); b,se,ll=cloglog(X.values,B.y.values,cl=B.loan_id.values); print('return: detectable HR <=%.2f or >=%.2f'%(np.exp(-2.8*se[1]),np.exp(2.8*se[1])))
print('== cross-transition equality'); bs=[];G=[];bb=[];GG=[]
for rs in (A,B,C):
    X=X_(rs,base+hist); b,se,IF=cloglog_if(X.values,rs.y.values,rs.loan_id.values); bs.append(b[1]); G.append(IF[1]); bb.append(b[1:1+K]); GG.append(IF[list(range(1,1+K))])
V=pd.concat(G,axis=1).fillna(0).values; V=V.T@V; bs=np.array(bs); R=np.array([[1,-1,0],[1,0,-1]]); dl=R@bs; RV=R@V@R.T; chi=dl@np.linalg.solve(RV,dl)
print('joint chi2 %.2f p=%.3f; init-return %.2f p=%.3f; init-redelinq %.2f p=%.3f'%(chi,chi2.sf(chi,2),-dl[0],2*norm.sf(abs(dl[0])/np.sqrt(RV[0,0])),-dl[1],2*norm.sf(abs(dl[1])/np.sqrt(RV[1,1]))))
V=pd.concat(GG,axis=1).fillna(0).values; V=V.T@V; b12=np.concatenate(bb); R=np.zeros((2*K,3*K))
for j in range(K): R[j,j]=1; R[j,K+j]=-1; R[K+j,j]=1; R[K+j,2*K+j]=-1
dl=R@b12; chi=dl@np.linalg.solve(R@V@R.T,dl); print('block heterogeneity chi2(%d)=%.2f p=%.2f'%(2*K,chi,chi2.sf(chi,2*K)))
def pval(rs,ctrl,cl):
    X=X_(rs,ctrl); b,se,ll=cloglog(X.values,rs.y.values,cl=rs[cl].values); return 2*norm.sf(abs(b[1]/se[1]))
def holm(p):
    p=np.array(p); o=np.argsort(p); adj=np.zeros(len(p)); run=0
    for r,i in enumerate(o): run=max(run,p[i]*(len(p)-r)); adj[i]=min(1,run)
    return np.round(adj,3)
print('== Table 6'); pa=[]; pf=[]
for thr in (1,3,7,14,30):
    rs=initiation(thr); pa.append(pval(rs,base+hist,'bgroup')); pf.append(pval(rs[rs.ord==0],base,'bgroup'))
print('Holm-corrected p, all loans:',holm(pa),'| first loans:',holm(pf))
for thr in (1,3,7,14,30):
    rs=initiation(thr); f=rs[rs.ord==0]; print('>=%dd'%thr,'events',int(rs.y.sum()),'| all loans',hr(rs,base+hist,'bgroup'),'| first loans (events %d)'%f.y.sum(),hr(f,base,'bgroup'))
for thr in (1,7):
    q=P.copy(); un=(q.unpaid==1)&(q.status!='중도상환'); q['e']=((q.delay>=thr)|un); m=d.join(q.groupby('loan_id').e.any().rename('evx'),on='loan_id'); m['ev']=m.first_delinq_k.notna() if thr==1 else m.evx
    print('raw share delinquent, threshold',thr,m.groupby('pay_sched_rx').ev.mean().round(3).to_dict())
print('== Online Appendix H')
ex=(A.pay_sched_rx*A.late_period).rename('dx'); X=X_(A,base+hist,extra=ex); b,se,ll=cloglog(X.values,A.y.values,cl=A.bgroup.values); i=list(X.columns).index('dx')
print('2008-2013 HR %.2f (p=%.3f), 2014-2019 implied %.2f, interaction p=%.2f'%(np.exp(b[1]),2*norm.sf(abs(b[1]/se[1])),np.exp(b[1]+b[i]),2*norm.sf(abs(b[i]/se[i]))))
print('repeat-borrower loans',hr(A[A.nb>1],base+hist,'bgroup'),'| single-loan borrowers',hr(A[A.nb==1],base+hist,'bgroup'))
top=d[d.pay_sched_rx==1].groupby('bgroup').size().sort_values(ascending=False); print('excluding three most frequent disclosers',hr(A[~A.bgroup.isin(top.index[:3])],base+hist,'bgroup'))
print('disclosing loans',int(d.pay_sched_rx.sum()),'borrowers',len(top),'with one disclosing loan',int((top==1).sum()))
print('== due-day mismatch (Section 5.3, Appendix H)')
p2=p.copy(); p2['dom']=p2.sched.dt.day; due=p2.groupby('loan_id').dom.agg(lambda x:x.mode().iloc[0] if x.notna().any() else np.nan).rename('due'); m=d.join(due,on='loan_id'); m['ref']=m.day_v3.where(m.day_v3.notna(),m.salday); m=m[m.ref.notna()].copy()
m['g']=((m.ref.clip(upper=30)-m.due.clip(upper=30)+15)%30)-15; m['after']=(m.g>=4).astype(float); m['before']=(m.g<=-4).astype(float); PM=p.merge(m,on='loan_id',suffixes=('','_l'))
for thr in (1,7):
    q=PM.copy()
    if thr==1: q['fk']=q.first_delinq_k
    else:
        un=(q.unpaid==1)&(q.status!='중도상환'); q['e']=((q.delay>=thr)|un).astype(int); q=q.join(q[q.e==1].groupby('loan_id').k.min().rename('fk'),on='loan_id')
    rs=q[(q.fk.isna())|(q.k<=q.fk)].copy(); rs['y']=(rs.k==rs.fk).astype(float); t=rs.k/rs.n_inst
    X=pd.concat([pd.Series(1.0,index=rs.index,name='c'),rs[['after','before']],t.rename('t'),(t**2).rename('t2'),(t**3).rename('t3'),rs[['lnamt','dur_m','rate','male','lnlen','h_nloans','h_ontime','h_late','h_none']].astype(float)],axis=1); X=X.loc[:,(X.std()>0)|(X.columns=='c')]
    b,se,ll=cloglog(X.values,rs.y.values,cl=rs.bgroup.values); print('threshold',thr,'loans',len(m),'obs',len(rs),'events',int(rs.y.sum()),'after %.2f (p=%.2f) before %.2f (p=%.2f)'%(np.exp(b[1]),2*norm.sf(abs(b[1]/se[1])),np.exp(b[2]),2*norm.sf(abs(b[2]/se[2]))))
