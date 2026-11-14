"""Table 4, Section 5.2, and Online Appendix D/F/G/H: repayment success and equivalence tests."""
import numpy as np, pandas as pd
from scipy.stats import norm, f as Fd
from R5_lib import load
d,p=load(); d=d.merge(pd.read_csv('data/listing_history.csv'),on='loan_id'); d=d.sort_values(['bgroup','list_date']); d['ord']=d.groupby('bgroup').cumcount()
hist=['h_nloans','h_ontime','h_late','h_overdue','h_none']; blk=['pay_sched_rx','P2','P3b']; obs=['male','age_n','age_miss','lnlen']; sty=['S2','S3']; con=['lnamt','dur_m','rate']
def ols(s,y,cols,fe=True):
    parts=[pd.Series(1.0,index=s.index,name='c'),s[cols].astype(float)]
    if fe: parts.append(pd.get_dummies(s.year,drop_first=True,prefix='y').astype(float))
    X=pd.concat(parts,axis=1); X=X.loc[:,(X.std()>0)|(X.columns=='c')]; Xv=X.values; n,k=Xv.shape
    XtXi=np.linalg.pinv(Xv.T@Xv); b=XtXi@Xv.T@y; e=y-Xv@b; V=XtXi@(Xv.T*(e**2))@Xv@XtXi*n/(n-k)
    return X,b,V,1-e@e/((y-y.mean())@(y-y.mean())),n,k
def ft(X,b,V,names,n,k):
    idx=[list(X.columns).index(c) for c in names]; bb=b[idx]; F=bb@np.linalg.solve(V[np.ix_(idx,idx)],bb)/len(idx); return F,Fd.sf(F,len(idx),n-k)
def coef(X,b,V,c):
    i=list(X.columns).index(c); se=np.sqrt(V[i,i]); return '%s %+.3f [%+.3f, %+.3f] 90%%[%+.3f, %+.3f] p=%.3f'%(c,b[i],b[i]-1.96*se,b[i]+1.96*se,b[i]-1.645*se,b[i]+1.645*se,2*norm.sf(abs(b[i]/se)))
s=d[d.outcome.isin(['완납','손실'])].copy(); y=(s.outcome=='완납').astype(float).values
X0,b0,V0,r0,n,k0=ols(s,y,obs+hist); X1,b1,V1,r1,n,k1=ols(s,y,obs+sty+hist); X2,b2,V2,r2,n,k2=ols(s,y,blk+obs+sty+hist); X3,b3,V3,r3,n,k3=ols(s,y,blk+con+obs+sty+hist)
print('== Table 4  N',n,'R2 M1 %.3f M2 %.3f M3 %.3f'%(r1,r2,r3))
print('style: dR2 %.4f F %.2f p=%.3f'%(r1-r0,*ft(X1,b1,V1,sty,n,k1)),'| content block: dR2 %.4f F %.2f p=%.3f'%(r2-r1,*ft(X2,b2,V2,blk,n,k2)),'| contract: dR2 %.4f F %.2f p=%.3f'%(r3-r2,*ft(X3,b3,V3,con,n,k3)))
for nm,(X,b,V) in [('M2',(X2,b2,V2)),('M3',(X3,b3,V3))]: print(nm,' | '.join(coef(X,b,V,c) for c in blk))
i=list(X2.columns).index('pay_sched_rx'); print('smallest detectable effect (80%% power): %.3f; disclosing resolved loans %d; charge-off rate %.3f'%(2.8*np.sqrt(V2[i,i]),s.pay_sched_rx.sum(),1-y.mean()))
print('style coefficients in M1:',coef(X1,b1,V1,'S2'),'|',coef(X1,b1,V1,'S3'))
f=s[s.ord==0]; yf=(f.outcome=='완납').astype(float).values; X,b,V,r,nf,k=ols(f,yf,blk+obs+sty)
print('== first loans  N',nf,coef(X,b,V,'pay_sched_rx'),'| disclosing',int(f.pay_sched_rx.sum()),'| raw full-repayment',f.groupby('pay_sched_rx').apply(lambda g:(g.outcome=='완납').mean()).round(3).to_dict())
er=d[(d.first_delinq_k.notna())&d.outcome.isin(['완납','손실'])]; b4=blk
for lab,yy in [('full repayment vs charge-off',(er.outcome=='완납').astype(float).values),('return vs no return',er.return_to_current_k.notna().astype(float).values)]:
    X,b,V,r,n_,k=ols(er,yy,b4+obs+con+hist); print('== after delinquency:',lab,'N',n_,'block F %.2f p=%.2f'%ft(X,b,V,b4,n_,k))
print('delinquent loans, full-repayment rate by disclosure:',er.groupby('pay_sched_rx').apply(lambda g:(g.outcome=='완납').mean()).round(3).to_dict())
st=d.day_v3.notna(); print('== Table C.2: earlier loan in sample (%%): date-stating %.1f, others %.1f'%((1-d[st].h_none.mean())*100,(1-d[~st].h_none.mean())*100))
print('== plan-capacity gap, equivalence tests (Appendix F)')
g=pd.read_csv('data/loan_level_pcg.csv'); g=g[g.loan_id.isin(d.loan_id)].merge(d[['loan_id','lnamt','male','lnlen']+hist],on='loan_id'); delta=0.10*g.on_time_rate.std()
for meas in ['pcg_ratio','pcg_abs']:
    q=g[g[meas].notna()].copy(); x=q[meas].clip(q[meas].quantile(.01),q[meas].quantile(.99)); q['z']=(x-x.mean())/x.std()
    for lab,cols,fe in [('unadjusted',['z'],False),('adjusted',['z','lnamt','dur_m','rate','male','lnlen']+hist,True)]:
        X,b,V,r,n_,k=ols(q,q.on_time_rate.values,cols,fe); se=np.sqrt(V[1,1]); print(meas,lab,n_,'%+.4f 90%% CI [%+.4f, %+.4f] TOST p=%.4f'%(b[1],b[1]-1.645*se,b[1]+1.645*se,max(norm.sf((b[1]+delta)/se),norm.cdf((b[1]-delta)/se))))
