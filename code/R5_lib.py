"""Shared helpers: data loading, cloglog estimator with cluster-robust SE."""
import pandas as pd, numpy as np
R='data/'  # run from the repository root
def load():
    d=pd.read_csv(R+'analysis_v3_public.csv'); p=pd.read_csv(R+'panel_long.csv')
    age=pd.to_numeric(d.age.str.replace('세',''),errors='coerce'); d['age_n']=age.fillna(0); d['age_miss']=age.isna().astype(float)
    d['list_date']=pd.to_datetime('20'+d.loan_id.str[1:7],format='%Y%m%d',errors='coerce')
    p['sched']=pd.to_datetime(p.sched_date,errors='coerce'); p['act']=pd.to_datetime(p.actual_date,errors='coerce')
    return d,p
def cloglog(X,y,cl=None,maxit=200):
    mu=X.mean(0); sd=X.std(0); sd[sd==0]=1; Z=(X-mu)/sd; Z[:,0]=1
    b=np.zeros(Z.shape[1]); b[0]=np.log(-np.log(1-y.mean()))
    def ll(b):
        eta=np.clip(Z@b,-30,3.5); pr=1-np.exp(-np.exp(eta)); pr=np.clip(pr,1e-12,1-1e-12); return (y*np.log(pr)+(1-y)*np.log(1-pr)).sum()
    cur=ll(b)
    for it in range(maxit):
        eta=np.clip(Z@b,-30,3.5); ex=np.exp(eta); pr=np.clip(1-np.exp(-ex),1e-12,1-1e-12)
        dpr=ex*np.exp(-ex); s=(y-pr)/(pr*(1-pr))*dpr; g=Z.T@s
        W=dpr**2/(pr*(1-pr)); H=(Z.T*W)@Z
        step=np.linalg.solve(H+1e-8*np.eye(len(b)),g); t=1.0
        while t>1e-6:
            nb=b+t*step; new=ll(nb)
            if new>=cur-1e-10: break
            t/=2
        b=nb; 
        if abs(new-cur)<1e-9: cur=new; break
        cur=new
    eta=np.clip(Z@b,-30,3.5); ex=np.exp(eta); pr=np.clip(1-np.exp(-ex),1e-12,1-1e-12); dpr=ex*np.exp(-ex)
    s=(y-pr)/(pr*(1-pr))*dpr; W=dpr**2/(pr*(1-pr)); Hi=np.linalg.inv((Z.T*W)@Z)
    S=Z*s[:,None]
    if cl is not None:
        S=pd.DataFrame(S).groupby(np.asarray(cl)).sum().values; G=S.shape[0]; adj=G/(G-1)
    else: adj=1
    V=Hi@(S.T@S)@Hi*adj
    # back-transform slopes
    bb=b/sd; se=np.sqrt(np.diag(V))/sd
    return bb,se,cur

def cloglog_if(X,y,cl):
    """returns slopes b, and per-cluster influence (clusters x k) in original scale (slopes only valid)"""
    b,se,ll=cloglog(X,y,cl=cl)
    mu=X.mean(0); sd=X.std(0); sd[sd==0]=1; Z=(X-mu)/sd; Z[:,0]=1
    bz=b*sd
    eta=np.clip(Z@bz,-30,3.5); ex=np.exp(eta); pr=np.clip(1-np.exp(-ex),1e-12,1-1e-12); dpr=ex*np.exp(-ex)
    s=(y-pr)/(pr*(1-pr))*dpr; W=dpr**2/(pr*(1-pr)); Hi=np.linalg.inv((Z.T*W)@Z)
    S=pd.DataFrame(Z*s[:,None]).groupby(np.asarray(cl)).sum()
    IF=pd.DataFrame((S.values@Hi)/sd,index=S.index)
    return b,se,IF
