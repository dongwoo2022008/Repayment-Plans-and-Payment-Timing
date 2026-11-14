"""Calibration of the sibling-loan test under a true null (Online Appendix H).
Borrowers whose plans never give a date receive a pseudo stated day on one loan:
that loan's due day plus a gap drawn from the real stated-day gaps."""
import numpy as np, pandas as pd, sys
from R7_sibling_loan_alignment import *
S3=table(3); base=d.index[d.day_v3.notna()&d.loan_id.isin(pay)]; R=pairs(base,'day_v3'); M=R[R.dist>=4]
a=M.drop_duplicates('A'); real_gaps=sgn(np.minimum(a.s,30),np.minimum(a.A.map(due),30)).astype(float)
ok=d.groupby('bgroup').apply(lambda g: bool(nodate[g.index].all()) and len(g)>=2); cand=ok.index[ok.values]
loans={bg:d.index[d.bgroup==bg].to_numpy() for bg in cand}
NS=int(sys.argv[1]) if len(sys.argv)>1 else 1000; rng=np.random.default_rng(11); out=[]
for it in range(NS):
    rows=[]
    for bg in rng.choice(cand,28,replace=False):
        A_=rng.choice(loans[bg]); s=wrap(due[A_]+rng.choice(real_gaps))
        for b in loans[bg]:
            if b!=A_: rows.append(dict(A=A_,B=b,bg=bg,s=float(s),dueB=float(due[b]),g=float(sgn(s,min(due[b],30)))))
    r=pd.DataFrame(rows); r=r[r.g.abs()>=4].reset_index(drop=True)
    if r.bg.nunique()<10: continue
    out.append(evaluate(r,S3,rng,B=199,verbose=False)+(r.bg.nunique(),len(r)))
o=pd.DataFrame(out,columns=['obs','b1','p1','b2','p2','nb','np'])
print('placebo borrowers available',len(cand),'| simulated samples',len(o),'| mean borrowers %.1f, pairs %.1f'%(o.nb.mean(),o.np.mean()))
print('benchmark 1: mean excess %+.2f pp, rejection rate at 5%%: %.1f%%'%(100*(o.obs-o.b1).mean(),100*(o.p1<=.05).mean()))
print('benchmark 2: mean excess %+.2f pp, rejection rate at 5%%: %.1f%%'%(100*(o.obs-o.b2).mean(),100*(o.p2<=.05).mean()))
