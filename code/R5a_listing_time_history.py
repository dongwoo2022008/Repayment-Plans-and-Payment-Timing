"""Repayment history at listing, built from the installment panel.
For each loan: the borrower's earlier loans in the sample and the payments
recorded on them before the listing date. Output: data/listing_history.csv"""
import pandas as pd
from R5_lib import load
d,p=load()
pp=p.merge(d[['loan_id','bgroup','list_date']],on='loan_id')
byb={g:x for g,x in pp.groupby('bgroup')}; lb={g:x for g,x in d.groupby('bgroup')}
rows=[]
for _,r in d.iterrows():
    L=r.list_date; x=byb[r.bgroup]; x=x[(x.loan_id!=r.loan_id)&(x.list_date<L)]
    paid=x[x.act<L]
    od=x[(x.sched<L)&((x.act.isna())|(x.act>=L))&(x.status!='중도상환')]
    rows.append(dict(loan_id=r.loan_id,h_nloans=int((lb[r.bgroup].list_date<L).sum()),h_ontime=int((paid.ontime==1).sum()),h_late=int((paid.late==1).sum()),h_overdue=len(od)))
h=pd.DataFrame(rows); h['h_none']=(h.h_nloans==0).astype(int); h.to_csv('data/listing_history.csv',index=False)
print(h.describe().loc[['mean','std','max']].round(2)); print('loans without an earlier sample loan:',int(h.h_none.sum()))
