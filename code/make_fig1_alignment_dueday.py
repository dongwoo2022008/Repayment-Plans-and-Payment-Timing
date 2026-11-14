#!/usr/bin/env python3
"""Figure 1 - payment timing relative to the stated repayment day (Panel A) and to the
contractual due day (Panel B), for the 854 paid installments of the 91 date-stating loans.
Run from the repository root:  python code/make_fig1_alignment_dueday.py figures/Figure_1.png
"""
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd, numpy as np
OUT = sys.argv[1]
plt.rcParams['font.family'] = 'DejaVu Sans'
d = pd.read_csv('data/analysis_v3_public.csv'); p = pd.read_csv('data/panel_long.csv')
day = dict(zip(d['loan_id'], d['day_v3'])); ids = d.loc[d['day_v3'].notna(), 'loan_id']
q = p[p['loan_id'].isin(ids) & p['actual_date'].notna()].copy()
q['aday'] = pd.to_datetime(q['actual_date']).dt.day
q['sday'] = pd.to_datetime(q['sched_date']).dt.day
circ = lambda a, b: np.minimum(np.abs(a - b), 30 - np.abs(a - b))
q['d_st'] = circ(q['aday'].to_numpy(), q['loan_id'].map(day).to_numpy())
q['d_due'] = circ(q['aday'].to_numpy(), q['sday'].to_numpy())
n_inst, n_loan = len(q), q['loan_id'].nunique()
stat = {c: 100 * q.groupby('loan_id')[c].apply(lambda s: (s <= 3).mean()).mean() for c in ('d_st', 'd_due')}
unif = np.array([100 / 30 if k in (0, 15) else 200 / 30 for k in range(16)])
DARK, LIGHT, RULE = '#4d4d4d', '#c9c9c9', '#1a1a1a'
fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), dpi=150, sharey=True)
panels = [('d_st', 'A. Distance to the stated repayment day', 'Circular distance between actual payment day\nand stated repayment day (days)',
           f"Within ±3 days of the stated day:\nloan-level mean share {stat['d_st']:.1f}%\n(permutation null 24.3%, p = .0005)"),
          ('d_due', 'B. Distance to the contractual due day', 'Circular distance between actual payment day\nand contractual due day (days)',
           f"Within ±3 days of the due day:\nloan-level mean share {stat['d_due']:.1f}%")]
ymax = 0
for ax, (c, title, xl, note) in zip(axes, panels):
    share = (q[c].value_counts(normalize=True).sort_index() * 100).reindex(range(16), fill_value=0.0)
    ymax = max(ymax, share.max())
    ax.bar(range(16), share.to_numpy(), color=[DARK if k <= 3 else LIGHT for k in range(16)], width=0.86, zorder=2)
    ax.step(np.arange(-0.5, 16.0), np.append(unif, unif[-1]), where='post', color=RULE, lw=1.8, ls=(0, (5, 3)), zorder=3)
    ax.set_title(title, fontsize=12, fontweight='bold', loc='left', pad=8)
    ax.set_xlabel(xl, fontsize=11); ax.set_xticks(range(16)); ax.set_xlim(-0.7, 15.7); ax.tick_params(labelsize=10)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.set_yticks(range(0, 25, 5)); ax.set_ylim(0, 22)
    ax.yaxis.grid(True, color='#e6e6e6', zorder=0); ax.set_axisbelow(True)
    ax.text(0.36, 0.86, note, transform=ax.transAxes, fontsize=10.5, color='#222222', va='top')
    ax.text(8.3, unif[9] + 0.6, 'Uniform benchmark', fontsize=10, color='#7a8590')
    print(c, 'pooled shares d=0..3:', [round(v, 1) for v in share.iloc[:4]], 'pooled within3: %.1f' % share.iloc[:4].sum())
axes[0].set_ylabel('Share of installments (%)', fontsize=11)
axes[1].tick_params(labelleft=True)
fig.subplots_adjust(wspace=0.12)
fig.savefig(OUT, dpi=1000, bbox_inches='tight', pad_inches=0.10, facecolor='white')
print(f'{n_loan} loans, {n_inst} paid installments | loan-level mean within +/-3: stated {stat["d_st"]:.2f}%, due {stat["d_due"]:.2f}%')
