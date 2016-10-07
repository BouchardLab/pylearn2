#!/usr/bin/env python
import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
from pylab import rcParams
import matplotlib.pyplot as plt


rcParams.update({'figure.autolayout': True,
                 'font.size': 24})

folder = '/home/jesse/plots/model/data'
raw_files = ['ec2_new2_ec2_lin3_corr_raw.npz',
             'ec9_new2_ec9_lin1_corr_raw.npz',
             'gp31_new2_gp31_lin0_corr_raw.npz',
             'gp33_new2_gp33_lin0_corr_raw.npz']
linear_files = ['ec2_new2_ec2_lin3_corr_y_hat.npz',
                'ec9_new2_ec9_lin1_corr_y_hat.npz',
                'gp31_new2_gp31_lin0_corr_y_hat.npz',
                'gp33_new2_gp33_lin0_corr_y_hat.npz']
deep_files = ['ec2_new2_ec2_fc1_corr_y_hat.npz',
              'ec9_new2_ec9_fc1_corr_y_hat.npz',
              'gp31_new2_gp31_fc1_corr_y_hat.npz',
              'gp33_new2_gp33_fc0_corr_y_hat.npz']

colors = ['green', 'black', 'red']

def load_data(path):
    data = np.load(path)
    return data['ccp'], data['ccm'], data['ccv']

rp = []
rm = []
rv = []
lp = []
lm = []
lv = []
dp = []
dm = []
dv = []

for fname in raw_files:
    path = os.path.join(folder, fname)
    data = load_data(path)
    rp.extend(data[0])
    rm.extend(data[1])
    rv.extend(data[2])
for fname in linear_files:
    path = os.path.join(folder, fname)
    data = load_data(path)
    lp.extend(data[0])
    lm.extend(data[1])
    lv.extend(data[2])
for fname in deep_files:
    path = os.path.join(folder, fname)
    data = load_data(path)
    dp.extend(data[0])
    dm.extend(data[1])
    dv.extend(data[2])


box_params = {'notch': False,
              'sym': '',
              'vert': False,
              'whis': 0,
              'labels': ('',
                         'Vowel',
                         '',
                         'Manner',
                         '',
                         'Place'),
              'positions': [2-.375, 2+.375, 4-.375, 4+.375, 6-.375, 6+.375],
              'medianprops': {'color': 'black', 'linewidth': 2},
              'boxprops': {'color': 'black', 'linewidth': 2}}

#data = [rv, lv, dv, rm, lm, dm, rp, lp, dp]
data = [lv, dv, lm, dm, lp, dp]
f = plt.figure(figsize=(15, 8))
bp = plt.boxplot(data, **box_params)
for ii in range(len(bp['boxes'])):
    c = colors[ii % (len(colors)-1)+1]
    plt.setp(bp['boxes'][ii], color=c)
    plt.setp(bp['caps'][2*ii], color=c)
    plt.setp(bp['caps'][2*ii+1], color=c)
    plt.setp(bp['whiskers'][2*ii], color=c)
    plt.setp(bp['whiskers'][2*ii+1], color=c)
    plt.setp(bp['medians'][ii], color=c)
plt.plot(0,0, '-', c='red', label='Deep')
plt.plot(0,0, '-', c='black', label='Linear')
plt.xlim([-.06, .45])
#plt.plot(0,0, '-', c='red', label='Neural Data')
plt.legend(loc='best', prop={'size': 20})
plt.xlabel('Correlation Coefficient')
plt.savefig('state_correlation.png')
plt.savefig('state_correlation.pdf')


positions = [1-.375, 1+.375, 3-.375, 3+.375, 5-.375, 5+.375]
box_params = {'vert': False,
              'showmedians': True,
              'positions': positions}
data = [lv, dv, lm, dm, lp, dp]
f = plt.figure(figsize=(15, 8))
vp = plt.violinplot(data, **box_params)
vp['cbars'].set_color(['black', 'red', 'black', 'red', 'black', 'red'])
vp['cbars'].set_linewidths(2)
vp['cmedians'].set_color(['black', 'red', 'black', 'red', 'black', 'red'])
vp['cmedians'].set_linewidths(2)
vp['cmins'].set_color(['black', 'red', 'black', 'red', 'black', 'red'])
vp['cmins'].set_linewidths(2)
vp['cmaxes'].set_color(['black', 'red', 'black', 'red', 'black', 'red'])
vp['cmaxes'].set_linewidths(2)
"""
from IPython import embed
embed()
"""
for ii in range(len(vp['bodies'])):
    c = colors[ii % (len(colors)-1)+1]
    vp['bodies'][ii].set_color(c)

plt.yticks(positions,
           ['', 'Vowel', '', 'Manner', '', 'Place']),
plt.ylim([positions[0]-.5, positions[-1]+.5])
min_x = np.inf
max_x = -np.inf
for d in data:
    d = np.array(d)
    min_x = min(min_x, d.min())
    max_x = max(max_x, d.max())
plt.xlim([min_x-.025, max_x+.025])
plt.plot(0,0, '-', c='red', label='Deep')
plt.plot(0,0, '-', c='black', label='Linear')
plt.axvline(0, linestyle='dotted', c='black')
#plt.plot(0,0, '-', c='red', label='Neural Data')
plt.legend(loc='lower right', prop={'size': 20})
plt.xlabel('Correlation Coefficient')
plt.savefig('state_correlation_violin.png')
plt.savefig('state_correlation_violin.pdf')
