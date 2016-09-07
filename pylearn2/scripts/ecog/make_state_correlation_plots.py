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

colors = ['red', 'blue', 'green']

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
                         '',
                         'Manner',
                         '',
                         '',
                         'Place',
                         ''),
              'positions': [1.22, 2, 2.75, 4.25, 5, 5.75, 7.25, 8, 8.75],
              'medianprops': {'color': 'black', 'linewidth': 2},
              'boxprops': {'color': 'black', 'linewidth': 2}}
data = [rv, lv, dv, rm, lm, dm, rp, lp, dp]
f = plt.figure(figsize=(15, 8))
bp = plt.boxplot(data, **box_params)
for ii in range(len(bp['boxes'])):
    c = colors[ii % len(colors)]
    plt.setp(bp['boxes'][ii], color=c)
    plt.setp(bp['caps'][2*ii], color=c)
    plt.setp(bp['caps'][2*ii+1], color=c)
    plt.setp(bp['whiskers'][2*ii], color=c)
    plt.setp(bp['whiskers'][2*ii+1], color=c)
    plt.setp(bp['medians'][ii], color=c)
plt.plot(0,0, '-', c='green', label='Deep')
plt.plot(0,0, '-', c='blue', label='Linear')
plt.plot(0,0, '-', c='red', label='Neural Data')
plt.legend(loc='best', prop={'size': 18})
plt.xlabel('Correlation Coefficient')
#plt.savefig('state_correlation.png')
plt.savefig('state_correlation.pdf')
