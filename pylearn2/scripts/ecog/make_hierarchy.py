#!/usr/bin/env python
import os, cPickle
import numpy as np
import functools

import matplotlib
matplotlib.use('Agg')
from pylab import rcParams
import matplotlib.pyplot as plt

from plotting import create_dendrogram


folder = '/home/jesse/plots/model/data'
files = ['ec2_ec2_hg_a_model_output.pkl']
""",
              'ec9_ec9_hg_a_model_output.pkl',
              'gp31_gp31_hg_a_model_output.pkl',
              'gp33_gp33_hg_a_model_output.pkl']
              """

consonants = sorted(['b', 'd', 'f', 'g', 'h', 'k', 'l', 'm', 'n', 'p', 'r',
                     's', 'sh', 't', 'th', 'v', 'w', 'y', 'z'])
vowels = sorted(['aa', 'ee', 'oo'])

cvs = []
for c in consonants:
    for v in vowels:
        cvs.append(c+v)

def load_data(path):
    with open(path) as f:
        dicts, dicts2, y_dims, has_data = cPickle.load(f)
    indices_dicts, y_hat_dicts, logits_dicts = dicts2
    return indices_dicts, y_hat_dicts

indices = []
y_hats = []
for f in files:
    indices_dicts, y_hat_dicts = load_data(os.path.join(folder, f))
    indices_dicts, y_hat_dicts = indices_dicts[0], y_hat_dicts[0]
    for key in sorted(y_hat_dicts.keys()):
        y_hats.append(y_hat_dicts[key][0])
    for key in sorted(indices_dicts.keys()):
        indices.append(indices_dicts[key][0])

yhs = np.zeros((57, 57))
correct = np.zeros((len(indices), 57))
total = np.zeros_like(correct)
for ii, (idxs, pys) in enumerate(zip(indices, y_hats)):
    for (y, yh), py in zip(idxs, pys):
        yhs[y] += py
        if y == yh:
            correct[ii, y] += 1
        total[ii, y] += 1
cv_accuracy = correct / total
yhs /= yhs.sum(axis=1, keepdims=True)

top_edge = .02
bot_edge = .125
v_gap = .01
h_gap = .01
l_edge = .13
r_edge = .01
width_cm = .65

figsize = (5, 5)
f = plt.figure(figsize=figsize)

height = width_cm * figsize[0] / float(figsize[1])
ax0 = f.add_axes([l_edge, bot_edge, width_cm, height])

height_d = 1 - bot_edge - v_gap - height - top_edge
ax1 = f.add_axes([l_edge, bot_edge + height + v_gap, width_cm, height_d])
threshold = .3
ax1.plot([0, 1e10], [threshold, threshold], '--', c='k')

z, r = create_dendrogram(yhs, threshold, ax=ax1, labels=cvs)
ax1.set_xticks([])
ax1.set_ylabel('Distance')


old_idx = []
for cv in r['ivl']:
    old_idx.append(cvs.index(cv))
yhs = yhs[old_idx]
yhs = yhs[:, old_idx]
cv_accuracy = cv_accuracy[:, old_idx]

im = ax0.imshow(yhs, cmap='gray_r', interpolation='nearest',
        vmin=0, vmax=yhs.max())
ax0.spines['right'].set_visible(False)
ax0.spines['top'].set_visible(False)
ax0.spines['left'].set_visible(False)
ax0.spines['bottom'].set_visible(False)
ax0.set_xticks(np.linspace(0, 56, 57))
ax0.set_xticklabels(r['ivl'], rotation='vertical', fontsize=6)
ax0.set_yticks(np.linspace(0, 56, 57))
ax0.set_yticklabels(r['ivl'], fontsize=6)
ax0.set_ylabel('Target CV')
ax0.set_xlabel('Predicted CV')

tick_offset = .02
tick_scale = .05
pos = 0
for label in ax0.yaxis.get_majorticklabels():
    label.set_position([tick_scale*(((pos+1)%2)-.5)-tick_offset, 0])
    pos += 1

pos = 0
for label in ax0.xaxis.get_majorticklabels():
    label.set_position([0, tick_scale*(((pos+1)%2)-.5)-tick_offset])
    pos += 1


width = 1 - l_edge - width_cm - h_gap - r_edge
ax2 = f.add_axes([l_edge + width_cm + h_gap, bot_edge, width, height])
folds, n_cvs = cv_accuracy.shape
ax2.barh(range(n_cvs), cv_accuracy.mean(axis=0)[::-1], color='k')
#        yerr=(cv_accuracy.std(axis=0)/np.sqrt(folds))[::-1])
ax2.set_ylim(np.array([0, 57])-.5)
ax2.set_yticks([])
ax2.set_xticks([0, .5])
ax2.set_xticklabels([0, .5])
ax2.tick_params(labelsize=8)
ax2.set_xlabel('Accuracy')


ax3 = f.add_axes([l_edge + width_cm + h_gap, bot_edge + height + v_gap,
                  width, height_d])
ds = z[:, 2]
bins = np.linspace(0, ds.max(), 1000)
h, b = np.histogram(ds, bins, density=False)
cs = np.cumsum(h[::-1])[::-1]
ax3.plot(cs, b[1:], c='k')
ax3.set_xticks([])
ax3.set_yticks([])

ax4 = f.add_axes([l_edge + width_cm + h_gap + width / 3,
                  bot_edge + height + v_gap + height_d / 3,
                  width / 4, height_d / 2])
c = f.colorbar(im, cax=ax4)
c.set_ticks([0, .12])
c.ax.tick_params(labelsize=8)
#c.set_lim([0, .12])
#ax4.set_ticklabels([0, .12])

max_d = .65
for ax in [ax1, ax3]:
    ax.set_ylim(None, max_d)
ax1.set_yticks([0, max_d])
ax1.set_yticklabels([0, max_d], fontsize=8)

f.text(.01, .97, 'A', fontsize=10)
f.text(.81, .95, 'B', fontsize=10)
f.text(.01, .75, 'C', fontsize=10)
f.text(.95, .7, 'D', fontsize=10)


plt.savefig('/home/jesse/Downloads/hierarchy.pdf')
