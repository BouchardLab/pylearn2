#!/usr/bin/env python
import cPickle, os
import numpy as np

import matplotlib
matplotlib.use('Agg')
from pylab import rcParams
import matplotlib.pyplot as plt

import analysis


rcParams.update({'figure.autolayout': True,
                 'font.size': 24})

folder = '/home/jesse/plots/model/data'
linear_files = ['ec2_new2_ec2_lin3_model_output.pkl',
                'ec9_new2_ec9_lin1_model_output.pkl',
                'gp31_new2_gp31_lin0_model_output.pkl',
                'gp33_new2_gp33_lin0_model_output.pkl']
deep_files = ['ec2_new2_ec2_fc1_model_output.pkl',
              'ec9_new2_ec9_fc1_model_output.pkl',
              'gp31_new2_gp31_fc1_model_output.pkl',
              'gp33_new2_gp33_fc0_model_output.pkl']
random_files = ['ec2_new2_ec2_random1_model_output.pkl',
                'ec9_new2_ec9_random0_model_output.pkl',
                'gp31_new2_gp31_random0_model_output.pkl',
                'gp33_new2_gp33_random1_model_output.pkl']

subj_colors = ['red', 'darkgray', 'pink', 'black']

def load_data(path):
    with open(path) as f:
        dicts, dicts2, y_dims, has_data = cPickle.load(f)
    (accuracy_dicts, indices_dicts, y_hat_dicts, logits_dicts,
     hidden_dicts) = dicts
    indices_dicts2, y_hat_dicts2, logits_dicts2 = dicts2
    mats = analysis.indx_dict2conf_mat(indices_dicts2, y_dims)
    c_mat, v_mat, cv_mat = mats
    accuracy = analysis.conf_mat2accuracy(c_mat, v_mat, cv_mat)
    (c_accuracy, v_accuracy, cv_accuracy, accuracy_per_cv,
     p_accuracy, m_accuracy) = accuracy
    return (c_accuracy, v_accuracy, cv_accuracy, accuracy_per_cv,
            p_accuracy, m_accuracy)

fig = plt.figure(figsize=(5, 8))
for ii, (linear_fname, deep_fname, random_fname) in enumerate(zip(linear_files,
                                                                  deep_files,
                                                                  random_files)):
    path = os.path.join(folder, linear_fname)
    (lc, lv, lcv, _, lp, lm) = load_data(path)
    path = os.path.join(folder, deep_fname)
    (dc, dv, dcv, _, dp, dm) = load_data(path)
    path = os.path.join(folder, random_fname)
    (rc, rv, rcv, _, rp, rm) = load_data(path)

    for jj, (l, d, r) in enumerate(zip([lcv, lc, lv, lp, lm],
                                       [dcv, dc, dv, dp, dm],
                                       [rcv, rc, rv, rp, rm])):
        if jj < 1:
            chance = np.nanmean(r)
            plt.errorbar(jj+1, np.nanmean(l)/chance, fmt='^',
                         yerr=np.nanstd(l)/chance,
                         c=subj_colors[ii])
            plt.errorbar(jj+1.25, np.nanmean(d)/chance, fmt='o',
                         yerr=np.nanstd(d)/chance,
                         c=subj_colors[ii])
            plt.plot([jj+1, jj+1.25], [np.nanmean(l)/chance, np.nanmean(d)/chance],
                     '-', c=subj_colors[ii], lw=2)
            plt.plot([jj+1, jj+1.25], [1/chance, 1/chance],
                    ':', c=subj_colors[ii], lw=2)
plt.plot([0, 10], [1, 1], '--', c='gray', lw=1, label='Chance')
plt.xticks(np.arange(1)+1.125, ['CV'])
plt.ylabel('Accuracy/chance')
plt.xlim([.75, 1.5])
plt.savefig('linear_vs_deep_accuracy1.pdf')
plt.savefig('linear_vs_deep_accuracy1.png')
plt.close()


rcParams.update({'figure.autolayout': True,
                 'font.size': 18})

fig = plt.figure(figsize=(7, 5))
for ii in range(4):
    label = 'Subject '+str(ii+1)
    plt.plot(0, 0, '-', c=subj_colors[ii], label=label)
plt.plot(0, 0, '^', c='gray', label='Linear')
plt.plot(0, 0, 'o', c='gray', label='Deep')
for ii, (linear_fname, deep_fname, random_fname) in enumerate(zip(linear_files,
                                                                  deep_files,
                                                                  random_files)):
    path = os.path.join(folder, linear_fname)
    (lc, lv, lcv, _, lp, lm) = load_data(path)
    path = os.path.join(folder, deep_fname)
    (dc, dv, dcv, _, dp, dm) = load_data(path)
    path = os.path.join(folder, random_fname)
    (rc, rv, rcv, _, rp, rm) = load_data(path)

    for jj, (l, d, r) in enumerate(zip([lcv, lc, lv, lp, lm],
                                       [dcv, dc, dv, dp, dm],
                                       [rcv, rc, rv, rp, rm])):
        if jj >= 1 and jj < 3:
            chance = np.nanmean(r)
            plt.errorbar(jj+1, np.nanmean(l)/chance, fmt='^',
                         yerr=np.nanstd(l)/chance,
                         c=subj_colors[ii])
            plt.errorbar(jj+1.25, np.nanmean(d)/chance, fmt='o',
                         yerr=np.nanstd(d)/chance,
                         c=subj_colors[ii])
            plt.plot([jj+1, jj+1.25], [np.nanmean(l)/chance, np.nanmean(d)/chance],
                     '-', c=subj_colors[ii], lw=2)
            plt.plot([jj+1, jj+1.25], [1/chance, 1/chance],
                    ':', c=subj_colors[ii], lw=2)
plt.plot([0, 10], [1, 1], '--', c='gray', lw=1, label='Chance')
plt.plot([0, 0], [0, 0], ':', c='gray', lw=1, label='Max')
plt.xticks(np.arange(1, 3)+1.125, ['Cons.', 'Vowel'])
plt.xlim([1.75, 3.5])
plt.ylim([.7, 23])
plt.legend(bbox_to_anchor=(1, 1), prop={'size': 14},
           loc='upper right')
plt.savefig('linear_vs_deep_accuracy2.pdf')
plt.savefig('linear_vs_deep_accuracy2.png')
plt.close()

fig = plt.figure(figsize=(7, 3))
for ii in range(4):
    label = 'Subject '+str(ii+1)
    plt.plot(0, 0, '-', c=subj_colors[ii], label=label)
plt.plot(0, 0, '^', c='gray', label='Linear')
plt.plot(0, 0, 'o', c='gray', label='Deep')
for ii, (linear_fname, deep_fname, random_fname) in enumerate(zip(linear_files,
                                                                  deep_files,
                                                                  random_files)):
    path = os.path.join(folder, linear_fname)
    (lc, lv, lcv, _, lp, lm) = load_data(path)
    path = os.path.join(folder, deep_fname)
    (dc, dv, dcv, _, dp, dm) = load_data(path)
    path = os.path.join(folder, random_fname)
    (rc, rv, rcv, _, rp, rm) = load_data(path)

    for jj, (l, d, r) in enumerate(zip([lcv, lc, lv, lp, lm],
                                       [dcv, dc, dv, dp, dm],
                                       [rcv, rc, rv, rp, rm])):
        if jj >= 3:
            chance = np.nanmean(r)
            plt.errorbar(jj+1, np.nanmean(l)/chance, fmt='^',
                         yerr=np.nanstd(l)/chance,
                         c=subj_colors[ii])
            plt.errorbar(jj+1.25, np.nanmean(d)/chance, fmt='o',
                         yerr=np.nanstd(d)/chance,
                         c=subj_colors[ii])
            plt.plot([jj+1, jj+1.25], [np.nanmean(l)/chance, np.nanmean(d)/chance],
                     '-', c=subj_colors[ii], lw=2)
            plt.plot([jj+1, jj+1.25], [1/chance, 1/chance],
                    ':', c=subj_colors[ii], lw=2)
plt.plot([2.5, 5.75], [1, 1], '--', c='gray', lw=1, label='Chance')
plt.xticks(np.arange(3, 5)+1.125, ['Place', 'Manner'])
plt.xlim([3.5, 5.75])
plt.ylim([.7, 3.4])
plt.savefig('linear_vs_deep_accuracy3.pdf')
plt.savefig('linear_vs_deep_accuracy3.png')
plt.close()
