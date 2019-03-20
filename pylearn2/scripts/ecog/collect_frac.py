from __future__ import print_function
import cPickle, sys, os
import numpy as np

from best_folder import best_folder

folder = sys.argv[1]

subjects = ['ec2', 'ec9', 'gp31', 'gp33']
fracs = ['_05', '_06', '_07', '_08', '_09', '']
frac_names = [.5, .6, .7, .8, .9, 1.0]
n_files = 400

deep_all = {}
linear_all = {}
#random_all = {}

for fr,fr_name in zip(fracs, frac_names):
    print(fr)
    d_all = []
    l_all = []
    #r_all = []
    for subject in subjects:
        f = '{}_hg{}'.format(subject, fr)
        f_name = os.path.join(folder, f)
        try:
            file_data, results, results_all, f_id = best_folder(f_name, n_files)
            if file_data[1, 0] < n_files:
                print(f_name, file_data[1, 0])
        except ValueError as e:
            print('not done: ', f_name)
            print(e)
            results_all = np.zeros((3, 10))
        d_all.append(results_all)

        f = '{}_hg_lin{}'.format(subject, fr)
        f_name = os.path.join(folder, f)
        try:
            file_data, results, results_all, f_id = best_folder(f_name, n_files)
            if file_data[1, 0] < n_files:
                print(f_name, file_data[1, 0])
        except ValueError as e:
            print('not done: ', f_name)
            print(e)
            results_all = np.zeros((3, 10))
        l_all.append(results_all)

        """
        f = '{}_hg_a_random{}'.format(subject, fr)
        f_name = os.path.join(folder, f)
        try:
            file_data, results, results_all, f_id = best_folder(f_name, n_files)
            if file_data[1, 0] < n_files:
                print(f_name, file_data[1, 0])
        except ValueError as e:
            print('not done: ', f_name)
            print(e)
            results_all = np.zeros((3, 10))
        r_all.append(results_all)
        """

    deep_all[fr_name] = np.array(d_all)
    linear_all[fr_name] = np.array(l_all)
    #random_all[fr_name] = np.array(r_all)

with open('frac_results_avg_ff.pkl', 'w') as f:
    cPickle.dump((deep_all, linear_all), f)
