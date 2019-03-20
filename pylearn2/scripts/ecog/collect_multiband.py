from __future__ import print_function
import cPickle, sys, os
import numpy as np

from best_folder import best_folder

folder = sys.argv[1]

subjects = ['ec2', 'ec9', 'gp31', 'gp33']
bands = ['alpha', 'beta', 'theta', 'gamma', 'high gamma']
band_abbreviations = ['a', 'b', 't', 'g', 'hg']
n_files = 400

single_band = {}
single_band_all = {}
multi_band = {}
multi_band_all = {}

if '1f' in folder:
    file_types = ['{}_1f_{}_ds_pca', '{}_1f_{}_ds']
    file_types = ['{}_1f_{}_ds']
else:
    file_types = ['{}_{}_ds_pca', '{}_{}_ds']

for fto in file_types:
    print(fto)
    for b, ba in zip(bands, band_abbreviations):
        if (fto[-2:] == 'ds') and (ba == 'hg'):
            ft = '{}_{}'
        elif (fto == '{}_1f_{}_ds_pca') and (ba == 'hg'):
            print('here')
            ft = '{}_{}_ds_pca'
        else:
            ft = fto
        print(ba)
        sb = []
        sb_all = []
        for subject in subjects:
            f = ft.format(subject, ba)
            if (fto == '{}_1f_{}_ds_pca') and (ba == 'hg'):
                f_name = os.path.join(folder.replace('_1f', ''), f)
            elif (fto == '{}_1f_{}_ds') and (ba == 'hg'):
                f_name = os.path.join(folder.replace('_1f', ''), f)
            else:
                f_name = os.path.join(folder, f)
            try:
                file_data, results, results_all, f_id = best_folder(f_name, n_files)
                if file_data[1, 0] < n_files:
                    print(f_name, file_data[1, 0])
            except ValueError as e:
                print('not done: ', f_name)
                print(e)
                results = np.zeros((3, 2))
                results_all = np.zeros((3, 10))
            sb.append(results)
            sb_all.append(results_all)
        single_band[ba] = np.array(sb)
        single_band_all[ba] = np.array(sb_all)
        if ba != 'hg':
            mb = []
            mb_all = []
            for subject in subjects:
                f = ft.format(subject, 'hg_{}'.format(ba))
                f_name = os.path.join(folder, f)
                try:
                    file_data, results, results_all, f_id = best_folder(f_name, n_files)
                    if file_data[1, 0] < n_files:
                        print(f_name, file_data[1, 0])
                except ValueError as e:
                    print('not done: ', f_name)
                    print(e)
                    results = np.zeros((3, 2))
                    results_all = np.zeros((3, 10))
                mb.append(results)
                mb_all.append(results_all)
            multi_band[ba] = np.array(mb)
            multi_band_all[ba] = np.array(mb_all)
    name = fto.replace('{}_', '')

    with open('multiband_results_{}.pkl'.format(name), 'w') as f:
        cPickle.dump((single_band, multi_band, single_band_all, multi_band_all), f)
    print()
