#!/usr/bin/env python
from pylearn2.datasets import ecog, ecog_new

import copy, os, h5py, argparse
import numpy as np
import scipy as sp

import matplotlib
matplotlib.use('Agg')
from pylab import rcParams
import matplotlib.pyplot as plt

import analysis
import plotting


rcParams.update({'figure.autolayout': True})

def main(data_file, plot_folder, min_cvs=10, model_file_base='.pkl',
         overwrite=False):
    subject = os.path.basename(data_file).split('_')[0].lower()
    fname_base = subject
    data_folder = os.path.join(plot_folder, 'data')

    with h5py.File(os.path.join(os.environ['HOME'],
                                'Development/data/ecog/EC2_CV.h5'), 'r') as f:
        ecog_E_lbls = f['Descriptors']['Event_ELbls'].value

    kwargs = {'move': .1,
              'center': True,
              'level_classes': True,
              'consonant_prediction': False,
              'vowel_prediction': False,
              'two_headed': False,
              'randomize_labels': False}
    if new:
        kwargs['min_cvs'] = min_cvs

    if new:
        ds = ecog_new.ECoG(data_file,
                           which_set='train',
                           **kwargs)
        ec = ecog_new
        has_data = []
        for ii in range(len(ecog_E_lbls)):
            if (ds.y == ii).sum() > 0:
                has_data.append(ii)


    else:
        ds = ecog.ECoG(data_file,
                       which_set='train',
                       **kwargs)
        ec = ecog

    X, y0 = analysis.load_raw_data(ds)
    
    # CV counts
    fname = fname_base + '_example_hist.pdf'
    data_fname = os.path.join(data_folder, fname_base +
                              '_example_hist.npz')
    f, h = plotting.plot_cv_counts(y0, subject, os.path.join(plot_folder, fname))
    np.savez(data_fname, hist=h)


    # Temporal Classification
    kwargs_copy = copy.deepcopy(kwargs)
    data_fname = os.path.join(data_folder, fname_base +
                              '_cv_time_indep.npz')
    if (not os.path.exists(data_fname) or overwrite):
        c_ita, v_ita, cv_ita, c_v_ita = analysis.time_accuracy(data_file, ec,
                                                               kwargs_copy,
                                                               has_data)
        np.savez(data_fname, c_ita=c_ita, v_ita=v_ita,
                 cv_ita=cv_ita, c_v_ita=c_v_ita)
    else:
        with np.load(data_fname) as f:
            c_ita, v_ita, cv_ita, c_v_ita = (
                    f['c_ita'], f['v_ita'], f['cv_ita'], f['c_v_ita'])

    data_fname = os.path.join(data_folder, fname_base +
                              '_cv_time_all.npz')
    if (not os.path.exists(data_fname) or overwrite):
        c_ata, v_ata, cv_ata, c_v_ata = analysis.time_accuracy(data_file, ec,
                                                               kwargs_copy,
                                                               has_data,
                                                               train_all_time=True)
        np.savez(data_fname, c_ata=c_ata, v_ata=v_ata,
                 cv_ata=cv_ata, c_v_ata=c_v_ata)
    else:
        with np.load(data_fname) as f:
            c_ata, v_ata, cv_ata, c_v_ata = (
                    f['c_ata'], f['v_ata'], f['cv_ata'], f['c_v_ata'])

    kwargs_copy['randomize_labels'] = True
    data_fname = os.path.join(data_folder, fname_base +
                              '_scv_time_indep.npz')
    if (not os.path.exists(data_fname) or overwrite):
        sc_ita, sv_ita, scv_ita, sc_v_ita = analysis.time_accuracy(data_file, ec,
                                                               kwargs_copy,
                                                               has_data)
        np.savez(data_fname, sc_ita=sc_ita, sv_ita=sv_ita,
                 scv_ita=scv_ita, sc_v_ita=sc_v_ita)
    else:
        with np.load(data_fname) as f:
            sc_ita, sv_ita, scv_ita, sc_v_ita = (
                    f['sc_ita'], f['sv_ita'], f['scv_ita'], f['sc_v_ita'])

    data_fname = os.path.join(data_folder, fname_base +
                              '_scv_time_all.npz')
    if (not os.path.exists(data_fname) or overwrite):
        sc_ata, sv_ata, scv_ata, sc_v_ata = analysis.time_accuracy(data_file, ec,
                                                                   kwargs_copy,
                                                                   has_data,
                                                                   train_all_time=True)
        np.savez(data_fname, sc_ata=sc_ata, sv_ata=sv_ata,
                 scv_ata=scv_ata, sc_v_ata=sc_v_ata)
    else:
        with np.load(data_fname) as f:
            sc_ata, sv_ata, scv_ata, sc_v_ata = (
                    f['sc_ata'], f['sv_ata'], f['scv_ata'], f['sc_v_ata'])
    # C and V
    fname = os.path.join(plot_folder, fname_base +
            '_consonant_and_vowel_indep.pdf')
    plotting.plot_time_accuracy_c_v(c_ita, sc_ita, v_ita, sv_ita,
                                    title=subject + ' ' + 'Independent Classifiers',
                                    save_path=fname)
    fname = os.path.join(plot_folder, fname_base +
            '_consonant_and_vowel_one.pdf')
    plotting.plot_time_accuracy_c_v(c_ata, sc_ata, v_ata, sv_ata,
                                    title=subject + ' ' + 'One Classifier',
                                    save_path=fname)
    # CV
    fname = os.path.join(plot_folder, fname_base + '_cv_indep.pdf')
    plotting.plot_time_accuracy_cv(cv_ita, scv_ita, c_v_ita, sc_v_ita,
                                    title=subject + ' ' + 'Independent Classifiers',
                                    save_path=fname)
    fname = os.path.join(plot_folder, fname_base + '_cv_one.pdf')
    plotting.plot_time_accuracy_cv(cv_ata, scv_ata, c_v_ata, sc_v_ata,
                                    title=subject + ' ' + 'One Classifier',
                                    save_path=fname)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make plots for an ECoG DNN model.')
    parser.add_argument('subject', choices=['ec2', 'ec9', 'gp31', 'gp33'], default='ec2')
    parser.add_argument('-p', '--plot_folder', type=str,
                        default=os.path.join(os.environ['HOME'], 'plots', 'ds'))
    parser.add_argument('-a', '--audio', type=bool, default=False)
    parser.add_argument('-m', '--min_cvs', type=int, default=10)
    parser.add_argument('-o', '--overwrite', type=bool, default=0)
    args = parser.parse_args()
    
    if args.audio:
        raise NotImplemetedError
    
    data_base = '${PYLEARN2_DATA_PATH}/data/ecog'
    data_files = {'ec2': 'EC2_blocks_1_8_9_15_76_89_105_CV_HG_align_window_-0.5_to_0.79_file_nobaseline.h5', 
                  'ec9': 'EC9_blocks_15_39_46_49_53_60_63_CV_HG_align_window_-0.5_to_0.79_file_nobaseline.h5',
                  'gp31': 'GP31_blocks_1_2_4_6_9_21_63_65_67_69_71_78_82_83_CV_HG_align_window_-0.5_to_0.79_file_nobaseline.h5',
                  'gp33': 'GP33_blocks_1_5_30_CV_HG_align_window_-0.5_to_0.79_file_nobaseline.h5'}
    
    data_file = os.path.join(data_base, 'hdf5', new_data_files[args.subject])
    
    main(data_file, args.plot_folder,
         args.min_cvs, args.overwrite)
