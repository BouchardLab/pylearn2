#!/usr/bin/env python
from pylearn2.datasets import ecog_neuro

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

def main(subject, bands, data_types, plot_folder, model_file_base='.pkl',
         overwrite=False, dim0=0, dim1=None):
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

    ds = ecog_neuro.ECoG(subject,
                         bands,
                         data_types,
                         'train',
                         dim0,
                         dim1,
                         **kwargs)
    ec = ecog_neuro
    has_data = []
    for ii in range(len(ecog_E_lbls)):
        if (ds.y == ii).sum() > 0:
            has_data.append(ii)

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
    print data_fname
    print overwrite, not os.path.exists(data_fname)
    if (not os.path.exists(data_fname) or overwrite):
        (c_ita, v_ita,
         cv_ita, c_v_ita) = analysis.time_accuracy(subject, bands, data_types,
                                                   dim0, dim1,
                                                   ec, kwargs_copy,
                                                   has_data)
        np.savez(data_fname, c_ita=c_ita, v_ita=v_ita,
                 cv_ita=cv_ita, c_v_ita=c_v_ita)
    else:
        with np.load(data_fname) as f:
            c_ita, v_ita, cv_ita, c_v_ita = (
                    f['c_ita'], f['v_ita'], f['cv_ita'], f['c_v_ita'])

    kwargs_copy['randomize_labels'] = True
    data_fname = os.path.join(data_folder, fname_base +
                              '_scv_time_indep.npz')
    if (not os.path.exists(data_fname) or overwrite):
        (sc_ita, sv_ita,
         scv_ita, sc_v_ita) = analysis.time_accuracy(subject, bands, data_types,
                                                     dim0, dim1,
                                                     ec, kwargs_copy,
                                                     has_data)
        np.savez(data_fname, sc_ita=sc_ita, sv_ita=sv_ita,
                 scv_ita=scv_ita, sc_v_ita=sc_v_ita)
    else:
        with np.load(data_fname) as f:
            sc_ita, sv_ita, scv_ita, sc_v_ita = (
                    f['sc_ita'], f['sv_ita'], f['scv_ita'], f['sc_v_ita'])

    # C and V
    fname = os.path.join(plot_folder, fname_base +
            '_consonant_and_vowel_indep.pdf')
    plotting.plot_time_accuracy_c_v(c_ita, sc_ita, v_ita, sv_ita,
                                    title=subject + ' ' + 'Independent Classifiers',
                                    save_path=fname)
    # CV
    fname = os.path.join(plot_folder, fname_base + '_cv_indep.pdf')
    plotting.plot_time_accuracy_cv(cv_ita, scv_ita, c_v_ita, sc_v_ita,
                                    title=subject + ' ' + 'Independent Classifiers',
                                    save_path=fname)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make plots for an ECoG DNN model.')
    parser.add_argument('subject', choices=['ec2', 'ec9', 'gp31', 'gp33'], default='ec2')
    parser.add_argument('bands', type=str)
    parser.add_argument('data_types', type=str)
    parser.add_argument('-p', '--plot_folder', type=str,
                        default=os.path.join(os.environ['HOME'], 'plots', 'ds'))
    parser.add_argument('-o', '--overwrite', action='store_true')
    args = parser.parse_args()
    
    main(args.subject, args.bands, args.data_types, args.plot_folder,
         overwrite=args.overwrite)
