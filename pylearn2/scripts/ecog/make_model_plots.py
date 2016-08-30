#!/usr/bin/env python
from pylearn2.datasets import ecog, ecog_new

import os, h5py, argparse, cPickle
import numpy as np
import scipy as sp

import matplotlib
matplotlib.use('Agg')
from pylab import rcParams
import matplotlib.pyplot as plt

import analysis
import plotting


rcParams.update({'figure.autolayout': True})

def main(data_file, model_folders, plot_folder, new, subset, min_cvs=10,
         model_file_base='.pkl', overwrite=False):
    subject = os.path.basename(data_file).split('_')[0].lower()
    print(subject)
    print(model_folders)
    run = '_'.join([os.path.basename(f) for f in model_folders])
    fname_base = subject + '_' + run
    data_folder = os.path.join(plot_folder, 'data')
    files = [sorted([f for f in os.listdir(model_folder)
                     if ((model_file_base in f) and
                         (subset in f))])
             for model_folder in model_folders]
    
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
    """
        kwargs['condense'] = True
    """

    data_fname = os.path.join(data_folder, fname_base + '_model_output.pkl')
    if (not os.path.exists(data_fname) or overwrite):
        # Run data through the models
        accuracy_dicts = []
        indices_dicts = []
        y_hat_dicts = []
        logits_dicts = []
        hidden_dicts = []
        for file_list in files:
            accuracy_dict = {}
            accuracy_dicts.append(accuracy_dict)
            indices_dict = {}
            indices_dicts.append(indices_dict)
            y_hat_dict = {}
            y_hat_dicts.append(y_hat_dict)
            logits_dict = {}
            logits_dicts.append(logits_dict)
            hidden_dict = {}
            hidden_dicts.append(hidden_dict)
            for ii, filename in enumerate(file_list):
                misclass, indices, y_hat, logits, hidden = analysis.get_model_results(model_folder, filename, ii,
                                                                                      kwargs, data_file, new)
                accuracy_dict[filename] = [1.-m for m in misclass]
                indices_dict[filename] = indices
                y_hat_dict[filename] = y_hat
                logits_dict[filename] = logits
                hidden_dict[filename] = hidden

        # Format model data
        y_dims = None
        for yd in y_hat_dicts:
            for key in yd.keys():
                ydim = tuple(ydi.shape[1] for ydi in yd[key])
                if y_dims == None:
                    y_dims = ydim
                else:
                    assert all(yds == ydi for yds, ydi in zip(y_dims, ydim))
                    
        if new:
            ds = ecog_new.ECoG(data_file,
                               which_set='train',
                               **kwargs)
            has_data = []
            for ii in range(len(ecog_E_lbls)):
                if (ds.y == ii).sum() > 0:
                    has_data.append(ii)
            y_dims = [57]
        else:
            None
        dicts = (accuracy_dicts, indices_dicts, y_hat_dicts, logits_dicts,
                 hidden_dicts)
        dicts2 = analysis.condensed_2_dense(new, indices_dicts,
                                            y_hat_dicts, logits_dicts, ds)
        with open(data_fname, 'w') as f:
            cPickle.dump((dicts, dicts2, y_dims, has_data), f)
    else:
        with open(data_fname) as f:
            dicts, dicts2, y_dims, has_data = cPickle.load(f)
    (accuracy_dicts, indices_dicts, y_hat_dicts, logits_dicts,
     hidden_dicts) = dicts
    indices_dicts2, y_hat_dicts2, logits_dicts2 = dicts2
    mats = analysis.indx_dict2conf_mat(indices_dicts2, y_dims)
    c_mat, v_mat, cv_mat = mats
    accuracy = analysis.conf_mat2accuracy(c_mat, v_mat, cv_mat)
    (c_accuracy, v_accuracy, cv_accuracy, accuracy_per_cv,
     p_accuracy, m_accuracy) = accuracy

    if cv_accuracy is not None:
        print('cv mean: ',cv_accuracy.mean())
        print('cv std: ',cv_accuracy.std())
    if c_accuracy is not None:
        print('c mean: ',c_accuracy.mean())
        print('c std: ',c_accuracy.std())
    if v_accuracy is not None:
        print('v mean: ',v_accuracy.mean())
        print('v std: ',v_accuracy.std())
    if p_accuracy is not None:
        print('p mean: ',np.nanmean(p_accuracy))
        print('p std: ',np.nanstd(p_accuracy))
    if m_accuracy is not None:
        print('m mean: ',np.nanmean(m_accuracy))
        print('m std: ',np.nanstd(m_accuracy))
        
    # Basic plots
    fname = fname_base + '_' + 'cv_accuracy.pdf'
    plotting.plot_cv_accuracy(accuracy_per_cv, ecog_E_lbls, has_data, os.path.join(plot_folder, fname))
    
    # Clustering Plots
    if new:
        ds = ecog_new.ECoG(data_file,
                           which_set='train',
                           **kwargs)

    else:
        ds = ecog.ECoG(data_file,
                       which_set='train',
                       **kwargs)
    cvs, labels, pmv, features = analysis.get_phonetic_feature_matrix()
    p = features[has_data].T[pmv['place']].T
    p_dist = analysis.compute_pairwise_distances(p,
            sp.spatial.distance.hamming)
    m = features[has_data].T[pmv['manner']].T
    m_dist = analysis.compute_pairwise_distances(m,
            sp.spatial.distance.hamming)
    v = features[has_data].T[pmv['vowel']].T
    v_dist = analysis.compute_pairwise_distances(v,
            sp.spatial.distance.hamming)
    # Raw data
    X, y0 = analysis.load_raw_data(ds)
    fname = fname_base + '_' + 'dend_raw.pdf'
    plotting.create_dendrogram(X, y0, ecog_E_lbls, has_data, title=subject+' raw',
                               save_path=os.path.join(plot_folder, fname))

    classes = sorted(set(y0.ravel()))
    Xp = np.zeros((len(classes), X.shape[1]))
    for ii in range(len(classes)):
        Xp[ii] = X[y0 == ii].mean(axis=0)

    X_dist = analysis.compute_pairwise_distances(Xp,
            sp.spatial.distance.euclidean)

    ccp = analysis.correlate(X_dist, p_dist)
    ccm = analysis.correlate(X_dist, m_dist)
    ccv = analysis.correlate(X_dist, v_dist)
    fname = fname_base + '_' + 'corr_raw.pdf'
    plotting.corr_box_plot(ccp, ccm, ccv, title=subject+' raw',
                           save_path=os.path.join(plot_folder, fname))
    np.savez(os.path.join(data_folder, fname_base + '_corr_raw'), ccp=ccp,
             ccm=ccm, ccv=ccv)
    # Logits + Y_hat
    lgs = tuple()
    yhs = tuple()
    ys = tuple()
    for ind, lg, yh in zip(indices_dicts, logits_dicts, y_hat_dicts):
        for key in ind.keys():
            lgs += lg[key][0],
            yhs += yh[key][0],
            ys += ind[key][0][:,0],
    logits = np.concatenate(lgs, axis=0)
    y_hat = np.concatenate(yhs, axis=0)
    y = np.concatenate(ys, axis=0)
    classes = sorted(set(y.ravel()))
    # Logits
    fname = fname_base + '_' + 'dend_logits.pdf'
    plotting.create_dendrogram(logits, y, ecog_E_lbls, has_data, title=subject+' logits',
                               save_path=os.path.join(plot_folder, fname))
    logitsp = np.zeros((len(classes), logits.shape[1]))
    for ii in range(len(classes)):
        logitsp[ii] = logits[y == ii].mean(axis=0)

    logits_dist = analysis.compute_pairwise_distances(logitsp,
            sp.spatial.distance.euclidean)

    ccp = analysis.correlate(logits_dist, p_dist)
    ccm = analysis.correlate(logits_dist, m_dist)
    ccv = analysis.correlate(logits_dist, v_dist)
    fname = fname_base + '_' + 'corr_logits.pdf'
    plotting.corr_box_plot(ccp, ccm, ccv, title=subject+' logits',
                           save_path=os.path.join(plot_folder, fname))
    # Y_hat
    fname = fname_base + '_' + 'dend_yhat.pdf'
    plotting.create_dendrogram(y_hat, y, ecog_E_lbls, has_data, title=subject+' y_hat',
                               save_path=os.path.join(plot_folder, fname))
    y_hatp = np.zeros((len(classes), y_hat.shape[1]))
    for ii in range(len(classes)):
        y_hatp[ii] = y_hat[y == ii].mean(axis=0)

    y_hat_dist = analysis.compute_pairwise_distances(y_hatp,
            sp.spatial.distance.euclidean)

    ccp = analysis.correlate(y_hat_dist, p_dist)
    ccm = analysis.correlate(y_hat_dist, m_dist)
    ccv = analysis.correlate(y_hat_dist, v_dist)
    fname = fname_base + '_' + 'corr_y_hat.pdf'
    plotting.corr_box_plot(ccp, ccm, ccv, title=subject+' y_hat',
                           save_path=os.path.join(plot_folder, fname))
    np.savez(os.path.join(data_folder, fname_base + '_corr_y_hat'), ccp=ccp,
             ccm=ccm, ccv=ccv)
    np.savez(os.path.join(data_folder, fname_base + '_y_hat'), y_hat=y_hat, y=y)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make plots for an ECoG DNN model.')
    parser.add_argument('subject', choices=['ec2', 'ec9', 'gp31', 'gp33'], default='ec2')
    parser.add_argument('model_folder')
    parser.add_argument('-p', '--plot_folder', type=str,
            default=os.path.join(os.environ['HOME'], 'plots', 'model'))
    parser.add_argument('-n', '--new', type=bool, default=True)
    parser.add_argument('-a', '--audio', type=bool, default=False)
    parser.add_argument('-o', '--overwrite', type=bool, default=False)
    parser.add_argument('-s', '--subset', type=str, default='')
    parser.add_argument('-m', '--min_cvs', type=int, default=10)
    args = parser.parse_args()
    
    if args.audio:
        raise NotImplemetedError
    
    data_base = '${PYLEARN2_DATA_PATH}/ecog/'
    new_data_files = {'ec2': 'EC2_blocks_1_8_9_15_76_89_105_CV_HG_align_window_-05_to_079_events_nobaseline.h5',
                      'ec9': 'EC9_blocks_15_39_46_49_53_60_63_CV_HG_align_window_-05_to_079_events_nobaseline.h5',
                      'gp33': 'GP33_blocks_1_5_30_CV_HG_align_window_-05_to_079_events_nobaseline.h5',
                      'gp31': 'GP31_blocks_1_2_4_6_9_21_63_65_67_69_71_78_82_83_CV_HG_align_window_-05_to_079_events_nobaseline.h5'}
    old_data_files = {'ec2': 'EC2_CV_85_nobaseline_aug.h5',
                      'ec9': None,
                      'gp33': None,
                      'gp31': None}
    
    if args.subject == 'ec2':
        if args.new:
            data_file = os.path.join(data_base, 'hdf5', new_data_files['ec2'])
        else:
            data_file = os.path.join(data_base, old_data_files['ec2'])
    elif args.subject == 'ec9':
        data_file = os.path.join(data_base, 'hdf5', new_data_files['ec9'])
    elif args.subject == 'gp31':
        data_file = os.path.join(data_base, 'hdf5', new_data_files['gp31'])
    elif args.subject == 'gp33':
        data_file = os.path.join(data_base, 'hdf5', new_data_files['gp33'])
    else:
        raise ValueError
    
    main(data_file, [args.model_folder], args.plot_folder, args.new,
            args.subset, args.min_cvs, overwrite=args.overwrite)
