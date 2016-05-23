#!/usr/bin/env python
from pylearn2.datasets import ecog, ecog_new

import os, h5py, argparse
import numpy as np
import scipy as sp

import matplotlib
matplotlib.use('Agg')
from pylab import rcParams
import matplotlib.pyplot as plt

import analysis
import plotting


rcParams.update({'figure.autolayout': True})

def main(data_file, model_folders, plot_folder, new, subset, min_cvs=10, model_file_base='.pkl'):
    subject = os.path.basename(data_file).split('_')[0].lower()
    run = '_'.join([os.path.basename(f) for f in model_folders])
    fname_base = subject + '_' + run
    data_folder = os.path.join(plot_folder, 'data')
    files = [sorted([f for f in os.listdir(model_folder) if ((model_file_base in f) and
                                                             (subset in f))])
             for model_folder in model_folders]

    fold_file = [[[f for f in folder if 'fold'+str(n) in f] for n in range(10)]
            for folder in files]
    for l in fold_file:
        for f in l:
            print f
            print ''
            print ''
    
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

    """
    files2 = []
    for x in files:
        order = np.random.permutation(len(x))
        nf = []
        for ii in range(5):
            nf.append(x[order[ii]])
        files2.append(nf)
    files = files2
    print files
    """
    
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
        for filename in file_list:
            print filename
            fold = int(filename.split('fold')[-1].split('_')[0])
            misclass, indices, y_hat, logits, hidden = analysis.get_model_results(model_folder, filename, fold,
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
        pass

    indices_dicts2, y_hat_dicts2, logits_dicts2 = analysis.condensed_2_dense(new, indices_dicts,
                                                                             y_hat_dicts, logits_dicts, ds)

    place = dict()
    manner = dict()
    vowel = dict()

    for idx_d in indices_dicts2:
        for f in idx_d.keys():
            pl = []
            ml = []
            vl = []
            idxs = idx_d[f][0]
            ys = idxs[:, 0]
            y_hats = idxs[:, 1]
            for y, y_hat in zip(ys, y_hats):
                p = analysis.place_equiv(y, y_hat)
                if p is not None:
                    pl.append(p)
                m = analysis.manner_equiv(y, y_hat)
                if m is not None:
                    ml.append(m)
                v = analysis.vowel_equiv(y, y_hat)
                if v is not None:
                    vl.append(v)
            place[f] = np.array(pl).astype(float).mean()
            manner[f] = np.array(ml).astype(float).mean()
            vowel[f] = np.array(vl).astype(float).mean()
    folds = 0
    for f in place.keys():
        folds = max(folds, int(f.split('fold')[-1].split('_')[0]))
    epoch_max = np.zeros(folds+1, dtype=int)
    for f in place.keys():
        fold = int(f.split('fold')[-1].split('_')[0])
        epoch_max[fold] = max(epoch_max[fold],
                int(f.split('.')[0].split('_')[-1]))
    epoch_pmv = [np.zeros((epochs+1, 3)) for epochs in epoch_max]
    for f in place.keys():
        fold = int(f.split('fold')[-1].split('_')[0])
        epoch = int(f.split('.')[0].split('_')[-1])
        epoch_pmv[fold][epoch, 0] = place[f]
        epoch_pmv[fold][epoch, 1] = manner[f]
        epoch_pmv[fold][epoch, 2] = vowel[f]
    np.savez(os.path.join(data_folder, fname_base + '_training_pmv'),
             *epoch_pmv)

    min_epoch = epoch_max.min()
    epoch_accuracies = np.zeros((min_epoch, 3))
    for pmv in epoch_pmv:
        epoch_accuracies += pmv[:min_epoch]
    epoch_accuracies /= len(epoch_pmv)
    fname = fname_base + '_place_epochs.pdf'
    plt.figure()
    plt.plot(epoch_accuracies[:, 0])
    plt.xlabel('epochs')
    plt.ylabel('Place Accuracy')
    plt.savefig(os.path.join(plot_folder, fname))

    fname = fname_base + '_manner_epochs.pdf'
    plt.figure()
    plt.plot(epoch_accuracies[:, 1])
    plt.xlabel('epochs')
    plt.ylabel('Manner Accuracy')
    plt.savefig(os.path.join(plot_folder, fname))

    fname = fname_base + '_vowel_epochs.pdf'
    plt.figure()
    plt.plot(epoch_accuracies[:, 2])
    plt.xlabel('epochs')
    plt.ylabel('Vowel Accuracy')
    plt.savefig(os.path.join(plot_folder, fname))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make plots for an ECoG DNN model.')
    parser.add_argument('subject', choices=['ec2', 'ec9', 'gp31'], default='ec2')
    parser.add_argument('model_folder')
    parser.add_argument('-p', '--plot_folder', type=str, default=os.path.join(os.environ['HOME'], 'plots'))
    parser.add_argument('-n', '--new', type=bool, default=True)
    parser.add_argument('-a', '--audio', type=bool, default=False)
    parser.add_argument('-s', '--subset', type=str, default='')
    parser.add_argument('-m', '--min_cvs', type=int, default=10)
    args = parser.parse_args()
    
    if args.audio:
        raise NotImplemetedError
    
    data_base = '${PYLEARN2_DATA_PATH}/ecog/'
    new_data_files = {'ec2': 'EC2_blocks_1_8_9_15_76_89_105_CV_HG_align_window_-05_to_079_events_nobaseline.h5',
                      'ec9': 'EC9_blocks_15_39_46_49_53_60_63_CV_HG_align_window_-05_to_079_events_nobaseline.h5',
                      'gp31': 'GP31_blocks_1_2_4_6_9_21_63_65_67_69_71_78_82_83_CV_HG_align_window_-05_to_079_events_nobaseline.h5'}
    old_data_files = {'ec2': 'EC2_CV_85_nobaseline_aug.h5',
                      'ec9': None,
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
    else:
        raise ValueError
    
    main(data_file, [args.model_folder], args.plot_folder, args.new,
            args.subset, args.min_cvs)
