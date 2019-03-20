#!/usr/bin/env python
from pylearn2.datasets import ecog_neuro

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

def main(subject, bands, model_folders, plot_folder,
         model_file_base='.pkl', overwrite=False, randomize_labels=False,
         audio=False):
    print(subject)
    print(model_folders)
    print('audio', audio)
    run = '_'.join([os.path.basename(f) for f in model_folders])
    fname_base = subject + '_' + run
    data_folder = os.path.join(plot_folder, 'data')
    files = [sorted([f for f in os.listdir(model_folder)
                     if model_file_base in f])
             for model_folder in model_folders]
    print(files)
    
    with h5py.File(os.path.join(os.environ['PYLEARN2_DATA_PATH'],
                                'ecog/EC2_CV.h5'), 'r') as f:
        ecog_E_lbls = f['Descriptors']['Event_ELbls'].value

    kwargs = {'consonant_prediction': False,
              'vowel_prediction': False,
              'randomize_labels': randomize_labels,
              'audio': audio}

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
                misclass, indices, y_hat, logits, hidden = analysis.get_model_results(filename,
                                                                                      model_folder, 
                                                                                      subject,
                                                                                      bands,
                                                                                      ii,
                                                                                      kwargs)
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
                    
        ds = ecog_neuro.ECoG(subject,
                             bands,
                             'train',
                             **kwargs)
        has_data = []
        for ii in range(len(ecog_E_lbls)):
            if (ds.y == ii).sum() > 0:
                has_data.append(ii)
        y_dims = [57]

        dicts = (accuracy_dicts, indices_dicts, y_hat_dicts, logits_dicts,
                 hidden_dicts)
        dicts2 = analysis.condensed_2_dense(indices_dicts,
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
        print('cv: ',cv_accuracy)
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
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make plots for an ECoG DNN model.')
    parser.add_argument('subject', choices=['ec2', 'ec9', 'gp31', 'gp33'], default='ec2')
    parser.add_argument('bands', type=str)
    parser.add_argument('model_folder')
    parser.add_argument('-p', '--plot_folder', type=str,
            default=os.path.join(os.environ['HOME'], 'plots', 'model'))
    parser.add_argument('-o', '--overwrite', action='store_true')
    parser.add_argument('-r', '--randomize_labels', action='store_true')
    parser.add_argument('-a', '--audio', action='store_true')
    args = parser.parse_args()

    main(args.subject, args.bands, [args.model_folder],
         args.plot_folder, overwrite=args.overwrite,
         randomize_labels=args.randomize_labels, audio=args.audio)
