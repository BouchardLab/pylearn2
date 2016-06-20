#!/usr/bin/env python
from pylearn2.datasets import ecog, ecog_new

import copy, os, h5py, argparse
import numpy as np
import scipy as sp
from scipy.optimize import minimize
import theano
import theano.tensor as T

import matplotlib
matplotlib.use('Agg')
from pylab import rcParams
import matplotlib.pyplot as plt

import analysis
import plotting


rcParams.update({'figure.autolayout': True})

def main(data_file, plot_folder, new, min_cvs=10,
         overwrite=False):
    subject = os.path.basename(data_file).split('_')[0].lower()
    fname_base = subject
    data_folder = os.path.join(plot_folder, 'data')

    with h5py.File(os.path.join(os.environ['HOME'],
                                'Development/data/ecog/EC2_CV.h5'), 'r') as f:
        ecog_E_lbls = f['Descriptors']['Event_ELbls'].value

    kwargs = {}
    if new:
        kwargs['min_cvs'] = min_cvs

    if new:
        ec = ecog_new
    else:
        ec = ecog

    kwargs['randomize_labels'] =  True
    # SVD analysis
    fname = fname_base + '_svd_random.pdf'
    data_fname = os.path.join(data_folder, fname_base +
                              '_svd_all_random.npz')
    if (not os.path.exists(data_fname) or overwrite):
        par, mar, var, usr, ssr, vsr, ilr, nlr = analysis.svd_accuracy(data_file, ec, kwargs)
        parf, prw = analysis.fit_accuracy_lognormal([par], ilr, nlr, check_nan=True)
        marf, mrw = analysis.fit_accuracy_lognormal([mar], ilr, nlr, check_nan=True)
        varf, vrw = analysis.fit_accuracy_lognormal([var], ilr, nlr, check_nan=True)
        np.savez(data_fname, pa=par, ma=mar, va=var, us=usr, ss=ssr, vs=vsr, il=ilr, nl=nlr,
                 paf=parf, maf=marf, vaf=varf, pw=prw, mw=mrw, vw=vrw)
    else:
        with np.load(data_fname) as f:
            (par, mar, var, usr, ssr, vsr, ilr, nlr,
             parf, marf, varf, prw, pmrw, vrw) = (
            f['pa'], f['ma'],f['va'],f['us'],f['ss'],f['vs'],f['il'],f['nl'],
            f['paf'], f['maf'],f['vaf'], f['pw'], f['mw'],f['vw'])
    plotting.plot_svd_accuracy(par, mar, var, ssr, ilr, nlr, title=subject,
                               save_path=os.path.join(plot_folder, fname),
                               over_chance=False)
    fname = fname_base + '_svd_random_fits.pdf'
    plotting.plot_svd_accuracy(parf/np.nanmean(par), marf/np.nanmean(mar), varf/np.nanmean(var),
                               ssr, ilr, nlr, title=subject,
                               save_path=os.path.join(plot_folder, fname))



    kwargs['randomize_labels'] =  False
    # SVD analysis
    fname = fname_base + '_svd.pdf'
    data_fname = os.path.join(data_folder, fname_base +
                              '_svd_all.npz')
    if (not os.path.exists(data_fname) or overwrite):
        pa, ma, va, us, ss, vs, il, nl = analysis.svd_accuracy(data_file, ec, kwargs)
        np.savez(data_fname, pa=pa, ma=ma, va=va, us=us, ss=ss, vs=vs, il=il, nl=nl)
    else:
        with np.load(data_fname) as f:
            pa, ma, va, us, ss, vs, il, nl = (
            f['pa'], f['ma'],f['va'],f['us'],f['ss'],f['vs'],f['il'],f['nl'])
    plotting.plot_svd_accuracy(pa/np.nanmean(par), ma/np.nanmean(mar), va/np.nanmean(var),
                               ss, il, nl, title=subject,
                               save_path=os.path.join(plot_folder, fname))

    fname = fname_base + '_svd_fits.pdf'
    data_fname = os.path.join(data_folder, fname_base +
                              '_svd_all_fits.npz')
    if (not os.path.exists(data_fname) or overwrite):
        paf, maf, vaf, pw, mw, vw = analysis.fit_accuracy_lognormal([pa, ma, va], il, nl)
        np.savez(data_fname, pa=paf, ma=maf, va=vaf, pw=pw, mw=mw, vw=vw, il=il, nl=nl)
    else:
        with np.load(data_fname) as f:
            paf, maf, vaf, pw, mw, vw, il, nl = (
            f['pa'], f['ma'],f['va'],f['pw'], f['mw'],f['vw'],f['il'],f['nl'])
    plotting.plot_svd_accuracy(paf/np.nanmean(par), maf/np.nanmean(mar), vaf/np.nanmean(var),
                               ss, il, nl, title=subject,
                               save_path=os.path.join(plot_folder, fname))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make svd plots for an ECoG dataset.')
    parser.add_argument('subject', choices=['ec2', 'ec9', 'gp31'], default='ec2')
    parser.add_argument('-p', '--plot_folder', type=str,
                        default=os.path.join(os.environ['HOME'], 'plots', 'svd'))
    parser.add_argument('-n', '--new', type=bool, default=True)
    parser.add_argument('-a', '--audio', type=bool, default=False)
    parser.add_argument('-m', '--min_cvs', type=int, default=10)
    parser.add_argument('-o', '--overwrite', type=bool, default=0)
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
    
    main(data_file, args.plot_folder, args.new,
         args.min_cvs, args.overwrite)
