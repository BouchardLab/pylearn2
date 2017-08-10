import h5py, os

import numpy as np
import scipy as sp

from ecog.utils import bands


def good_examples_and_channels(data):
    """Find good examples and channels.
    
    First removes all examples and channels that are completely NaN.
    The removes remaining examples and channels that are partially NaN.
    
    Parameters
    ----------
    data : ndarray (examples, channels, time)
        Data array.
    Returns
    -------
    good_examples : list
        Binary mask of examples that are not NaN.
        Same length as first dimension of data.
    good_channels : list
        Binary mask of channels that are not NaN.
        Same length as second dimension of data.
    """
    data = data.sum(axis=2)
    nan_time = np.isnan(data)
    
    # First exlude examples and channels that are all NaN
    bad_examples = np.all(nan_time, axis=1)
    bad_channels = np.all(nan_time, axis=0)
    data_good = data.copy()
    data_good[bad_examples] = 0.
    data_good[:, bad_channels] = 0.
    
    # Then exclude examples and channels with any NaNs from remaining group
    # Often a channel is bad only for a subset of blocks, so mask them first
    partial_bad_channels = np.isnan(data_good.sum(axis=0))
    data_good[:, partial_bad_channels] = 0.
    partial_bad_examples = np.isnan(data_good.sum(axis=1))
    good_examples = np.logical_not(partial_bad_examples) * np.logical_not(bad_examples)
    good_channels = np.logical_not(partial_bad_channels) * np.logical_not(bad_channels)
    
    return good_examples, good_channels

def get_cv_idxs(y, good_examples):
    y_counts = np.zeros(57)
    cv_idxs = []
    for ii in range(57):
        y_counts[ii] = (y == ii).sum()
        if (y_counts[ii] >= 10):
            cv_idxs.append(np.nonzero(y == ii)[0].tolist())
        else:
            cv_idxs.append([])

    keep_cvs = y_counts >= 10
    n_cv = keep_cvs.sum()

    cv_idxs = [sorted(list(set(idxs).intersection(good_examples))) for idxs in cv_idxs]
    return cv_idxs, n_cv

def save_power(f, channel, cv, subject):
    """Save the power spectrum matrix.
    
    Parameters
    ----------
    f : h5py file handle
    channel : int
        ECoG array channel.
    cv : str
        CV to select.
    subject : str
        Subject name for file name.
    """
    y = f['y'].value
    good_examples, good_channels = good_examples_and_channels(f['X0'].value)
    good_channels = np.nonzero(good_channels)[0].tolist()
    assert channel in good_channels
    n_time = f['X0'].shape[-1]
    cv_idx = f['tokens'].value.astype('str').tolist().index(cv)
    batch_idxs = np.nonzero(np.equal(y, cv_idx) * good_examples)[0].tolist()
    power_data = np.zeros((40, 258))
    for ii in range(40):
        power_data[ii] = np.nanmean(f['X{}'.format(ii)][batch_idxs][:, channel], axis=0)
    np.savez(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                          '{}_{}_{}_power.npz'.format(subject, cv, channel)), **{'power_data': power_data})

def plot_power(subject, channel, cv, vmin=None, vmax=None):
    """Plot the power spectrum matrix.
    
    Parameters
    ----------
    subject : str
        Subject name for file name.
    channel : int
        ECoG array channel.
    cv : str
        CV to select.
    """

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(5, 8))
    
    power_data = np.load(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                          '{}_{}_{}_power.npz'.format(subject, cv, channel)))['power_data']
    
    im = ax0.imshow(power_data[::-1, s], interpolation='nearest', cmap='afmhot', aspect='auto', vmin=vmin, vmax=vmax)
    ax0.set_yticks(np.arange(0, 40, 5))
    ax0.set_yticklabels(bands.chang_lab['cfs'][::-5].astype(int))
    #ax0.set_xticks([0, 100, 258])
    #ax0.set_xticklabels([-500, 0, 800])
    ax0.set_title('{}_{}_{}'.format(subject, channel, cv))
    ax0.set_ylabel('Freq.')
    ax0.set_xlabel('Time (ms)')
    
    hg_bands = np.logical_and(bands.chang_lab['cfs'] >= bands.neuro['min_freqs'][-1],
                              bands.chang_lab['cfs'] <= bands.neuro['max_freqs'][-1])
    b_bands = np.logical_and(bands.chang_lab['cfs'] >= bands.neuro['min_freqs'][2],
                             bands.chang_lab['cfs'] <= bands.neuro['max_freqs'][2])

    hb_bands = np.logical_and(bands.chang_lab['cfs'] >= bands.neuro['min_freqs'][3],
                              bands.chang_lab['cfs'] <= bands.neuro['max_freqs'][3])
    b_bands = np.logical_or(b_bands, hb_bands)
    b_bands = range(10, 21)
    hg = power_data[hg_bands].mean(axis=0)
    b = power_data[b_bands].mean(axis=0)

    b = b[s]
    hg = hg[s]

    hg -= hg.min()
    hg /= hg.max()
    hg = 2. * hg - 1
    b -= b.min()
    b /= b.max()
    b = 2. * b - 1

    ax1.plot(hg, c='r', lw=4)
    ax1.plot(b, c='k', lw=4)
    for ax in [ax0, ax1]:
        ax.set_xticks([0, 100, plot_idx[-1]])
        ax.set_xticklabels([-500, 0, int(1000 * plot_time[-1])-500])
    ax1.set_ylabel('Normalized Power')
    ax1.set_xlabel('Time (ms)')
    ax1.set_xlim([0, plot_idx[-1]])
    fig.tight_layout()

    plt.savefig(os.path.join(os.environ['HOME'], 'plots/xfreq',
                             '{}_{}_{}.pdf'.format(subject, channel, cv)))

def save_correlations(f, subject, channel=None):
    good_examples, good_channels = good_examples_and_channels(f['X0'].value)
    n_time = f['X0'].shape[-1]
    assert plot_idx[-1] <= n_time
    n_time = plot_idx[-1]

    vsmc = np.concatenate([f['anatomy']['preCG'].value, f['anatomy']['postCG'].value])
    vsmc_electrodes = np.zeros(256)
    vsmc_electrodes[vsmc] = 1

    good_examples = np.nonzero(good_examples)[0].tolist()

    good_channels = np.nonzero(vsmc_electrodes * good_channels)[0].tolist()
    if channel is not None:
        assert channel in good_channels
        good_channels = [channel]
    
    cv_idxs, n_cv = get_cv_idxs(f['y'].value, good_examples)

    n_ch = len(good_channels)
    n_ex = len(good_examples)
    
    def normalize(a):
        a -= np.mean(a, axis=-1, keepdims=True)
        a /= np.linalg.norm(a, axis=-1, keepdims=True)
        return a
    
    hg_bands = np.logical_and(bands.chang_lab['cfs'] >= bands.neuro['min_freqs'][-1],
                              bands.chang_lab['cfs'] <= bands.neuro['max_freqs'][-1])
    b_bands = np.logical_and(bands.chang_lab['cfs'] >= bands.neuro['min_freqs'][2],
                             bands.chang_lab['cfs'] <= bands.neuro['max_freqs'][2])
    hb_bands = np.logical_and(bands.chang_lab['cfs'] >= bands.neuro['min_freqs'][3],
                              bands.chang_lab['cfs'] <= bands.neuro['max_freqs'][3])
    b_bands = np.logical_or(b_bands, hb_bands)
    b_bands = range(10, 21)

    xcorr_freq = np.zeros((40, n_cv, n_ch))
    hg_ts = np.zeros((hg_bands.sum(), n_cv, n_ch, n_time))
    for ii, c in enumerate(np.nonzero(hg_bands)[0]):
        for jj, idxs in enumerate(cv_idxs):
            hg_ts[ii, jj] = f['X{}'.format(c)][idxs][:, good_channels].mean(axis=0)[..., s]
    hg_ts = np.mean(hg_ts, axis=0)
    hg_ts = normalize(hg_ts)
    print('loaded hg')

    for ii in range(40):
        for jj, idxs in enumerate(cv_idxs):
            other_ts = normalize(f['X{}'.format(ii)][idxs][:, good_channels].mean(axis=0)[..., s])
            xcorr_freq[ii, jj] = np.sum(hg_ts[jj] * other_ts, axis=-1)
            #print(np.sum(hg_ts[jj] * hg_ts[jj], axis=-1), np.sum(other_ts * other_ts, axis=-1))
    print('freq')
    
    b_ts = np.zeros((len(b_bands), n_cv, n_ch, n_time))
    for ii, c in enumerate(b_bands):
        for jj, idxs in enumerate(cv_idxs):
            b_ts[ii, jj] = f['X{}'.format(c)][idxs][:, good_channels].mean(axis=0)[..., s]
    b_ts = np.mean(b_ts, axis=0)
    b_ts = normalize(b_ts)
    ones = np.ones_like(hg_ts[0, 0])
    n_overlap = np.correlate(ones, ones, mode='full')
    xcorr_time = np.zeros((n_overlap.size, n_cv, n_ch))
    acorr_time = np.zeros((2, n_overlap.size, n_cv, n_ch))
    for ii in range(n_cv):
        for jj in range(n_ch):
            hg = hg_ts[ii, jj]
            b = b_ts[ii, jj]
            xcorr_time[:, ii, jj] = np.correlate(hg, b, mode='full')
            acorr_time[0, :, ii, jj] = np.correlate(hg, hg, mode='full')
            acorr_time[1, :, ii, jj] = np.correlate(b, b, mode='full')
    print('time')
    
    np.savez(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                          '{}_correlations.npz'.format(subject)), **{'xcorr_freq': xcorr_freq,
                                                                           'xcorr_time': xcorr_time,
                                                                    'acorr_time': acorr_time})

def plot_correlations(subject):
    d = np.load(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                '{}_correlations.npz'.format(subject)))
    xcorr_freq = d['xcorr_freq']
    xcorr_time = d['xcorr_time']
    acorr_time = d['acorr_time']

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(5, 2))
    
    mean = xcorr_freq.mean(axis=(1, 2))
    sem = xcorr_freq.std(axis=(1, 2)) / np.sqrt(np.prod(xcorr_freq.shape[1:]))
    ax0.plot(bands.chang_lab['cfs'], mean)
    ax0.fill_between(bands.chang_lab['cfs'], mean-sem, mean+sem, alpha=.5)
    ax0.set_xlim(0, 70)
    ax0.set_ylim(-.2, None)
    ax0.set_xlabel('Freq. (Hz)')
    ax0.set_ylabel(r'H$\gamma$ Corr. Coef.')
    
    mean = xcorr_time.mean(axis=(1, 2))
    sem = xcorr_time.std(axis=(1, 2)) / np.sqrt(np.prod(xcorr_time.shape[1:]))
    ax1.plot(mean)
    ax1.fill_between(np.arange(mean.size), mean-sem, mean+sem, alpha=.5)
    n_time = xcorr_time.shape[0]
    ax1.set_xticks([0, n_time // 2, n_time])
    ax1.set_xticklabels(int(1000 * (n_time // 2) * (1/200.)) * np.array([-1, 0, 1]))
    ax1.set_xlabel('Lag (ms)')
    ax1.set_ylabel(r'H$\gamma$-$\beta$ Corr. Coef.')
    fig.tight_layout()
    plt.savefig(os.path.join(os.environ['HOME'], 'plots/xfreq',
                             '{}_correlations.pdf'.format(subject)))
    
    fig, axes = plt.subplots(1, 2, figsize=(5, 2))
    
    for ii, ax in enumerate(axes):
        mean = acorr_time[ii].mean(axis=(1, 2))
        sem = acorr_time[ii].std(axis=(1, 2)) / np.sqrt(np.prod(xcorr_time.shape[1:]))
        ax.plot(mean)
        ax.fill_between(np.arange(mean.size), mean-sem, mean+sem, alpha=.5)
        n_time = xcorr_time.shape[0]
        ax.set_xticks([0, n_time // 2, n_time])
        ax.set_xticklabels(int(1000 * (n_time // 2) * (1/200.)) * np.array([-1, 0, 1]))
        ax.set_xlabel('Lag (ms)')
        ax.set_ylabel(r'H$\gamma$-$\beta$ Corr. Coef.')
        fig.tight_layout()
