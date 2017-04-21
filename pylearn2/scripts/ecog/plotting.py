import matplotlib.pyplot as plt
import numpy as np
from scipy import cluster
import functools


def plot_confusion_matrix(conf_matrix, title=None, save_path=None):
    phoneme_map = {'aa': 'a',
                   'ee': 'i',
                   'oo': 'u',
                   'y': 'j',
                   'th': 'Q',
                   'sh': 'L'}

    def to_phonetic(syllables):
        rval = []
        for syl in syllables:
            for k, v in phoneme_map.iteritems():
                syl = syl.replace(k, v)
            rval.append(syl)
        return rval

    tick_offset = .02
    tick_scale = .05
    ratio = 6

    ticks = [0,.25,.5,.75,1]
    f = plt.figure(figsize=(20, 8))
    clip = [0,.5]
    f = plt.figure()
    plt.imshow(np.clip(conf_matrix_consonant/conf_matrix_consonant.sum(1, keepdims=True),
                             clip[0], clip[1]),
                     interpolation='nearest', cmap='gray_r')
    plt.set_ylabel('Ground Truth')
    plt.set_xlabel('Predicted')
    plt.set_xticks(np.arange(y_dim))
    plt.set_xticklabels(to_phonetic(ecog_E_lbls[to_consonant]), rotation=90)
    plt.set_yticks(np.arange(y_dim))
    plt.set_yticklabels(to_phonetic(ecog_E_lbls[to_consonant]))
    pos = 0
    for label in f.axes[0].yaxis.get_majorticklabels():
        label.set_position([tick_scale*((pos%2)-.5)-tick_offset, 0])
        pos += 1
    pos = 0
    for label in f.axes[0].xaxis.get_majorticklabels():
        label.set_position([0, tick_scale*((pos%2)-.5)-tick_offset])
        pos += 1

    plt.colorbar(ticks=ticks)

    if title is not None:
        plt.title(title)
    if save_path is not None:
        plt.savefig(f, save_path)

    return f

def plot_svd_accuracy(pa, ma, va, ss, il, nl,
                      folds=10., over_chance=True,
                      title=None, save_path=None):

    colors = plt.cm.plasma(np.linspace(0, 1, len(nl)))
    figs = []

    fig = plt.figure()
    mean = ss.mean(axis=0)[:-1]
    std = ss.std(axis=0)[:-1]
    plt.fill_between(np.arange(mean.shape[0])+1, (mean-std/np.sqrt(ss.shape[0])),
                             (mean+std/np.sqrt(ss.shape[0])),
                                              facecolor='black',
                                                               edgecolor='black')
    plt.xlabel('Singular value index')
    plt.ylabel('Singular value, log scale')
    if title:
        plt.title(title)
    plt.title(title)
    plt.yscale('log')
    if save_path:
        print 'here'
        pre, post = save_path.split('.')
        print save_path
        plt.savefig(pre+'_svd.'+post)

    for l, rs in zip(['place', 'manner', 'vowel'], [pa, ma, va]):
        fig = plt.figure()
        figs.append(fig)
        for ii, n_svs in enumerate(nl):
            mean = rs[:, ii].mean(axis=0)
            std = rs[:, ii].std(axis=0)
            plt.fill_between(il, (mean-std/np.sqrt(folds)),
                             (mean+std/np.sqrt(folds)),
                             label='n S.V.: '+str(n_svs),
                             facecolor=colors[ii],
                             edgecolor='black')
        plt.legend(loc='upper right')
        plt.xlabel('Start of SV window')
        if over_chance:
            plt.ylabel('Accuracy/chance')
            plt.ylim([.5, 2])
        else:
            plt.ylabel('Accuracy')
            plt.ylim([0, 1.5])
        plt.xlim((il.min(),il.max()))
        if title:
            plt.title(title+' '+l)
        if save_path:
            pre, post = save_path.split('.')
            plt.savefig(pre+'_'+l+'.'+post)
    return figs

def plot_time_accuracy_c_v(ca, sca, va, sva, folds=10.,
                           title=None, save_path=None):
    x = np.arange(-100*5,158*5, 5)

    c_mean = ca.mean(axis=0)
    c_std = ca.std(axis=0)
    v_mean = va.mean(axis=0)
    v_std = va.std(axis=0)

    sc_mean = sca.mean()
    sv_mean = sva.mean()

    fig = plt.figure()
    plt.fill_between(x, (c_mean-c_std/np.sqrt(folds))/sc_mean,
            (c_mean+c_std/np.sqrt(folds))/sc_mean,
                             facecolor='black', edgecolor='black')

    plt.fill_between(x, (v_mean-v_std/np.sqrt(folds))/sv_mean,
            (v_mean+v_std/np.sqrt(folds))/sv_mean,
                             facecolor='red', edgecolor='red')

    plt.plot(x, np.ones_like(x), color='black')
    plt.xlabel('Time (ms)')
    plt.ylabel('Accuracy/chance')
    plt.xlim((x.min(),x.max()))
    plt.ylim((.5, max(5.5, max(c_mean.max(), v_mean.max()))+.5))
    if title:
        plt.title(title)
    if save_path:
        plt.savefig(save_path)
    return fig

def plot_time_accuracy_cv(cva, scva, c_va, sc_va, folds=10.,
                           title=None, save_path=None):
    x = np.arange(-100*5,158*5, 5)

    cv_mean = cva.mean(axis=0)
    cv_std = cva.std(axis=0)
    c_v_mean = c_va.mean(axis=0)
    c_v_std = c_va.std(axis=0)

    scv_mean = scva.mean()
    sc_v_mean = sc_va.mean()

    fig = plt.figure()
    plt.fill_between(x, (cv_mean-cv_std/np.sqrt(folds))/scv_mean,
            (cv_mean+cv_std/np.sqrt(folds))/scv_mean,
                             facecolor='black', edgecolor='black')

    plt.fill_between(x, (c_v_mean-c_v_std/np.sqrt(folds))/sc_v_mean,
            (c_v_mean+c_v_std/np.sqrt(folds))/sc_v_mean,
                             facecolor='grey', edgecolor='grey')

    plt.plot(x, np.ones_like(x), color='black')
    plt.xlabel('Time (ms)')
    plt.ylabel('Accuracy/chance')
    plt.xlim((x.min(),x.max()))
    if title:
        plt.title(title)
    if save_path:
        plt.savefig(save_path)
    return fig

def plot_cv_counts(y, title, save_path, cvs=57):
    nums = np.zeros(cvs)
    for ii in range(cvs):
        nums[ii] = (y == ii).sum()
    min_n = nums.min().astype(int)
    max_n = nums.max().astype(int)
    hist = np.zeros(max_n+1)
    for n in nums:
        hist[n] += 1
    fig = plt.figure()
    plt.bar(range(max_n+1), hist)
    plt.xlabel('Examples per CV')
    plt.ylabel('Counts')
    if title:
        plt.title(title)
    if save_path:
        plt.savefig(save_path)
    return fig, hist

def plot_cv_accuracy(accuracy_per_cv, labels, has_data=None, save_path=None):
    folds, cvs = accuracy_per_cv.shape
    if has_data is not None:
        has_data = np.arange(cvs)
    accuracy_per_cv = accuracy_per_cv[:, has_data]
    labels = labels[has_data]
    fig = plt.figure()
    plt.bar(range(cvs), accuracy_per_cv.mean(axis=0), color='g',
            yerr=accuracy_per_cv.std(axis=0)/np.sqrt(folds))
    plt.title('Accuracy (Fold Averaged)')
    plt.xticks(np.arange(cvs)+.5, labels, rotation=90)
    if save_path:
        plt.savefig(save_path)
    return fig
    

def plot_trials(trials, labels, label_to_string, time=None, onset=None, pp=None):
    """
    Plot all trials individually.
    
    Parameters
    ----------
    trials : ndarray
        Trial data (dim(optional), trial, electrode, time).
        If 3D, trials are plotted. If 4D, first dimension
        assumed to index (mean, std).
    labels : ndarray
        Integer label for trials.
    label_to_string : list of str
        Titles for different labels.
    """
    cm = plt.get_cmap('viridis')
    n_labels = len(label_to_string)
    if time is None:
        time = np.arange(trials.shape[-1])
    indices = {}
    for ii, string in enumerate(label_to_string):
        indices[string] = np.nonzero(labels == ii)
        
    if trials.ndim == 3:
        data = trials
        std = None
    elif trials.ndim == 4:
        data = trials[0]
        std = trials[1]
    else:
        raise ValueError("trials should be 3 or 4 dimensional")
    spacing = 3.*data.std()
    
    means = {}
    stds = {}
    for string in label_to_string:
        indxs = indices[string]
        if len(indxs) > 0:
            means[string] = data[indxs]
            try:
                stds[string] = std[indxs]
            except TypeError:
                stds[string] = None
        else:
            means[string] = None
            stds[string] = None
    for string in label_to_string:
        if len(indices[string]) > 0:
            if stds[string] is not None:
                for trial, std in zip(means[string], stds[string]):
                    plt.figure()
                    for ii, (e_m, e_s) in enumerate(zip(trial, std)):
                        plt.plot(time, e_m/spacing+ii, c='black')
                        plt.fill_between(time, (e_m-e_s)/spacing+ii,
                                         (e_m+e_s)/spacing+ii,
                                         facecolor=cm(float(ii)/len(trial)))
                        plt.title(string)
                        plt.xlabel('time')
                        plt.ylabel('electrodes')
            else:
                for trial in means[string]:
                    plt.figure()
                    for ii, e in enumerate(trial):
                        plt.plot(time, e/spacing+ii, c=cm(float(ii)/len(trial)))
                        plt.title(string)
                        plt.xlabel('time')
                        plt.ylabel('electrodes')
    
def create_dendrogram(features, color_threshold, labels,
                      title=None, save_path=None, ax=None):
    """
    Create dendrogram from data X. Averages over labels y.
    """
    def color(z, thresh, groups, k):
        dist = z[k-57, 2]
        child = z[k-57, 0].astype('int')
        while child > 56:
            child = z[child-57, 0].astype(int)
        if dist > thresh:
            set_c = 'k'
        else:
            for c, idxs in groups.iteritems():
                if child in idxs:
                    set_c = c
        return set_c

    z = cluster.hierarchy.ward(features)
    r = cluster.hierarchy.dendrogram(z, labels=labels,
                                     no_plot=True)
    old_idx = []
    for cv in r['ivl']:
        old_idx.append(labels.index(cv))
    groups = {'#1f77b4': old_idx[0:11],
              '#ff7f0e': old_idx[11:23],
              '#2ca02c': old_idx[23:39],
              '#9467bd': old_idx[39:57]}


    r = cluster.hierarchy.dendrogram(z, labels=labels,
                                     link_color_func=functools.partial(color,
                                         z, color_threshold, groups),
                                     ax=ax)

    if title:
        plt.title(title)
    if save_path:
        plt.savefig(save_path)
    return z, r

def corr_box_plot(p, m, v, title=None, save_path=None):
    place_25 = np.sort(p)[np.round(int(p.size*.25))]
    place_med = np.median(p)
    place_75 = np.sort(p)[np.round(int(p.size*.75))]
    manner_25 = np.sort(m)[np.round(int(m.size*.25))]
    manner_med = np.median(m)
    manner_75 = np.sort(m)[np.round(int(m.size*.75))]
    vowel_25 = np.sort(v)[np.round(int(v.size*.25))]
    vowel_med = np.median(v)
    vowel_75 = np.sort(v)[np.round(int(v.size*.75))]
    box_params = {'notch': False,
                  'sym': '',
                  'vert': False,
                  'whis': 0,
                  'labels': ('Vowel configuration', 'Constriction degree', 'Constriction location'),
                  'medianprops': {'color': 'black', 'linewidth': 2},
                  'boxprops': {'color': 'gray', 'linewidth': 2}}
    data = [v, m, p]
    f = plt.figure()
    plt.boxplot(data, **box_params)
    plt.xlabel('Correlation Coefficient')
    if title:
        plt.title(title)
    if save_path:
        plt.savefig(save_path)
    return f
