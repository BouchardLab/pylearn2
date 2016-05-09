import matplotlib.pyplot as plt
import numpy as np
from scipy import cluster


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
    
def create_dendrogram(X, y, labels, has_data, color_threshold=None,
                      title=None, save_path=None):
    """
    Create dendrogram from data X. Averages over labels y.
    """
    vecs = np.zeros((len(has_data), X.shape[1]))
    y = y.ravel()
    for ii, idx in enumerate(has_data):
        vecs[ii] = X[y == idx].mean(0)
    z = cluster.hierarchy.ward(vecs)
    r = cluster.hierarchy.dendrogram(z, labels = labels[has_data],
                                     orientation='left', color_threshold=color_threshold)
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
    if title:
        plt.title(title)
    if save_path:
        plt.savefig(save_path)
    return f
