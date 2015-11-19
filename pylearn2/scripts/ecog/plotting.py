import matplotlib.pyplot as plt
import numpy as np

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
    plt.show()

        