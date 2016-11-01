from __future__ import division
"""
ECoG dataset.
"""
__authors__ = "Jesse Livezey"

import numpy as N
np = N
import scipy as sp
import h5py, os
from theano.compat.six.moves import xrange
from pylearn2.datasets import hdf5, dense_design_matrix
from pylearn2.datasets import control
from pylearn2.datasets import cache
from pylearn2.utils import serial
from pylearn2.utils.rng import make_np_rng
from sklearn.decomposition import PCA


def complex_pca_function(X, final_dim=2000):
    """
    n_samples, n_features = X.shape
    u, s, v = np.linalg.svd(X, full_matrices=False)
    X_pca = u[:, :final_dim] * s[np.newaxis, :final_dim]
    K = v[:final_dim].conj().T

    class PCA(object):
        def __init__(self, K):
            self.K = K
        def transform(self, X):
            return X.dot(self.K)
    """
    model = PCA(n_components=final_dim)
    X_pca = model.fit_transform(X)

    return X_pca, model


_split = {'train': .8, 'valid': .1, 'test': .1}
assert np.allclose(_split['train']+_split['valid']+_split['test'],1.)

class ECoG(dense_design_matrix.DenseDesignMatrix):
    """
    ECoG dataset

    Parameters
    ----------
    filename : str
        Filename for data.
    which_set : str
        'train', 'valid', 'test', or 'augment'
    fold : int
        Which fold to use.
    center : bool
        If True, preprocess so that data has zero mean.
    move : float
        Fraction of data to move through for each fold.
    level_classes: bool
        Flag for making classes even over splits or just sampling randomly.
    consonant_prediction: bool
        Flag for just setting y to consonant class.
    vowel_prediction: bool
        Flag for just setting y to vowel class.
    two_headed: bool
        Flag for predicting consonant and vowel class in one network.
        Overrides consonant and vowel prediction.
    randomize_labels: bool
        Randomly permutes the labels for the examples.
        Meant for control runs.
    frac_train: float
        Percentage of training set to use during training.
    pm_aug_range: int
        Number of of time shifts to use in augmentation.
    """

    def __init__(self, subject, bands, data_types,
                 which_set, fold=0,
                 seed=20161022, center=True,
                 move = .1, level_classes=True,
                 randomize_labels=False,
                 frac_train=1.,
                 y_labels=57,
                 min_cvs=10,
                 condense=True,
                 total_dim=2000):
        self.args = locals()
        print( self.args)

        possible_subjects = ['EC2', 'EC9', 'GP31', 'GP33']
        possible_data_types = ['complex', 'amplitude', 'phase']
        possible_bands = ['alpha', 'theta', 'beta', 'high beta',
                          'gamma', 'high gamma']


        if which_set not in ['train', 'valid', 'test']:
            raise ValueError(
                'Unrecognized which_set value "%s".' % (which_set,) +
                '". Valid values are ["train","valid","test"].')

        print data_types, bands
        assert subject in possible_subjects
        if not isinstance(data_types, list):
            if ',' in data_types:
                data_types = data_types.replace(', ', ',')
                data_types = data_types.split(',')
            else:
                data_types = [data_types]
        if not  isinstance(bands, list):
            if ',' in bands:
                bands = bands.replace(', ', ',')
                bands = bands.split(',')
            else:
                bands = [bands]
        print data_types, bands
        assert all(d in possible_data_types for d in data_types)
        assert all(b in possible_bands for b in bands)

        if subject == 'EC2':
            filename = 'EC2_blocks_1_8_9_15_76_89_105_CV_AA_avg_align_window_-0.5_to_0.79_between_data_nobaseline.h5'
        else:
            raise ValueError
        filename = os.path.join('${PYLEARN2_DATA_PATH}/ecog/hdf5', filename)

        dims = [2*(int(np.ceil(total_dim/len(bands)))//2) for _ in bands]
        dims[0] = dims[0] + (total_dim - sum(dims))

        rng = np.random.RandomState(seed)

        filename = serial.preprocess(filename)
        stored = False

        with h5py.File(filename,'r') as f:
            try:
                raise KeyError
                X_train_tmp = [f['fold{}'.format(fold)][b]['train']['X'].value for b in bands]
                X_valid_tmp = [f['fold{}'.format(fold)][b]['valid']['X'].value for b in bands]
                X_test_tmp = [f['fold{}'.format(fold)][b]['test']['X'].value for b in bands]
                y_train = f['fold{}'.format(fold)][bands[0]]['train']['y'].value.astype(int)
                y_valid = f['fold{}'.format(fold)][bands[0]]['valid']['y'].value.astype(int)
                y_test = f['fold{}'.format(fold)][bands[0]]['test']['y'].value.astype(int)
                stored = True
            except KeyError:
                Xs = [f['X{}'.format(b)].value for b in bands]
                y = f['y'].value.astype(int)

        def split_indices(indices, frac_train, min_cvs):
            """
            Split indices into training/validation/testing groups.
            """
            num_idx = len(indices)
            if (num_idx >= min_cvs) and (num_idx > 3):
                indices = np.array(indices, dtype=int)
                order = rng.permutation(num_idx)

                n_test = max(int(np.round(num_idx*_split['test'])),1)
                n_valid = max(int(np.round(num_idx*_split['valid'])), 1)
                n_train = num_idx-n_valid-n_test
                assert num_idx == n_train + n_valid + n_test

                train_start = int(np.round(fold*num_idx*move))
                train_end = (train_start+n_train) % num_idx
                valid_start = train_end
                valid_end = (valid_start+n_valid) % num_idx
                test_start = valid_end
                test_end = (test_start+n_test) % num_idx

                if train_end > train_start:
                    train_idx = order[train_start:train_end]
                else:
                    train_idx = np.hstack((order[train_start:],order[:train_end]))
                assert train_idx.size == n_train, (train_start, train_end)

                if valid_end > valid_start:
                    valid_idx = order[valid_start:valid_end]
                else:
                    valid_idx = np.hstack((order[valid_start:],order[:valid_end]))
                assert valid_idx.size == n_valid

                if test_end > test_start:
                    test_idx = order[test_start:test_end]
                else:
                    test_idx = np.hstack((order[test_start:],order[:test_end]))
                assert test_idx.size == n_test

                if frac_train is not None:
                    assert frac_train > 0.
                    assert frac_train <= 1.
                    n_keep = int(np.round(frac_train*len(train_idx)))
                    extra_idx = train_idx[n_keep:]
                    train_idx = train_idx[:n_keep]
                else:
                    extra_idx = []

                return tuple([indices[idx].tolist() for idx in [train_idx, valid_idx, test_idx, extra_idx]])
            else:
                return tuple([[] for _ in range(3)]) + (indices,)

        def check_indices(tr, va, te, ex):
            """
            Check that all indices were included and the training/validation/testing
            splits are independent.
            """
            tr = set(tr)
            va = set(va)
            te = set(te)
            ex = set(ex)
            union = tr | va | te | ex
            max_val = max(union)
            assert len(union)-1 == max_val
            assert len(tr)+len(va)+len(te)+len(ex) == len(union)

        if not stored:

            n_examples = Xs[0].shape[0]
            assert all(n_examples == X.shape[0] for X in Xs)
            self.present_cvs = np.zeros(y_labels, dtype=int)
            if level_classes:
                n_classes = y_labels
                class_indices = {}
                for ii in xrange(n_classes):
                    class_indices[str(ii)] = np.nonzero(y == ii)[0].tolist()
                total = 0
                for indices in class_indices.values():
                    total += len(indices)
                assert total == y.shape[0]
                train_idx = []
                valid_idx = []
                test_idx = []
                extra_idx = []
                for ii, key in enumerate(sorted(class_indices.keys())):
                    tr, va, te, ex = split_indices(class_indices[key], frac_train, min_cvs)
                    if len(tr) > 0:
                        self.present_cvs[ii] = 1
                    train_idx += tr
                    valid_idx += va
                    test_idx += te
                    extra_idx += ex
            else:
                indices = range(n_examples)
                train_idx, valid_idx, test_idx, extra_idx = split_indices(indices, frac_train, min_cvs)

            check_indices(train_idx, valid_idx, test_idx, extra_idx)

            self.indices = (train_idx, valid_idx, test_idx, extra_idx)


            n_classes = y_labels

            if randomize_labels:
                in_idx = np.concatenate((train_idx, valid_idx, test_idx))
                order = rng.permutation(in_idx.shape[0])
                for X in Xs:
                    X[in_idx] = X[in_idx][order]

            X_train = [X[train_idx] for X in Xs]
            X_valid = [X[valid_idx] for X in Xs]
            X_test = [X[test_idx] for X in Xs]
            y_train = y[train_idx]
            y_valid = y[valid_idx]
            y_test = y[test_idx]

            X_train_tmp = []
            X_valid_tmp = []
            X_test_tmp = []
            for ii, (X, dim, dt) in enumerate(zip(X_train, dims, data_types)):
                n_ex = X.shape[0]
                X_pca, pca = complex_pca_function(X.reshape(n_ex, -1))
                X_train_tmp.append(X_pca)
                n_ex = X_valid[ii].shape[0]
                X_valid_tmp.append(pca.transform(X_valid[ii].reshape(n_ex,-1)))
                n_ex = X_test[ii].shape[0]
                X_test_tmp.append(pca.transform(X_test[ii].reshape(n_ex,-1)))
                with h5py.File(filename) as f:
                    for e in ['train', 'valid', 'test']:
                        for d in ['X', 'y']:
                            try:
                                del f['fold{}'.format(fold)][bands[ii]][e][d]
                            except KeyError:
                                pass
                        try:
                            del f['fold{}'.format(fold)][bands[ii]][e]
                        except KeyError:
                            pass
                    try:
                        g0 = f.create_group('fold{}'.format(fold))
                    except ValueError:
                        g0 = f['fold{}'.format(fold)]
                    try:
                        g1 = g0.create_group(bands[ii])
                    except ValueError:
                        g1 = g0[bands[ii]]

                    g2 = g1.create_group('train')
                    g2.create_dataset('X', data=X_train_tmp[ii])
                    g2.create_dataset('y', data=y_train)

                    g2 = g1.create_group('valid')
                    g2.create_dataset('X', data=X_valid_tmp[ii])
                    g2.create_dataset('y', data=y_valid)

                    g2 = g1.create_group('test')
                    g2.create_dataset('X', data=X_test_tmp[ii])
                    g2.create_dataset('y', data=y_test)

        X_train = np.hstack([X[:, :d] for X, d in zip(X_train_tmp, dims)])
        X_valid = np.hstack([X[:, :d] for X, d in zip(X_valid_tmp, dims)])
        X_test = np.hstack([X[:, :d] for X, d in zip(X_test_tmp, dims)])

        self.train_mean = X_train.mean(axis=0, keepdims=True)
        print X_train.shape, X_valid.shape, X_test.shape, self.train_mean.shape

        if which_set == 'train':
            topo_view = X_train
            y_final = y_train
        elif which_set == 'valid':
            topo_view = X_valid
            y_final = y_valid
        else:
            topo_view = X_test
            y_final = y_test
        print topo_view.shape
        if center:
            topo_view = topo_view-self.train_mean

        shape = topo_view.shape
        topo_view = topo_view[:, np.newaxis, np.newaxis, :]

        order = rng.permutation(topo_view.shape[0])
        topo_view = topo_view[order]
        y_final = y_final[order]
        self.y_final = y_final
        
        if condense:
            available_indxs = sorted(set(y_final))
            curr_idx = 0
            y_condensed = np.zeros_like(y_final)
            self.mapping = np.inf * np.ones(y_labels, dtype=int)
            for old_idx in range(max(available_indxs)+1):
                if old_idx in available_indxs:
                    y_condensed[y_final == old_idx] = curr_idx
                    self.mapping[old_idx] = curr_idx
                    curr_idx += 1
            n_classes = curr_idx
            y_final = y_condensed

        super(ECoG, self).__init__(topo_view=topo_view.astype('float32'),
                                    y=y_final[:, np.newaxis],
                                    axes=('b',0,1,'c'),
                                    y_labels=n_classes)

    def get_valid_set(self):
        """
        .. todo::

            WRITEME
        """
        args = {}
        args.update(self.args)
        del args['self']
        args['which_set'] = 'valid'
        return ECoG(**args)

    def get_test_set(self):
        """
        .. todo::

            WRITEME
        """
        args = {}
        args.update(self.args)
        del args['self']
        args['which_set'] = 'test'
        return ECoG(**args)
