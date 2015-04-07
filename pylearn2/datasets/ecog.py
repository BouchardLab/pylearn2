"""
ECoG dataset.
"""
__authors__ = "Jesse Livezey"

import numpy as N
np = N
import h5py
from theano.compat.six.moves import xrange
from pylearn2.datasets import hdf5, dense_design_matrix
from pylearn2.datasets import control
from pylearn2.datasets import cache
from pylearn2.utils import serial
from pylearn2.utils.rng import make_np_rng


_split = {'train': .8, 'valid': .1, 'test': .1}
assert np.allclose(_split['train']+_split['valid']+_split['test'],1.)

#class ECoG(hdf5.HDF5Dataset):
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
    randomize_label: bool
        Randomly permutes the labels for the examples.
        Meant for control runs.
    """

    def __init__(self, filename, which_set,
                 fold=0, seed=20141210, center=False,
                 move = .1, level_classes=False,
                 consonant_prediction=False,
                 vowel_prediction=False,
                 two_headed=False,
                 randomize_label=False,
                 load_all=None, cache_size=400000000):
        self.args = locals()


        if load_all is None:
            if which_set == 'augment':
                load_all = False
            else:
                load_all = True

        if which_set not in ['train', 'valid', 'test', 'augment']:
            raise ValueError(
                'Unrecognized which_set value "%s".' % (which_set,) +
                '". Valid values are ["train","valid","test"].')

        filename = serial.preprocess(filename)
        with h5py.File(filename,'r') as f:
            X = f['X'].value
            y = f['y'].value
            if two_headed:
                y_consonant = f['y_consonant'].value
                y_vowel = f['y_vowel'].value
            elif consonant_prediction:
                assert not vowel_prediction
                y_consonant = f['y_consonant'].value
            elif vowel_prediction:
                assert not consonant_prediction
                y_vowel = f['y_vowel'].value
            if which_set == 'augment':
                X_aug = f['X_aug'].value
                assert X_aug.shape[0] % y.shape[0] == 0
                tile_len = int(X_aug.shape[0]/y.shape[0])
                y_aug = np.tile(y, (tile_len, 1))
            
        rng = np.random.RandomState(seed)

        def split_indices(indices):
            num_idx = len(indices)
            indices = np.array(indices, dtype=int)
            order = rng.permutation(num_idx)

            n_train = int(np.round(num_idx*_split['train']))
            n_valid = int(np.round(num_idx*_split['valid']))
            n_test = num_idx-n_train-n_valid

            train_start = fold*num_idx*move
            train_end = (train_start+n_train) % num_idx
            valid_start = train_end
            valid_end = (valid_start+n_valid) % num_idx
            test_start = valid_end
            test_end = (test_start+n_test) % num_idx

            if train_end > train_start:
                train_idx = order[train_start:train_end]
            else:
                train_idx = np.hstack((order[train_start:],order[:train_end]))
            if valid_end > valid_start:
                valid_idx = order[valid_start:valid_end]
            else:
                valid_idx = np.hstack((order[valid_start:],order[:valid_end]))
            if test_end > test_start:
                test_idx = order[test_start:test_end]
            else:
                test_idx = np.hstack((order[test_start:],order[:test_end]))
            return tuple([indices[idx].tolist() for idx in [train_idx, valid_idx, test_idx]])

        def check_indices(tr, va, te):
            tr = set(tr)
            va = set(va)
            te = set(te)
            union = tr | va | te
            assert len(tr)+len(va)+len(te) == len(union)
            max_val = max(union)
            assert len(union)-1 == max_val

        if level_classes:
            n_classes = y.shape[1]
            class_indices = {}
            classes = y.argmax(axis=1)
            for ii in xrange(n_classes):
                class_indices[str(ii)] = np.nonzero(classes == ii)[0].tolist()
            total = 0
            for indices in class_indices.values():
                total += len(indices)
            assert total == y.shape[0]
            train_idx = []
            valid_idx = []
            test_idx = []
            for indices in class_indices.values():
                tr, va, te = split_indices(indices)
                train_idx += tr
                valid_idx +=va
                test_idx += te
        else:
            n_examples = X.shape[0]
            indices = range(n_examples)
            train_idx, valid_idx, test_idx = split_indices(indices)

        check_indices(train_idx, valid_idx, test_idx)

        if two_headed:
            y = np.hstack((y_consonant, y_vowel))
        elif consonant_prediction:
            assert not vowel_prediction
            y = y_consonant
        elif vowel_prediction:
            assert not consonant_prediction
            y = y_vowel

        if randomize_label:
            print 'here'
            order = rng.permutation(y.shape[0])
            y = y[order]

        X_train = X[train_idx]
        X_valid = X[valid_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_valid = y[valid_idx]
        y_test = y[test_idx]

        self.train_mean = X_train.mean(0)

        if which_set == 'augment':
            img_shape = X_train.shape[1:]
            y_shape = y_train.shape[1:]
            X_aug = X_aug[:,train_idx]
            X_aug = X_aug.reshape(-1,*img_shape)
            y_aug = y_aug[:,train_idx]
            y_aug = y_aug.reshape(-1,*y_shape)
            X_aug = np.concatenate((X_train, X_aug))
            y_aug = np.concatenate((y_train, y_aug))
            order = rng.permutation(X_aug.shape[0])
            X_train = X_aug[order]
            y_train = y_aug[order]
            del X_aug
            del y_aug

        if (which_set == 'train') or (which_set == 'augment'):
            topo_view = X_train
            y_final = y_train
        elif which_set == 'valid':
            topo_view = X_valid
            y_final = y_valid
        else:
            topo_view = X_test
            y_final = y_test
        if center:
            topo_view = topo_view-self.train_mean[np.newaxis,...]

        super(ECoG, self).__init__(topo_view=topo_view.astype('float32'),
                                    y=y_final.astype('float32'),
                                    #load_all=load_all,
                                    #cache_size=cache_size,
                                    axes=('b',0,1,'c'))

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
