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
    """

    def __init__(self, filename, which_set,
                 fold=0, seed=20141210, center=False,
                 move = .1, load_all=None, cache_size=400000000):
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
            if which_set == 'augment':
                X_aug = f['X_aug'].value
                y_aug = f['y_aug'].value
            
        rng = np.random.RandomState(seed)
        n_examples = X.shape[0]
        order = rng.permutation(n_examples)

        n_train = int(n_examples*_split['train'])
        n_valid = int(n_examples*_split['valid'])
        n_test = n_examples-n_train-n_valid

        train_start = fold*n_examples*move
        train_end = (train_start+n_train) % n_examples
        valid_start = train_end
        valid_end = (valid_start+n_valid) % n_examples
        test_start = valid_end
        test_end = (test_start+n_test) % n_examples

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
            print 'in train'
            print which_set
            print ''
            topo_view = X_train
            y_final = y_train
        elif which_set == 'valid':
            print 'in valid'
            print which_set
            print ''
            topo_view = X_valid
            y_final = y_valid
        else:
            print 'in test'
            print which_set
            print ''
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
