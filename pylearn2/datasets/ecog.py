"""
ECoG dataset.
"""
__authors__ = "Jesse Livezey"

import numpy as N
np = N
import h5py
from theano.compat.six.moves import xrange
from pylearn2.datasets import dense_design_matrix
from pylearn2.datasets import control
from pylearn2.datasets import cache
from pylearn2.utils import serial
from pylearn2.utils.rng import make_np_rng


_split = {'train': .8, 'valid': .1, 'test': .1, 'move': .1}
assert np.allclose(_split['train']+_split['valid']+_split['test'],1.)

class ECoG(dense_design_matrix.DenseDesignMatrix):
    """
    ECoG dataset

    Parameters
    ----------
    filename : str
        Filename for data.
    which_set : str
        'train' or 'valid'
    fold : int
        Which fold to use.
    center : bool
        If True, preprocess so that data has zero mean.
    """

    def __init__(self, filename, which_set,
                 fold=0, seed=20141210, center=False, normalize=False):
        self.args = locals()

        if which_set not in ['train', 'valid', 'test']:
            raise ValueError(
                'Unrecognized which_set value "%s".' % (which_set,) +
                '". Valid values are ["train","valid","test"].')
        filename = serial.preprocess(filename)
        with h5py.File(filename,'r') as f:
            X = f['X'].value
            y = f['y'].value
            
        rng = np.random.RandomState(seed)
        n_examples = X.shape[0]
        order = rng.permutation(n_examples)

        n_train = int(n_examples*_split['train'])
        n_valid = int(n_examples*_split['valid'])
        n_test = n_examples-n_train-n_valid

        train_start = fold*n_examples*_split['move']
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
        self.train_std = X_train.std(0)

        if which_set == 'train':
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
        if normalize:
            topo_view = topo_view/self.train_std[np.newaxis,...]

        super(ECoG, self).__init__(topo_view=topo_view.astype('float32'),
                                    y=y_final.astype('float32'),
                                    axes=('b',0,1,'c'))

        assert not N.any(N.isnan(self.X))
        assert not N.any(N.isnan(self.y))

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
