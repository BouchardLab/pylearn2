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


_split = {'train': .8, 'valid': .1, 'test': .1, 'move': .2}
assert np.allclose(_split['train']+__split['valid']+split['test'],1.)
assert np.allclose(_split['valid']+_split['test'], _split['move'])

class ECoG(dense_design_matrix.DenseDesignMatrix):
    """
    ECoG dataset

    Parameters
    ----------
    filename : str
        Filename for data.
    which_set : str
        'train' or 'valid'
    frac_train : float
        Fraction of data for training. Remaining data is for validation.
    center : bool
        If True, preprocess so that data has zero mean.
    """

    def __init__(self, filename, which_set, fold=0, seed=20141210, center=False):
        self.args = locals()

        if which_set not in ['train', 'valid']:
            raise ValueError(
                'Unrecognized which_set value "%s".' % (which_set,) +
                '". Valid values are ["train","valid"].')
        with h5py.File(filename,'r') as f:
            X = f['X'].value
            y = f['y'].value
        rng = make_np_rng(seed)
        n_examples = X.shape[0]
        order = rng.permutation(n_examples)
        train_start = fold*n_examples*(1.-_split['train'])

        n_train = int(n_examples*frac_train)
        X_train = X[order[:n_train]]
        X_valid = X[order[n_train:]]
        y_train = y[order[:n_train]]
        y_valid = y[order[n_train:]]
        if which_set == 'train':
            topo_view = X_train[...,np.newaxis]
            y_final = y_train
        else:
            topo_view = X_valid[...,np.newaxis]
            y_final = y_valid
        if center:
            topo_view = topo_view-X_train[...,np.newaxis].mean(0)

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
