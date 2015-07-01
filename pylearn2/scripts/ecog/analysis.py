from pylearn2.utils import serial
from pylearn2.config import yaml_parse
from pylearn2.datasets import ecog
from pylearn2.space import VectorSpace, Conv2DSpace, CompositeSpace
from pylearn2.expr import nnet
import os, h5py, theano, cPickle, copy
import numpy as np
import matplotlib.pyplot as plt
import theano.tensor as T


def get_model_results(model_folder, filename, fold, kwargs):
    kwargs = copy.deepcopy(kwargs)
    file_loc = os.path.join(model_folder, filename)
    model = serial.load(file_loc)
    X_sym = model.get_input_space().make_theano_batch()
    y_sym = model.get_target_space().make_theano_batch()
    input_space = model.get_input_space()
    if kwargs['audio']:
        data_file = '${PYLEARN2_DATA_PATH}/ecog/audio_EC2_CV_mcep.h5'
    else:
        data_file = '${PYLEARN2_DATA_PATH}/ecog/EC2_CV_85_nobaseline_aug.h5'
    del kwargs['audio']
    ds = ecog.ECoG(data_file,
                   which_set='train',
                   fold=fold,
                   **kwargs)
    ts = ds.get_test_set()
    y_hat = model.fprop(X_sym)
    misclass_sym = nnet.Misclass(y_sym, y_hat)
    indices_sym = T.join(1, T.argmax(y_sym, axis=1, keepdims=True), T.argmax(y_hat, axis=1, keepdims=True))
    f = theano.function([X_sym, y_sym], [misclass_sym, indices_sym, y_hat])
    it = ts.iterator(mode = 'sequential',
                     batch_size = ts.X.shape[0],
                     num_batches = 1,
                     data_specs = (CompositeSpace((model.get_input_space(),
                                                 model.get_target_space())),
                                   (model.get_input_source(), model.get_target_source())))
    X, y = it.next()
    return f(X, y)
