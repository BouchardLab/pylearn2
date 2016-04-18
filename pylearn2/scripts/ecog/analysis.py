from pylearn2.utils import serial
from pylearn2.config import yaml_parse
from pylearn2.datasets import ecog, ecog_new
from pylearn2.space import VectorSpace, Conv2DSpace, CompositeSpace
from pylearn2.expr import nnet
from pylearn2.models.mlp import FlattenerLayer
import os, h5py, theano, cPickle, copy, itertools
import numpy as np
import matplotlib.pyplot as plt
import theano.tensor as T
from sklearn.linear_model import LogisticRegression as LR


def load_raw_data(ds):
    ts = ds.get_test_set()
    vs = ds.get_valid_set()
    X = np.concatenate((ds.X, ts.X, vs.X), axis=0)
    y = np.concatenate((ds.y, ts.y, vs.y), axis=0).ravel()
    return X, y

def condensed_2_dense(new, indices_dicts, y_hat_dicts, logits_dicts, ds):
    if new:
        y_dims = [57]
        indices_dicts2 = []
        y_hat_dicts2 = []
        logits_dicts2 = []
        for ii, (ind, yhd, lgd) in enumerate(zip(indices_dicts, y_hat_dicts, logits_dicts)):
            ind2 = {}
            yhd2 = {}
            lgd2 = {}
            for key in ind.keys():
                indices2 = np.zeros_like(ind[key])
                y_hat2 = np.zeros((yhd[key][0].shape[0], y_dims[0]))
                logits2 = -np.inf * np.ones_like(y_hat2) 
                for old, new in enumerate(ds.mapping):
                    if not np.isinf(new):
                        indices2[ind[key] == new] = old
                        y_hat2[:, old] = yhd[key][0][:, int(new)]
                        logits2[:, old] = lgd[key][0][:, int(new)]
                ind2[key] = indices2
                yhd2[key] = [y_hat2]
                lgd2[key] = logits2
            indices_dicts2.append(ind2)
            y_hat_dicts2.append(yhd2)
            logits_dicts2.append(lgd2)
    else:
        indices_dicts2 = indices_dicts
        y_hat_dicts2 = y_hat_dicts
        logits_dicts2 = logits_dicts
    return (indices_dicts2, y_hat_dicts2, logits_dicts2)

def time_accuracy(file_name, ec, kwargs, folds=10):
    """
    Classify data independently at each point in time.
    """
    ds = ecog.ECoG(file_name, which_set='train', **kwargs)
    X_shape = ds.get_topological_view().shape
    n_time = X_shape[2]
    accuracy = np.zeros((10, n_time))
    for fold in range(folds):
        print('fold: {}'.format(fold))
        ds = ec.ECoG(file_name,
                     which_set='train',
                     fold=fold,
                     **kwargs)
        ts = ds.get_test_set()
        vs = ds.get_valid_set()
        train_X = np.concatenate((ds.get_topological_view(), vs.get_topological_view()), axis=0)
        train_y = np.concatenate((ds.y, vs.y), axis=0)
        test_X = ts.get_topological_view()
        test_y = ts.y
        for tt in range(n_time):
            svm = LR(solver='lbfgs', multi_class='multinomial').fit(train_X[:, 0, tt], train_y.ravel())
            accuracy[fold, tt] = svm.score(test_X[:, 0, tt], test_y.ravel())
    return accuracy

def conf_mat2accuracy(c_mat, v_mat, cv_mat):
    c_accuracy = None
    v_accuracy = None
    cv_accuracy = None
    accuracy_per_cv = None

    if cv_mat is not None:
        cv_accuracy = np.zeros(len(cv_mat))
        accuracy_per_cv = np.zeros((len(cv_mat), 57))
        for ii, cvf in enumerate(cv_mat):
            cv_accuracy[ii] = np.diag(cvf).sum()/cvf.sum()
            for jj in range(57):
                accuracy_per_cv[ii,jj] = cvf[jj,jj]/cvf[jj].sum()
    if c_mat is not None:
        c_accuracy = np.zeros(len(c_mat))
        for ii, cf in enumerate(c_mat):
            c_accuracy[ii] = np.diag(cf).sum()/cf.sum()
    if v_mat is not None:
        v_accuracy = np.zeros(len(v_mat))
        for ii, vf in enumerate(v_mat):
            v_accuracy[ii] = np.diag(vf).sum()/vf.sum()

    return c_accuracy, v_accuracy, cv_accuracy, accuracy_per_cv

def indx_dict2conf_mat(indices_dicts, y_dims):
    c = None
    c_dim = 19
    v = None
    v_dim = 3
    cv = None
    cv_dim = c_dim*v_dim
    n_files = len(indices_dicts)
    n_targets = None
    n_folds = None

    def c_v_from_cv(cv, v_dim):
        return int(cv/v_dim), cv % v_dim 
    def cv_from_c_v(c, v, v_dim):
        return c*v_dim+v

    for idx_dict in indices_dicts:
        if n_folds is None:
            n_folds = len(idx_dict.keys())
        else:
            assert n_folds == len(idx_dict.keys())
        for key in idx_dict.keys():
            nt = len(idx_dict[key])
            assert not (nt != 1 and n_files != 1)
            if n_targets is None:
                n_targets = nt
            else:
                assert nt == n_targets
    if cv_dim in y_dims:
        cv = np.zeros((n_folds, cv_dim, cv_dim))
        c = np.zeros((n_folds, c_dim, c_dim))
        v = np.zeros((n_folds, v_dim, v_dim))
        idx_dict = indices_dicts[0]
        for key in idx_dict.keys():
            fold = int(key.split('fold')[1].split('.')[0])
            assert fold < n_folds
            indices = idx_dict[key][0]
            y_true = indices[:,0]
            y_pred = indices[:,1]
            for yt, yp in zip(y_true, y_pred):
                ct, vt = c_v_from_cv(yt, v_dim)
                cp, vp = c_v_from_cv(yp, v_dim)
                cv[fold, yt, yp] += 1
                c[fold, ct, cp] += 1
                v[fold, vt, vp] += 1
    elif (c_dim in y_dims) and (v_dim in y_dims) and n_files == 1:
        cv = np.zeros((n_folds, cv_dim, cv_dim))
        c = np.zeros((n_folds, c_dim, c_dim))
        v = np.zeros((n_folds, v_dim, v_dim))
        idx_dict = indices_dicts[0]
        for key in idx_dict.keys():
            fold = int(key.split('fold')[1].split('.')[0])
            assert fold < n_folds
            ci, vi = idx_dict[key]
            cti = ci[:,0]
            cpi = ci[:,1]
            vti = vi[:,0]
            vpi = vi[:,1]
            for ct, cp, vt, vp in zip(cti, cpi, vti, vpi):
                cvt = cv_from_c_v(ct, vt, v_dim)
                cvp = cv_from_c_v(cp, vp, v_dim)
                cv[fold, cvt, cvp] += 1
                c[fold, ct, cp] += 1
                v[fold, vt, vp] += 1
    elif (c_dim in y_dims) and (v_dim in y_dims) and n_files != 1:
        def get_key(keys, string):
            rval = [key for key in keys if string in key]
            assert len(rval) == 1
            return rval[0]

        cm, vm = indices_dicts
        for ii in xrange(n_folds):
            fold_str = 'fold'+str(ii)
            ck = get_key(cm.keys(), fold_str)
            vk = get_key(vm.keys(), fold_str)
            ci = cm[ck][0]
            vi = vm[vk][0]
        raise NotImplementedError

        cv = np.zeros((n_folds, cv_dim, cv_dim))
        c = np.zeros((n_folds, c_dim, c_dim))
        v = np.zeros((n_folds, v_dim, v_dim))
    elif c_dim in y_dims:
        c = np.zeros((n_folds, c_dim, c_dim))
        raise NotImplementedError
    elif v_dim in y_dims:
        v = np.zeros((n_folds, v_dim, v_dim))
        raise NotImplementedError
    else:
        raise ValueError('Data does not match dimensionality expectations')

    return c, v, cv


def get_model_results(model_folder, filename, fold, kwargs, data_file, new):
    kwargs = copy.deepcopy(kwargs)
    file_loc = os.path.join(model_folder, filename)
    model = serial.load(file_loc)
    X_sym = model.get_input_space().make_theano_batch()
    target_space = model.get_target_space()
    y_inpt = target_space.make_theano_batch()
    y_sym = y_inpt
    input_space = model.get_input_space()
    if new:
        ec = ecog_new
    else:
        ec = ecog
        try:
            del kwargs['condense']
        except:
            pass
        try:
            del kwargs['min_cvs']
        except:
            pass

    ds = ec.ECoG(data_file,
                 which_set='train',
                 fold=fold,
                 **kwargs)
    ts = ds.get_test_set()
    acts = model.fprop(X_sym, return_all=True)
    y_hat = acts[-1]
    hidden = list(acts[:-1])
    n_hidden = len(hidden)
    if isinstance(model.layers[-1], FlattenerLayer):
        comp_space = model.layers[-1].raw_layer.get_output_space()
        y_hat_list = list(comp_space.undo_format_as(y_hat, target_space))
        y_sym_list = list(target_space.format_as(y_inpt, comp_space))
        n_targets = len(y_hat_list)
    else:
        n_targets = 1
        y_hat_list = [y_hat]
        y_sym_list = [y_sym]
    misclass_sym = []
    indices_sym = []
    logits_sym = []
    for yh, ys in zip(y_hat_list, y_sym_list):
        misclass_sym.append(nnet.Misclass(ys, yh))
        indices_sym.append(T.join(1, T.argmax(ys, axis=1, keepdims=True), T.argmax(yh, axis=1, keepdims=True)))
        logits_sym.append(nnet.arg_of_softmax(yh))

    f = theano.function([X_sym, y_inpt], misclass_sym+indices_sym+y_hat_list+logits_sym+hidden)
    it = ts.iterator(mode = 'sequential',
                     batch_size = ts.X.shape[0],
                     num_batches = 1,
                     data_specs = (CompositeSpace((model.get_input_space(),
                                                 model.get_target_space())),
                                   (model.get_input_source(), model.get_target_source())))
    X, y = it.next()
    rvals = f(X, y)
    misclass = list(rvals[:n_targets])
    indices = list(rvals[n_targets:2*n_targets])
    y_hats = list(rvals[2*n_targets:3*n_targets])
    logits = list(rvals[3*n_targets:4*n_targets])
    hidden = list(rvals[4*n_targets:4*n_targets+n_hidden])
    return misclass, indices, y_hats, logits, hidden

def get_articulator_state_matrix():
    consonants = sorted(['b', 'd', 'f', 'g', 'h', 'k', 'l', 'm', 'n', 'p', 'r', 's', 'sh', 't', 'th', 'v', 'w', 'y', 'z'])
    vowels = sorted(['aa', 'ee', 'oo'])
    assert len(set(consonants)) == 19
    assert len(set(vowels)) == 3
    cvs = sorted([c+v for c, v in itertools.product(consonants, vowels)])
    assert len(set(cvs)) == 57
    labels = ['lips', 'tongue', 'larynx', 'jaw', 'back tounge', 'high tongue']
    
    features = np.zeros((57, 6), dtype=int)
    # b
    features[:3, (0, 2, 3)] = 1
    # d
    features[3:6, (1, 2, 3)] = 1
    # f
    features[6:9, (0, 3)] = 1
    # g
    features[9:12, (1, 2)] = 1
    # h
    # None
    # k
    features[15:18, 1] = 1
    # l
    features[18:21, (1, 2, 3)] = 1
    # m
    features[21:24, (0, 2, 3)] = 1
    # n
    features[24:27, (1, 2, 3)] = 1
    # p
    features[27:30, (0, 3)] = 1
    # r
    features[30:33, (1, 2, 3)] = 1
    # s
    features[33:36, (1, 3)] = 1
    # sh
    features[36:39, (1, 3)] = 1
    # t
    features[39:42, (1, 3)] = 1
    # th
    features[42:45, (1, 3)] = 1
    # v
    features[45:48, (0, 2, 3)] = 1
    # w
    features[48:51, (0, 1, 2, 3)] = 1
    # y
    features[51:54, (1, 2, 3)] = 1
    # z
    features[54:, (1, 2, 3)] = 1

    # Vowels
    # aa
    features[::3, 4] = 1
    # ee
    features[1::3, 5] = 1
    # oo
    features[2::3, (4, 5)] = 1
    import matplotlib.pyplot as plt
    plt.imshow(features.T, cmap='gray', interpolation='nearest')
    return (cvs, labels, features)

def get_phonetic_feature_matrix():
    consonants = sorted(['b', 'd', 'f', 'g', 'h', 'k', 'l', 'm', 'n', 'p',
                         'r', 's', 'sh', 't', 'th', 'v', 'w', 'y', 'z'])
    vowels = sorted(['aa', 'ee', 'oo'])
    assert len(set(consonants)) == 19
    assert len(set(vowels)) == 3
    cvs = sorted([c+v for c, v in itertools.product(consonants, vowels)])
    assert len(set(cvs)) == 57
    labels = ['bilabial', 'secondary labial', 'labiodental', 'dental',
              'alveolar', 'post alveolar', 'velar', 'voiced', 'mandibular',
              'oral stop', 'fricative', 'approximate', 'nasal stop',
              'lateral', 'rhotic', 'back tongue', 'high tongue',
              'lip rounding', 'jaw open']
    assert len(set(labels)) == 19
    pmv = {'place': slice(0, 9), 'manner': slice(9, 15), 'vowel': slice(15, 19)}
    
    features = np.zeros((57, 19), dtype=int)
    # b
    features[:3, (0, 7, 8, 9)] = 1
    # d
    features[3:6, (4, 7, 8, 9)] = 1
    # f
    features[6:9, (2, 8, 10)] = 1
    # g
    features[9:12, (6, 7, 9)] = 1
    # h
    # None
    # k
    features[15:18, (6, 9)] = 1
    # l
    features[18:21, (4, 7, 8, 13)] = 1
    # m
    features[21:24, (0, 7, 8, 12)] = 1
    # n
    features[24:27, (4, 7, 8, 9, 12)] = 1
    # p
    features[27:30, (0, 8, 9)] = 1
    # r
    features[30:33, (1, 5, 7, 8, 11, 14)] = 1
    # s
    features[33:36, (4, 8, 10)] = 1
    # sh
    features[36:39, (1, 5, 8, 10)] = 1
    # t
    features[39:42, (4, 8, 9)] = 1
    # th
    features[42:45, (3, 8, 10)] = 1
    # v
    features[45:48, (2, 7, 8, 10)] = 1
    # w
    features[48:51, (0, 6, 7, 8, 11)] = 1
    # y
    features[51:54, (5, 7, 8, 11)] = 1
    # z
    features[54:, (4, 7, 8, 10)] = 1

    # Vowels
    # aa
    features[::3, (15, 18)] = 1
    # ee
    features[1::3, 16] = 1
    # oo
    features[2::3, (15, 16, 17)] = 1
    import matplotlib.pyplot as plt
    plt.imshow(features.T, cmap='gray', interpolation='nearest')
    return (cvs, labels, pmv, features)

def cross_correlate(X1, X2):
    """
    Calculates cross correlation matrix.
    
    X1: ndarray
        First set of variables (n, features)
    X2 : ndarray
        Second set of variables (m, features)
    """
    print X1.shape, X1.mean(1).shape
    print X2.shape, X2.mean(1).shape
    X1 = X1 - X1.mean(axis=1, keepdims=True)
    X1 = X1 / X1.std(axis=1, keepdims=True)
    X2 = X2 - X2.mean(axis=1, keepdims=True)
    X2 = X2 / X2.std(axis=1, keepdims=True)
    return X1.dot(X2.T)/X1.shape[1]