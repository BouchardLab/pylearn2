print 'Starting up...'
import os, time
from pylearn2.config import yaml_parse
from pylearn2.utils import serial
from yaml_builder import build_yaml
import numpy as np

def get_final_val(fname, key):
    model = serial.load(fname)
    channels = model.monitor.channels
    return 1.-float(channels[key].val_record[-1])

def get_result(train_params, ins_dict, fixed_parameters):
    start = time.time()
    if ins_dict['n_conv_layers'] > 0:
        fixed_parameters['conv'] = True
        fixed_parameters['in_shape'] = in_shape
        fixed_parameters['in_channels'] = channels
    else:
        fixed_parameters['conv'] = False
        fixed_parameters['in_shape'] = np.prod(in_shape)*channels

    valid_accuracy = np.zeros(n_folds)
    test_accuracy = np.zeros(n_folds)
    train_accuracy = np.zeros(n_folds)
    n_folds = train_params['n_folds']
    scratch = train_params['scratch']
    exp_name = train_params['exp_name']
    job_id = train_params['job_id']
    ins_dict = ins_dict.copy()
    fixed_parameters = fixed_parameters.copy()
    print 'Starting training...'
    for fold in xrange(n_folds):
        ins_dict['fold'] = fold
        ins_dict['filename'] = os.path.join(scratch, exp_name, str(job_id)+'_fold'+str(fold)+'.pkl')
        train = build_yaml(ins_dict, fixed_parameters)
        print train
        train = yaml_parse.load(train)
        train.main_loop()
        del train
        valid_accuracy[fold] = get_final_val(ins_dict['filename'], 'valid_y_misclass')
        test_accuracy[fold] = get_final_val(ins_dict['filename'], 'test_y_misclass')
        train_accuracy[fold] = get_final_val(ins_dict['filename'], 'train_y_misclass')
    for fold in xrange(n_folds):
        print '--------------------------------------'
        print 'Accuracy fold '+str(fold)+':'
        print 'train: ',train_accuracy[fold]
        print 'valid: ',valid_accuracy[fold]
        print 'test: ',test_accuracy[fold]
    print '--------------------------------------'
    print 'final_train_mean: ',train_accuracy.mean()
    print 'final_valid_mean: ',valid_accuracy.mean()
    print 'final_test_mean: ',test_accuracy.mean()
    print '--------------------------------------'
    print 'final_train_std: ',train_accuracy.std()
    print 'final_valid_std: ',valid_accuracy.std()
    print 'final_test_std: ',test_accuracy.std()
    print '--------------------------------------'
    print 'Total time in seconds'
    print time.time()-start
    return valid_accuracy.mean()

