import os, time
from pylearn2.config import yaml_parse
from pylearn2.utils import serial
from yaml_builder import build_yaml
import numpy as np

def get_final_val(fname, key):
    model = serial.load(fname)
    channels = model.monitor.channels
    return 1.-float(channels[key].val_record[-1])

def get_result(ins_dict, fixed_params):
    start = time.time()
    if ins_dict['n_conv_layers'] > 0:
        fixed_params['conv'] = True
        fixed_params['in_shape'] = fixed_params['shape']
        fixed_params['in_channels'] = fixed_params['channels']
    else:
        fixed_params['conv'] = False
        fixed_params['in_shape'] = np.prod(fixed_params['shape'])*fixed_params['channels']

    n_folds = fixed_params['n_folds']
    scratch = fixed_params['scratch']
    exp_name = fixed_params['exp_name']
    job_id = fixed_params['job_id']
    ins_dict = ins_dict.copy()
    fixed_params = fixed_params.copy()
    print 'Starting training...'
    valid_accuracy = np.zeros(n_folds)
    test_accuracy = np.zeros(n_folds)
    train_accuracy = np.zeros(n_folds)
    for fold in xrange(n_folds):
        ins_dict['fold'] = fold
        ins_dict['filename'] = os.path.join(scratch, exp_name, str(job_id)+'_fold'+str(fold)+'.pkl')
        train = build_yaml(ins_dict, fixed_params)
        yaml_file = os.path.join(scratch, exp_name, str(job_id)+'_fold'+str(fold)+'.yaml')
        with open(yaml_file, 'w') as f:
            f.write(train)
        print train
        train = yaml_parse.load(train)
        train.main_loop()
        del train
        train_accuracy[fold] = get_final_val(ins_dict['filename'], 'train_y_misclass')
        valid_accuracy[fold] = get_final_val(ins_dict['filename'], 'valid_y_misclass')
        test_accuracy[fold] = get_final_val(ins_dict['filename'], 'test_y_misclass')

    for fold in xrange(n_folds):
        print '--------------------------------------'
        print 'Accuracy fold '+str(fold)+':'
        print 'train: ',train_accuracy[fold]
        print 'valid: ',valid_accuracy[fold]
        print 'test: ',test_accuracy[fold]
    print '--------------------------------------'
    print 'final_train_mean_'+str(job_id)+': ',train_accuracy.mean()
    print 'final_valid_mean'+str(job_id)+': ',valid_accuracy.mean()
    print 'final_test_mean'+str(job_id)+': ',test_accuracy.mean()
    print '--------------------------------------'
    print 'final_train_std'+str(job_id)+': ',train_accuracy.std()
    print 'final_valid_std'+str(job_id)+': ',valid_accuracy.std()
    print 'final_test_std'+str(job_id)+': ',test_accuracy.std()
    print '--------------------------------------'
    print 'Total training time in seconds'
    print time.time()-start
    return valid_accuracy.mean()

