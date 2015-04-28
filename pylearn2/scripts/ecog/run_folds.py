#!/usr/bin/env python
print 'Starting up...'
import math, sys, os, time
from pylearn2.config import yaml_parse
from pylearn2.utils import serial
from yaml_builder import build_yaml
import numpy as np
import whetlab

print 'Imports done...'

script_folder = '.'
in_shape = [1, 258]
channels = 85
out_dim = 57
consonant_dim = 19
vowel_dim = 3
max_dim = 1000
n_folds = 10
exp_name = 'fc_run_new_aug'
description='FC nets on new augmented ecog.'
scratch = "exps"
test = True
rng = np.random.RandomState(20150427)

fixed_parameters = {'center': True,
                    'level_classes': True,
                    'consonant_prediction': False,
                    'vowel_prediction': False,
                    'init_type': 'istdev',
                    'train_set': 'augment',
                    'data_file': 'EC2_CV_85_nobaseline_aug.h5'}

if fixed_parameters['consonant_prediction']:
    out_dim = consonant_dim
elif fixed_parameters['vowel_prediction']:
    out_dim = vowel_dim
fixed_parameters['out_dim'] = out_dim

if test:
    min_dim = 2
    max_dim = out_dim
else:
    min_dim = out_dim
fixed_parameters['min_dim'] = min_dim

parameters = {'n_fc_layers': {'min': 1, 'max': 1, 'type': 'int'},
	      'fc_dim0': {'min': out_dim, 'max': max_dim, 'type': 'int'},
	      'fc_dim1': {'min': out_dim, 'max': max_dim, 'type': 'int'},
	      'fc_dim2': {'min': out_dim, 'max': max_dim, 'type': 'int'},
	      'fc_dim3': {'min': out_dim, 'max': max_dim, 'type': 'int'},
              'n_conv_layers': {'min': 1, 'max': 1, 'type': 'int'},
              'channels_0': {'min': 8, 'max':64, 'type': 'int'},
              'channels_grow': {'min': 1., 'max':8., 'type': 'float'},
              'conv_0_shp': {'min': 3, 'max':50, 'type': 'int'},
              'conv_0_pshp': {'min': 3, 'max':50, 'type': 'int'},
              'conv_0_pstrd': {'min': 3, 'max':50, 'type': 'int'},
              'conv_1_shp': {'min': 3, 'max':50, 'type': 'int'},
              'conv_1_pshp': {'min': 3, 'max':50, 'type': 'int'},
              'conv_1_pstrd': {'min': 3, 'max':50, 'type': 'int'},
	      'batch_size': {'min': 15, 'max': 256, 'type': 'int'},
	      'fc_layer_type': {'options': ['RectifiedLinear', 'Tanh', 'Sigmoid'], 'type': 'enum'},
	      'cost_type': {'options': ['xent', 'h1', 'h2'], 'type': 'enum'},
	      'log_fc_irange': {'min': -5., 'max': 0., 'type': 'float'},
	      'log_conv_irange': {'min': -5., 'max': 0., 'type': 'float'},
	      'log_lr': {'min': -3., 'max': -1., 'type': 'float'},
	      'log_min_lr': {'min': -5., 'max': -1., 'type': 'float'},
	      'log_decay_eps': {'min': -5., 'max': -1., 'type': 'float'},
	      'max_epochs': {'min': 10, 'max': 100, 'type': 'int'},
	      'mom_sat': {'min': 1, 'max': 50, 'type': 'int'},
	      'log_final_mom_eps': {'min': -2., 'max': -.30102, 'type': 'float'},
	      'input_dropout': {'min': .3, 'max': 1., 'type': 'float'},
	      'input_scale': {'min': 1., 'max': 3., 'type': 'float'},
	      'default_input_include_prob': {'min': .3, 'max': 1., 'type': 'float'},
	      'default_input_scale': {'min': 1., 'max': 3., 'type': 'float'},
	      'log_weight_decay': {'min': -7., 'max': 0., 'type': 'float'},
	      'max_col_norm': {'min': 0., 'max': 3., 'type': 'float'}}

def random_params(rng, kwargs):
    params = {}
    for key, value in kwargs.iteritems():
        if value['type'] == 'float':
            start = value['max']
            width = value['max']-start
            params[key] = width*rng.rand()+start
        elif value['type'] == 'int':
            low = value['min']
            high = value['max']
            params[key] = rng.randint(low=low, high=high+1)
        elif value['type'] == 'enum':
            n = len(value['options'])
            idx = rng.randint(n)
            params[key] = value['options'][idx]
        else:
            raise ValueError("Bad type '"+str(value['type'])
                             +"' for parameter "+str(key)+'.')
    return params



outcome = {'name': 'accuracy'}

with open('access_token.txt', 'r') as f:
    access_token = f.read().splitlines()[0]

start = time.time()

if not test:
    scientist = whetlab.Experiment(name=exp_name,
                                   description=description,
                                   parameters=parameters,
                                   outcome=outcome,
                                   access_token=access_token)
print 'Scientist created...'

def get_final_val(fname, key):
    model = serial.load(fname)
    channels = model.monitor.channels
    return 1.-float(channels[key].val_record[-1])

if test:
    job = random_params(rng, parameters)
    job_id = 0
else:
    job = scientist.suggest()
    job_id = scientist.get_id(job)

if job['n_conv_layers'] > 0:
    fixed_parameters['conv'] = True
    fixed_parameters['in_shape'] = in_shape
    fixed_parameters['in_channels'] = channels
else:
    fixed_parameters['conv'] = False
    fixed_parameters['in_dim'] = np.prod(in_shape)*channels

valid_accuracy = np.zeros(n_folds)
test_accuracy = np.zeros(n_folds)
train_accuracy = np.zeros(n_folds)
ins_dict = job.copy()

target_folder = os.path.join(scratch,exp_name)
if not (os.path.exists(target_folder) or test):
    os.mkdir(target_folder)

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

if not test:
    scientist.update(job, valid_accuracy.mean())
print 'Total time in seconds'
print time.time()-start
