#!/usr/bin/env python
print 'Starting up...'
import math, sys, os, time
from pylearn2.config import yaml_parse
from pylearn2.utils import serial
import numpy as np
import whetlab

print 'Imports done...'

script_folder = '.'
in_dim = 258*85
out_dim = 57
min_dim = 2
max_dim = 1000
n_folds = 10
exp_name = 'fc_run2'
description='FC nets on ecog.'
scratch = "exps"
test = False

parameters = {'n_layers': {'min': 1, 'max': 2, 'type': 'int'},
	      'dim_0': {'min': out_dim, 'max': max_dim, 'type': 'int'},
	      'dim_shrink': {'min': 0., 'max': 1., 'type': 'float'},
	      'batch_size': {'min': 15, 'max': 128, 'type': 'int'},
	      'layer_type': {'options': ['RectifiedLinear', 'Tanh', 'Sigmoid'], 'type': 'enum'},
	      'cost_type': {'options': ['xent', 'h1', 'h2'], 'type': 'enum'},
	      'log_irange': {'min': -5., 'max': 0., 'type': 'float'},
	      'log_lr': {'min': -3., 'max': -1., 'type': 'float'},
	      'log_min_lr': {'min': -5., 'max': -1., 'type': 'float'},
	      'log_decay_eps': {'min': -5., 'max': -1., 'type': 'float'},
	      'max_epochs': {'min': 10, 'max': 100, 'type': 'int'},
	      'mom_sat': {'min': 1, 'max': 50, 'type': 'int'},
	      'final_mom': {'min': .5, 'max': 1., 'type': 'float'},
	      'input_dropout': {'min': .3, 'max': 1., 'type': 'float'},
	      'input_scale': {'min': 1., 'max': 3., 'type': 'float'},
	      'default_input_include_prob': {'min': .3, 'max': 1., 'type': 'float'},
	      'default_input_scale': {'min': 1., 'max': 3., 'type': 'float'},
	      'log_weight_decay': {'min': -7., 'max': 0., 'type': 'float'},
	      'max_col_norm': {'min': 0., 'max': 3., 'type': 'float'}}


fixed_parameters = {'center': True,
                    'init_type': 'istdev'}

test_parameters = {'n_layers': 1,
                  'dim_0':150,
                  'dim_shrink': .5,
                  'batch_size': 20,
                  'layer_type': 'Tanh',
                  'center': False,
                  'cost_type': 'xent',
                  'init_type': 'istdev',
                  'log_irange': -3.,
                  'log_lr': -3.,
                  'log_min_lr': -3.,
                  'log_decay_eps': -3,
                  'log_decay_eps': -3.,
                  'mom_sat': 20,
                  'final_mom': .9,
                  'input_dropout': .5,
                  'input_scale': 1.8,
                  'default_input_include_prob': .8,
                  'default_input_scale': 1.,
                  'log_weight_decay': -5.,
                  'max_col_norm': 2.}

cost_type_map = {}
cost_type_map['xent'] = 'mlp.dropout.Dropout'
cost_type_map['h1'] = 'hinge_loss.DropoutHingeLossL1'
cost_type_map['h2'] = 'hinge_loss.DropoutHingeLossL2'

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

with open(os.path.join(script_folder,'ecog_nersc.yaml'), 'rb') as f:
    train_string = f.read()

def get_final_val(fname, key):
    model = serial.load(fname)
    channels = model.monitor.channels
    return 1.-float(channels[key].val_record[-1])

def make_layers(in_dim, **kwargs):
    layer_string = ("!obj:pylearn2.models.mlp.%(layer_type)s {\n"
                    +"layer_name: %(name)s,\n"
                    +"dim: %(dim)i,\n"
                    +"%(init_type)s: %(range)f,\n"
                    +"max_col_norm: %(max_col_norm)f,\n"
                    +"},\n")
    out_string = ""
    dim = int(kwargs['dim_0'])
    for ii in xrange(kwargs['n_layers']):
        this_dict = kwargs.copy()
        this_dict['dim'] = max(int(math.ceil(dim)), min_dim)
        this_dict['name'] = 'h'+str(ii)
        this_dict['range'] = np.power(10., kwargs['log_irange'])
        out_string += layer_string % this_dict
        dim = dim*kwargs['dim_shrink']
    return out_string

def make_last_layer_and_cost(out_dim, **kwargs):
    layer_string = ("!obj:pylearn2.models.mlp.%(final_layer_type)s {\n"
                    +"layer_name: y,\n"
                    +"%(string)s: %(dim)i,\n"
                    +"%(init_type)s: %(range)f,\n"
                    +"max_col_norm: %(max_col_norm)f,\n"
                    +"},\n")
    cost_string = ("!obj:pylearn2.costs.cost.SumOfCosts {\n"
                   +"costs: [\n"
                   +"!obj:pylearn2.costs.%(cost_obj)s {\n"
                   +"default_input_include_prob: %(default_input_include_prob)f,\n"
                   +"default_input_scale: %(default_input_scale)f,\n"
                   +"input_include_probs: { 'h0': %(input_dropout)f },\n"
                   +"input_scales: { 'h0': %(input_scale)f },\n"
                   +"},\n"
                   +"!obj:pylearn2.costs.mlp.WeightDecay {\n"
                   +"coeffs: { 'h0': %(wd)f,\n"
                   +"'y': %(wd)f,\n")
    wd_string = "%(name)s: %(wd)f,\n"
    end_cost_string = ("},\n"
                       +"},\n"
                       +"],\n"
                       +"},\n")

    # Create final string and dict
    this_dict = kwargs.copy()
    this_dict['dim'] = out_dim
    this_dict['range'] = np.power(10., kwargs['log_irange'])
    this_dict['wd'] = np.power(10., kwargs['log_weight_decay'])
    if kwargs['cost_type'] == 'xent':
        this_dict['string'] = 'n_classes'
        this_dict['final_layer_type'] = 'Softmax'
    else:
        this_dict['string'] = 'dim'
        this_dict['final_layer_type'] = 'Linear'

    out_layer_string = layer_string % this_dict

    out_cost_string = cost_string
    for ii in xrange(1, kwargs['n_layers']):
        out_cost_string += wd_string % {'name': 'h'+str(ii),
                                        'wd': this_dict['wd']}
    out_cost_string += end_cost_string
    out_cost_string = out_cost_string % this_dict
    return out_layer_string, out_cost_string


if test:
    job = test_parameters
    job_id = 0
else:
    job = scientist.suggest()
    job_id = scientist.get_id(job)

valid_accuracy = np.zeros(n_folds)
test_accuracy = np.zeros(n_folds)
train_accuracy = np.zeros(n_folds)
ins_dict = job.copy()
ins_dict['lr'] = np.power(10., ins_dict['log_lr'])
ins_dict['cost_obj'] = cost_type_map[ins_dict['cost_type']]
ls = make_layers(in_dim, **ins_dict)
lsf, cs = make_last_layer_and_cost(out_dim, **ins_dict)
ins_dict['layer_string'] = ls+lsf
ins_dict['cost_string'] = cs
ins_dict['decay_factor'] = 1.+np.power(10., ins_dict['log_decay_eps'])
ins_dict['min_lr'] = np.power(10., ins_dict['log_min_lr'])
ins_dict.update(fixed_parameters)

target_folder = os.path.join(scratch,exp_name)
if not os.path.exists(target_folder):
    os.mkdir(target_folder)

print 'Starting training...'

for fold in xrange(n_folds):
    ins_dict['fold'] = fold
    ins_dict['filename'] = os.path.join(scratch, exp_name, str(job_id)+'_fold'+str(fold)+'.pkl')
    train = train_string % ins_dict
    print train
    train = yaml_parse.load(train)
    train.main_loop()
    valid_accuracy[fold] = get_final_val(ins_dict['filename'], 'valid_y_misclass')
    test_accuracy[fold] = get_final_val(ins_dict['filename'], 'test_y_misclass')
    train_accuracy[fold] = get_final_val(ins_dict['filename'], 'train_y_misclass')
for fold in xrange(n_folds):
    print '--------------------------------------'
    print 'Accuracy fold '+str(fold)+':'
    print 'train: ',train_accuracy[fold]
    print 'valid: ',valid_accuracy[fold]
    print 'test: ',test_accuracy[fold]

if not test:
    scientist.update(job, valid_accuracy.mean())
print 'Total time in seconds'
print time.time()-start
