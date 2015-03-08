#!/usr/bin/env python
print 'Starting up...'
import math, sys, os, time
from pylearn2.config import yaml_parse
from pylearn2.utils import serial
import numpy as np
import whetlab

print 'Imports done...'

script_folder = '.'
in_shape = [1, 258]
in_channels = 85
out_dim = 57
max_dim = 1000
n_folds = 10
exp_name = 'first_conv_run'
description='First run of Conv nets on ecog.'
scratch = "exps"
test = False
if test:
    min_dim = 2
else:
    min_dim = out_dim

parameters = {'n_conv_layers': {'min': 1, 'max': 4, 'type': 'int'},
              'n_fc_layers': {'min': 0, 'max': 4, 'type': 'int'},
              'channels_0': {'min': 8, 'max':64, 'type': 'int'},
              'channels_grow': {'min': 1., 'max':8., 'type': 'float'},
              'conv_0_shp': {'min': 3, 'max':50, 'type': 'int'},
              'conv_0_pshp': {'min': 3, 'max':50, 'type': 'int'},
              'conv_0_pstrd': {'min': 3, 'max':50, 'type': 'int'},
              'conv_1_shp': {'min': 3, 'max':50, 'type': 'int'},
              'conv_1_pshp': {'min': 3, 'max':50, 'type': 'int'},
              'conv_1_pstrd': {'min': 3, 'max':50, 'type': 'int'},
              'conv_2_shp': {'min': 3, 'max':50, 'type': 'int'},
              'conv_2_pshp': {'min': 3, 'max':50, 'type': 'int'},
              'conv_2_pstrd': {'min': 3, 'max':50, 'type': 'int'},
              'conv_3_shp': {'min': 3, 'max':50, 'type': 'int'},
              'conv_3_pshp': {'min': 3, 'max':50, 'type': 'int'},
              'conv_3_pstrd': {'min': 3, 'max':50, 'type': 'int'},
	      'fc_dim_0': {'min': out_dim, 'max': max_dim, 'type': 'int'},
	      'dim_shrink': {'min': 0., 'max': 1., 'type': 'float'},
	      'batch_size': {'min': 15, 'max': 128, 'type': 'int'},
	      'max_epochs': {'min': 10, 'max': 100, 'type': 'int'},
	      'cost_type': {'options': ['xent', 'h1', 'h2'], 'type': 'enum'},
	      'fc_layer_type': {'options': ['Linear', 'Tanh', 'Sigmoid'], 'type': 'enum'},
	      'log_conv_irange': {'min': -5., 'max': 0., 'type': 'float'},
	      'log_fc_irange': {'min': -5., 'max': 0., 'type': 'float'},
	      'log_lr': {'min': -5., 'max': -1., 'type': 'float'},
	      'log_min_lr': {'min': -5., 'max': -3., 'type': 'float'},
	      'log_decay_eps': {'min': -6., 'max': -3., 'type': 'float'},
	      'mom_sat': {'min': 1, 'max': 50, 'type': 'int'},
	      'final_mom': {'min': .5, 'max': 1., 'type': 'float'},
	      'input_dropout': {'min': .3, 'max': 1., 'type': 'float'},
	      'input_scale': {'min': 1., 'max': 3., 'type': 'float'},
	      'default_input_include_prob': {'min': .3, 'max': 1., 'type': 'float'},
	      'default_input_scale': {'min': 1., 'max': 3., 'type': 'float'},
	      'log_weight_decay': {'min': -7., 'max': 0., 'type': 'float'},
	      'max_kernel_norm': {'min': 0., 'max': 3., 'type': 'float'},
	      'max_col_norm': {'min': 0., 'max': 3., 'type': 'float'}}


fixed_parameters = {'center': True,
                    'init_type': 'istdev',
                    'train_set': 'train'}

test_parameters = {'n_conv_layers': 1,
                   'n_fc_layers': 1,
                   'channels_0': 8,
                   'channels_grow': 1,
                  'conv_0_shp': 3,
                  'conv_0_pshp': 3,
                  'conv_0_pstrd': 3,
                  'conv_1_shp': 3,
                  'conv_1_pshp': 3,
                  'conv_1_pstrd': 3,
                  'conv_2_shp': 3,
                  'conv_2_pshp': 3,
                  'conv_2_pstrd': 3,
                  'conv_3_shp': 3,
                  'conv_3_pshp': 3,
                  'conv_3_pstrd': 3,
                  'fc_dim_0': 2,
                  'dim_shrink': .5,
                  'batch_size': 20,
                  'max_epochs': 2,
                  'fc_layer_type': 'Tanh',
                  'center': False,
                  'cost_type': 'xent',
                  'init_type': 'istdev',
                  'log_conv_irange': -3.,
                  'log_fc_irange': -3.,
                  'log_lr': -3.,
                  'log_min_lr': -5.,
                  'log_decay_eps': -5.,
                  'mom_sat': 20,
                  'final_mom': .9,
                  'input_dropout': .5,
                  'input_scale': 1.8,
                  'default_input_include_prob': .8,
                  'default_input_scale': 1.,
                  'log_weight_decay': -5.,
                  'max_kernel_norm': 2.,
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

with open(os.path.join(script_folder,'ecog_conv_nersc.yaml'), 'rb') as f:
    train_string = f.read()

def get_final_val(fname, key):
    model = serial.load(fname)
    channels = model.monitor.channels
    return 1.-float(channels[key].val_record[-1])

def make_layers(in_shape, **kwargs):
    conv_layer_string = ("!obj:pylearn2.models.mlp.ConvRectifiedLinear {\n"
                    +"layer_name: %(name)s,\n"
                    +"kernel_shape: [%(conv_shp0)i,%(conv_shp1)i],\n"
                    +"output_channels: %(channels)i,\n"
                    +"pool_shape: [%(pool_shp0)i,%(pool_shp1)i],\n"
                    +"pool_stride: [%(pool_strd0)i,%(pool_strd1)i],\n"
                    +"irange: %(range)f,\n"
                    +"max_kernel_norm: %(max_col_norm)f,\n"
                    +"},\n")
    fc_layer_string = ("!obj:pylearn2.models.mlp.%(fc_layer_type)s {\n"
                    +"layer_name: %(name)s,\n"
                    +"dim: %(dim)i,\n"
                    +"%(init_type)s: %(range)f,\n"
                    +"max_col_norm: %(max_col_norm)f,\n"
                    +"},\n")
    def get_shapes(in_shp, ker_shp, pool_shp, pool_strd):
        detector_shp = [in_s-ker_s+1 for in_s, ker_s in zip(in_shp, ker_shp)]
        print 'det'
        print detector_shp
        out_shp = [int(1+math.ceil((d_s-p_s)/float(p_st))) for d_s, p_s, p_st in zip(detector_shp,
                                                                                   pool_shp,
                                                                                   pool_strd)]
        print 'out'
        print out_shp
        out_shp = [o_s-1 if (o_s-1)*p_st >= i_s else o_s for o_s, p_st, i_s in zip(out_shp,
                                                                             pool_shp,
                                                                             in_shp)]
        return out_shp

    out_string = ""
    channels = int(kwargs['channels_0'])
    cur_shp = in_shape
    for ii in xrange(kwargs['n_conv_layers']):
        this_dict = kwargs.copy()
        k_shp = [1,this_dict['conv_'+str(ii)+'_shp']]
        p_shp = [1,this_dict['conv_'+str(ii)+'_pshp']]
        p_strd = [1,this_dict['conv_'+str(ii)+'_pstrd']]
        if k_shp[1] >= cur_shp[1]:
            k_shp[1] = cur_shp[1]
            p_shp[1] = 1
            p_strd[1] = 1
        print 'pre'
        print cur_shp
        cur_shp = get_shapes(cur_shp, k_shp, p_shp, p_strd)
        print 'post'
        print cur_shp
        print ''
        this_dict['conv_shp0'] = k_shp[0]
        this_dict['conv_shp1'] = k_shp[1]
        this_dict['channels'] = channels
        this_dict['pool_shp0'] = p_shp[0]
        this_dict['pool_shp1'] = p_shp[1]
        this_dict['pool_strd0'] = p_strd[0]
        this_dict['pool_strd1'] = p_strd[1]
        this_dict['name'] = 'c'+str(ii)
        this_dict['range'] = np.power(10., kwargs['log_conv_irange'])
        out_string += conv_layer_string % this_dict
        channels = channels*kwargs['channels_grow']

    out_dim = np.prod(cur_shp)*channels
    dim = kwargs['fc_dim_0']
    print 'dims'
    print min_dim
    print out_dim
    print dim

    for ii in xrange(kwargs['n_fc_layers']):
        this_dict = kwargs.copy()
        this_dict['dim'] = min(max(int(math.ceil(dim)), min_dim), int(kwargs['dim_shrink']*out_dim))
        this_dict['name'] = 'f'+str(ii)
        this_dict['range'] = np.power(10., kwargs['log_fc_irange'])
        dim = dim*kwargs['dim_shrink']
        out_string += fc_layer_string % this_dict
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
                   +"input_include_probs: { 'c0': %(input_dropout)f },\n"
                   +"input_scales: { 'c0': %(input_scale)f },\n"
                   +"},\n"
                   +"!obj:pylearn2.costs.mlp.WeightDecay {\n"
                   +"coeffs: { 'c0': %(wd)f,\n"
                   "'y': %(wd)f,\n")
    wd_string = "%(name)s: %(wd)f,\n"
    end_cost_string = ("},\n"
                       +"},\n"
                       +"],\n"
                       +"},\n")

    # Create final string and dict
    this_dict = kwargs.copy()
    this_dict['dim'] = out_dim
    this_dict['range'] = np.power(10., kwargs['log_fc_irange'])
    this_dict['wd'] = np.power(10., kwargs['log_weight_decay'])
    if kwargs['cost_type'] == 'xent':
        this_dict['string'] = 'n_classes'
        this_dict['final_layer_type'] = 'Softmax'
    else:
        this_dict['string'] = 'dim'
        this_dict['final_layer_type'] = 'Linear'

    out_layer_string = layer_string % this_dict

    out_cost_string = cost_string
    for ii in xrange(1, kwargs['n_conv_layers']):
        out_cost_string += wd_string % {'name': 'c'+str(ii),
                                        'wd': this_dict['wd']}
    for ii in xrange(0, kwargs['n_fc_layers']):
        out_cost_string += wd_string % {'name': 'f'+str(ii),
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
ins_dict.update(fixed_parameters)
ins_dict['lr'] = np.power(10., job['log_lr'])
ins_dict['cost_obj'] = cost_type_map[ins_dict['cost_type']]
ls = make_layers(in_shape, **ins_dict)
lsf, cs = make_last_layer_and_cost(out_dim, **ins_dict)
ins_dict['layer_string'] = ls+lsf
ins_dict['cost_string'] = cs
ins_dict['min_lr'] = np.power(10, job['log_min_lr'])
ins_dict['decay_factor'] = 1.+np.power(10, job['log_decay_eps'])

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
