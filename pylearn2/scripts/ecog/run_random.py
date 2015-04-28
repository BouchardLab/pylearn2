#!/usr/bin/env python
import os
from run_folds import get_result
import numpy as np

def main():
    print 'Imports done...'

    script_folder = '.'
    in_shape = [1, 258]
    channels = 85
    out_dim = 57
    consonant_dim = 19
    vowel_dim = 3
    max_dim = 1000
    n_folds = 10
    exp_name = 'test'
    description='test'
    scratch = "exps"
    test = True
    seed = 20150427
    rng = np.random.RandomState(seed)

    fixed_params = {'center': True,
                        'level_classes': True,
                        'consonant_prediction': False,
                        'vowel_prediction': False,
                        'init_type': 'istdev',
                        'train_set': 'train',
                        'data_file': 'EC2_CV_85_nobaseline_aug.h5'}

    if fixed_params['consonant_prediction']:
        out_dim = consonant_dim
    elif fixed_params['vowel_prediction']:
        out_dim = vowel_dim
    fixed_params['out_dim'] = out_dim

    if test:
        min_dim = 2
        max_dim = out_dim
    else:
        min_dim = out_dim
    fixed_params['min_dim'] = min_dim

    params = {'n_fc_layers': {'min': 1, 'max': 1, 'type': 'int'},
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
                  'max_col_norm': {'min': 0., 'max': 3., 'type': 'float'},
                  'max_kernel_norm': {'min': 0., 'max': 3., 'type': 'float'}}


    job = random_params(rng, params)
    job_id = seed

    target_folder = os.path.join(scratch,exp_name)
    if not (os.path.exists(target_folder) or test):
        os.mkdir(target_folder)

    train_params = {'n_folds': n_folds,
                    'scratch': scratch,
                    'exp_name': exp_name,
                    'job_id': job_id,
                    'in_shape': in_shape,
                    'channels': channels}

    get_result(train_params, job, fixed_params)

def random_params(rng, kwargs):
    params = {}
    for key, value in kwargs.iteritems():
        if value['type'] == 'float':
            start = value['min']
            width = value['max']-start
            params[key] = float(width)*rng.rand()+start
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

if __name__ == "__main__":
    main()
