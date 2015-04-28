import os
from run_folds import get_result
import numpy as np

def main(job_id, params):
    in_shape = [1, 258]
    channels = 85
    out_dim = 57
    consonant_dim = 19
    vowel_dim = 3
    n_folds = 10
    exp_name = 'fc_run_new_aug'
    scratch = "exps"

    fixed_params = {'center': True,
                        'level_classes': True,
                        'consonant_prediction': False,
                        'vowel_prediction': False,
                        'init_type': 'istdev',
                        'train_set': 'augment',
                        'data_file': 'EC2_CV_85_nobaseline_aug.h5'}

    if fixed_params['consonant_prediction']:
        out_dim = consonant_dim
    elif fixed_params['vowel_prediction']:
        out_dim = vowel_dim
    fixed_params['out_dim'] = out_dim
    fixed_params['min_dim'] = min_dim

    target_folder = os.path.join(scratch,exp_name)
    if not (os.path.exists(target_folder) or test):
        os.mkdir(target_folder)

    train_params = {'n_folds': n_folds,
                    'scratch': scratch,
                    'exp_name': exp_name,
                    'job_id': job_id,
                    'in_shape': in_shape,
                    'channels': channels}

    return get_result(train_params, job, fixed_params)
