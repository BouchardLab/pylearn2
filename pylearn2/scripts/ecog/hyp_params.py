import decimal, json, os, yaml

def get_params(json_file):

    fixed_params = {'train_set': 'train',
                    'frac_train': .5,
                    'pm_aug_range': 10,
                    'shape': [1, 258],
                    'channels': 85,
                    'consonant_dim': 19,
                    'vowel_dim': 3,
                    'n_folds': 10,
                    'level_classes': True,
                    'randomize_labels': False,
                    'consonant_prediction': False,
                    'vowel_prediction': False,
                    'center': True,
                    'test': False,
                    'factorize': False,
                    'data_file': 'EC2_CV_85_nobaseline_aug.h5',
                    'init_type': 'istdev',
                    'script_folder': '.',
                    'exp_name': 'conv_point5',
                    'description':'conv_point5',
                    'scratch': 'exps'}

    out_dim = 57
    if fixed_params['consonant_prediction']:
        out_dim = consonant_dim
    elif fixed_params['vowel_prediction']:
        out_dim = vowel_dim
    fixed_params['out_dim'] = out_dim

    if fixed_params['test']:
        min_dim = 2
    else:
        min_dim = out_dim
    fixed_params['min_dim'] = min_dim

    with open(json_file, 'r') as f:
        exp = yaml.safe_load(f)
    opt_params = exp['variables']
    fixed_params['exp_name'] = exp['experiment-name']
    fixed_params['description'] = exp['experiment-name']

    return opt_params, fixed_params

def make_dir(fixed_params):
    scratch = fixed_params['scratch']
    exp_name = fixed_params['exp_name']
    target_folder = os.path.join(scratch,exp_name)
    if not (os.path.exists(target_folder) or fixed_params['test']):
        os.mkdir(target_folder)
    return

