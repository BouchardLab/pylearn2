import os, yaml

from pylearn2.datasets import ecog_neuro

def get_params(json_file, subject=None, bands=None,
               frac_train=None,
               scratch=None, randomize_labels=None):

    fixed_params = {'train_set': 'train',
                    'subject': 'EC2',
                    'bands': 'high gamma',
                    'frac_train': 1.,
                    'pm_aug_range': 10,
                    'consonant_dim': 19,
                    'vowel_dim': 3,
                    'n_folds': 10,
                    'level_classes': True,
                    'randomize_labels': False,
                    'consonant_prediction': False,
                    'vowel_prediction': False,
                    'two_headed': False,
                    'audio_features': False,
                    'center': True,
                    'test': False,
                    'factorize': False,
                    'init_type': 'istdev',
                    'script_folder': '.',
                    'scratch': os.path.join(os.environ['SAVE'], 'exps')}

    if subject is not None:
        fixed_params['subject'] = subject
    if bands is not None:
        fixed_params['bands'] = bands
    if frac_train is not None:
        fixed_params['frac_train'] = float(frac_train)
    if scratch is not None:
        fixed_params['scratch'] = scratch
    if randomize_labels is not None:
        fixed_params['randomize_labels'] = randomize_labels

    if fixed_params['audio_features']:
        fixed_params['data_file'] = fixed_params['audio_file']
    
    ds = ecog_neuro.ECoG(fixed_params['subject'],
                         fixed_params['bands'],
                         'train')
    X_shape = ds.get_topological_view().shape
    n_cvs = len(set(ds.y.ravel()))

    out_dim = n_cvs
    if fixed_params['consonant_prediction']:
        out_dim = fixed_params['consonant_dim']
    elif fixed_params['vowel_prediction']:
        out_dim = fixed_params['vowel_dim']
    elif fixed_params['two_headed']:
        out_dim = fixed_params['consonant_dim']+fixed_params['vowel_dim']
    fixed_params['out_dim'] = out_dim

    if fixed_params['test']:
        min_dim = 2
    else:
        min_dim = out_dim
    fixed_params['min_dim'] = min_dim

    fixed_params['shape'] = list(X_shape[1:3])
    fixed_params['channels'] = X_shape[-1]

    with open(json_file, 'r') as f:
        exp = yaml.safe_load(f)
    opt_params = exp['variables']
    fixed_params['exp_name'] = os.environ.get('SLURM_JOB_NAME', exp['experiment-name'])
    fixed_params['description'] = os.environ.get('SLURM_JOB_NAME', exp['experiment-name'])

    return opt_params, fixed_params

def make_dir(fixed_params):
    scratch = fixed_params['scratch']
    exp_name = fixed_params['exp_name']
    target_folder = os.path.join(scratch,exp_name)
    if not (os.path.exists(target_folder) or fixed_params['test']):
        os.mkdir(target_folder)
    return

