import os, yaml

from pylearn2.datasets import ecog_neuro

def get_params(json_file, subject=None, bands=None,
               frac_train=None,
               scratch=None, randomize_labels=None, pca=None,
               avg_ff=None, avg_1f=None, ds=None):

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
                    'pca': False,
                    'avg_ff': False,
                    'avg_1f': False,
                    'ds': False,
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
    if pca is not None:
        fixed_params['pca'] = pca
    if avg_ff is not None:
        fixed_params['avg_ff'] = avg_ff
    if avg_1f is not None:
        fixed_params['avg_1f'] = avg_1f
    if ds is not None:
        fixed_params['ds'] = ds

    if fixed_params['audio_features']:
        fixed_params['data_file'] = fixed_params['audio_file']
    
    dset = ecog_neuro.ECoG(fixed_params['subject'],
                         fixed_params['bands'],
                         'train',
                         pca=fixed_params['pca'],
                         avg_ff=fixed_params['avg_ff'],
                         avg_1f=fixed_params['avg_1f'],
                         ds=fixed_params['ds'])
    X_shape = dset.get_topological_view().shape
    n_cvs = len(set(dset.y.ravel()))

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
    if not fixed_params['test']:
        try:
            os.mkdir(target_folder)
        except OSError:
            pass
    return
