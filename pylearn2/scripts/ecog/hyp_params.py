import decimal, json, os, yaml

from pylearn2.datasets import ecog, ecog_new

def get_params(json_file):

    fixed_params = {'train_set': 'train',
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
                    'data_file': 'hdf5/GP31_blocks_1_2_4_6_9_21_63_65_67_69_71_78_82_83_CV_HG_align_window_-05_to_079_events_nobaseline.h5',
                    'audio_file': 'audio_EC2_CV_mcep.h5',
                    'init_type': 'istdev',
                    'script_folder': '.',
                    'scratch': 'exps'}
    """
                    'data_file': 'hdf5/EC9_blocks_15_39_46_49_53_60_63_CV_HG_align_window_-05_to_079_events_nobaseline.h5',
                    'data_file': 'hdf5/EC2_blocks_1_8_9_15_76_89_105_CV_HG_align_window_-05_to_079_events_nobaseline.h5',
                    'data_file': 'EC2_CV_85_nobaseline_aug.h5',
                    'audio_file': 'audio_EC2_CV_mcep.h5',
    """

    if fixed_params['audio_features']:
        fixed_params['data_file'] = fixed_params['audio_file']
    
    ds = ecog_new.ECoG(os.path.join('${PYLEARN2_DATA_PATH}', 'ecog', fixed_params['data_file']), 'train')
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

