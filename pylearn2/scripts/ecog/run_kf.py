#!/usr/bin/env python
import os, cPickle
from run_folds import get_result
from run_random import random_params
from hyp_params import get_params, make_dir
import numpy as np

print 'Imports done...'

json_file = 'config.json'
opt_params, fixed_params = get_params(json_file)
outcome = {'name': 'accuracy'}

dims = range(1,20)

seed = 20160201
rng = np.random.RandomState(seed)

job_id = 0
fixed_params['job_id'] = job_id

make_dir(fixed_params)

results = []
for dim in dims:
    lda_job = {}
    lda_job['state_dim'] = dim
    lda_job['obs_dim'] = fixed_params['channels']

    valid_accuracy = get_result(lda_job, fixed_params, kf=True)
    results.append((valid_accuracy, lda_job))
    job_id += 1

with open('kf.pkl', 'w') as f:
    cPickle.dump(results, f)
