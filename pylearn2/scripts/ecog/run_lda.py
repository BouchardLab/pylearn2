#!/usr/bin/env python
import os, cPickle
from run_folds import get_result
from hyp_params import get_params, make_dir
import numpy as np

print 'Imports done...'

seed = 20150427
rng = np.random.RandomState(seed)

json_file = 'config.json'
for subject in ['EC2', 'EC9', 'GP31', 'GP33']:
    opt_params, fixed_params = get_params(json_file, subject, 'high gamma')
    fixed_params['exp_name'] = subject + '_lda'

    make_dir(fixed_params)

    for job_id in range(200):
        print(subject, job_id)
        fixed_params['job_id'] = job_id
        solver = rng.choice(['svd', 'lsqr', 'eigen'])
        lda_job = {}
        lda_job['solver'] = solver
        lda_job['shrinkage'] = None
        if solver != 'svd':
            shrinkage = rng.choice([None, 'auto', 'float'])
            if shrinkage == 'auto':
                lda_job['shrinkage'] = 'auto'
            else:
                fl = np.power(10, rng.uniform(-2, 0))
                lda_job['shrinkage'] = fl

        get_result(lda_job, fixed_params, lda=True)
        job_id += 1
