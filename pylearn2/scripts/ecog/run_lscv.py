#!/usr/bin/env python
import os, cPickle
from run_folds import get_result
from hyp_params import get_params, make_dir
import numpy as np

print 'Imports done...'
seed = 20150427
rng = np.random.RandomState(seed)

json_file = 'config.json'
#for subject in ['EC2', 'EC9', 'GP31', 'GP33']:
for subject in ['GP31']:
    opt_params, fixed_params = get_params(json_file, subject, 'high gamma')
    fixed_params['exp_name'] = subject + '_lsvc'

    make_dir(fixed_params)

    for job_id in range(200):
        print(subject, job_id)
        fixed_params['job_id'] = job_id
        penalty = rng.choice(['l1', 'l2'])
        loss = rng.choice(['hinge', 'squared_hinge'])
        dual = True
        if penalty == 'l1':
            loss = 'squared_hinge'
            dual = False
        C = np.power(10, rng.uniform(-2, 1))
        lsvc_job = {}
        lsvc_job['dual'] = dual
        lsvc_job['penalty'] = penalty
        lsvc_job['loss'] = loss
        lsvc_job['C'] = C

        get_result(lsvc_job, fixed_params, lsvc=True)
        job_id += 1
