#!/usr/bin/env python
import os
from run_folds import get_result
from run_random import random_params
from hyp_params import get_params, make_dir
import numpy as np
import whetlab

print 'Imports done...'

json_file = 'spearmint/config.json'
opt_params, fixed_params = get_params(json_file)
outcome = {'name': 'accuracy'}


seed = 20150427
rng = np.random.RandomState(seed)

with open('access_token.txt', 'r') as f:
    access_token = f.read().splitlines()[0]

if not fixed_params['test']:
    scientist = whetlab.Experiment(name=fixed_params['exp_name'],
                                   description=fixed_params['description'],
                                   parameters=opt_params,
                                   outcome=outcome,
                                   access_token=access_token)
print 'Scientist created...'

if fixed_params['test']:
    job = random_params(rng, params)
    job_id = seed
else:
    job = scientist.suggest()
    job_id = scientist.get_id(job)
fixed_params['job_id'] = job_id

make_dir(fixed_params)

valid_accuracy = get_result(job, fixed_params)

if not fixed_params['test']:
    scientist.update(job, valid_accuracy)
