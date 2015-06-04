#!/usr/bin/env python
import os
from hyp_params import get_params
import numpy as np
import whetlab

print 'Imports done...'

json_file = 'config.json'
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
    raise NotImplementedError
else:
    job = scientist.best()
    job_id = scientist.get_id(job)
print job_id
