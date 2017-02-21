#!/usr/bin/env python
import os
from run_folds import get_result
from hyp_params import get_params, make_dir
import numpy as np

def main(seed, json_file=None):
    print 'Imports done...'
    if json_file is None:
        json_file = 'spearmint/config.json'
    opt_params, fixed_params = get_params(json_file)

    rng = np.random.RandomState(seed)

    job = random_params(rng, opt_params)
    fixed_params['job_id'] = seed

    make_dir(fixed_params)

    get_result(job, fixed_params)

def random_params(rng, kwargs):
    params = {}
    for key, value in kwargs.iteritems():
        if value['type'] == 'float':
            start = value['min']
            width = value['max']-start
            params[key] = float(width)*rng.rand()+start
        elif value['type'] == 'int':
            low = value['min']
            high = value['max']
            params[key] = rng.randint(low=low, high=high+1)
        elif value['type'] == 'enum':
            n = len(value['options'])
            idx = rng.randint(n)
            params[key] = value['options'][idx]
        else:
            raise ValueError("Bad type '"+str(value['type'])
                             +"' for parameter "+str(key)+'.')
    return params

if __name__ == "__main__":
    main()
