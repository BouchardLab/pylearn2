#!/usr/bin/env python
import os, sys
from run_folds import get_result
from hyp_params import get_params, make_dir
import numpy as np

def main(seed, json_file, subject, scratch, bands, data_types):
    print 'Imports done...'
    opt_params, fixed_params = get_params(json_file)
    fixed_params['subject'] = subject
    fixed_params['scratch'] = scratch
    fixed_params['bands'] = bands
    fixed_params['data_types'] = data_types

    rng = np.random.RandomState(int(seed))

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
    main(*sys.argv[1:])
