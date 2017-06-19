#!/usr/bin/env python
import glob, sys, os
import numpy as np

folder = sys.argv[1]
if len(sys.argv[1:]) > 1:
    keep_n = int(sys.argv[2])
else:
    keep_n = 'all'

files = glob.glob(os.path.join(folder, '*.o*'))
files = [f for f in files if os.path.isfile(os.path.join(folder, f))]

error = np.full((3,len(files)), np.nan)
std = np.full((3,len(files)), np.nan)

for ii, f in enumerate(files):
    with open(f, 'r') as fh:
        lines = fh.readlines()
        if len(lines) > 12:
            if 'final_train_mean' in lines[-12]:
                delta = 0
            elif 'final_train_mean' in lines[-10]:
                delta = 2
            else:
                continue
            error[0,ii] = float(lines[-12+delta].split(' ')[-1])
            error[1,ii] = float(lines[-11+delta].split(' ')[-1])
            error[2,ii] = float(lines[-10+delta].split(' ')[-1])
            std[0,ii] = float(lines[-8+delta].split(' ')[-1])
            std[1,ii] = float(lines[-7+delta].split(' ')[-1])
            std[2,ii] = float(lines[-6+delta].split(' ')[-1])


print (np.nanmax(error[0]), np.nanmax(error[1]), np.nanmax(error[1]))

nan_idx = np.isnan(error[1])
good_error = error[:, ~nan_idx]
good_std = std[:, ~nan_idx]
files = np.array(files)
files = files[~nan_idx]
print '{} out of {} finished'.format(good_error.shape[1], error.shape[1])

if len(sys.argv[1:]) > 1:
    keep_n = int(sys.argv[2])
else:
    keep_n = good_error.shape[1]

keep_good_error = good_error[:, :keep_n]
keep_good_std = good_std[:, :keep_n]

max_idx = np.nanargmax(keep_good_error[1])
print 'using {} out of {} results'.format(keep_good_error.shape[1], good_error.shape[1])
print os.path.join(folder,files[max_idx])
print 'train: ', keep_good_error[0, max_idx], keep_good_std[0, max_idx]
print 'valid: ', keep_good_error[1, max_idx], keep_good_std[1, max_idx]
print 'test: ', keep_good_error[2, max_idx], keep_good_std[2, max_idx]
