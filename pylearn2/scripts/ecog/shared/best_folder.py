#!/usr/bin/env python
import glob, sys, os
import numpy as np

def best_folder(folder, keep_n):

    keep_n = int(keep_n)


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

    nan_idx = np.isnan(error[1])
    good_error = error[:, ~nan_idx]
    good_std = std[:, ~nan_idx]
    files = np.array(files)
    if nan_idx.sum() > 0:
        print('nan', nan_idx.sum(), folder)
    files = files[~nan_idx]
    file_data = np.zeros((2, 2), dtype=int)
    results = np.zeros((3, 2), dtype=float)
    file_data[0, 0] = good_error.shape[1]
    file_data[0, 1] = error.shape[1]

    keep_good_error = good_error[:, :keep_n]
    keep_good_std = good_std[:, :keep_n]

    file_data[1, 0] = keep_good_error.shape[1]
    file_data[1, 1] = good_error.shape[1]

    max_idx = np.nanargmax(keep_good_error[1])
    f_id = os.path.join(folder,files[max_idx])
    results = np.array([[keep_good_error[0, max_idx], keep_good_std[0, max_idx]],
                        [keep_good_error[1, max_idx], keep_good_std[1, max_idx]],
                        [keep_good_error[2, max_idx], keep_good_std[2, max_idx]]],
                       dtype=float)
    return file_data, results, f_id


if __name__ == '__main__':
    file_data, results, f_id = best_folder(*sys.argv[1:])
    print '{} out of {} finished'.format(file_data[0, 1], file_data[0, 1])
    print 'using {} out of {} results'.format(file_data[1, 0], file_data[1, 1])
    print('best')
    print f_id
    print 'train: ', results[0, 0], results[0, 1]
    print 'valid: ', results[1, 0], results[1, 1]
    print 'test: ', results[2, 0], results[2, 1]
