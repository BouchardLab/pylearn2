from pylearn2.config import yaml_parse
import numpy as np
import sys
with open('conv_ecog.yaml', 'rb') as f:
    train = f.read()
    
fold = int(sys.argv[1])
print 'fold: '+str(fold)
filename = 'ecog_85_conv_mean_f'+str(fold)+'.pkl'
init_alpha = .01
dim = 32*258
targets = 57

chan_0 = 24
chan_1 = 32
chan_2 = 64
max_col_norm = .863105108422
max_ker_norm = max_col_norm
irange = .0001
f0 = 500
f1 = 500
f0_std = np.sqrt(init_alpha/(f0+1000))
f1_std = np.sqrt(init_alpha/(f0+f1))
y_std = np.sqrt(init_alpha/(f1+targets))

params = {'chan_0': chan_0,
          'chan_1': chan_1,
          'chan_2': chan_2,
          'f0': f0,
          'f1': f1,
          'max_col_norm': max_col_norm,
          'max_ker_norm': max_ker_norm,
          'irange': irange,
          'f0_std': f0_std,
          'f1_std': f1_std,
          'y_std': y_std,
          'fold': fold,
          'filename': filename,
          }
train = train % params
print train
train = yaml_parse.load(train)
train.main_loop()
