from pylearn2.config import yaml_parse
import numpy as np
with open('ecog.yaml', 'rb') as f:
    train = f.read()
    
init_alpha = .1
dim = 784

chan_0 = 32
chan_1 = 32
max_col_norm = .863105108422
max_ker_norm = max_col_norm
irange = .1
istdev = .1
ystd = .1

params = {'chan_0': chan_0,
          'chan_1': chan_1,
          'max_col_norm': max_col_norm,
          'max_ker_norm': max_ker_norm,
          'irange': irange,
          'istdev': istdev,
          'ystd': ystd,
          }
train = train % params
print train
train = yaml_parse.load(train)
train.main_loop()
