from pylearn2.config import yaml_parse
import numpy as np
import sys
with open('ecog.yaml', 'rb') as f:
    train = f.read()
    
fold = int(sys.argv[1])
print 'fold: '+str(fold)
filename = 'exps/ecog_85_reg_lin_f'+str(fold)+'.pkl'
init_alpha = .01
dim = 784

L0 = 150
L1 = 75
max_col_norm = .9
L0_std = np.sqrt(init_alpha/(dim+L0))
L1_std = np.sqrt(init_alpha/(L0+L1))
y_std = np.sqrt(init_alpha/(L1+targets))

params = {'chan_0': chan_0,
          'chan_1': chan_1,
          'max_col_norm': max_col_norm,
          'L0_std': L0_std,
          'L1_std': L1_std,
          'y_std': y_std,
          'fold': fold,
          'filename': filename,
          }
train = train % params
print train
train = yaml_parse.load(train)
train.main_loop()
