from pylearn2.config import yaml_parse
import numpy as np
import sys
with open('ecog_aug.yaml', 'rb') as f:
    train = f.read()
    
fold = int(sys.argv[1])
print 'fold: '+str(fold)
filename = 'ecog_85_mean_reg_aug_f'+str(fold)+'.pkl'
init_alpha = .01
dim = 32*258
targets = 57

L0 = 150
L1 = 75
max_col_norm = .9
L0_std = np.sqrt(init_alpha/(dim+L0))
L1_std = np.sqrt(init_alpha/(L0+L1))
y_std = np.sqrt(init_alpha/(L1+targets))

params = {'L0': L0,
          'L1': L1,
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
