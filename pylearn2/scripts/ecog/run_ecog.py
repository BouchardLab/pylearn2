from pylearn2.config import yaml_parse
import numpy as np
import sys
with open('linear.yaml', 'rb') as f:
    train = f.read()
    
fold = int(sys.argv[1])
print 'fold: '+str(fold)
filename = 'exps/ecog_85_linear_level_true_f'+str(fold)+'.pkl'
init_alpha = .01
dim = 784

L0 = 150
L1 = 75
max_col_norm = .9

params = {'L0': L0,
          'fold': fold,
          'filename': filename,
          }
train = train % params
print train
train = yaml_parse.load(train)
train.main_loop()
