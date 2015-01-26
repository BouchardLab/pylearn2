from pylearn2.blocks import Block
import numpy as np
from pylearn2.utils.rng import make_np_rng
from pylearn2.space import Conv2DSpace
from scipy.ndimage.interpolation import rotate, shift, zoom

class DataAugmentation(Block):

    def __init__(self, space, seed=20150111, spline_order=0, cval=0.):
        self.rng = make_np_rng(np.random.RandomState(seed),
                               which_method=['rand', 'randint'])
        assert isinstance(space, Conv2DSpace)
        self.space = space
        self.spline_order = spline_order
        self.cval = cval
        super(DataAugmentation, self).__init__()


    def perform(self, X):
        axis1 = self.rng.uniform(low=-5., high=5.)
        X = shift(X,
                  shift=(0, 0, axis1, 0),
                  order=self.spline_order,
                  mode='nearest')
        return X

    def get_input_space(self):
        return self.space

    def get_output_space(self):
        return self.space

