from pylearn2.blocks import Block
from pylearn2.utils.rng import make_theano_rng
from pylearn2.space import Conv2DSpace, VectorSpace
import theano

clas ScaleAugmentation(Block):

    def __init__(self, space, seed=20150111, mean=1., std=.05):
        self.rng = make_theano_rng(seed, which_method=['normal'])
        self.mean = mean
        self.std = std
        self.space = space
        super(ScaleAugmentation, self).__init__()

    def create_theano_function(self):
        if hasattr(self, 'f'):
            return self.f
        else:
            X = self.space.make_theano_batch()
            dim = X.ndim
            arg = (dim-1)*('x',)
            scale = self.rng.normal(size=[X.shape[0]], avg=self.mean, std=self.std)
            scale = scale.dimshuffle(0,*arg)
            out = X*scale
            return theano.function([X], out)

    def perform(self, X):
        f = self.create_theano_function()
        return f(X)

    def get_input_space(self):
        return self.space

    def get_output_space(self):
        return self.space

