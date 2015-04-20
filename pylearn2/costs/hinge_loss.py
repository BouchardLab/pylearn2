"""
Hinge loss costs.
"""
__authors__ = 'Jesse Livezey, Brian Cheung'

from theano.compat.python2x import OrderedDict

from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin
from pylearn2.expr.nnet import HingeL2, HingeL1, Misclass

class HingeLoss(DefaultDataSpecsMixin, Cost):
    supervised = True

    def get_monitoring_channels(self, model, data):
        space, source = self.get_data_specs(model)
        space.validate(data)
        X, Y = data
        Y_hat = model.fprop(X)
        rval = OrderedDict()
        name = model.layers[-1].layer_name+'_misclass'
        rval[name] = Misclass(Y, Y_hat)
        return rval

    def expr(self, model, data):
        raise ValueError('Abstract HingeLoss class '
                        +'should not be used directly.')


class HingeLossL2(HingeLoss):
    def expr(self, model, data):
        space, source = self.get_data_specs(model)
        space.validate(data)
        X, Y = data
        Y_hat = model.fprop(X)
        cost = HingeL2(Y, Y_hat)
        cost.name = 'hingel2'
        return cost

class HingeLossL1(HingeLoss):
    def expr(self, model, data):
        space, source = self.get_data_specs(model)
        space.validate(data)
        X, Y = data
        Y_hat = model.fprop(X)
        cost = HingeL1(Y, Y_hat)
        cost.name = 'hingel1'
        return cost

class DropoutHingeLossL2(HingeLoss):
    def __init__(self, default_input_include_prob=.5, input_include_probs=None,
                 default_input_scale=2., input_scales=None, per_example=True):

        if input_include_probs is None:
            input_include_probs = {}

        if input_scales is None:
            input_scales = {}

        self.__dict__.update(locals())
        del self.self

    def expr(self, model, data):
        space, source = self.get_data_specs(model)
        space.validate(data)
        X, Y = data
        Y_hat = model.dropout_fprop(
                            X,
                            default_input_include_prob=self.default_input_include_prob,
                            input_include_probs=self.input_include_probs,
                            default_input_scale=self.default_input_scale,
                            input_scales=self.input_scales,
                            per_example=self.per_example)
        cost = HingeL2(Y, Y_hat)
        cost.name = 'hingel2'
        return cost

class DropoutHingeLossL1(HingeLoss):
    def __init__(self, default_input_include_prob=.5, input_include_probs=None,
                 default_input_scale=2., input_scales=None, per_example=True):

        if input_include_probs is None:
            input_include_probs = {}

        if input_scales is None:
            input_scales = {}

        self.__dict__.update(locals())
        del self.self

    def expr(self, model, data):
        space, source = self.get_data_specs(model)
        space.validate(data)
        X, Y = data
        Y_hat = model.dropout_fprop(
                            X,
                            default_input_include_prob=self.default_input_include_prob,
                            input_include_probs=self.input_include_probs,
                            default_input_scale=self.default_input_scale,
                            input_scales=self.input_scales,
                            per_example=self.per_example)
        cost = HingeL1(Y, Y_hat)
        cost.name = 'hingel1'
        return cost
