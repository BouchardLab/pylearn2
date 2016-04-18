from pylearn2.utils import serial
from pylearn2.config import yaml_parse
from pylearn2.datasets import ecog, ecog_new
from pylearn2.space import VectorSpace, Conv2DSpace, CompositeSpace
from pylearn2.expr import nnet

import os, h5py, theano, cPickle, argparse
import numpy as np
import theano.tensor as T

import matplotlib
matplotlib.use('Agg')
from pylab import rcParams
from matplotlib import rc
import matplotlib.pyplot as plt

import analysis
import plotting
from scipy.spatial import distance
from scipy import cluster

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def main(args):
    pass

if __name__ == '__main__':
    main(args)