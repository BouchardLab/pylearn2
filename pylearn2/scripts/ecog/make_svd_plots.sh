#!/usr/bin/env python
THEANO_FLAGS='device=cpu' python make_svd_plots.py ec2
THEANO_FLAGS='device=cpu' python make_svd_plots.py ec9
THEANO_FLAGS='device=cpu' python make_svd_plots.py gp31
