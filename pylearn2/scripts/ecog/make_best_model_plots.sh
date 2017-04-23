#!/usr/bin/env bash
export MKL_NUM_THREADS=2
#./make_model_plots.py ec2 'high gamma' 'amplitude' $SCRATCH/exps/ec2_hg_a
./make_model_plots.py ec9 'high gamma' 'amplitude' $SCRATCH/exps/ec9_hg_a
./make_model_plots.py gp31 'high gamma' 'amplitude' $SCRATCH/exps/gp31_hg_a
./make_model_plots.py gp33 'high gamma' 'amplitude' $SCRATCH/exps/gp33_hg_a
