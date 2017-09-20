#!/usr/bin/env bash
export MKL_NUMTHREADS=1
# HG best deep models
./make_model_plots.py ec2 'high gamma' $HOME/exps/ec2_hg_a -o
#./make_model_plots.py ec9 'high gamma' $HOME/exps/ec9_hg_a -o
#./make_model_plots.py gp31 'high gamma' $HOME/exps/gp31_hg_a -o
#./make_model_plots.py gp33 'high gamma' $HOME/exps/gp33_hg_a -o

# HG best linear models
#./make_model_plots.py ec2 'high gamma' $HOME/exps/ec2_hg_a_lin -o
#./make_model_plots.py ec9 'high gamma' $HOME/exps/ec9_hg_a_lin -o
#./make_model_plots.py gp31 'high gamma' $HOME/exps/gp31_hg_a_lin -o
#./make_model_plots.py gp33 'high gamma' $HOME/exps/gp33_hg_a_lin -o

# HG best random models
#./make_model_plots.py ec2 'high gamma' $HOME/exps/ec2_hg_a_random -o -r
#./make_model_plots.py ec9 'high gamma' $HOME/exps/ec9_hg_a_random -o -r
#./make_model_plots.py gp31 'high gamma' $HOME/exps/gp31_hg_a_random -o -r
#./make_model_plots.py gp33 'high gamma' $HOME/exps/gp33_hg_a_random -o -r
