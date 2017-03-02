#!/bin/bash -l
#SBATCH -p regular
#SBATCH --qos=premium
#SBATCH -N 5
#SBATCH -t 00:10:00
#SBATCH -J train
#SBATCH -o /global/homes/j/jlivezey/output/train_output.o%j


if [ "$NERSC_HOST" == "edison" ]
then
  cores=24
fi
if [ "$NERSC_HOST" == "cori" ]
then
  cores=32
fi

export PATH="$HOME/anaconda/bin:$PATH"

echo $(which python)
echo $PATH

export PYLEARN2_DATA_PATH="$SCRATCH/data"
export SAVE="$SCRATCH/exps"

export MKL_NUM_THREADS="$cores"
export OMP_NUM_THREADS="$cores"

for i in {1..5};
do
  export THEANO_FLAGS="floatX=float32,base_compiledir=$SCRATCH/.theano/$SLURM_JOBID/$i,openmp=True,blas.ldflags=-lmkl_rt,allow_gc=False"
#  srun -N 1 -n 1 -c "$cores" python -u run_random.py "$i" config.json &
  srun -N 1 -n 1 -c "$cores" python -u /global/homes/j/jlivezey/conv_timing.py &
done
wait
