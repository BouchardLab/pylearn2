#!/bin/bash -l
#SBATCH -p shared
#SBATCH -n 4
#SBATCH --mem=6GB

cores=4
#cores=24

export PATH="$HOME/anaconda/bin:$PATH"
export PYLEARN2_DATA_PATH="$SCRATCH/data"

export MKL_NUM_THREADS="$cores"
export OMP_NUM_THREADS="$cores"
export SAVE=/scratch2/scratchdirs/jlivezey/exps_ff_1f

mkdir -p "/scratch2/scratchdirs/jlivezey/output_ff_1f/$SLURM_JOB_NAME"

export THEANO_FLAGS="floatX=float32,base_compiledir=$SCRATCH/.theano/$SLURM_JOBID/$1,openmp=True,blas.ldflags=-lmkl_rt,allow_gc=False"
srun -o "/scratch2/scratchdirs/jlivezey/output_ff_1f/$SLURM_JOB_NAME/$SLURM_JOBID.out" -N 1 -n 1 -c "$cores" python -u run_random.py $SLURM_JOBID "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "$10"
