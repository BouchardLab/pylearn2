#!/bin/bash -l
#SBATCH -p shared
#SBATCH -n 4
#SBATCH --mem=12GB
##SBATCH -p regular
##SBATCH --qos=premium
##SBATCH -N 1
#SBATCH -t 10:00:00
#SBATCH -J gp33_hg_a_random_09
#SBATCH -o /scratch2/scratchdirs/jlivezey/output/%J.out

cores=4
#cores=24

export PATH="$HOME/anaconda/bin:$PATH"
export PYLEARN2_DATA_PATH="$SCRATCH/data"

export MKL_NUM_THREADS="$cores"
export OMP_NUM_THREADS="$cores"

mkdir -p "/scratch2/scratchdirs/jlivezey/output/$SLURM_JOB_NAME"

export THEANO_FLAGS="floatX=float32,base_compiledir=$SCRATCH/.theano/$SLURM_JOBID/$1,openmp=True,blas.ldflags=-lmkl_rt,allow_gc=False"
srun -o "/scratch2/scratchdirs/jlivezey/output/$SLURM_JOB_NAME/$SLURM_JOBID.out" -N 1 -n 1 -c "$cores" python -u run_random.py "$SLURM_JOBID" "$1" "$2" "$3" "$4" "$5" "$6"
