#!/bin/bash
#SBATCH --job-name=pin
#SBATCH --output=pin.%j.out
#SBATCH --error=pin.%j.err
#SBATCH --time=5-05:00:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH -p amd  

source ~/miniforge/etc/profile.d/conda.sh
conda activate mkl-gcc

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}"

srun -n ${SLURM_NTASKS} ./pin.x