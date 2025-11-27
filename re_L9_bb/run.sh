#!/bin/bash -l
#SBATCH --job-name=CFMC
#SBATCH --partition=neptune,pluto,charon
#SBATCH --time=7-0:00:00
#SBATCH --nodes=4
#SBATCH --ntasks=32
#SBATCH --output=slurm_logs/%x.out.%j
#SBATCH --error=slurm_logs/%x.err.%j

mkdir -p slurm_logs

# Load modules (needed on ALL nodes)
module load gcc/11.3
module load openmpi/4.1.4-gcc11.3
module load mkl/latest

export OMPI_MCA_pmix=^s1,s2,cray

# DO NOT ADD conda paths
# DO NOT modify LD_LIBRARY_PATH unless necessary

mpirun -np "$SLURM_NTASKS" ./pin.x
