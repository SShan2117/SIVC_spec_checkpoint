#!/bin/bash -l
#SBATCH --job-name=CFMC
#SBATCH --partition=neptune,pluto,charon
#SBATCH --time=7-0:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --output=slurm_logs/%x.out.%j
#SBATCH --error=slurm_logs/%x.err.%j

set -euo pipefail
mkdir -p slurm_logs

# module load gcc/11.3
# module load openmpi/4.1.4-gcc11.3
# module load mkl/latest

DAT_DIR="results"
mkdir -p "$DAT_DIR"

export OMPI_MCA_pmix=^s1,s2,cray

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/data/home/shan2117/.conda/envs/shan/lib"

mpirun --mca plm rsh -np "$SLURM_NTASKS" ./pin.x

mv *.dat "$DAT_DIR/"

echo "All .dat files have been moved to the $DAT_DIR directory."