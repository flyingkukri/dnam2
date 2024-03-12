#!/bin/bash -l
#SBATCH --job-name=train_300
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --time=24:00:00
# need to set mem https://stackoverflow.com/questions/76641146/using-a-slurm-script-to-run-other-snakemake-slurm-jobs
#SBATCH --mem-per-cpu=2G
#SBATCH --mail-user=lukas.huan@tum.de
#SBATCH --mail-type=BEGIN,END,FAIL

srun python fullds.py