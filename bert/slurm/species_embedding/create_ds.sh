#!/bin/bash
#SBATCH --job-name=dataframe # create a short name for your job
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --cpus-per-task=16        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=64GB               # total memory per node
#SBATCH --partition=standard               # total memory per node

python3 create_ds.py
