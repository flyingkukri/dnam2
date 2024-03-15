#!/bin/bash
#SBATCH --job-name=eval_specieslm   # create a short name for your job
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --gres=gpu:a40:1            # number of gpus per node
#SBATCH --cpus-per-task=16        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=20GB               # total memory per node
#SBATCH --partition=standard               # total memory per node

srun python train-noglu.py
