#!/bin/bash
#SBATCH --job-name=m2ds300_6gpu   # create a short name for your job
#SBATCH --ntasks-per-node=6      # total number of tasks per node
#SBATCH --gres=gpu:a40:6           # number of gpus per node
#SBATCH --cpus-per-task=16        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=180GB               # total memory per node
#SBATCH --partition=standard               # total memory per node

#https://discuss.huggingface.co/t/using-transformers-with-distributeddataparallel-any-examples/10775/9

# we have to change the port https://pytorch.org/docs/stable/elastic/run.html
# but the correct option is rdzv_backend not rdzv-backend
srun torchrun --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 train.py

