#!/usr/bin/env bash

#name the job pybench33 and place it's output in a file named slurm-<jobid>.out
# allow 40 minutes to run (it should not take 40 minutes however)
# set partition to 'all' so it runs on any available node on the cluster

#SBATCH -J 'slurm_aw'
#SBATCH -o slurm_aw-%j.out
#SBATCH -t 72:00:00
#SBATCH --mem 32gb
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH -p ctn

nvidia-smi

. activate directpose

export NCCL_P2P_DISABLE=1

python  test_bee.py
#python Interactive_training_centroids.py

#python Interactive_training_topdown.py
