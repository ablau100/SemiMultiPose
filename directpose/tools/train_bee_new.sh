#!/usr/bin/env bash

#name the job pybench33 and place it's output in a file named slurm-<jobid>.out
# allow 40 minutes to run (it should not take 40 minutes however)
# set partition to 'all' so it runs on any available node on the cluster

#SBATCH -J 'slurm_bsb'
#SBATCH -o slurm_bsb-%j.out
#SBATCH -t 02:30:00
#SBATCH --mem 32gb
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH -p all

nvidia-smi

. activate directpose

export NCCL_P2P_DISABLE=1 

VAR250="250"
VAR135="135"
VAR100="100"
VAR50="50"
VAR25="25"
VARL="combined"
VARS="standard"

if [ "$VAR135" = "$2" ]; then
  python  nc_250_10.py "$1" "$2" "$3" "$4"
fi

if [ "$VAR250" = "$2" ]; then
  python  train_bee_250_lrp2.py "$1" "$2" "$3" "$4"
fi

if [ "$VAR100" = "$2" ]; then
  python  train_bee_100.py "$1" "$2" "$3" "$4"
fi

if [ "$VAR50" = "$2" ]; then
  python  train_bee_50.py "$1" "$2" "$3" "$4"
fi

# if [ "$VAR25" = "$2" ]; then
#   python nc_f69.py "$1" "$2" "$3" "$4" 
# fi

if [[ "$VAR25" = "$2" ]] && [[ "$VARS" = "$1" ]]; then
  python nc_f5_fb.py "$1" "$2" "$3" "$4" "$5"
fi

if [[ "$VAR25" = "$2" ]] && [[ "$VARL" = "$1" ]]; then
  python nc_f5f6.py "$1" "$2" "$3" "$4" "$5"
fi




#f5_fb.py
#python Interactive_training_centroids.py

#python Interactive_training_topdown.py

