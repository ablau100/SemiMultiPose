#!/usr/bin/env bash


helpFunction()
{
   echo ""
   echo "Usage: $0 -o out_dir -d data_type -a alpha -b beta -n max_iters"
   echo -e "\t-o Path to output directory"
   echo -e "\t-d Data type to train on i.e. type of animal"
   echo -e "\t-a Value of hyperparameter alpha (default is 0.01)"
   echo -e "\t-b Value of hyperparameter beta (default is 0.1)"
   echo -e "\t-n Number of iterations to train. Should be a multiple of 1k (default is 40000)"
   exit 1 # Exit script after printing help
}

while getopts "o:d:a:b:n:" opt
do
   case "$opt" in
      o ) out_dir="$OPTARG" ;;
      d ) data_type="$OPTARG" ;;
      a ) alpha="$OPTARG" ;;
      b ) beta="$OPTARG" ;;
      n ) max_iters="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$out_dir" ] || [ -z "$data_type" ]
then
   echo "Some or all of the parameters are empty. out_dir and data_type are non-optional parameters.";
   helpFunction
fi

# default for alpha
if [ -z "$alpha" ]
then
   alpha=".01"
fi

# default for beta
if [ -z "$beta" ]
then
   beta=".1"
fi

# default for max_iters
if [ -z "$max_iters" ]
then
   max_iters="40000"
fi

# Begin script in case all parameters are correct

nvidia-smi

. activate directpose

export NCCL_P2P_DISABLE=1 

python train_model.py "$out_dir" "$data_type" "$alpha" "$beta" "$max_iters"


