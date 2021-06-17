#!/usr/bin/env bash

# plots all environemnts
# this will make videos comparing step 1 and step 2
#source activate /home/ekellbuch/anaconda3/envs/dgp
# alphas: = [.0001, .001, .01, .1, .5, 1]
#betas = ".001" ".01" ".5" "1" "5" "10"

call_run_dp() {
    echo "$1"
    echo "$2"
    echo "$3"
    echo "$4"
    echo "$5"
    sbatch ./train_bee_new.sh "$1" "$2" "$3" "$4" "$5"
    #python run_dpg_demo.py $1 
}

for loss in "combined" 
    do
    for number in "25" 
        do
        for version in  "1"  "3" 
            do
            for alpha in   ".001"
                do
                for beta in ".001" 
                    do
                    call_run_dp "$loss" "$number" "$version" "$alpha" "$beta"
                    done
                done
            done
        done
    done


