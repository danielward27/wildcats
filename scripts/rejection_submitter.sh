#!/bin/bash
# request resources:
#PBS -N run_sim_0_99
#PBS -l nodes=1:ppn=1,walltime=100:00:00
#PBS -l mem=8gb 
#PBS -t 0-99
#PBS -o ../output/logs/error/
#PBS -e ../output/logs/out/

cd $PBS_O_WORKDIR
source activate wildcats_env
python ./run_rejection.py
