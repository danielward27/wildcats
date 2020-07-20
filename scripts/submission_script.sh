#!/bin/bash
# request resources:
#PBS -N run_sim_0_99
#PBS -l nodes=1:ppn=1,walltime=50:00:00
#PBS -l mem=8gb 
#PBS -t 0-99
#PBS -o ../output/logs/
#PBS -e ../output/logs/
# on compute node, change directory to 'submission directory':
cd $PBS_O_WORKDIR
conda activate wildcats_summer_env
python ./run_simulation_parallel.py
