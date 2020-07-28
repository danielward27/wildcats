#!/bin/bash
# request resources:
#PBS -N abc_smc
#PBS -l nodes=10:ppn=16,walltime=50:00:00
#PBS -o ../output/logs/
#PBS -e ../output/logs/
# on compute node, change directory to 'submission directory':
cd $PBS_O_WORKDIR
# conda activate wildcats_summer_env
Rscript ./detect_cores.R
