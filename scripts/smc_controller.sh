#!/bin/bash
# request resources:
#PBS -N controller
#PBS -l nodes=1:ppn=1
#PBS -l walltime=04:00:00
#PBS -o ../output/logs/out/
#PBS -e ../output/logs/error/

cd $PBS_O_WORKDIR
conda activate wildcats_summer_env

ipcontroller --profile=pbs --nodb
