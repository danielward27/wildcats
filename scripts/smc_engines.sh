#!/bin/bash
# request resources:
#PBS -N engines
#PBS -l nodes=4:ppn=16
#PBS -l walltime=05:00:00
#PBS -o ../output/logs/out/
#PBS -e ../output/logs/error/

cd $PBS_O_WORKDIR
conda activate wildcats_summer_env

sleep 30  # Ensure controller job has time to get running
mpiexec -n 64 ipengine --profile=pbs
