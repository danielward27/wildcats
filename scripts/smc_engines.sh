#!/bin/bash
# request resources:
#PBS -N engines
#PBS -l nodes=4:ppn=16
#PBS -l walltime=12:00:00
#PBS -o ../output/logs/out/
#PBS -e ../output/logs/error/

cd $PBS_O_WORKDIR
conda activate wildcats_summer_env

mpiexec -n 64 ipengine --profile=pbs --quiet
