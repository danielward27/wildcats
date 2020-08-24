#!/bin/bash
# request resources:
#PBS -N engines
#PBS -l nodes=12:ppn=8,pmem=8gb
#PBS -l walltime=04:00:00
#PBS -o ../output/logs/out/
#PBS -e ../output/logs/error/

cd $PBS_O_WORKDIR
conda activate wildcats_summer_env

mpiexec -n 96 ipengine --profile=pbs --quiet
