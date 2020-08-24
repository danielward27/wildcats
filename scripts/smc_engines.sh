#!/bin/bash
# request resources:
#PBS -N engines
#PBS -l nodes=2:ppn=8,pmem=4gb
#PBS -l walltime=02:00:00
#PBS -o ../output/logs/out/
#PBS -e ../output/logs/error/

cd $PBS_O_WORKDIR
conda activate wildcats_summer_env

mpiexec -n 16 ipengine --profile=pbs --quiet
