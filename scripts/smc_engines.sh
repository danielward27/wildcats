#!/bin/bash
# request resources:
#PBS -N engines
#PBS -l nodes=2:ppn=16
#PBS -l walltime=10:00:00
#PBS -o ../output/logs/out/
#PBS -e ../output/logs/error/

cd $PBS_O_WORKDIR
source activate wildcats_env

mpiexec -n 32 ipengine --profile=pbs
