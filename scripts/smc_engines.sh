#!/bin/bash
# request resources:
#PBS -N engines
#PBS -l nodes=8:ppn=16
#PBS -l walltime=100:00:00
#PBS -o ../output/logs/out/
#PBS -e ../output/logs/error/

cd $PBS_O_WORKDIR
conda activate wildcats_env

mpiexec -n 128 ipengine --profile=pbs --quiet
