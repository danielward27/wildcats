#!/bin/bash
# request resources:
#PBS -N python
#PBS -l nodes=1:ppn=16
#PBS -l walltime=100:00:00
#PBS -o ../output/logs/out/
#PBS -e ../output/logs/error/

cd $PBS_O_WORKDIR
conda activate wildcats_env

sleep 10
python ./run_smc.py
