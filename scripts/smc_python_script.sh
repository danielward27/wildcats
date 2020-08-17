#!/bin/bash
# request resources:
#PBS -N python
#PBS -l nodes=1:ppn=1
#PBS -l walltime=05:00:00
#PBS -o ../output/logs/out/
#PBS -e ../output/logs/error/

cd $PBS_O_WORKDIR
conda activate wildcats_summer_env

sleep 60  # Ensure engines and controller have time to start (probably better way?)
python ./run_smc.py
ipcluster stop --profile=pbs
