#!/bin/bash
# request resources:
#PBS -N smc
#PBS -l nodes=2:ppn=16
#PBS -l walltime=3:00:00
#PBS -o ../output/logs/out/
#PBS -e ../output/logs/error/

cd $PBS_O_WORKDIR
conda activate wildcats_summer_env

if [[ $$OMPI_COMM_WORLD_RANK = 0 ]]; then
	ipcontroller --profile=pbs --nodb;

elif [[ $$OMPI_COMM_WORLD_RANK = 1 ]]; then
	sleep 10
	python ./run_smc.py

else
	mpiexec -n 30 ipengine --profile=pbs #--quiet
fi

