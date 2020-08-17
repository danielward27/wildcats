#!/bin/bash
# request resources:
#PBS -N ipython
#PBS -l nodes=2:ppn=16
#PBS -l walltime=00:07:00
#PBS -o ../output/logs/
#PBS -e ../output/logs/
# on compute node, change directory to 'submission directory':
cd $PBS_O_WORKDIR
conda activate wildcats_summer_env

# if [[ $RANK = 0 ]]; then
#	ipcontroller --profile=pbs;
#	exit 123;
#else
#	mpiexec -n 31 ipengine --profile=pbs;
#fi

FILE="~/.ipython/profile_pbs/security/ipcontroller-client.json"
if [[ -f $FILE ]]; then
	echo starting engine;
	mpiexec -n 31 ipengine --profile=pbs;
else
	echo starting controller;
	(ipcontroller --profile=pbs &);
	sleep 10;
fi

sleep 30

python ./run_smc.py

ipcluster stop
