#!/bin/bash
# request resources:
#PBS -N submitter
#PBS -l nodes=1:ppn=1,mem=1mb
#PBS -l walltime=00:01:00
#PBS -o ../output/logs/out/
#PBS -e ../output/logs/error/

cd $PBS_O_WORKDIR
conda activate wildcats_summer_env

controller=$(qsub smc_controller.sh)
engines=$(qsub -W depend=after:$controller smc_engines.sh)
python_script=$(qsub -W depend=after:$engines smc_python_script.sh)
