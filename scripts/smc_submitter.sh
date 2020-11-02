#!/bin/bash
# request resources:
#PBS -N submitter
#PBS -l nodes=1:ppn=1,mem=1mb
#PBS -l walltime=00:01:00
#PBS -o ../output/logs/out/
#PBS -e ../output/logs/error/

cd $PBS_O_WORKDIR
source activate wildcats_env

# Give some extra time to the controller as it waits for engines to start
controller=$(qsub smc_controller.sh)
engines=$(qsub -W depend=after:$controller smc_engines.sh)
python_script=$(qsub -W depend=after:$engines smc_python_script.sh)

# The below order also works, but have to be careful that engines don't timeout before controller starts...
# engines=$(qsub smc_engines.sh)
# controller=$(qsub -W depend=after:$engines smc_controller.sh)
# python_script=$(qsub -W depend=after:$controller smc_python_script.sh)
