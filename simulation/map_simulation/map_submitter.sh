#!/bin/bash

#SBATCH --job-name=wildcats
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=25G
#SBATCH --account=BISC019342


cd "${SLURM_SUBMIT_DIR}"

echo Time is "$(date)"

module load lang/python/miniconda/3.9.7
. ~/.bashrc

conda activate bp1_envA
python run_sim_map.py
conda deactivate

echo Time is "$(date)"
