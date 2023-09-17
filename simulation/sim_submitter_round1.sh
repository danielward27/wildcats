#!/bin/bash

#SBATCH --job-name=wildcats
#SBATCH --partition=htp
#SBATCH --array=1-9999
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=45G
#SBATCH --account=BISC019342
#SBATCH --output ./output/slurm/slurm-%A_%a.out

cd "${SLURM_SUBMIT_DIR}"

echo Time is "$(date)"

module load lang/python/miniconda/3.9.7
. ~/.bashrc

conda activate bp1_envA
python run_sim.py
conda deactivate

echo Time is "$(date)"
