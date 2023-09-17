#!/bin/bash

#SBATCH --job-name=tsinfer
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --mem=10G
#SBATCH --account=BISC019342

cd "${SLURM_SUBMIT_DIR}"

echo Time is "$(date)"

module load lang/python/miniconda/3.9.7
. ~/.bashrc

conda activate bp1_envA
python infer_tree.py
conda deactivate

echo Time is "$(date)"
