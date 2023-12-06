#!/bin/bash

#SBATCH -p nvidia
#SBATCH -C 80g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1:59:59
#SBATCH --mem=10GB
#SBATCH --job-name=run_infer

module purge

# Load the Conda module
source ~/.bashrc

# Activate your Conda environment
conda activate [your conda env]

# Set the transformers cache path
export TRANSFORMERS_CACHE="/scratch/[your NetID]/huggingface_cache"

# Call python script
python -u infer_codegen.py