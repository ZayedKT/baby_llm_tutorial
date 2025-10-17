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

## Load the Conda module
#source ~/.bashrc

# This ensures 'conda activate' works even in batch jobs.
source ~/miniconda3/etc/profile.d/conda.sh

# Activate your Conda environment. By default we named it baby
conda activate baby

# Set the transformers cache path. Replace [your NetID] with your actual NetID. If you already included this in your .bashrc, you can comment it out here
export TRANSFORMERS_CACHE="/scratch/zka224/huggingface_cache"

# Call python script. -u is for logging the output in a sbatch job. 
python -u huggingface_transformers.py
