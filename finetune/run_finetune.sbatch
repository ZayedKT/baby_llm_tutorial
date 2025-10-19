#!/bin/bash

#SBATCH -p nvidia
#SBATCH -C 80g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3          # Increased for data loading + tokenization
#SBATCH --gres=gpu:a100:3          # Request 3 A100 GPUs
#SBATCH --time=11:59:59             # Increase time, large model takes longer
#SBATCH --mem=100GB                 # Increase memory to handle 8B model + LoRA + 3 GPUs
#SBATCH --job-name=finetune_LLama3

module purge

# Load the Conda module
#source ~/.bashrc

# Initialize conda for bash
eval "$(conda shell.bash hook)"

# Activate your Conda environment
conda activate baby

# Set the transformers cache path
export TRANSFORMERS_CACHE="/scratch/zka224/huggingface_cache"

# Ensure Python sees the 3 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2

# Call the fine-tuning Python script
python -u finetune_llama3.py \
    --train_file "/scratch/zka224/Fine-tuning/train_converted.jsonl" \
    --dev_file "/scratch/zka224/Fine-tuning/dev_converted.jsonl" \
    --output_dir "/scratch/zka224/Fine-tuning/checkpoints" \
    --model_name "meta-llama/Llama-3.1-8B" \
    --batch_size 1 \
    --epochs 1 \
    --use_lora True \
    --load_in_8bit True
