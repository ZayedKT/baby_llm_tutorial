#!/bin/bash

#SBATCH -p nvidia
#SBATCH -C 80g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1:59:59
#SBATCH --mem=50GB           # Increased to handle 8B model + 8-bit/LoRA
#SBATCH --job-name=finetune_llama3

module purge

# Load the Conda module
source ~/.bashrc

# Activate your Conda environment
conda activate baby

# Set the transformers cache path
export TRANSFORMERS_CACHE="/scratch/zka224/huggingface_cache"

# Call python script
#python -u infer_codegen.py

# Call the fine-tuning Python script
python -u finetune_llama3.py \
    --train_file "/scratch/zka224/train_converted.jsonl" \
    --dev_file "/scratch/zka224/dev_converted.jsonl" \
    --output_dir "/scratch/zka224/checkpoints" \
    --model_name "meta-llama/Llama-3-8b-hf" \
    --batch_size 1 \
    --epochs 1 \
    --use_lora True \
    --load_in_8bit True
