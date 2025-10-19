#!/bin/bash

#SBATCH -p nvidia
#SBATCH -C 80g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:a100:3
#SBATCH --time=11:59:59
#SBATCH --mem=100GB
#SBATCH --job-name=finetune_LLaMA3

module purge
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate baby

# Set Hugging Face token
export HUGGINGFACE_HUB_TOKEN="your_token_here"

# Optionally set cache location
export HF_HOME="/scratch/zka224/huggingface_cache"

# Run fine-tuning
python -u finetune_llama3.py \
    --train_file "/scratch/zka224/Fine-tuning/train_converted.jsonl" \
    --dev_file "/scratch/zka224/Fine-tuning/dev_converted.jsonl" \
    --output_dir "/scratch/zka224/Fine-tuning/checkpoints" \
    --model_name "meta-llama/Llama-3.1-8B" \
    --batch_size 1 \
    --epochs 1 \
    --use_lora True \
    --load_in_8bit True \
    --prompt_field "prompt" \
    --completion_field "completion"
