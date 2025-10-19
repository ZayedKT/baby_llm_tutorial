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
#SBATCH --output=slurm-%j.out

# -----------------------------
# Environment setup
# -----------------------------
module purge
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate baby

# -----------------------------
# Hugging Face token & cache
# -----------------------------
export HUGGINGFACE_HUB_TOKEN="The Token Here"
echo "HF token set: $HUGGINGFACE_HUB_TOKEN"

# Optional: set HF cache to a scratch folder for large models
export HF_HOME="/scratch/zka224/huggingface_cache"
mkdir -p "$HF_HOME"

# -----------------------------
# Check Python sees HF token
# -----------------------------
python - <<EOF
import os
token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
print("HF token in Python:", token is not None)
EOF

# -----------------------------
# Run finetuning
# -----------------------------
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
