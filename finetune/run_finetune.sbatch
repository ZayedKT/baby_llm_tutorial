#!/bin/bash
#SBATCH -p nvidia
#SBATCH -C 80g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:a100:3
#SBATCH --time=02:00:00   # short test run
#SBATCH --mem=100GB
#SBATCH --job-name=finetune_LLaMA3_test
#SBATCH --output=slurm-%j.out

# -----------------------------
# Environment setup
# -----------------------------
module purge
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate baby

# -----------------------------
# Optional HF cache (local only)
# -----------------------------
export HF_HOME="/scratch/zka224/huggingface_cache"
mkdir -p "$HF_HOME"


# -----------------------------
# Run small test fine-tuning
# -----------------------------

# --- Change: Added CUDA_VISIBLE_DEVICES to explicitly allow all 3 GPUs ---
export CUDA_VISIBLE_DEVICES=0,1,2

python -u /scratch/zka224/baby_llm_tutorial/finetune/finetune_llama3.py \
    --train_file "/scratch/zka224/Fine-tuning/train_converted.jsonl" \
    --dev_file "/scratch/zka224/Fine-tuning/dev_converted.jsonl" \
    --output_dir "/scratch/zka224/Fine-tuning/checkpoints_test" \
    --model_name "/scratch/zka224/Llama-3.1-8B" \
    --batch_size 1 \
    --epochs 1 \
    --use_lora True \
    --load_in_8bit True \
    --prompt_field "prompt" \
    --completion_field "completion"
