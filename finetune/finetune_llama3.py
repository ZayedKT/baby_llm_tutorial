#!/usr/bin/env python3
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer
import torch

# -----------------------------
# Parse command-line arguments
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--train_file", type=str, required=True)
parser.add_argument("--dev_file", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--use_lora", type=bool, default=True)
parser.add_argument("--load_in_8bit", type=bool, default=True)
parser.add_argument("--prompt_field", type=str, default="prompt")
parser.add_argument("--completion_field", type=str, default="completion")
args = parser.parse_args()

# -----------------------------
# Load dataset
# -----------------------------
dataset = load_dataset(
    "json",
    data_files={"train": args.train_file, "validation": args.dev_file}
)
print("Dataset columns:", dataset["train"].column_names)

# -----------------------------
# Load tokenizer
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# -----------------------------
# 8-bit config
# -----------------------------
bnb_config = BitsAndBytesConfig(load_in_8bit=True) if args.load_in_8bit else None

# -----------------------------
# Load model
# -----------------------------
model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True,
    dtype=torch.float16 if args.load_in_8bit else None
)
model.config.use_cache = False

# -----------------------------
# LoRA config
# -----------------------------
peft_config = None
if args.use_lora:
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

# -----------------------------
# Tokenize dataset
# -----------------------------
def tokenize(batch):
    texts = [p + c for p, c in zip(batch[args.prompt_field], batch[args.completion_field])]
    return tokenizer(texts, truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize, batched=True)

# -----------------------------
# Minimal TrainingArguments (hub-free)
# -----------------------------
training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=args.epochs,
    save_strategy="epoch",
    fp16=True,
    logging_strategy="steps",
    logging_steps=50,
    report_to="none"  # disables wandb/tensorboard
)

# -----------------------------
# Initialize SFTTrainer (hub-free)
# -----------------------------
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    peft_config=peft_config,
    args=training_args,
    # Do NOT pass push_to_hub, tokenizer, or hub args
)

# -----------------------------
# Train
# -----------------------------
trainer.train()

# -----------------------------
# Save model
# -----------------------------
trainer.save_model(args.output_dir)
print(f"Model saved to {args.output_dir}")
