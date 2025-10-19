import argparse
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

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
# Read Hugging Face token
# -----------------------------
hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
if hf_token is None:
    raise ValueError("HUGGINGFACE_HUB_TOKEN environment variable not set!")

print("HF token in Python:", hf_token)

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
tokenizer = AutoTokenizer.from_pretrained(
    args.model_name,
    trust_remote_code=True,
    token=hf_token  # updated API
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# -----------------------------
# 8-bit quantization config
# -----------------------------
bnb_config = None
if args.load_in_8bit:
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

# -----------------------------
# Load model
# -----------------------------
model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True,
    token=hf_token  # updated API
)
model.config.use_cache = False

# -----------------------------
# LoRA configuration
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
# Tokenize function
# -----------------------------
def tokenize(batch):
    if args.prompt_field in batch and args.completion_field in batch:
        texts = [p + c for p, c in zip(batch[args.prompt_field], batch[args.completion_field])]
        return tokenizer(texts, truncation=True, padding="max_length", max_length=512)
    else:
        raise ValueError(f"Dataset must contain '{args.prompt_field}' and '{args.completion_field}' fields.")

tokenized_dataset = dataset.map(tokenize, batched=True)

# -----------------------------
# Training arguments
# -----------------------------
training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=args.epochs,
    save_strategy="epoch",
    fp16=True,
    logging_dir=f"{args.output_dir}/logs",
    report_to="none"
)

# -----------------------------
# Initialize Trainer
# -----------------------------
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    peft_config=peft_config,
    tokenizer=tokenizer,
    args=training_args,
    dataset_text_field=args.prompt_field
)

# -----------------------------
# Train model
# -----------------------------
trainer.train()

# -----------------------------
# Save final model
# -----------------------------
print(f"Model saved to {args.output_dir}")
