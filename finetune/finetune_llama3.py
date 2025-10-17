import argparse
import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
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
args = parser.parse_args()

# -----------------------------
# Load dataset
# -----------------------------
dataset = load_dataset("json", data_files={"train": args.train_file, "validation": args.dev_file})

# -----------------------------
# Load tokenizer and model
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    device_map="auto",
    load_in_8bit=args.load_in_8bit
)

# -----------------------------
# LoRA configuration
# -----------------------------
if args.use_lora:
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
else:
    peft_config = None

# -----------------------------
# Tokenize function
# -----------------------------
def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512)

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
    logging_dir=f"{args.output_dir}/logs"
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
    dataset_text_field="text"
)

# -----------------------------
# Train model
# -----------------------------
trainer.train()

# Save final model
trainer.model.save_pretrained(args.output_dir)
print(f"Model saved to {args.output_dir}")
