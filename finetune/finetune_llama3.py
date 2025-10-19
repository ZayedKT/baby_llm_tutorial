import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
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
# Load tokenizer
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

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
# BitsAndBytes 8-bit config
# -----------------------------
bnb_config = None
if args.load_in_8bit:
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )

# -----------------------------
# Load model with 8-bit + LoRA support
# -----------------------------
model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True,
    use_auth_token=True
)
model.config.use_cache = False

# -----------------------------
# Tokenize function
# -----------------------------
def tokenize(batch):
    if "text" in batch:
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512)
    else:
        raise ValueError("Dataset must contain a 'text' field for fine-tuning.")

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
    dataset_text_field="text"
)

# -----------------------------
# Train model
# -----------------------------
trainer.train()

# Save final model
print(f"Model saved to {args.output_dir}")
