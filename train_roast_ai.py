# ==============================
# train_roast_ai.py
# Fine-tune a funny roast generator model using PEFT (LoRA)
# ==============================

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
import torch

# ------------------------------
# 1. SETTINGS
# ------------------------------
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"   # base model
dataset_path = "roasts_train.jsonl"                  # your dataset file
output_dir = "./roast-ai-lora"                       # where to save model

# ------------------------------
# 2. LOAD DATASET
# ------------------------------
print("🔹 Loading dataset...")
dataset = load_dataset("json", data_files=dataset_path)
dataset["train"] = dataset["train"].select(range(1000))  
print(f"✅ Loaded {len(dataset['train'])} examples.")

# ------------------------------
# 3. LOAD TOKENIZER + BASE MODEL
# ------------------------------
print("🔹 Loading base model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
     torch_dtype=torch.float32,  # normal CPU precision
    device_map=None 
)

# ------------------------------
# 4. PREPARE PEFT (LoRA) CONFIG
# ------------------------------
print("🔹 Applying LoRA configuration (CPU mode)...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    use_rslora=False,   # disables some heavy GPU ops
    init_lora_weights="gaussian"  # safer for CPU
)

# Important: disable bitsandbytes loading on CPU
import os
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["BITSANDBYTES_OVERRIDE"] = "0"

model = get_peft_model(model, lora_config)

# ------------------------------
# 5. PREPARE DATASET FOR TRAINING
# ------------------------------
def format_example(example):
    prompt = f"{example['instruction']}\n"
    response = example["response"]
    return {"input_text": prompt, "labels": response}

dataset = dataset.map(format_example)

def tokenize_function(example):
    result = tokenizer(
        example["input_text"],
        text_target=example["labels"],
        truncation=True,
        max_length = 128  ,
        padding="max_length"
    )
    return result

print("🔹 Tokenizing dataset (this may take a few minutes)...")
tokenized_ds = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)

# ------------------------------
# 6. TRAINING SETUP
# ------------------------------
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=False,
    dataloader_pin_memory=False,
    logging_steps=20,
    save_strategy="no",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"]
)

# ------------------------------
# 7. TRAIN MODEL
# ------------------------------
print("🚀 Starting training...")
trainer.train()
print("✅ Training complete!")

# ------------------------------
# 8. SAVE MODEL + TOKENIZER
# ------------------------------
print("💾 Saving model...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"✅ Model saved to {output_dir}")

# ------------------------------
# 9. OPTIONAL: QUICK TEST
# ------------------------------
# from transformers import pipeline

# print("\n🤖 Testing the model...\n")
# pipe = pipeline("text-generation", model=output_dir, tokenizer=output_dir)

# prompt = "Roast a person who still uses Internet Explorer"
# output = pipe(prompt, max_new_tokens=60)[0]["generated_text"]

# print("🔥 Example Roast:")
# print(output)
