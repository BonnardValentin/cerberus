#!/usr/bin/env python3

import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


logger.info("Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.pad_token = tokenizer.eos_token
logger.info("Tokenizer loaded successfully")


logger.info("Loading dataset")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir="./datasets")
logger.info("Dataset loaded successfully")



def preprocess_function(examples):
    inputs = tokenizer(
        examples["text"], truncation=True, padding="max_length", max_length=512
    )
    inputs["labels"] = inputs["input_ids"] 
    return {key: torch.tensor(value, dtype=torch.long) for key, value in inputs.items()}


logger.info("Tokenizing dataset")
tokenized_datasets = dataset.map(preprocess_function, batched=True)
tokenized_datasets.set_format(
    "torch", columns=["input_ids", "attention_mask", "labels"]
)
logger.info("Dataset tokenized successfully")

# Define the model
logger.info("Loading model")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
logger.info("Model loaded successfully")

# Define training arguments
logger.info("Setting up training arguments")
training_args = TrainingArguments(
    output_dir="./text-generation/results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    logging_dir="./text-generation/logs",
    logging_strategy="epoch",
    save_total_limit=2,
)
logger.info("Training arguments configured")

# Initialize Trainer
logger.info("Initializing the Trainer")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)
logger.info("Trainer initialized successfully")

# Train the model
logger.info("Starting training")
trainer.train()
logger.info("Training completed successfully")

# Save the fine-tuned model
logger.info("Saving the fine-tuned model and tokenizer")
model.save_pretrained("./text-generation/models/fine-tuned")
tokenizer.save_pretrained("./text-generation/models/fine-tuned")
logger.info("Model and tokenizer saved successfully")
