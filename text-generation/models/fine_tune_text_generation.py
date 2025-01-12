from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset

# Load the WikiText dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-small")


# Tokenize the dataset
def preprocess_function(examples):
    return tokenizer(
        examples["text"], truncation=True, padding="max_length", max_length=512
    )


tokenized_datasets = dataset.map(preprocess_function, batched=True)
tokenized_datasets.set_format("torch", columns=["input_ids"])

# Define the model
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-small")

# Training arguments
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

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./text-generation/models/fine-tuned")
tokenizer.save_pretrained("./text-generation/models/fine-tuned")
