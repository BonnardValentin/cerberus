from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset

# Load the CNN/DailyMail dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")
tokenizer = AutoTokenizer.from_pretrained("t5-small")


# Preprocess the data
def preprocess_function(examples):
    inputs = [doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["highlights"],
            max_length=128,
            truncation=True
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Define the model
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

# Training arguments
training_args = TrainingArguments(
    output_dir="./summarization/results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    logging_dir="./summarization/logs",
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
model.save_pretrained("./summarization/models/fine-tuned")
tokenizer.save_pretrained("./summarization/models/fine-tuned")
