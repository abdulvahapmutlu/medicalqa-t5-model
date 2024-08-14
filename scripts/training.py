import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd

# Load the dataset
data_path = "/content/drive/My Drive/train.csv"
df = pd.read_csv(data_path)

# Initialize the tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Tokenize the dataset
def tokenize_data(examples):
    inputs = [q for q in examples['Question']]
    targets = [a for a in examples['Answer']]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding='max_length')

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=512, truncation=True, padding='max_length')

    model_inputs['labels'] = labels['input_ids']
    return model_inputs

# Create and tokenize the dataset
dataset = Dataset.from_pandas(df)
tokenized_dataset = dataset.map(tokenize_data, batched=True)

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=8,
    weight_decay=0.01,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
)

# Start training
trainer.train()
