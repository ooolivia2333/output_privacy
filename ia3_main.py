import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict
from transformers import set_seed, DistilBertTokenizerFast, TrainingArguments
from config import device, seed, ia3_num_epochs, ia3_train_batch_size, ia3_eval_batch_size, ia3_lr
from data_preparation import load_and_prepare_dataset, modify_samples, verify_indices, data_collator, data_collator_with_idx_text
from ia3_training import train_ia3_model, compute_metrics, SaveModelCallback, load_model_from_epoch
from mia import perform_mia_and_log_results, collect_logits_labels_losses, evaluate, calculate_metrics_on_modified_samples
from gradient_plotting import plot_gradient_norms

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import transformers
transformers.set_seed(42)
import tensorflow as tf
import numpy as np
import random

def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

# Example usage:
set_seed(42)

# Load dataset
datasets = load_dataset("imdb")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Tokenize dataset
datasets = datasets.map(lambda x: {
    "input_ids": tokenizer(x["text"], truncation=True, padding="max_length", max_length=256)["input_ids"], 
    "attention_mask": tokenizer(x["text"], truncation=True, padding="max_length", max_length=256)["attention_mask"], 
    "labels": x["label"], 
    "text": x["text"]
}, batched=True)

# Add 'idx' column
datasets = datasets.map(lambda example, idx: {"idx": idx}, with_indices=True)
datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "idx", "text"])

# Load QNLI dataset
datasets = load_dataset("glue", "qnli")

# Initialize the tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
# Randomly select half of the dataset indices
train_dataset = datasets['train']
selected_indices = random.sample(range(len(train_dataset)), len(train_dataset) // 2)
selected_train_dataset = train_dataset.select(selected_indices)
datasets["train"] = selected_train_dataset

# Tokenize the dataset
def preprocess_function(examples):
    combined_text = [q + " " + s for q, s in zip(examples['question'], examples['sentence'])]
    tokenized = tokenizer(
        examples['question'],
        examples['sentence'],
        truncation=True,
        padding='max_length',
        max_length=256
    )
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": examples["label"],
        "text": combined_text
    }

datasets = datasets.map(preprocess_function, batched=True)

# Add 'idx' column
datasets = datasets.map(lambda example, idx: {"idx": idx}, with_indices=True)

# Set the format for PyTorch
datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "idx", "text"])

# Modify samples
train_indices, modified_train_dataset = modify_samples(datasets["train"], num_samples=30)
test_indices, modified_test_dataset = modify_samples(datasets["test"], num_samples=30)

# Verify original and modified indices
print("Verifying original Train Indices:")
verify_indices(datasets["train"], train_indices)
print("Verifying original Test Indices:")
verify_indices(datasets["test"], test_indices)
print("Verifying modified Train Indices:")
verify_indices(modified_train_dataset, train_indices)
print("Verifying modified Test Indices:")
verify_indices(modified_test_dataset, test_indices)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=ia3_num_epochs,
    per_device_train_batch_size=ia3_train_batch_size,
    per_device_eval_batch_size=ia3_eval_batch_size,
    evaluation_strategy="epoch",
    optim='adamw_hf',
    learning_rate=ia3_lr,
    report_to=["none"],
    seed=seed
)

# Train model with standard dataset
print("Training model with standard dataset")
train_ia3_model(datasets["train"], datasets["test"], tokenizer, training_args, device, model_type="distilbert-base-uncased-ia3")

# Load models and perform MIA
train_loader = DataLoader(datasets["train"], shuffle=False, batch_size=32, collate_fn=data_collator_with_idx_text)
test_loader = DataLoader(datasets["test"], shuffle=False, batch_size=64, collate_fn=data_collator_with_idx_text)

print("attack model without modified dataset")
for epoch in [1.0, 3.0]:
    model = load_model_from_epoch(epoch, model_type="QNLI-distilbert-base-uncased-ia3")
    accuracy, precision, recall, f1 = evaluate(model, test_loader)
        
    # Calculate metrics on the original training samples
    train_accuracy, train_loss = calculate_metrics_on_modified_samples(model, datasets["train"], train_indices, torch.nn.CrossEntropyLoss(), device)
    print(f"original Training Samples - Accuracy: {train_accuracy}, Loss: {train_loss}")
    # Calculate metrics on the original testing samples
    test_accuracy, test_loss = calculate_metrics_on_modified_samples(model, datasets["test"], test_indices, torch.nn.CrossEntropyLoss(), device)
    print(f"original Testing Samples - Accuracy: {test_accuracy}, Loss: {test_loss}")

    log_file_name = "model_performance.log"
    with open(log_file_name, "a") as log_file:
        log_file.write(f"Epoch {epoch}\n")
        log_file.write("ia3 model non-dp trained with not modified dataset\n")
        log_file.write(f"Accuracy: {accuracy}\n")
        log_file.write(f"Precision: {precision}\n")
        log_file.write(f"Recall: {precision}\n")
        log_file.write(f"f1: {f1}\n")
        log_file.write("on modified subset: \n")
        log_file.write(f"original Training Samples - Accuracy: {train_accuracy}, Loss: {train_loss}\n")
        log_file.write(f"original Testing Samples - Accuracy: {test_accuracy}, Loss: {test_loss}\n")
        log_file.write(f"\n")
    
    train_logits, train_labels, train_losses, _ = collect_logits_labels_losses(train_loader, model, torch.nn.CrossEntropyLoss(reduction="none"), train_indices, device)
    test_logits, test_labels, test_losses, _ = collect_logits_labels_losses(test_loader, model, torch.nn.CrossEntropyLoss(reduction="none"), test_indices, device)
    perform_mia_and_log_results(model, epoch, train_logits, train_labels, train_losses, test_logits, test_labels, test_losses, train_indices, test_indices, datasets, model_type="QNLI-distilbert-base-uncased-ia3")

# Train model with modified dataset
print("Training model with modified dataset")
train_ia3_model(modified_train_dataset, modified_test_dataset, tokenizer, training_args, device, model_type="QNLI-distilbert-base-uncased-ia3-modified")

# Load models and perform MIA with modified dataset
modified_train_loader = DataLoader(modified_train_dataset, shuffle=False, batch_size=32, collate_fn=data_collator_with_idx_text)
modified_test_loader = DataLoader(modified_test_dataset, shuffle=False, batch_size=64, collate_fn=data_collator_with_idx_text)

print("attack model without modified dataset")
for epoch in [1.0, 3.0]:
    model = load_model_from_epoch(epoch, model_type="QNLI-distilbert-base-uncased-ia3-modified")
    accuracy, precision, recall, f1 = evaluate(model, modified_test_loader)
        
    # Calculate metrics on the modified training samples
    train_accuracy, train_loss = calculate_metrics_on_modified_samples(model, modified_train_dataset, train_indices, torch.nn.CrossEntropyLoss(), device)
    print(f"Modified Training Samples - Accuracy: {train_accuracy}, Loss: {train_loss}")
    # Calculate metrics on the modified testing samples
    test_accuracy, test_loss = calculate_metrics_on_modified_samples(model, modified_test_dataset, test_indices, torch.nn.CrossEntropyLoss(), device)
    print(f"Modified Testing Samples - Accuracy: {test_accuracy}, Loss: {test_loss}")

    log_file_name = "model_performance.log"
    with open(log_file_name, "a") as log_file:
        log_file.write(f"Epoch {epoch}\n")
        log_file.write("ia3 model not dp trained with modified dataset\n")
        log_file.write(f"Accuracy: {accuracy}\n")
        log_file.write(f"Precision: {precision}\n")
        log_file.write(f"Recall: {precision}\n")
        log_file.write(f"f1: {f1}\n")
        log_file.write("on modified subset: \n")
        log_file.write(f"Modified Training Samples - Accuracy: {train_accuracy}, Loss: {train_loss}\n")
        log_file.write(f"Modified Testing Samples - Accuracy: {test_accuracy}, Loss: {test_loss}\n")
        log_file.write(f"\n")

    train_logits, train_labels, train_losses, _ = collect_logits_labels_losses(modified_train_loader, model, torch.nn.CrossEntropyLoss(reduction="none"), train_indices, device)
    test_logits, test_labels, test_losses, _ = collect_logits_labels_losses(modified_test_loader, model, torch.nn.CrossEntropyLoss(reduction="none"), test_indices, device)
    perform_mia_and_log_results(model, epoch, train_logits, train_labels, train_losses, test_logits, test_labels, test_losses, train_indices, test_indices, datasets, model_type="QNLI-distilbert-base-uncased-ia3-modified")
