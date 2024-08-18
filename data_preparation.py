import os
import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import DistilBertTokenizerFast
import torch

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

def load_and_prepare_dataset(path, test_size=0.15, seed=42):
    dataset = load_dataset(path)
    datasets = DatasetDict({
        'train': subset_train['train'],
        'test': subset_test['test']
    })
    return datasets

def modify_samples(dataset, num_samples=5):
    num_samples = min(num_samples, len(dataset))
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    def flip_label(example, indices_set):
        if int(example['idx']) in indices_set:
            print(f"Index: {int(example['idx'])} | Text: {' '.join(example['text'].split()[:15])}... | Original Label : {example['labels']}")
            example['labels'] = torch.tensor(1) - example['labels']
        return example
    indexed_dataset = dataset.map(lambda example, idx: {"idx": idx}, with_indices=True)
    indices_set = set(indices)
    modified_dataset = indexed_dataset.map(lambda x: flip_label(x, indices_set))
    return indices, modified_dataset

def verify_indices(original_dataset, modified_indices):
    for idx in modified_indices:
        idx = int(idx)
        original_text = original_dataset[idx]['text']
        label = original_dataset[idx]['labels']
        print(f"Index: {idx} | Text: {' '.join(original_text.split()[:15])}... | Label : {label}")

def data_collator(batch):
    return {
        'input_ids': torch.stack([item['input_ids'].clone().detach() for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'].clone().detach() for item in batch]),
        'labels': torch.tensor([item['labels'] for item in batch])
    }

def data_collator_with_idx_text(batch):
    return {
        'input_ids': torch.stack([item['input_ids'].clone().detach() for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'].clone().detach() for item in batch]),
        'labels': torch.tensor([item['labels'] for item in batch]),
        'idx': torch.tensor([item['idx'] for item in batch]),
        'text': [item['text'] for item in batch]
    }
