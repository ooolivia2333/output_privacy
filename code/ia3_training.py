import os
from transformers import Trainer, TrainingArguments, TrainerCallback, DistilBertForSequenceClassification
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from data_preparation import load_and_prepare_dataset, modify_samples, verify_indices, data_collator, data_collator_with_idx_text
from plot_gradients import CaptureGradientsCallback, plot_gradient_norms, GradientCaptureCallback, plot_dp_gradient_norms
import time
from config import device
import numpy as np
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoraConfig,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
    IA3Config,
)

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


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    print({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def load_model_from_epoch(epoch, model_type="distilbert-base-uncased-ia3", base_model_type="distilbert-base-uncased"):
    checkpoint_dir = f'{model_type}_model_checkpoint_epoch_{epoch}'
    
    ia3_config = IA3Config(task_type="SEQ_CLS", inference_mode=False, target_modules=['q_lin', 'v_lin', 'out_lin'], feedforward_modules=['out_lin'])

    model = DistilBertForSequenceClassification.from_pretrained(base_model_type, num_labels=2)
    ia3_model = get_peft_model(model, ia3_config)
    
    # Load the adapter weights
    adapter_weights = torch.load(os.path.join(checkpoint_dir, "adapter_model.bin"))
    ia3_model.load_state_dict(adapter_weights, strict=False)

    # Load the full model weights (including classification layer)
    model_state = torch.load(os.path.join(checkpoint_dir, "model_state.bin"))
    ia3_model.load_state_dict(model_state, strict=False)

    ia3_model.to(device)
    return ia3_model

class TrainingTimeMemoryCallback(TrainerCallback):
    def __init__(self, model_type):
        self.model_type = model_type
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        torch.cuda.reset_peak_memory_stats(device)
        self.max_memory_allocated = 0

    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.start_time
        max_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)  # Convert to GB
        
        # Print the information
        print(f"Total training time: {total_time:.2f} seconds")
        print(f"Max memory allocated: {max_memory_allocated:.2f} GB")
        
        # Save the information to a file
        checkpoint_dir = f'{self.model_type}_checkpoint_final'
        os.makedirs(checkpoint_dir, exist_ok=True)
        training_info_file = os.path.join(checkpoint_dir, "training_info.txt")
        with open(training_info_file, 'w') as f:
            f.write(f"Total training time: {total_time:.2f} seconds\n")
            f.write(f"Max memory allocated: {max_memory_allocated:.2f} GB\n")

class SaveModelCallback(TrainerCallback):
    def __init__(self, model_type):
        self.model_type = model_type

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs['model']
        checkpoint_dir = f'{self.model_type}_model_checkpoint_epoch_{state.epoch}'
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model_state.bin"))

def train_ia3_model(train_dataset, eval_dataset, tokenizer, training_args, device, model_type="distilbert-base-uncased"):
    ia3_config = IA3Config(task_type="SEQ_CLS", inference_mode=False, target_modules=['q_lin', 'v_lin', 'out_lin'], feedforward_modules=['out_lin'])
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    model = get_peft_model(model, ia3_config)
    
    model.to(device)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset.remove_columns(["idx", "text"]),
        eval_dataset=eval_dataset.remove_columns(["idx", "text"]),
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.add_callback(SaveModelCallback(model_type))
    trainer.add_callback(TrainingTimeMemoryCallback(model_type))
    gradient_capture_callback=GradientCaptureCallback()
    trainer.add_callback(gradient_capture_callback)
    trainer.train()
    
    plot_gradient_norms(gradient_capture_callback.gradient_stats, ['attention'], model_type="ia3")
