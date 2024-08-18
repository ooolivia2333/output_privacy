import os
from transformers import Trainer, TrainingArguments, TrainerCallback, DistilBertForSequenceClassification
import torch
from peft import get_peft_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time
from config import device, make_adapter, count_parameters, bottleneck_size, num_labels
import tempfile
from plot_gradients import CaptureGradientsCallback, plot_gradient_norms, GradientCaptureCallback, plot_dp_gradient_norms

import transformers
transformers.set_seed(42)
import tensorflow as tf
import numpy as np
import random
from data_preparation import load_and_prepare_dataset, modify_samples, verify_indices, data_collator, data_collator_with_idx_text
from dp_transformers.dp_utils import OpacusDPTrainer, DPCallback

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

def load_model_from_epoch(epoch, model_type="distilbert-base-uncased-adapter", base_model_type="distilbert-base-uncased"):
    checkpoint_dir = f'{model_type}_model_checkpoint_epoch_{epoch}'
    
    with tempfile.TemporaryDirectory() as temp_dir:
        adapter_model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", 
            num_labels=num_labels, 
            cache_dir=temp_dir
        )
        
    total_size = 0

    for block_idx in range(6):

        ###################################################
        # insert 1st adapter layer into transformer block
        ###################################################

        orig_layer_1 = adapter_model.distilbert.transformer.layer[block_idx].attention.out_lin

        adapter_layers_1 = make_adapter(
            in_dim=orig_layer_1.out_features,
            bottleneck_dim=bottleneck_size,
            out_dim=orig_layer_1.out_features)

        new_1 = torch.nn.Sequential(orig_layer_1, *adapter_layers_1)
        adapter_model.distilbert.transformer.layer[block_idx].attention.out_lin = new_1

        total_size += count_parameters(adapter_layers_1)

        ###################################################
        # insert 2nd adapter layer into transformer block
        ###################################################

        orig_layer_2 = adapter_model.distilbert.transformer.layer[block_idx].ffn.lin2

        adapter_layers_2 = make_adapter(
            in_dim=orig_layer_2.out_features,
            bottleneck_dim=bottleneck_size,
            out_dim=orig_layer_2.out_features)

        new_2 = torch.nn.Sequential(orig_layer_2, *adapter_layers_2)
        adapter_model.distilbert.transformer.layer[block_idx].ffn.lin2 = new_2

        total_size += count_parameters(adapter_layers_2)


    print("Number of adapter parameters added:", total_size)

    # Load the full model weights (including classification layer)
    model_state = torch.load(os.path.join(checkpoint_dir, "model_state.bin"))
    adapter_model.load_state_dict(model_state, strict=False)

    adapter_model.to(device)
    return adapter_model

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

def train_adapter_model(train_dataset, eval_dataset, tokenizer, training_args, device, model_type="distilbert-base-uncased"):
    with tempfile.TemporaryDirectory() as temp_dir:
        adapter_model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", 
            num_labels=num_labels, 
            cache_dir=temp_dir
        )
    # frozen other parts
    
    total_size = 0
    for param in adapter_model.parameters():
        param.requires_grad = False

    for block_idx in range(6):

        ###################################################
        # insert 1st adapter layer into transformer block
        ###################################################

        orig_layer_1 = adapter_model.distilbert.transformer.layer[block_idx].attention.out_lin

        adapter_layers_1 = make_adapter(
            in_dim=orig_layer_1.out_features,
            bottleneck_dim=bottleneck_size,
            out_dim=orig_layer_1.out_features)

        new_1 = torch.nn.Sequential(orig_layer_1, *adapter_layers_1)
        adapter_model.distilbert.transformer.layer[block_idx].attention.out_lin = new_1

        total_size += count_parameters(adapter_layers_1)

        ###################################################
        # insert 2nd adapter layer into transformer block
        ###################################################

        orig_layer_2 = adapter_model.distilbert.transformer.layer[block_idx].ffn.lin2

        adapter_layers_2 = make_adapter(
            in_dim=orig_layer_2.out_features,
            bottleneck_dim=bottleneck_size,
            out_dim=orig_layer_2.out_features)

        new_2 = torch.nn.Sequential(orig_layer_2, *adapter_layers_2)
        adapter_model.distilbert.transformer.layer[block_idx].ffn.lin2 = new_2

        total_size += count_parameters(adapter_layers_2)

    print("Number of adapter parameters added:", total_size)
    # Print the number of trainable parameters
    def count_trainable_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(count_trainable_parameters(adapter_model))

    adapter_model.to(device)
    trainer = Trainer(
        model=adapter_model,
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
    
    plot_gradient_norms(gradient_capture_callback.gradient_stats, ['attention'], model_type="adapter")

class SaveDPModelCallback(TrainerCallback):
    def __init__(self, model_type):
        self.model_type = model_type

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs['model']
        checkpoint_dir = f'{self.model_type}_model_checkpoint_epoch_{state.epoch}'
        os.makedirs(checkpoint_dir, exist_ok=True)
        model._module.save_pretrained(checkpoint_dir)
        torch.save(model._module.state_dict(), os.path.join(checkpoint_dir, "model_state.bin"))
        
class ManualMetricsCallback(transformers.TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        print("Calculating manual metrics...")
        eval_dataloader = kwargs["eval_dataloader"]
        model = kwargs["model"]
        
        all_preds = []
        all_labels = []
        model.eval()
        
        for batch in eval_dataloader:
            with torch.no_grad():
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                preds = torch.argmax(logits, dim=-1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
        
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        
        log_file_name = "model_performance.log"
        with open(log_file_name, "a") as log_file:
            log_file.write(f"Accuracy: {accuracy}\n")
            log_file.write(f"Precision: {precision}\n")
            log_file.write(f"Recall: {precision}\n")
        
def dp_train_adapter_model(train_dataset, eval_dataset, tokenizer, training_args, privacy_args, device, model_type="dp-distilbert-base-uncased"):
    with tempfile.TemporaryDirectory() as temp_dir:
        adapter_model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", 
            num_labels=num_labels, 
            cache_dir=temp_dir
        )
    # frozen other parts
    total_size = 0
    for param in adapter_model.parameters():
        param.requires_grad = False

    for block_idx in range(6):

        ###################################################
        # insert 1st adapter layer into transformer block
        ###################################################

        orig_layer_1 = adapter_model.distilbert.transformer.layer[block_idx].attention.out_lin

        adapter_layers_1 = make_adapter(
            in_dim=orig_layer_1.out_features,
            bottleneck_dim=bottleneck_size,
            out_dim=orig_layer_1.out_features)

        new_1 = torch.nn.Sequential(orig_layer_1, *adapter_layers_1)
        adapter_model.distilbert.transformer.layer[block_idx].attention.out_lin = new_1

        total_size += count_parameters(adapter_layers_1)

        ###################################################
        # insert 2nd adapter layer into transformer block
        ###################################################

        orig_layer_2 = adapter_model.distilbert.transformer.layer[block_idx].ffn.lin2

        adapter_layers_2 = make_adapter(
            in_dim=orig_layer_2.out_features,
            bottleneck_dim=bottleneck_size,
            out_dim=orig_layer_2.out_features)

        new_2 = torch.nn.Sequential(orig_layer_2, *adapter_layers_2)
        adapter_model.distilbert.transformer.layer[block_idx].ffn.lin2 = new_2

        total_size += count_parameters(adapter_layers_2)

    print("Number of adapter parameters added:", total_size)
    # Print the number of trainable parameters
    def count_trainable_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(count_trainable_parameters(adapter_model))

    adapter_model.to(device)
    
    trainer = OpacusDPTrainer(
        model=adapter_model,
        args=training_args,
        train_dataset=train_dataset.remove_columns(["idx", "text"]),
        eval_dataset=eval_dataset.remove_columns(["idx", "text"]),
        data_collator=data_collator,
        privacy_args = privacy_args
    )
    trainer.add_callback(ManualMetricsCallback())
    trainer.add_callback(SaveDPModelCallback(model_type))
    trainer.add_callback(TrainingTimeMemoryCallback(model_type))
    gradient_capture_callback=CaptureGradientsCallback()
    trainer.add_callback(gradient_capture_callback)
    trainer.train()
    
    plot_dp_gradient_norms(gradient_capture_callback.gradient_stats, privacy_args.target_epsilon, ["attention"], model_type="adapter")