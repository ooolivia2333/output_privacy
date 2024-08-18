import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
import datasets
import transformers
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, Trainer, TrainingArguments, AdamW, TrainerCallback
from datasets import load_dataset
from dp_transformers.dp_utils import OpacusDPTrainer, DPCallback
import dp_transformers
from peft import get_peft_model, IA3Config, LoraConfig
import logging
import sys
from dataclasses import dataclass, field
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import membership_inference_attack as mia
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackInputData, SlicingSpec, AttackType
import pandas as pd
import tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.plotting as plotting
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import time
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# for non dp
class GradientCaptureCallback(TrainerCallback):
    def __init__(self):
        self.gradient_stats = {}

    def on_optimizer_step(self, args, state, control, **kwargs):
        model = kwargs['model'] 

        current_epoch = state.epoch
        if current_epoch not in self.gradient_stats:
            self.gradient_stats[current_epoch] = []

        grad_data = {}
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad = param.grad.detach()
                grad_norm = grad.norm().item()
                grad_data[name] = {'norm': grad_norm}

        self.gradient_stats[current_epoch].append(grad_data)
        
class CaptureGradientsCallback(TrainerCallback):
    def __init__(self):
        self.gradient_stats = {}
        
    def on_optimizer_step(self, args, state, control, **kwargs):
        model = kwargs['model']
        capture_per_sample_gradients(model, self.gradient_stats, current_epoch=state.epoch, phase="before_clipping")
        # Assuming the clipping happens here
        capture_per_sample_gradients_after_clipping(model, self.gradient_stats, current_epoch=state.epoch, max_norm=args.max_grad_norm)
        
        
# Gradient capturing functions
def calc_clipping_factors(norms, max_norm):
    return torch.clamp(max_norm / (norms + 1e-6), max=1.0)

def capture_per_sample_gradients(model, gradient_stats, current_epoch, phase="before_clipping"):
    grad_data = {}
    for name, param in model.named_parameters():
        if param.requires_grad and hasattr(param, 'grad_sample') and param.grad_sample is not None:
            grad_norms = param.grad_sample.view(param.grad_sample.size(0), -1).norm(2, dim=1)
            grad_data[name] = grad_norms.mean().item()

    if current_epoch not in gradient_stats:
        gradient_stats[current_epoch] = {"before_clipping": [], "after_clipping": []}

    gradient_stats[current_epoch][phase].append(grad_data)

def calc_sample_norms(named_params):
    return torch.stack([p.grad_sample.view(len(p.grad_sample), -1).norm(2, dim=1) for _, p in named_params], dim=1).norm(2, dim=1)

def is_close_to_integer(value, target):
    return math.isclose(value, target, abs_tol=1e-9)

def capture_per_sample_gradients_after_clipping(model, gradient_stats, current_epoch, max_norm):
    grad_data = {}
    named_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad and hasattr(param, 'grad_sample') and param.grad_sample is not None]
    all_norms = calc_sample_norms(named_params)
    clipping_factors = calc_clipping_factors(all_norms, max_norm)
    
    for (name, param) in named_params:
        if param.requires_grad and hasattr(param, 'grad_sample') and param.grad_sample is not None:
            grad_norms = (param.grad_sample.view(param.grad_sample.size(0), -1) * clipping_factors.unsqueeze(1).to(param.grad_sample.device)).norm(2, dim=1)
            grad_data[name] = grad_norms.mean().item()

    if current_epoch not in gradient_stats:
        gradient_stats[current_epoch] = {"before_clipping": [], "after_clipping": []}

    gradient_stats[current_epoch]["after_clipping"].append(grad_data)

# Plot gradient norms
def plot_gradient_norms(gradient_stats, type_filters, save_dir='figs', model_type="adapter", end_epoch=None):
    os.makedirs(save_dir, exist_ok=True)

    if end_epoch is None:
        end_epoch = max(gradient_stats.keys())

    start_epochs = [epoch for epoch in gradient_stats.keys() if 0 <= epoch <= 1]
    end_epochs = [epoch for epoch in gradient_stats.keys() if end_epoch - 1 < epoch <= end_epoch]
    start_grads = [gradient_stats[epoch] for epoch in start_epochs]
    end_grads = [gradient_stats[epoch] for epoch in end_epochs]

    for ftype in type_filters:
        layer_names = set()
        for grads in start_grads:
            for grad in grads:
                layer_names.update(grad.keys())

        layer_names = sorted(layer_names)

        filtered_layer_names = [name for name in layer_names if ftype in name]
        
        cleaned_layer_names = [
            name.replace("module.distilbert.transformer.", "")
                .replace("module.base_model.model.distilbert.transformer.", "")
                .replace("_module.base_model.model", "")
                .replace("distilbert.embeddings", "")
                .replace("distilbert.transformer.layer", "")
                .replace("base_model.model..", "")
            for name in filtered_layer_names
        ]

        start_means = [
            np.mean([grad[layer]['norm'] for grads in start_grads for grad in grads if layer in grad])
            for layer in filtered_layer_names
        ]
        end_means = [
            np.mean([grad[layer]['norm'] for grads in end_grads for grad in grads if layer in grad])
            for layer in filtered_layer_names
        ]
        start_stds = [
            np.std([grad[layer]['norm'] for grads in start_grads for grad in grads if layer in grad])
            for layer in filtered_layer_names
        ]
        end_stds = [
            np.std([grad[layer]['norm'] for grads in end_grads for grad in grads if layer in grad])
            for layer in filtered_layer_names
        ]

        width = 0.35
        x = np.arange(len(cleaned_layer_names))

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width/2, start_means, width, label="Start Epoch")
        ax.bar(x + width/2, end_means, width, label="End Epoch")
        ax.set_xlabel("Layers")
        ax.set_ylabel("Gradient Norm Mean")
        ax.set_title(f"Gradient Norm Mean by Layer - {ftype.capitalize()}")
        ax.set_xticks(x)
        ax.set_xticklabels(cleaned_layer_names, rotation=90)
        ax.legend()
        fig.tight_layout()
        file_name = os.path.join(save_dir, f"{model_type}_{ftype}_mean.png")
        fig.savefig(file_name)
        print(f"Gradient norms mean plot saved to {file_name}")

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width/2, start_stds, width, label="Start Epoch")
        ax.bar(x + width/2, end_stds, width, label="End Epoch")
        ax.set_xlabel("Layers")
        ax.set_ylabel("Gradient Norm Std")
        ax.set_title(f"Gradient Norm Std by Layer - {ftype.capitalize()}")
        ax.set_xticks(x)
        ax.set_xticklabels(cleaned_layer_names, rotation=90)
        ax.legend()
        fig.tight_layout()
        file_name = os.path.join(save_dir, f"{model_type}_{ftype}_std.png")
        fig.savefig(file_name)
        print(f"Gradient norms std plot saved to {file_name}")


# Plotting function
def plot_dp_gradient_norms(gradient_stats, epsilon, type_filters, save_dir='figs', model_type="adapter", end_epoch=None):
    # Create the directory if it does not exist
    os.makedirs(save_dir, exist_ok=True)

    if end_epoch is None:
        end_epoch = max(gradient_stats.keys())

    start_epochs = [epoch for epoch in gradient_stats.keys() if 0 <= epoch <= 1]
    end_epochs = [epoch for epoch in gradient_stats.keys() if end_epoch - 1 < epoch <= end_epoch]

    start_grads_after = [grad for epoch in start_epochs for grad in gradient_stats[epoch]['after_clipping']]
    end_grads_after = [grad for epoch in end_epochs for grad in gradient_stats[epoch]['after_clipping']]
    start_grads_before = [grad for epoch in start_epochs for grad in gradient_stats[epoch]['before_clipping']]
    end_grads_before = [grad for epoch in end_epochs for grad in gradient_stats[epoch]['before_clipping']]
    
    for type_filter in type_filters:
        layer_names = set()
        for grads in start_grads_after:
            layer_names.update(grads.keys())

        layer_names = sorted(layer_names)
        layer_names = [name for name in layer_names if type_filter in name]

        cleaned_layer_names = [
            name.replace("module.distilbert.transformer.", "")
                .replace("module.base_model.model.distilbert.transformer.", "")
                .replace("_module.base_model.model", "")
            for name in layer_names
        ]

        # Helper function to calculate means and stds
        def calculate_means_and_stds(grads, layer_names):
            means = [
                np.mean([grad[layer] for grad in grads if layer in grad])
                for layer in layer_names
            ]
            stds = [
                np.std([grad[layer] for grad in grads if layer in grad])
                for layer in layer_names
            ]
            return means, stds

        # Calculate means and stds for start and end epochs after clipping
        start_means_after, start_std_after = calculate_means_and_stds(start_grads_after, layer_names)
        end_means_after, end_std_after = calculate_means_and_stds(end_grads_after, layer_names)

        # Calculate means and stds for start and end epochs before clipping
        start_means_before, start_std_before = calculate_means_and_stds(start_grads_before, layer_names)
        end_means_before, end_std_before = calculate_means_and_stds(end_grads_before, layer_names)

        # Plotting function
        def plot_bars(x, start_vals, end_vals, y_label, title, file_suffix):
            width = 0.35
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(x - width/2, start_vals, width, label="Start Epoch")
            ax.bar(x + width/2, end_vals, width, label="End Epoch")
            ax.set_xlabel("Layers")
            ax.set_ylabel(y_label)
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels(cleaned_layer_names, rotation=90)
            ax.legend()
            fig.tight_layout()
            file_name = os.path.join(save_dir, f"dp_{model_type}_{type_filter}_epsilon_{epsilon}_{file_suffix}.png")
            fig.savefig(file_name)
            print(f"{title} plot saved to {file_name}")

        x = np.arange(len(cleaned_layer_names))

        # Plot mean and std after clipping
        plot_bars(x, start_means_after, end_means_after, "Gradient Norm Mean", "Gradient Norm Mean by Layer - After Clipping", "mean_after_clipping")
        plot_bars(x, start_std_after, end_std_after, "Gradient Norm Std", "Gradient Norm Std by Layer - After Clipping", "std_after_clipping")

        # Plot mean and std before clipping
        plot_bars(x, start_means_before, end_means_before, "Gradient Norm Mean", "Gradient Norm Mean by Layer - Before Clipping", "mean_before_clipping")
        plot_bars(x, start_std_before, end_std_before, "Gradient Norm Std", "Gradient Norm Std by Layer - Before Clipping", "std_before_clipping")