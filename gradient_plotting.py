import os
import numpy as np
import matplotlib.pyplot as plt

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
