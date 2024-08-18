import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_labels = 2

## standard
standard_num_epochs = 3
standard_train_batch_size = 32
standard_eval_batch_size = 64
standard_lr = 5e-5
seed = 42

## lora
lora_num_epochs = 3
lora_train_batch_size=32
lora_eval_batch_size=64
lora_lr = 3e-4

## adapter
bottleneck_size=512
adapter_num_epochs=5
adapter_train_batch_size=32
adapter_eval_batch_size=64
adapter_lr = 1e-4

def make_adapter(in_dim, bottleneck_dim, out_dim):
    adapter_layers = torch.nn.Sequential(
        torch.nn.Linear(in_dim, bottleneck_dim),
        torch.nn.GELU(),
        torch.nn.Linear(bottleneck_dim, out_dim),
    )
    return adapter_layers

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

## IA3
ia3_num_epochs=3
ia3_train_batch_size=32
ia3_eval_batch_size=64
ia3_lr=7e-3