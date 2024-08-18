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
from utils import binary_accuracy
import time
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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

# Plotting function
def plot_gradient_norms(gradient_stats, epsilon, type_filters, save_dir='figs', model_type="adapter", end_epoch=None):
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

@dataclass
class ModelArguments:
    model_name: str = field(default="distilbert-base-uncased", metadata={
        "help": "Model name in HuggingFace, e.g. 'distilbert-base-uncased'"
    })
    sequence_len: int = field(default=256, metadata={
        "help": "Maximum sequence length"
    })

@dataclass
class IA3Arguments:
    enable_ia3: bool = field(default=True, metadata={
        "help": "Whether to enable IA3"
    })
    ia3_dim: int = field(default=8, metadata={
        "help": "IA3 dimension"
    })

    def as_peft_config(self) -> IA3Config:
        return IA3Config(
            task_type="SEQ_CLS",
            inference_mode=False,
            target_modules=['q_lin', 'v_lin', 'out_lin'],
            feedforward_modules=['out_lin']
        )

@dataclass
class PrivacyArguments:
    target_epsilon: float = field(default=4.0, metadata={
        "help": "Target epsilon for differential privacy"
    })
    max_grad_norm: float = field(default=1.0, metadata={
        "help": "Max gradient norm for differential privacy"
    })
    target_delta: float = field(default=4e-5, metadata={
        "help": "Target delta for differential privacy"
    })

@dataclass
class Arguments:
    train: dp_transformers.TrainingArguments
    privacy: dp_transformers.PrivacyArguments
    model: ModelArguments
    ia3: IA3Arguments

class Args:
    batch_size = 64
    epochs = 3
    lr = 5e-4
    sigma = None
    max_per_sample_grad_norm = 1.5
    delta = 1e-5
    max_sequence_length = 256
    device = "cuda"
    save_model = False
    disable_dp = False
    secure_rng = False
    data_root = "../imdb"
    workers = 2
    accumulation_steps = 1
    log_interval = 100
    epsilon = 8.0

args = Args()

gradient_stats = {}
log_file_name = f"test_adapter_epsilon_{args.epsilon}_training.log"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def make_adapter(in_dim, bottleneck_dim, out_dim):
    adapter_layers = torch.nn.Sequential(
        torch.nn.Linear(in_dim, bottleneck_dim),
        torch.nn.GELU(),
        torch.nn.Linear(bottleneck_dim, out_dim),
    )
    return adapter_layers

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Function to collect logits, labels, and losses
def collect_logits_labels_losses(loader, model, loss_fn):
    logits_list = []
    labels_list = []
    losses_list = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            logits_list.extend(logits.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
            losses_list.extend(loss.cpu().numpy())  # Ensure each loss is recorded per sample
    logits_array = np.array(logits_list)
    labels_array = np.array(labels_list)
    losses_array = np.array(losses_list)

    print(f"logits shape: {logits_array.shape}")
    print(f"labels shape: {labels_array.shape}")
    print(f"losses shape: {losses_array.shape}")

    return logits_array, labels_array, losses_array

# Function to run MIA and log results
def run_mia_and_log(logits_train, labels_train, losses_train, logits_test, labels_test, losses_test, epoch, fig_name, log_file_name):
    attack_input = AttackInputData(
        logits_train=logits_train,
        logits_test=logits_test,
        loss_train=losses_train,
        loss_test=losses_test,
        labels_train=labels_train,
        labels_test=labels_test
    )
    slicing_spec = SlicingSpec(entire_dataset=True, by_class=True, by_classification_correctness=True)
    attack_types = [
        AttackType.THRESHOLD_ATTACK,
        AttackType.LOGISTIC_REGRESSION,
        AttackType.MULTI_LAYERED_PERCEPTRON,
        AttackType.RANDOM_FOREST,
        AttackType.K_NEAREST_NEIGHBORS
    ]
    attack_results = mia.run_attacks(attack_input=attack_input, slicing_spec=slicing_spec, attack_types=attack_types)

    with open(log_file_name, "w") as log_file:
        log_file.write(f"\nMembership inference attack results after epoch {epoch}:\n")
        log_file.write(f"{attack_results.summary(by_slices=True)}\n")

        pd.set_option("display.max_columns", None)
        df = attack_results.calculate_pd_dataframe()
        log_file.write(f"{df}\n")

    max_auc_attacker = attack_results.get_result_with_max_auc()
    figure = plotting.plot_roc_curve(max_auc_attacker.roc_curve)
    figure.savefig(fig_name)

class SaveModelCallback(TrainerCallback):
    def __init__(self, privacy_args):
        self.privacy_args = privacy_args

    def on_epoch_end(self, args, state, control, **kwargs):
        print("SaveModelCallback: on_epoch_end called")
        model = kwargs['model']
        checkpoint_dir = f'dp_adapter_model_epsilon_{self.privacy_args.target_epsilon}_checkpoint_epoch_{state.epoch}'
        os.makedirs(checkpoint_dir, exist_ok=True)
        model._module.save_pretrained(checkpoint_dir)
        # Save the entire state dictionary including classification layers
        torch.save(model._module.state_dict(), os.path.join(checkpoint_dir, "model_state.bin"))

class ManualMetricsCallback(transformers.TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        logger.info("Calculating manual metrics...")
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
        
        logger.info(f"Accuracy: {accuracy}")
        logger.info(f"Precision: {precision}")
        logger.info(f"Recall: {recall}")
        logger.info(f"F1 Score: {f1}")

class CaptureGradientsCallback(TrainerCallback):
    def on_optimizer_step(self, args, state, control, **kwargs):
        model = kwargs['model']
        capture_per_sample_gradients(model, gradient_stats, current_epoch=state.epoch, phase="before_clipping")
        # Assuming the clipping happens here
        capture_per_sample_gradients_after_clipping(model, gradient_stats, current_epoch=state.epoch, max_norm=args.max_grad_norm)

def evaluate(args, model, test_loader):
    all_preds = []
    all_labels = []
    model.eval()

    for batch in test_loader:
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

    logger.info(f"Accuracy: {accuracy}")
    logger.info(f"Precision: {precision}")
    logger.info(f"Recall: {recall}")
    logger.info(f"F1 Score: {f1}")

# Ensure each element in the batch is of equal size
def padded_collate(batch, padding_idx=0):
    x = pad_sequence(
        [torch.tensor(elem["input_ids"]) for elem in batch],
        batch_first=True,
        padding_value=padding_idx,
    )
    attention_mask = pad_sequence(
        [torch.tensor(elem["attention_mask"]) for elem in batch],
        batch_first=True,
        padding_value=0,
    )
    y = torch.stack([torch.tensor(elem["labels"]) for elem in batch]).long()
    return x, attention_mask, y

def main(args: Arguments):
    transformers.set_seed(args.train.seed)
    import tensorflow as tf
    import numpy as np
    import random

    def set_seed(seed):
        np.random.seed(seed)
        tf.random.set_seed(seed)
        random.seed(seed)

    # Example usage:
    set_seed(42)

    # Load model and tokenizer
    model = DistilBertForSequenceClassification.from_pretrained(args.model.model_name)
    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model.model_name)

    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Load and preprocess dataset
    dataset = load_dataset("imdb")
    dataset = dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=args.model.sequence_len), batched=True)
    # Rename 'label' column to 'labels'
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    print(dataset['train'].features)

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # Integrate IA3 if enabled
    if args.ia3.enable_ia3:
        logger.info("Using LoRA")
        lora_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1, target_modules=['q_lin', 'v_lin'])
        model = get_peft_model(model=model, peft_config=lora_config)
    else:
        logger.info("Not using LoRA")

    model = model.to(args.train.device)
    model.train()

    log_file_name = f"dp_lora_epsilon_{args.privacy.target_epsilon}_training.log"
    # Custom collate function to filter out unwanted keys
    def data_collator(batch):
        return {
            'input_ids': torch.stack([item['input_ids'].clone().detach() for item in batch]),
            'attention_mask': torch.stack([item['attention_mask'].clone().detach() for item in batch]),
            'labels': torch.tensor([item['labels'] for item in batch])
        }

    # Track training time and memory
    start_time = time.time()
    torch.cuda.reset_peak_memory_stats(device)
    
    trainer = OpacusDPTrainer(
        model=model,
        args=args.train,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        privacy_args=args.privacy
    )
    trainer.add_callback(ManualMetricsCallback())
    #trainer.add_callback(SaveModelCallback(args.privacy))
    trainer.add_callback(CaptureGradientsCallback())

    trainer.train()
    
    total_time = time.time() - start_time
    max_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 3)  # Convert to GB

    with open(log_file_name, "a") as log_file:
        log_file.write(f"Total training time: {total_time} seconds\n")
        log_file.write(f"Maximum memory allocated: {max_memory} GB\n")
    
    logger.info(f"Total training time: {total_time} seconds")
    logger.info(f"Maximum memory allocated: {max_memory} GB")
    
    plot_gradient_norms(gradient_stats, args.privacy.target_epsilon, ["attention"], model_type="lora")

#     # DataLoaders
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train.per_device_train_batch_size, collate_fn=data_collator)
#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.train.per_device_eval_batch_size, collate_fn=data_collator)
    
#     epochs_to_check = [1.0, 3.0]
#     loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
#     for epoch in epochs_to_check:
#         checkpoint_dir = f'dp_lora_model_epsilon_{args.privacy.target_epsilon}_checkpoint_epoch_{epoch}'
#         # Load the LoRA model
#         base_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
#         lora_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1, target_modules=['q_lin', 'v_lin'])
#         model = get_peft_model(base_model, lora_config)

#         # Load the adapter weights
#         adapter_weights = torch.load(os.path.join(checkpoint_dir, "adapter_model.bin"))
#         model.load_state_dict(adapter_weights, strict=False)

#         # Load the full model weights (including classification layer)
#         model_state = torch.load(os.path.join(checkpoint_dir, "model_state.bin"))
#         model.load_state_dict(model_state, strict=False)

#         model.to(device)

#         # Print accuracy before running MIA
#         evaluate(args, model, test_loader)
#         #print(f"Epoch {epoch} - Accuracy: {accuracy:.4f}")
#         logits_train, labels_train, losses_train = collect_logits_labels_losses(train_loader, model, loss_fn)
#         logits_test, labels_test, losses_test = collect_logits_labels_losses(test_loader, model, loss_fn)
#         run_mia_and_log(
#             logits_train,
#             labels_train,
#             losses_train,
#             logits_test,
#             labels_test,
#             losses_test,
#             epoch,
#             fig_name=f'roc_curve_dp_lora_{args.privacy.target_epsilon}_epoch_{epoch}.png',
#             log_file_name=f'mia_results_dp_lora_{args.privacy.target_epsilon}_epoch_{epoch}.log'
#         )

    eps_prv = trainer.get_prv_epsilon()
    eps_rdp = trainer.get_rdp_epsilon()
    trainer.log({
        "final_epsilon_prv": eps_prv,
        "final_epsilon_rdp": eps_rdp,
    })

    # Plot gradient norms

if __name__ == "__main__":
    # Define TrainingArguments
    training_args = dp_transformers.TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        logging_dir="./logs",
        logging_steps=100,
        learning_rate=9e-4,
        evaluation_strategy="epoch",  # Change to evaluate per epoch
        logging_strategy="steps",
        remove_unused_columns=False,
        disable_tqdm=True,
        optim='adamw_hf',
        dataloader_num_workers=2,
        seed=42,
        report_to=["none"],
    )

    # Define other arguments
    model_args = ModelArguments()
    ia3_args = IA3Arguments()

    # Define PrivacyArguments
    privacy_args = dp_transformers.PrivacyArguments(
        target_epsilon=1.0,
        per_sample_max_grad_norm=1.5,
        target_delta=4e-5
    )

    main(Arguments(train=training_args, privacy=privacy_args, model=model_args, ia3=ia3_args))
