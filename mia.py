import numpy as np
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import membership_inference_attack as mia
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import (
    AttackInputData, SlicingSpec, AttackType, RocCurve
)
import tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.plotting as plotting
import torch
import pandas as pd
from config import device
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
import transformers
transformers.set_seed(42)
import tensorflow as tf
import numpy as np
import random
from sklearn import metrics
from data_preparation import load_and_prepare_dataset, modify_samples, verify_indices, data_collator, data_collator_with_idx_text

def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

# Example usage:
set_seed(42)

def calculate_metrics_on_modified_samples(model, dataset, indices, loss_fn, device):
    # Filter the dataset to get only the modified samples
    indices = [int(idx) for idx in indices]
    modified_samples = [dataset[i] for i in indices]
    
    # Create DataLoader for the modified samples
    modified_loader = DataLoader(modified_samples, batch_size=32, collate_fn=data_collator)
    
    model.eval()
    all_logits = []
    all_labels = []
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for batch in modified_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = loss_fn(logits, labels)

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            total_loss += loss.sum().item()
            total_samples += labels.size(0)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    # Calculate accuracy
    predictions = torch.argmax(all_logits, dim=1)
    accuracy = accuracy_score(all_labels, predictions)

    # Calculate mean loss
    mean_loss = total_loss / total_samples
    
    log_file_name = "model_performance.log"
    with open(log_file_name, "a") as log_file:
        log_file.write(f"Accuracy: {accuracy}\n")
        log_file.write(f"mean_loss: {mean_loss}\n")
        log_file.write(f"\n")

    return accuracy, mean_loss

def perform_mia_and_log_results(model, epoch, train_logits, train_labels, train_losses, test_logits, test_labels, test_losses, train_indices, test_indices, datasets, model_type, num_bootstrap_samples=1000):
    log_file_name = f"{model_type}_mia_attack_results_epoch_{epoch}.log"
    fig_name = f"{model_type}_mia_attack_roc_curve_epoch_{epoch}.png"
    
    attack_input = AttackInputData(
        logits_train=train_logits,
        logits_test=test_logits,
        loss_train=train_losses,
        loss_test=test_losses,
        labels_train=train_labels,
        labels_test=test_labels
    )
    
    attack_types = [
        AttackType.THRESHOLD_ATTACK,
        AttackType.LOGISTIC_REGRESSION,
        AttackType.MULTI_LAYERED_PERCEPTRON,
        AttackType.RANDOM_FOREST,
        AttackType.K_NEAREST_NEIGHBORS
    ]
    
    for indices in train_indices:
        print(train_logits[indices])
    for indices in test_indices:
        print(test_logits[indices])
    
    slicing_spec = SlicingSpec(
        entire_dataset=True,
        by_class=False,
        by_classification_correctness=False,
        all_custom_train_indices=[np.array([2 if i in train_indices else 0 for i in range(len(datasets["train"]))])],
        all_custom_test_indices=[np.array([2 if i in test_indices else 0 for i in range(len(datasets["test"]))])],
        custom_slices_names = {2: 'modified_indices', 0: 'normal_indices'}
    )

    attack_results = mia.run_attacks(attack_input=attack_input, slicing_spec=slicing_spec, attack_types=attack_types)
    
    pd.set_option("display.max_columns", None)
    with open(log_file_name, "w") as log_file:
        df = attack_results.calculate_pd_dataframe()
        log_file.write(f"\nDataframe:\n{df}\n")

        max_auc_attacker = attack_results.get_result_with_max_auc()
        log_file.write(f"\nMax AUC Attacker Result:\n{max_auc_attacker}\n")

        figure = plotting.plot_roc_curve(max_auc_attacker.roc_curve)
    figure.savefig(fig_name)
    
#     # Extract results for specific indices
#     filtered_results = []
#     with open(log_file_name, "w") as log_file:
#         log_file.write(f"\nMembership inference attack results after epoch {epoch}:\n")
#         log_file.write(f"{attack_results.summary(by_slices=True)}\n")
        
#         for attack_result in attack_results.single_attack_results:
#             # Filter membership scores for the specified indices
#             filtered_train_scores = attack_result.membership_scores_train[train_indices]
#             filtered_test_scores = attack_result.membership_scores_test[test_indices]
            
#             # Combine the scores and labels
#             combined_scores = np.concatenate([filtered_train_scores, filtered_test_scores])
#             combined_labels = np.concatenate([np.ones(len(filtered_train_scores)), np.zeros(len(filtered_test_scores))])
            
#             # Bootstrap sampling to estimate AUC
#             bootstrapped_aucs = []
#             for _ in range(num_bootstrap_samples):
#                 indices = np.random.choice(len(combined_scores), len(combined_scores), replace=True)
#                 sample_scores = combined_scores[indices]
#                 sample_labels = combined_labels[indices]
#                 fpr, tpr, _ = metrics.roc_curve(sample_labels, sample_scores)
#                 bootstrapped_aucs.append(metrics.auc(fpr, tpr))
            
#             mean_auc = np.mean(bootstrapped_aucs)
#             std_auc = np.std(bootstrapped_aucs)
            
#             roc_curve = RocCurve(
#                 tpr=tpr,
#                 fpr=fpr,
#                 thresholds=_,
#                 test_train_ratio=len(filtered_test_scores) / len(filtered_train_scores)
#             )
            
#             filtered_result = {
#                 "Attack Type": attack_result.attack_type.name,
#                 "ROC Curve": {
#                     "FPR": fpr.tolist(),
#                     "TPR": tpr.tolist(),
#                     "Thresholds": _.tolist(),
#                 },
#                 "Mean AUC (Bootstrap)": mean_auc,
#                 "Std AUC (Bootstrap)": std_auc
#             }
            
#             filtered_results.append(filtered_result)
        
#         # Write detailed filtered results to log file
#         log_file.write("\nFiltered Results:\n")
#         for result in filtered_results:
#             log_file.write(f"\nAttack Type: {result['Attack Type']}\n")
#             log_file.write(f"ROC Curve:\n")
#             log_file.write(f"  FPR: {result['ROC Curve']['FPR']}\n")
#             log_file.write(f"  TPR: {result['ROC Curve']['TPR']}\n")
#             log_file.write(f"  Thresholds: {result['ROC Curve']['Thresholds']}\n")
#             log_file.write(f"Mean AUC (Bootstrap): {result['Mean AUC (Bootstrap)']}\n")
#             log_file.write(f"Std AUC (Bootstrap): {result['Std AUC (Bootstrap)']}\n")
       
    
def evaluate(model, test_loader):
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

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    
    return accuracy, precision, recall, f1

def collect_logits_labels_losses(loader, model, loss_fn, modified_indices, device):
    logits_list, labels_list, losses_list, indices_list, texts_list = [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids, attention_mask, labels, idx, text = batch["input_ids"], batch["attention_mask"], batch["labels"], batch["idx"], batch["text"]
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            logits_list.extend(logits.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
            losses_list.extend(loss.cpu().numpy())
            indices_list.extend(idx.cpu().numpy())
            texts_list.extend(text)
    logits_array = np.array(logits_list)
    labels_array = np.array(labels_list)
    losses_array = np.array(losses_list)
    indices_array = np.array(indices_list)
    for idx in modified_indices:
        first_few_words = ' '.join(texts_list[idx].split()[:15])
        print(f"Modified Index: {idx} | Text: {first_few_words} | Label: {labels_list[idx]} | Logits : {logits_list[idx]}")
    print(f"logits shape: {logits_array.shape}")
    print(f"labels shape: {labels_array.shape}")
    print(f"losses shape: {losses_array.shape}")
    print(f"indices shape: {indices_array.shape}")
    return logits_array, labels_array, losses_array, indices_array
