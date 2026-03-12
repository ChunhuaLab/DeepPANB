import os
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn as nn
from data_utils import load_dataset, transfer_to_device
from model import PAINN
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
from sklearn.model_selection import KFold
import psutil
import torch
import numpy as np
import time
import json
import copy
import warnings

warnings.filterwarnings('ignore', category=UserWarning)


def evaluate_on_dataset(model, dataset, device, name="Dataset", verbose=True, return_predictions=False):
    model.eval()
    all_true = []
    all_pred = []
    all_pred_probs = []

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=transfer_to_device
    )

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data)

            # Process prediction results for each residue
            pred_probs = pred.squeeze().cpu().numpy()
            predictions = (pred_probs > 0.5).astype(int)
            true_labels = data.y.squeeze().cpu().numpy().astype(int)

            all_pred_probs.extend(pred_probs.tolist())
            all_true.extend(true_labels.tolist())
            all_pred.extend(predictions.tolist())

    # Convert to numpy arrays
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    all_pred_probs = np.array(all_pred_probs)

    # Compute metrics
    precision = precision_score(all_true, all_pred)
    sensitivity = recall_score(all_true, all_pred)  # Recall = Sensitivity
    f1 = f1_score(all_true, all_pred)
    mcc = matthews_corrcoef(all_true, all_pred)

    # Handle specificity calculation
    cm = confusion_matrix(all_true, all_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
    else:
        specificity = 0.0
        accuracy = 0.0

    # Handle AUC and AUPRC calculation
    try:
        auc = roc_auc_score(all_true, all_pred_probs)
        auprc = average_precision_score(all_true, all_pred_probs)
    except ValueError:
        auc = 0.0
        auprc = 0.0

    if verbose:
        print(f"\n========== {name} Evaluation Results ==========")
        print(f"Precision: {precision:.4f}")
        print(f"Sensitivity/Recall: {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"AUPRC: {auprc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"MCC: {mcc:.4f}")

    results = {
        'precision': precision,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'accuracy': accuracy,
        'auc': auc,
        'auprc': auprc,
        'f1': f1,
        'mcc': mcc
    }

    if return_predictions:
        results['y_true'] = all_true
        results['y_pred_probs'] = all_pred_probs
        results['y_pred'] = all_pred

    return results


def train_model(train_data, device, params, num_epochs=30):
    """
    Train a single model
    """
    model = PAINN(
        input_dim=1583,
        hidden_dim=params['hidden_dim'],
        num_layers=params['num_layers'],
        dropout_rate=params['dropout_rate']
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    criterion = nn.BCELoss()

    train_loader = DataLoader(
        train_data,
        batch_size=8,
        shuffle=True,
        collate_fn=transfer_to_device,
        num_workers=8,
        pin_memory=True
    )

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | Loss: {total_loss / len(train_loader):.4f}")

    return model


def perform_cross_validation(dataset, device, params, n_folds=5, save_dir="cv_results"):
    """
    Perform K-fold cross validation
    """
    print(f"\n====== Starting {n_folds}-Fold Cross Validation ======")

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Prepare K-fold cross validation
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Store results for all folds
    fold_results = []
    all_y_true = []
    all_y_pred_probs = []

    # Convert dataset to index list for splitting
    dataset_indices = list(range(len(dataset)))

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset_indices)):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")

        # Create training and validation sets
        train_subset = [dataset[i] for i in train_idx]
        val_subset = [dataset[i] for i in val_idx]

        print(f"Training samples: {len(train_subset)}, Validation samples: {len(val_subset)}")

        # Train model
        model = train_model(train_subset, device, params, num_epochs=30)

        # Evaluate on validation set
        val_results = evaluate_on_dataset(
            model, val_subset, device,
            name=f"Fold {fold + 1} Validation",
            verbose=True, return_predictions=True
        )

        # Collect predictions for overall evaluation
        all_y_true.extend(val_results['y_true'])
        all_y_pred_probs.extend(val_results['y_pred_probs'])

        # Save results for this fold
        fold_result = {
            'fold': fold + 1,
            'precision': val_results['precision'],
            'sensitivity': val_results['sensitivity'],
            'specificity': val_results['specificity'],
            'accuracy': val_results['accuracy'],
            'auc': val_results['auc'],
            'auprc': val_results['auprc'],
            'f1': val_results['f1'],
            'mcc': val_results['mcc']
        }
        fold_results.append(fold_result)

        # Free memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Compute mean and std of cross validation results
    metrics = ['precision', 'sensitivity', 'specificity', 'accuracy', 'auc', 'auprc', 'f1', 'mcc']
    cv_summary = {}

    print(f"\n====== {n_folds}-Fold Cross Validation Results ======")
    for metric in metrics:
        values = [result[metric] for result in fold_results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        cv_summary[f'{metric}_mean'] = mean_val
        cv_summary[f'{metric}_std'] = std_val
        print(f"{metric.upper()}: {mean_val:.4f} +/- {std_val:.4f}")

    # Compute overall metrics using predictions from all validation folds
    all_y_true = np.array(all_y_true)
    all_y_pred_probs = np.array(all_y_pred_probs)

    print(f"\n====== Overall Cross Validation Performance ======")
    overall_metrics = evaluate_on_dataset_from_predictions(
        all_y_true, all_y_pred_probs,
        name="Overall CV Performance", verbose=True
    )

    # Save detailed results
    cv_results = {
        'parameters': params,
        'n_folds': n_folds,
        'fold_results': fold_results,
        'cv_summary': cv_summary,
        'overall_metrics': overall_metrics
    }

    with open(os.path.join(save_dir, 'cross_validation_results.json'), 'w') as f:
        json.dump(cv_results, f, indent=2)

    return cv_results


def evaluate_on_dataset_from_predictions(y_true, y_pred_probs, name="Dataset", verbose=True):
    """
    Compute evaluation metrics from predictions
    """
    y_pred = (y_pred_probs > 0.5).astype(int)

    # Compute metrics
    precision = precision_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    # Handle specificity and accuracy calculation
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
    else:
        specificity = 0.0
        accuracy = 0.0

    # AUC and AUPRC
    try:
        auc = roc_auc_score(y_true, y_pred_probs)
        auprc = average_precision_score(y_true, y_pred_probs)
    except ValueError:
        auc = 0.0
        auprc = 0.0

    if verbose:
        print(f"\n========== {name} ==========")
        print(f"Precision: {precision:.4f}")
        print(f"Sensitivity/Recall: {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"AUPRC: {auprc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"MCC: {mcc:.4f}")

    return {
        'precision': precision,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'accuracy': accuracy,
        'auc': auc,
        'auprc': auprc,
        'f1': f1,
        'mcc': mcc
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("\n====== Device Information ======")
    print(f"- CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"- Current GPU: {torch.cuda.current_device()}")
        print(f"- GPU name: {torch.cuda.get_device_name()}")

    print(f"Logical cores: {os.cpu_count()}")
    print(f"Physical cores: {psutil.cpu_count(logical=False)}")

    # ========== Configuration ==========
    train_txt = os.environ.get("TRAIN_TXT", "./datasets/DNA-573_Train.fa")
    train_pdb = os.environ.get("TRAIN_PDB", "./data/DNA_573_train")
    test129_txt = os.environ.get("TEST129_TXT", "./datasets/DNA-129_Test.fa")
    test129_pdb = os.environ.get("TEST129_PDB", "./data/DNA_129_test")
    test181_txt = os.environ.get("TEST181_TXT", "./datasets/DNA_Test_181.fa")
    test181_pdb = os.environ.get("TEST181_PDB", "./data/DNA_Test_181")
    use_cache = True

    # ========== Model Parameters ==========
    # Set parameters directly without grid search
    params = {
        'lr': 0.001,
        'hidden_dim': 64,
        'num_layers': 3,
        'dropout_rate': 0.1
    }

    print(f"\n====== Model Parameters ======")
    for key, value in params.items():
        print(f"{key}: {value}")

    # ========== Load Data ==========
    print("\n====== Loading Data ======")
    train_set = load_dataset(
        train_txt, train_pdb, device,
        use_cache=use_cache,
        max_workers=16
    )

    test129_set = load_dataset(
        test129_txt, test129_pdb, device,
        use_cache=use_cache,
        max_workers=16
    )

    test181_set = load_dataset(
        test181_txt, test181_pdb, device,
        use_cache=use_cache,
        max_workers=16
    )

    print(f"Training set size: {len(train_set)}")
    print(f"Test set 1 size: {len(test129_set)}")
    print(f"Test set 2 size: {len(test181_set)}")

    # ========== Perform Cross Validation on Training Set ==========
    cv_results = perform_cross_validation(
        train_set, device, params,
        n_folds=5, save_dir="cv_results"
    )

    # ========== Train Final Model on Full Training Set ==========
    print("\n====== Training Final Model on Full Training Set ======")
    final_model = train_model(train_set, device, params, num_epochs=30)

    # ========== Evaluate on Test Sets ==========
    print("\n====== Evaluating on Test Sets ======")

    # Test set 1
    test129_results = evaluate_on_dataset(
        final_model, test129_set, device,
        name="Test Set 1", verbose=True, return_predictions=True
    )

    # Test set 2
    test181_results = evaluate_on_dataset(
        final_model, test181_set, device,
        name="Test Set 2", verbose=True, return_predictions=True
    )

    # ========== Save Final Results ==========
    final_results = {
        'parameters': params,
        'cross_validation': cv_results['cv_summary'],
        'test_set_1': {k: v for k, v in test129_results.items() if k not in ['y_true', 'y_pred_probs', 'y_pred']},
        'test_set_2': {k: v for k, v in test181_results.items() if k not in ['y_true', 'y_pred_probs', 'y_pred']}
    }

    with open('cv_results/final_results_summary.json', 'w') as f:
        json.dump(final_results, f, indent=2)

    # Save model
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'params': params,
        'cv_results': cv_results['cv_summary']
    }, 'cv_results/final_model.pth')

    print("\n====== Training and Evaluation Complete! ======")
    print("Results saved in 'cv_results' folder:")
    print("- cross_validation_results.json: Detailed CV results")
    print("- final_results_summary.json: Summary of all results")
    print("- final_model.pth: Trained model")


if __name__ == "__main__":
    main()
