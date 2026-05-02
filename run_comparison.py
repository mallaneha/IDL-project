"""
Master Comparison Script — Run All Models & Compare
====================================================
This script trains all three models on FD001, evaluates them,
and produces comparison plots for the report.

Can also be extended to run cross-dataset (FD001-FD004) analysis.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from data_preprocessing import get_dataloaders
from evaluation import (train_model, predict, evaluate_all,
                        plot_training_curves, plot_predictions,
                        plot_model_comparison)
from model_lstm import LSTMModel
from model_transformer import TransformerRUL
from model_tcn import TCNModel


# ============================================================
# Global Config
# ============================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEQUENCE_LENGTH = 30
BATCH_SIZE = 64
MAX_RUL = 125
N_EPOCHS = 100
PATIENCE = 15
SEED = 42

# Dataset paths — update for FD002-FD004 as needed
DATASETS = {
    'FD001': {
        'train': 'train_FD001.txt',
        'test': 'test_FD001.txt',
        'rul': 'RUL_FD001.txt',
        'drop_settings': True,   # FD001 has constant op settings
    },
    # Uncomment below for cross-dataset analysis
    'FD002': {
        'train': 'train_FD002.txt',
        'test': 'test_FD002.txt',
        'rul': 'RUL_FD002.txt',
        'drop_settings': False,  # FD002 has 6 operating conditions!
    },
    'FD003': {
        'train': 'train_FD003.txt',
        'test': 'test_FD003.txt',
        'rul': 'RUL_FD003.txt',
        'drop_settings': True,   # FD003 has 1 condition, 2 fault modes
    },
    'FD004': {
        'train': 'train_FD004.txt',
        'test': 'test_FD004.txt',
        'rul': 'RUL_FD004.txt',
        'drop_settings': False,  # FD004 has 6 conditions, 2 fault modes
    },
}


def build_models(n_features):
    """Instantiate all three models with consistent configs."""
    models = {
        'LSTM': LSTMModel(
            n_features=n_features,
            hidden_size=64,
            n_layers=2,
            dropout=0.3,
            bidirectional=False
        ),
        'TCN': TCNModel(
            n_features=n_features,
            n_channels=[32, 32, 64, 64],
            kernel_size=3,
            dropout=0.2
        ),
        'Transformer': TransformerRUL(
            n_features=n_features,
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_ff=128,
            dropout=0.2,
            pooling='avg'
        ),
    }
    return models


def run_experiment(dataset_name, dataset_config):
    """Train and evaluate all models on one dataset."""
    print(f"\n{'#'*60}")
    print(f"  DATASET: {dataset_name}")
    print(f"{'#'*60}\n")

    # Load data
    train_loader, val_loader, test_loader, test_rul, n_features = get_dataloaders(
        dataset_config['train'],
        dataset_config['test'],
        dataset_config['rul'],
        sequence_length=SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE,
        max_rul=MAX_RUL,
        drop_settings=dataset_config['drop_settings'],
        seed=SEED
    )

    # Build models
    models = build_models(n_features)

    all_results = {}
    all_predictions = {}
    all_histories = {}

    for model_name, model in models.items():
        print(f"\n{'='*50}")
        print(f"  Training: {model_name} on {dataset_name}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"{'='*50}")

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        model, history = train_model(
            model, train_loader, val_loader, criterion, optimizer,
            n_epochs=N_EPOCHS, device=DEVICE, scheduler=scheduler,
            patience=PATIENCE, model_name=model_name
        )

        preds = predict(model, test_loader, device=DEVICE)
        results = evaluate_all(test_rul, preds, model_name=f"{model_name} ({dataset_name})")

        # Store results
        all_results[model_name] = results
        all_predictions[model_name] = preds
        all_histories[model_name] = history

        # Save checkpoint
        save_name = f"{model_name.lower()}_{dataset_name}.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'results': results,
        }, save_name)

        # Individual plots
        plot_training_curves(history, f"{model_name} ({dataset_name})",
                             save_path=f"{model_name.lower()}_{dataset_name}_training.png")
        plot_predictions(test_rul, preds, f"{model_name} ({dataset_name})",
                         save_path=f"{model_name.lower()}_{dataset_name}_predictions.png")

    # Comparison plot
    plot_model_comparison(all_results,
                          save_path=f"comparison_{dataset_name}.png")

    return all_results, all_predictions


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print(f"Device: {DEVICE}")
    print(f"PyTorch: {torch.__version__}")

    # Run on all configured datasets
    all_dataset_results = {}
    for ds_name, ds_config in DATASETS.items():
        results, predictions = run_experiment(ds_name, ds_config)
        all_dataset_results[ds_name] = results

    # If multiple datasets, create cross-dataset comparison table
    if len(all_dataset_results) > 1:
        print("\n\n" + "="*70)
        print("  CROSS-DATASET COMPARISON (RMSE)")
        print("="*70)
        header = f"{'Model':<15}" + "".join(f"{ds:<12}" for ds in all_dataset_results.keys())
        print(header)
        print("-" * len(header))

        model_names = list(next(iter(all_dataset_results.values())).keys())
        for model_name in model_names:
            row = f"{model_name:<15}"
            for ds_name in all_dataset_results:
                rmse_val = all_dataset_results[ds_name][model_name]['rmse']
                row += f"{rmse_val:<12.2f}"
            print(row)

    print("\n\nDone! All results saved.")
