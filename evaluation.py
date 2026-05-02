"""
Evaluation Utilities for C-MAPSS RUL Prediction
================================================
Shared metrics and plotting for all models.
Includes RMSE, PHM asymmetric scoring, and visualization functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch


# ============================================================
# Metrics
# ============================================================
def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def phm_score(y_true, y_pred):
    """
    PHM08 asymmetric scoring function.
    Late predictions (overestimating RUL) are penalized more heavily.

    s = sum( exp(-d/13) - 1 )  if d < 0  (early prediction)
    s = sum( exp( d/10) - 1 )  if d >= 0 (late prediction)

    where d = predicted_RUL - true_RUL
    """
    d = y_pred - y_true
    scores = np.where(d < 0,
                      np.exp(-d / 13.0) - 1,
                      np.exp(d / 10.0) - 1)
    return np.sum(scores)


def mae(y_true, y_pred):
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def evaluate_all(y_true, y_pred, model_name="Model"):
    """Compute and print all metrics."""
    r = rmse(y_true, y_pred)
    s = phm_score(y_true, y_pred)
    m = mae(y_true, y_pred)
    print(f"\n{'='*40}")
    print(f"  {model_name} Evaluation")
    print(f"{'='*40}")
    print(f"  RMSE:      {r:.4f}")
    print(f"  MAE:       {m:.4f}")
    print(f"  PHM Score: {s:.2f}")
    print(f"{'='*40}")
    return {'rmse': r, 'mae': m, 'phm_score': s}


# ============================================================
# Prediction Helper
# ============================================================
def predict(model, test_loader, device='cpu'):
    """Run inference on test set, return predictions as numpy array."""
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, (list, tuple)):
                X = batch[0].to(device)
            else:
                X = batch.to(device)
            preds = model(X).squeeze()
            predictions.append(preds.cpu().numpy())
    return np.concatenate(predictions)


# ============================================================
# Training Loop
# ============================================================
def train_model(model, train_loader, val_loader, criterion, optimizer,
                n_epochs=50, device='cpu', scheduler=None, patience=10,
                model_name="model"):
    """
    Generic training loop with early stopping and loss tracking.

    Returns:
        model: trained model (best validation loss checkpoint)
        history: dict with 'train_loss' and 'val_loss' lists
    """
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(n_epochs):
        # --- Training ---
        model.train()
        train_losses = []
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(X).squeeze()
            loss = criterion(preds, y)

            # Skip NaN batches instead of poisoning the model
            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()  # Clear any partial gradients
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # --- Validation ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                preds = model(X).squeeze()
                loss = criterion(preds, y)
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    val_losses.append(loss.item())

        avg_train = np.mean(train_losses) if train_losses else float('inf')
        avg_val = np.mean(val_losses) if val_losses else float('inf')
        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)

        if scheduler:
            scheduler.step(avg_val)

        # Early stopping
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{n_epochs} | "
                  f"Train Loss: {avg_train:.4f} | "
                  f"Val Loss: {avg_val:.4f} | "
                  f"Best Val: {best_val_loss:.4f} | "
                  f"Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\n{model_name} training complete. Best val loss: {best_val_loss:.4f}")
    return model, history


# ============================================================
# Plotting Functions
# ============================================================
def plot_training_curves(history, model_name="Model", save_path=None):
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title(f'{model_name} — Training Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_predictions(y_true, y_pred, model_name="Model", save_path=None):
    """Plot predicted vs true RUL for each test engine."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot
    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
    max_val = max(y_true.max(), y_pred.max()) + 10
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect prediction')
    ax.set_xlabel('True RUL')
    ax.set_ylabel('Predicted RUL')
    ax.set_title(f'{model_name} — Predicted vs True RUL')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Engine-by-engine bar chart
    ax = axes[1]
    engine_ids = np.arange(1, len(y_true) + 1)
    errors = y_pred - y_true
    colors = ['#e74c3c' if e > 0 else '#2ecc71' for e in errors]
    ax.bar(engine_ids, errors, color=colors, alpha=0.7, width=1.0)
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_xlabel('Test Engine ID')
    ax.set_ylabel('Prediction Error (Pred - True)')
    ax.set_title(f'{model_name} — Per-Engine Error (Red=Late, Green=Early)')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_model_comparison(results_dict, save_path=None):
    """
    Compare multiple models side by side.

    Args:
        results_dict: {'ModelName': {'rmse': float, 'mae': float, 'phm_score': float}, ...}
    """
    models = list(results_dict.keys())
    metrics = ['rmse', 'mae', 'phm_score']
    titles = ['RMSE (↓ better)', 'MAE (↓ better)', 'PHM Score (↓ better)']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

    for ax, metric, title in zip(axes, metrics, titles):
        values = [results_dict[m][metric] for m in models]
        bars = ax.bar(models, values, color=colors[:len(models)], alpha=0.8,
                      edgecolor='black', linewidth=0.5)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
