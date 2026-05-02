"""
LSTM Baseline Model for C-MAPSS RUL Prediction
===============================================
Baseline model: stacked LSTM -> fully connected -> RUL prediction.

This is the baseline for the course project.
"""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    LSTM-based RUL prediction model.

    Architecture:
        Input [batch, seq_len, n_features]
          - LSTM (2 layers, bidirectional optional)
          - Dropout
          - FC layers
          - RUL prediction (scalar)
    """
    def __init__(self, n_features, hidden_size=64, n_layers=2,
                 dropout=0.3, bidirectional=False):
        super().__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.n_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=bidirectional
        )

        self.dropout = nn.Dropout(dropout)

        fc_input = hidden_size * self.n_directions
        self.fc = nn.Sequential(
            nn.Linear(fc_input, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, n_features]
        Returns:
            out: [batch] — predicted RUL
        """
        # LSTM output: [batch, seq_len, hidden * n_directions]
        lstm_out, (h_n, _) = self.lstm(x)

        # Use the last hidden state
        if self.bidirectional:
            # Concatenate last hidden states from both directions
            h_forward = h_n[-2]   # [batch, hidden]
            h_backward = h_n[-1]  # [batch, hidden]
            last_hidden = torch.cat([h_forward, h_backward], dim=1)
        else:
            last_hidden = h_n[-1]  # [batch, hidden]

        out = self.dropout(last_hidden)
        out = self.fc(out)
        return out.squeeze(-1)


# ============================================================
# Training Script
# ============================================================
if __name__ == '__main__':
    import sys
    sys.path.append('.')
    from data_preprocessing import get_dataloaders
    from evaluation import train_model, predict, evaluate_all, \
        plot_training_curves, plot_predictions

    # ------ CONFIG ------
    TRAIN_PATH = 'train_FD004.txt'
    TEST_PATH = 'test_FD004.txt'
    RUL_PATH = 'RUL_FD004.txt'
    SEQUENCE_LENGTH = 30
    BATCH_SIZE = 64
    HIDDEN_SIZE = 64
    N_LAYERS = 2
    DROPOUT = 0.3
    LEARNING_RATE = 1e-3
    N_EPOCHS = 100
    PATIENCE = 15
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # --------------------

    print(f"Using device: {DEVICE}\n")

    # Load data
    train_loader, val_loader, test_loader, test_rul, n_features = get_dataloaders(
        TRAIN_PATH, TEST_PATH, RUL_PATH,
        sequence_length=SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE
    )

    # Build model
    model = LSTMModel(
        n_features=n_features,
        hidden_size=HIDDEN_SIZE,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
        bidirectional=False
    )
    print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(model)

    # Training setup
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Train
    model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        n_epochs=N_EPOCHS, device=DEVICE, scheduler=scheduler,
        patience=PATIENCE, model_name="LSTM Baseline"
    )

    # Evaluate on test set
    predictions = predict(model, test_loader, device=DEVICE)
    results = evaluate_all(test_rul, predictions, model_name="LSTM Baseline")

    # Save model checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'n_features': n_features,
            'hidden_size': HIDDEN_SIZE,
            'n_layers': N_LAYERS,
            'dropout': DROPOUT,
            'sequence_length': SEQUENCE_LENGTH,
        },
        'results': results,
    }, 'lstm_baseline_FD004.pth')
    print("Model saved to lstm_baseline_FD004.pth")

    # Plot results
    plot_training_curves(history, "LSTM Baseline", save_path='lstm_training_FD004.png')
    plot_predictions(test_rul, predictions, "LSTM Baseline", save_path='lstm_predictions_FD004.png')
