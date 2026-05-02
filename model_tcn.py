"""
Temporal Convolutional Network (TCN) for C-MAPSS RUL Prediction
===============================================================
Variant 1: TCN uses dilated causal convolutions to capture
long-range temporal dependencies without recurrence.

Key paper: Bai et al., "An Empirical Evaluation of Generic Convolutional
and Recurrent Networks for Sequence Modeling" (2018).
"""

import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    """
    Causal convolution: pads only on the left so the output at time t
    depends only on inputs at times <= t.
    Uses BatchNorm instead of weight_norm for numerical stability.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=self.padding
        )
        self.bn = nn.BatchNorm1d(out_channels)

        # Kaiming init for ReLU activations
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        """x: [batch, channels, seq_len]"""
        out = self.conv(x)
        # Remove right padding to maintain causal property
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        out = self.bn(out)
        return out


class TemporalBlock(nn.Module):
    """
    One TCN block: two causal convolutions with residual connection.
        Input -> CausalConv+BN -> ReLU -> Dropout
              -> CausalConv+BN -> (+Residual) -> ReLU -> Dropout -> Output
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()

        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Residual connection (1x1 conv if channel sizes differ)
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        """x: [batch, channels, seq_len]"""
        # First conv block
        out = self.relu(self.conv1(x))
        out = self.dropout(out)

        # Second conv block (BN is inside CausalConv1d)
        out = self.conv2(out)

        # Residual + activation
        res = self.residual(x)
        out = self.relu(out + res)
        out = self.dropout(out)
        return out


class TCNModel(nn.Module):
    """
    TCN-based RUL prediction model.
    Architecture:
        Input [batch, seq_len, n_features]
          -> Transpose to [batch, n_features, seq_len]
          -> Stack of TemporalBlocks with increasing dilation
          -> Global Average Pooling
          -> FC -> RUL prediction
    Dilation schedule: 1, 2, 4, 8, ... (doubles each layer)
    This gives the network an exponentially growing receptive field.
    """
    def __init__(self, n_features, n_channels=[32, 32, 64, 64],
                 kernel_size=3, dropout=0.2):
        super().__init__()

        layers = []
        num_levels = len(n_channels)

        for i in range(num_levels):
            in_ch = n_features if i == 0 else n_channels[i - 1]
            out_ch = n_channels[i]
            dilation = 2 ** i  # Exponentially increasing dilation

            layers.append(TemporalBlock(
                in_ch, out_ch, kernel_size, dilation, dropout
            ))

        self.tcn = nn.Sequential(*layers)

        # Output head
        self.fc = nn.Sequential(
            nn.Linear(n_channels[-1], 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

        # Receptive field info
        rf = 1
        for i in range(num_levels):
            rf += 2 * (kernel_size - 1) * (2 ** i)
        self._receptive_field = rf

    @property
    def receptive_field(self):
        return self._receptive_field

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, n_features]
        Returns:
            out: [batch] -- predicted RUL
        """
        # TCN expects [batch, channels, seq_len]
        x = x.transpose(1, 2)       # [batch, n_features, seq_len]
        x = self.tcn(x)             # [batch, n_channels[-1], seq_len]

        # Global average pooling over time
        x = x.mean(dim=2)           # [batch, n_channels[-1]]

        out = self.fc(x)
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
    N_CHANNELS = [32, 32, 64, 64]   # Channel sizes per TCN block
    KERNEL_SIZE = 3
    DROPOUT = 0.2
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
    model = TCNModel(
        n_features=n_features,
        n_channels=N_CHANNELS,
        kernel_size=KERNEL_SIZE,
        dropout=DROPOUT
    )
    print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Receptive Field: {model.receptive_field} time steps")
    print(model)

    # Training setup
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)

    # Train
    model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        n_epochs=N_EPOCHS, device=DEVICE, scheduler=scheduler,
        patience=PATIENCE, model_name="TCN"
    )

    # Evaluate
    predictions = predict(model, test_loader, device=DEVICE)
    results = evaluate_all(test_rul, predictions, model_name="TCN")

    # Save
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'n_features': n_features,
            'n_channels': N_CHANNELS,
            'kernel_size': KERNEL_SIZE,
            'dropout': DROPOUT,
            'sequence_length': SEQUENCE_LENGTH,
        },
        'results': results,
    }, 'tcn_FD004.pth')
    print("Model saved to tcn_FD004.pth")

    # Plot
    plot_training_curves(history, "TCN", save_path='tcn_training_FD004.png')
    plot_predictions(test_rul, predictions, "TCN", save_path='tcn_predictions_FD004.png')
