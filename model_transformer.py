"""
Transformer Model for C-MAPSS RUL Prediction
=============================================
Variant 2: Encoder-only Transformer with positional encoding.

Uses self-attention to learn which time steps matter most for RUL.
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding from 'Attention is All You Need'.
    Adds positional information to the input embeddings.
    """
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """x: [batch, seq_len, d_model]"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerRUL(nn.Module):
    """
    Encoder-only Transformer for RUL prediction.
    Architecture:
        Input [batch, seq_len, n_features]
          - Linear projection to d_model
          - Positional Encoding
          - N × Transformer Encoder Layers
          - Global average pooling (or last token)
          - FC -> RUL prediction
    """
    def __init__(self, n_features, d_model=64, n_heads=4, n_layers=2,
                 d_ff=128, dropout=0.2, pooling='avg'):
        super().__init__()

        self.pooling = pooling

        # Project input features to d_model dimensions
        self.input_projection = nn.Linear(n_features, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        # Layer norm
        self.layer_norm = nn.LayerNorm(d_model)

        # Output head
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
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
        # Project to d_model
        x = self.input_projection(x)         # [batch, seq_len, d_model]
        x = self.pos_encoder(x)

        # Transformer encoder
        x = self.transformer_encoder(x)      # [batch, seq_len, d_model]
        x = self.layer_norm(x)

        # Pool across time dimension
        if self.pooling == 'avg':
            x = x.mean(dim=1)                # [batch, d_model]
        elif self.pooling == 'last':
            x = x[:, -1, :]                  # [batch, d_model]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

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
    D_MODEL = 64
    N_HEADS = 4
    N_LAYERS = 2
    D_FF = 128
    DROPOUT = 0.2
    POOLING = 'avg'      # 'avg' or 'last'
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
    model = TransformerRUL(
        n_features=n_features,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT,
        pooling=POOLING
    )
    print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(model)

    # Training setup
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, )

    # Train
    model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        n_epochs=N_EPOCHS, device=DEVICE, scheduler=scheduler,
        patience=PATIENCE, model_name="Transformer"
    )

    # Evaluate
    predictions = predict(model, test_loader, device=DEVICE)
    results = evaluate_all(test_rul, predictions, model_name="Transformer")

    # Save
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'n_features': n_features,
            'd_model': D_MODEL,
            'n_heads': N_HEADS,
            'n_layers': N_LAYERS,
            'd_ff': D_FF,
            'dropout': DROPOUT,
            'sequence_length': SEQUENCE_LENGTH,
            'pooling': POOLING,
        },
        'results': results,
    }, 'transformer_FD004.pth')
    print("Model saved to transformer_FD004.pth")

    # Plot
    plot_training_curves(history, "Transformer", save_path='transformer_training_FD004.png')
    plot_predictions(test_rul, predictions, "Transformer", save_path='transformer_predictions_FD004.png')
