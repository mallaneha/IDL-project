"""
C-MAPSS Data Preprocessing & PyTorch Dataset
=============================================
Shared module for all models (LSTM, TCN, Transformer).
Handles loading, cleaning, normalization, RUL labeling, and windowing.

Usage:
    from data_preprocessing import get_dataloaders, load_and_preprocess

    train_loader, test_loader, test_rul = get_dataloaders(
        train_path='train_FD001.txt',
        test_path='test_FD001.txt',
        rul_path='RUL_FD001.txt',
        sequence_length=30,
        batch_size=64
    )
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler


# ============================================================
# Column Definitions
# ============================================================
COLUMN_NAMES = (
    ['unit', 'cycle'] +
    [f'os_{i}' for i in range(1, 4)] +        # 3 operational settings
    [f'sensor_{i}' for i in range(1, 22)]       # 21 sensor measurements
)

# Sensors that are nearly constant in FD001 and carry no useful info.
# These are: sensor_1, sensor_5, sensor_6, sensor_10, sensor_16, sensor_18, sensor_19
DROP_SENSORS = ['sensor_1', 'sensor_5', 'sensor_6', 'sensor_10',
                'sensor_16', 'sensor_18', 'sensor_19']

# Operational settings are constant in FD001 (single operating condition),
# so we drop them too. For FD002-FD004, you may want to KEEP these.
DROP_SETTINGS_FD001 = ['os_1', 'os_2', 'os_3']


# ============================================================
# Loading & Preprocessing
# ============================================================
def load_data(train_path, test_path, rul_path):
    """Load raw C-MAPSS text files into DataFrames."""
    train_df = pd.read_csv(train_path, sep=r'\s+', header=None, names=COLUMN_NAMES)
    test_df = pd.read_csv(test_path, sep=r'\s+', header=None, names=COLUMN_NAMES)
    rul_df = pd.read_csv(rul_path, sep=r'\s+', header=None, names=['rul'])
    return train_df, test_df, rul_df


def add_rul_column(df):
    """
    Add RUL labels to training data.
    For each engine, RUL at cycle t = (max_cycle - t).
    """
    max_cycles = df.groupby('unit')['cycle'].max().reset_index()
    max_cycles.columns = ['unit', 'max_cycle']
    df = df.merge(max_cycles, on='unit', how='left')
    df['rul'] = df['max_cycle'] - df['cycle']
    df.drop('max_cycle', axis=1, inplace=True)
    return df


def clip_rul(df, max_rul=125):
    """
    Apply piecewise linear RUL: clip RUL at max_rul.
    Early in engine life, degradation signal is negligible,
    so we cap the target to avoid learning noise.
    """
    df['rul'] = df['rul'].clip(upper=max_rul)
    return df


def load_and_preprocess(train_path, test_path, rul_path,
                        max_rul=125, drop_settings=True):
    """
    Full preprocessing pipeline:
    1. Load data
    2. Add & clip RUL for training set
    3. Drop uninformative columns
    4. Normalize sensor features (fit on train, transform both)

    Returns:
        train_df: preprocessed training DataFrame with 'rul' column
        test_df:  preprocessed test DataFrame (no 'rul' column)
        test_rul: numpy array of true RUL values for test engines
        feature_cols: list of feature column names used
        scaler: fitted MinMaxScaler (for reference)
    """
    train_df, test_df, rul_df = load_data(train_path, test_path, rul_path)
    test_rul = rul_df['rul'].values

    # Add RUL labels to training data
    train_df = add_rul_column(train_df)
    train_df = clip_rul(train_df, max_rul=max_rul)

    # Determine columns to drop
    drop_cols = DROP_SENSORS.copy()
    if drop_settings:
        drop_cols += DROP_SETTINGS_FD001

    # Feature columns = everything except unit, cycle, rul, and dropped cols
    all_cols = [c for c in train_df.columns if c not in ['unit', 'cycle', 'rul']]
    feature_cols = [c for c in all_cols if c not in drop_cols]

    # Normalize features: fit on training data only
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    return train_df, test_df, test_rul, feature_cols, scaler


# ============================================================
# PyTorch Datasets
# ============================================================
class CMAPSSTrainDataset(Dataset):
    """
    Training dataset with sliding window.
    Each sample is a (sequence, rul_target) pair where:
        - sequence: [seq_len, n_features] tensor
        - rul_target: scalar (RUL at the last time step of the window)
    """
    def __init__(self, df, feature_cols, sequence_length=30):
        self.sequence_length = sequence_length
        self.feature_cols = feature_cols
        self.samples = []
        self.targets = []

        # Build windows for each engine
        for unit_id in df['unit'].unique():
            unit_df = df[df['unit'] == unit_id].sort_values('cycle')
            features = unit_df[feature_cols].values
            rul = unit_df['rul'].values

            # Sliding window
            for i in range(len(features) - sequence_length + 1):
                self.samples.append(features[i:i + sequence_length])
                self.targets.append(rul[i + sequence_length - 1])  # RUL at end of window

        self.samples = np.array(self.samples, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return (torch.tensor(self.samples[idx]),
                torch.tensor(self.targets[idx]))


class CMAPSSTestDataset(Dataset):
    """
    Test dataset: one sample per engine.
    Takes the LAST `sequence_length` time steps of each engine's trajectory.
    """
    def __init__(self, df, feature_cols, sequence_length=30):
        self.sequence_length = sequence_length
        self.samples = []

        for unit_id in sorted(df['unit'].unique()):
            unit_df = df[df['unit'] == unit_id].sort_values('cycle')
            features = unit_df[feature_cols].values

            # If trajectory is shorter than window, pad with first row
            if len(features) < sequence_length:
                pad_len = sequence_length - len(features)
                padding = np.repeat(features[:1], pad_len, axis=0)
                features = np.concatenate([padding, features], axis=0)

            # Take last `sequence_length` steps
            self.samples.append(features[-sequence_length:])

        self.samples = np.array(self.samples, dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx])


# ============================================================
# Convenience Function
# ============================================================
def get_dataloaders(train_path, test_path, rul_path,
                    sequence_length=30, batch_size=64, max_rul=125,
                    drop_settings=True, val_split=0.2, seed=42):
    """
    End-to-end: load data → preprocess → create DataLoaders.

    Returns:
        train_loader: DataLoader for training
        val_loader:   DataLoader for validation (from training data)
        test_loader:  DataLoader for test (one sample per engine)
        test_rul:     numpy array of true test RUL values
        n_features:   number of input features
    """
    train_df, test_df, test_rul, feature_cols, scaler = load_and_preprocess(
        train_path, test_path, rul_path,
        max_rul=max_rul, drop_settings=drop_settings
    )

    n_features = len(feature_cols)

    # Create full training dataset
    full_train_dataset = CMAPSSTrainDataset(train_df, feature_cols, sequence_length)

    # Train/validation split by engine (not by sample, to avoid data leakage)
    np.random.seed(seed)
    all_units = train_df['unit'].unique()
    np.random.shuffle(all_units)
    split_idx = int(len(all_units) * (1 - val_split))
    train_units = set(all_units[:split_idx])
    val_units = set(all_units[split_idx:])

    train_subset_df = train_df[train_df['unit'].isin(train_units)]
    val_subset_df = train_df[train_df['unit'].isin(val_units)]

    train_dataset = CMAPSSTrainDataset(train_subset_df, feature_cols, sequence_length)
    val_dataset = CMAPSSTrainDataset(val_subset_df, feature_cols, sequence_length)
    test_dataset = CMAPSSTestDataset(test_df, feature_cols, sequence_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Dataset Summary:")
    print(f"  Features: {n_features}")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Train engines: {len(train_units)}, Val engines: {len(val_units)}")
    print(f"  Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"  Test engines: {len(test_dataset)}")
    print(f"  Max RUL cap: {max_rul}")

    return train_loader, val_loader, test_loader, test_rul, n_features


# ============================================================
# Quick sanity check
# ============================================================
if __name__ == '__main__':
    train_loader, val_loader, test_loader, test_rul, n_features = get_dataloaders(
        train_path='train_FD001.txt',
        test_path='test_FD001.txt',
        rul_path='RUL_FD001.txt',
    )

    # Check one batch
    for X, y in train_loader:
        print(f"\nBatch shape: X={X.shape}, y={y.shape}")
        print(f"X range: [{X.min():.3f}, {X.max():.3f}]")
        print(f"y range: [{y.min():.1f}, {y.max():.1f}]")
        break
