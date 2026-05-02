# C-MAPSS RUL Prediction — Course Project (24-788)
## Team: Nisarga, Neha Malla, Daniel Wicklund

### Project Structure
```
├── data_preprocessing.py    # Shared: loading, normalization, windowing, DataLoaders
├── evaluation.py            # Shared: RMSE, PHM score, training loop, plotting
├── model_lstm.py            # Baseline: LSTM 
├── model_transformer.py     # Variant 2: Transformer 
├── model_tcn.py             # Variant 1: TCN 
├── run_comparison.py        # Master script: train all models + comparison plots
└── README.md
```

### Setup (Google Colab)

1. **Upload data files** to your Colab session or Google Drive:
   - `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`
   - (Later: FD002, FD003, FD004 for cross-dataset analysis)

2. **Upload all `.py` files** to the same directory.

3. **Install dependencies** (most are pre-installed in Colab):
   ```python
   # These should already be available in Colab:
   # torch, numpy, pandas, scikit-learn, matplotlib
   ```

4. **Run individual models:**
   ```python
   !python model_lstm.py
   !python model_tcn.py
   !python model_transformer.py
   ```

5. **Run full comparison:**
   ```python
   !python run_comparison.py
   ```

### Metrics
- **RMSE** (primary): Root Mean Squared Error on RUL predictions
- **PHM Score** (secondary): Asymmetric scoring - penalizes late predictions (a1=10, a2=13)
- **MAE**: Mean Absolute Error

### Cross-Dataset Analysis (3rd Contribution)
To run on FD002-FD004, uncomment the dataset entries in `run_comparison.py`.
**Important**: For FD002 and FD004, set `drop_settings=False` since operational
settings vary across 6 conditions and carry useful information.

### Reproducing Results
```python
# After training, load saved checkpoints:
checkpoint = torch.load('lstm_FD001.pth')
model.load_state_dict(checkpoint['model_state_dict'])
print(checkpoint['results'])
```
