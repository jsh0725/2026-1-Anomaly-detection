# LSTM-AE Anomaly Detection for SWaT Dataset

An implementation of LSTM-based Autoencoder (LSTM-AE) for detecting cyber-physical system anomalies using the SWaT (Secure Water Treatment) dataset.

## Overview

This project implements an anomaly detection system using LSTM autoencoders to identify abnormal patterns in industrial control system (ICS) data. The model is trained on normal operational data and learns to reconstruct normal patterns. Anomalies are detected when reconstruction error exceeds a threshold.

### Key Features

- **LSTM-based Autoencoder**: Captures temporal dependencies in time-series data
- **Windowing Strategy**: Converts 1D time-series into fixed-length sequences (default: 100 timesteps, 50% overlap)
- **StandardScaler Normalization**: Ensures consistent feature scaling across datasets
- **NaN Imputation**: Handles missing values through linear interpolation
- **Batch Processing**: Memory-efficient evaluation on large datasets
- **Comprehensive Analysis**: Performance metrics including AUC, F1-score, Precision, Recall, and ROC curves

## Dataset

The SWaT dataset contains:
- **normal.csv**: ~1.4M rows of normal operational data
- **attack.csv**: ~1.1M rows of cyberattack data
- **merged.csv**: ~1.4M rows combining normal and attack data

Each file has 51 sensor/actuator measurements recorded every second.

## Project Structure

```
lstm_ae/
├── preprocess.py      # Data preprocessing: windowing, scaling, interpolation
├── model.py          # LSTM-AE model architecture
├── train.py          # Model training script
├── evaluate.py       # Evaluation and reconstruction error computation
├── analyze.py        # Performance analysis and threshold optimization
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

## Installation

### Prerequisites
- Python 3.10+
- CUDA 11.8+ (optional, for GPU acceleration)
- Anaconda or Miniconda

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd SWaT

# Create virtual environment
conda create -n swat_env python=3.10
conda activate swat_env

# Install dependencies
pip install -r lstm_ae/requirements.txt
```

## Usage

### Step 1: Preprocessing

Preprocess normal.csv and create train/test windows:

```bash
python lstm_ae/preprocess.py --in Datasets/normal.csv --window 100 --step 50
```

This generates:
- `lstm_ae/data/normal_windows.npy` - Training data (windows)
- `lstm_ae/data/scaler.pkl` - Fitted StandardScaler for consistent normalization

**Parameters:**
- `--in`: Input CSV file path (required)
- `--window`: Window size in timesteps (default: 100)
- `--step`: Step size for window sliding (default: 50)
- `--out`: Output directory (default: lstm_ae/data)

### Step 2: Model Training

Train the LSTM-AE on normal data:

```bash
python lstm_ae/train.py --data lstm_ae/data/normal_windows.npy --epochs 100
```

**Parameters:**
- `--data`: Path to training data (required)
- `--epochs`: Number of training epochs (default: 20)
- `--batch`: Batch size (default: 64)
- `--lr`: Learning rate (default: 1e-3)
- `--hidden`: LSTM hidden size (default: 64)
- `--latent`: Latent bottleneck size (default: 16)
- `--out`: Checkpoint save directory (default: lstm_ae/checkpoints)

**Output:** `lstm_ae/checkpoints/model_final.pt`

### Step 3: Preprocess Evaluation Data

Transform attack and merged datasets using the fitted scaler from Step 1:

```bash
python lstm_ae/preprocess.py --in Datasets/attack.csv
python lstm_ae/preprocess.py --in Datasets/merged.csv
```

Generates:
- `lstm_ae/data/attack_windows.npy`
- `lstm_ae/data/merged_windows.npy`

### Step 4: Evaluate

Compute reconstruction errors on all datasets:

```bash
python lstm_ae/evaluate.py --model lstm_ae/checkpoints/model_final.pt --data lstm_ae/data/normal_windows.npy
python lstm_ae/evaluate.py --model lstm_ae/checkpoints/model_final.pt --data lstm_ae/data/attack_windows.npy
python lstm_ae/evaluate.py --model lstm_ae/checkpoints/model_final.pt --data lstm_ae/data/merged_windows.npy
```

**Output:** `.npy` files with reconstruction errors:
- `normal_windows_errs.npy`
- `attack_windows_errs.npy`
- `merged_windows_errs.npy`

### Step 5: Analyze Performance

Run comprehensive analysis with threshold optimization and performance metrics:

```bash
python lstm_ae/analyze.py
```

**Output:**
- Error statistics (mean, std, min-max) for each dataset
- Multiple threshold optimization methods:
  - Mean ± 1σ, 2σ, 3σ
  - 95th and 99th percentiles
- Best F1-score with optimal threshold
- Final performance metrics (AUC, Precision, Recall, F1)
- ROC curve visualization (saved as `lstm_ae/results_roc.png`)

## Model Architecture

### LSTM-AE Structure

```
Input (batch, 100, 51)
  ↓
LSTM Encoder (51 → 64)
  ↓
Linear Projection (64 → 16)  [Bottleneck/Latent]
  ↓
Linear Projection (16 → 64)
  ↓
LSTM Decoder (64 → 64)
  ↓
Linear Output (64 → 51)
Output (batch, 100, 51)
```

**Reconstruction Loss:** Mean Squared Error (MSE)

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Window Size | 100 | Timesteps per sequence |
| Window Step | 50 | Overlapping ratio (50%) |
| Hidden Size | 64 | LSTM hidden dimension |
| Latent Size | 16 | Bottleneck dimension |
| Batch Size | 64 | Training batch size |
| Learning Rate | 1e-3 | Adam optimizer learning rate |
| Epochs | 20 | Training iterations |

## Expected Performance

Based on the SWaT dataset with optimal hyperparameters:

| Metric | Value |
|--------|-------|
| AUC-ROC | ~0.85-0.95 |
| F1-Score | ~0.70-0.85 |
| Precision | ~0.75-0.90 |
| Recall | ~0.65-0.80 |

*Performance varies based on data preprocessing and model tuning*

## Data Preprocessing Details

1. **Numeric Extraction**: Selects only numeric columns (excludes timestamps, labels)
2. **NaN Imputation**: 
   - Linear interpolation within each feature
   - Remaining NaNs filled with 0
3. **Standardization**: Zero-mean, unit-variance normalization per feature
4. **Windowing**: Sliding windows with 50% overlap for temporal context

## Anomaly Detection Mechanism

1. **Training**: Model learns to reconstruct normal data with low error
2. **Inference**: 
   - Pass window through model
   - Compute reconstruction error: MSE between input and output
   - High error → anomaly detected
3. **Thresholding**: 
   - Optimal threshold determined from normal data statistics
   - Default: Mean + 3σ from training set

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size: `--batch 32`
- Reduce window size: `--window 50`
- Reduce hidden size: `--hidden 32`

### NaN in Errors
- Check data quality in source CSVs
- Ensure interpolation was applied: `preprocess.py` handles this automatically

### Low Performance (F1 < 0.5)
- Increase epochs: `--epochs 200`
- Try different window sizes: `--window 150`
- Tune learning rate: `--lr 5e-4`
- Consider alternative models (Isolation Forest, One-Class SVM)

## Dependencies

```
numpy>=1.26.4
pandas>=2.3.3
scikit-learn>=1.7.1
joblib>=1.5.1
torch>=2.5.1
matplotlib>=3.10.0
```

## References

### Dataset
- SWaT: [SUTD Data Repository](https://itrust.sutd.edu.sg/datasets/)

### LSTM-AE for Anomaly Detection
- Malhotra et al., "Long Short Term Memory Networks for Anomaly Detection in Time Series", ESANN 2015

## Author

Developed for cyber-physical system anomaly detection research.

## License

This project is provided as-is for research and educational purposes.

## Contact & Issues

For questions or issues, please open a GitHub issue.
