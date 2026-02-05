# Anomaly Detection for SWaT Dataset

Cybersecurity anomaly detection system for the Secure Water Treatment (SWaT) industrial control system using machine learning. This repository includes implementations of LSTM-based Autoencoder and Isolation Forest models with comprehensive performance analysis.

## Overview

This project implements anomaly detection on the SWaT dataset (1.4M+ samples) containing normal operational and cyberattack data. We compare two approaches:
1. **LSTM-AE**: Deep learning-based reconstruction error detection
2. **Isolation Forest**: Ensemble-based anomaly scoring (recommended)

Both models are trained on normal data only and detect anomalies through statistical thresholding.

### Key Contributions

-  **Isolation Forest achieves 67.6% F1-Score** with 61% recall on attack detection
-  **Data preprocessing pipeline**: NaN imputation, linear interpolation, standardization
-  **Comprehensive evaluation**: AUC-ROC, confusion matrix, threshold optimization
-  **Model comparison**: LSTM-AE vs Isolation Forest analysis
-  **Production-ready code**: Efficient batch processing, GPU support

## Dataset

The SWaT dataset contains:
- **normal.csv**: ~1.4M rows of normal operational data
- **attack.csv**: ~1.1M rows of cyberattack data
- **merged.csv**: ~1.4M rows combining normal and attack data

Each file has 51 sensor/actuator measurements recorded every second.

## Project Structure

```
lstm_ae/
├── preprocess.py           # Data preprocessing: windowing, scaling, interpolation
├── model.py               # LSTM-AE model architecture
├── train.py               # LSTM-AE training script
├── evaluate.py            # Reconstruction error computation
├── analyze.py             # LSTM-AE performance analysis & threshold optimization
├── isolation_forest.py    # Isolation Forest (recommended model)
├── __init__.py
├── requirements.txt       # Python dependencies
└── README.md              # This file
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

### Quick Start (Recommended)

For immediate anomaly detection using Isolation Forest:

```bash
# Install dependencies
pip install -r lstm_ae/requirements.txt

# Run Isolation Forest (fastest & best performance)
python lstm_ae/isolation_forest.py
```

Output: Anomaly scores saved to `lstm_ae/isolation_forest_scores.npy`

---

### Full Pipeline (with preprocessing)

**Step 1: Data Preprocessing**

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

### Step 3: Isolation Forest (Recommended)

For direct anomaly detection without preprocessing:

```bash
python lstm_ae/isolation_forest.py
```

**Output:**
- Anomaly scores saved to `lstm_ae/isolation_forest_scores.npy`
- Model saved to `lstm_ae/isolation_forest_model.pkl`
- Performance metrics (AUC, F1, confusion matrix)

### Step 4: LSTM-AE Pipeline (Alternative)

Preprocess evaluation data:

```bash
python lstm_ae/preprocess.py --in Datasets/attack.csv
python lstm_ae/preprocess.py --in Datasets/merged.csv
```

Evaluate reconstruction errors:

```bash
python lstm_ae/evaluate.py --model lstm_ae/checkpoints/model_final.pt --data lstm_ae/data/normal_windows.npy
python lstm_ae/evaluate.py --model lstm_ae/checkpoints/model_final.pt --data lstm_ae/data/attack_windows.npy
python lstm_ae/evaluate.py --model lstm_ae/checkpoints/model_final.pt --data lstm_ae/data/merged_windows.npy
```

Run analysis:

```bash
python lstm_ae/analyze.py
```

**Output:**
- Threshold optimization results
- ROC curve visualization (`lstm_ae/results_roc.png`)
- Performance metrics

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

## Performance Comparison

### Experimental Results on SWaT Dataset

| Model | AUC-ROC | F1-Score | Precision | Recall | Speed |
|-------|---------|----------|-----------|--------|-------|
| **Isolation Forest** | **0.8635** | **0.6758** | **75.69%** | **61.04%** | ⚡ Very Fast |
| LSTM-AE | 0.8488 | 0.2500 | 96.32% | 14.36% | Slow |

### Key Metrics Breakdown

**Isolation Forest (1.4M samples, 51 features):**
-  True Negatives: 1,376,393 (99.2% specificity)
-  True Positives: 33,339 (61% recall - catches most attacks)
-  False Positives: 10,705 (manageable)
-  False Negatives: 21,282 (39% miss rate)

**LSTM-AE (windowed data, 100 timestep windows):**
-  High precision (96.3%) but
-  **Critical issue**: Only 14% recall - misses 86% of attacks 

### Recommendation

**Use Isolation Forest** because:
1. **Higher Recall** (61% vs 14%) - catches more attacks
2. **Better F1-Score** (0.676 vs 0.25) - balanced performance
3. **Production-Ready** - no deep learning overhead
4. **Interpretable** - easier to debug and explain
5. **10-100x Faster** - suitable for real-time systems

## Data Preprocessing Details

1. **Numeric Extraction**: Selects only numeric columns (excludes timestamps, labels)
2. **NaN Handling**: 
   - Linear interpolation within each feature (handles ~6.9M NaN values in normal.csv)
   - Remaining NaNs filled with 0
3. **Standardization**: Zero-mean, unit-variance normalization per feature
4. **Windowing** (LSTM-AE only): Sliding windows with 50% overlap for temporal context

### Data Statistics

| Dataset | Rows | Features | Normal | Attack |
|---------|------|----------|--------|--------|
| normal.csv | 1,387,098 | 51 | 100% | - |
| attack.csv | 54,621 | 51 | - | 100% |
| merged.csv | 1,441,719 | 51 | 96.2% | 3.8% |

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

### Isolation Forest (Fast & Recommended)
-  No GPU needed
-  No preprocessing needed
-  Direct on raw data

### LSTM-AE Pipeline

#### CUDA Out of Memory
- Reduce batch size: `--batch 32`
- Reduce window size: `--window 50`
- Reduce hidden size: `--hidden 32`

#### NaN in Reconstruction Errors
- Check data quality in source CSVs
- `preprocess.py` automatically handles NaN via interpolation

#### Low F1-Score (< 0.5)
- **Known issue**: LSTM-AE has recall ~14% on this dataset
- **Solution**: Use Isolation Forest instead
- Alternative: Increase epochs, tune hyperparameters, use different architecture

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
- Paper: Mathur et al., "SWaT: A Water Treatment Testbed for Devise- and Cyber-Physical Attacks on Cyber-Physical Systems" (2016)

### Methods
- **Isolation Forest**: Liu et al., "Isolation Forest", ICDM 2008
- **LSTM-AE**: Malhotra et al., "Long Short Term Memory Networks for Anomaly Detection in Time Series", ESANN 2015

### Tools
- PyTorch, scikit-learn, pandas, numpy

## Author

Developed for cyber-physical system anomaly detection research and comparison of deep learning vs ensemble methods.

## License

This project is provided as-is for research and educational purposes.

## Citation

If you use this work, please cite:

```bibtex
@software{swat_anomaly_detection_2026,
  title={Anomaly Detection for SWaT: Isolation Forest vs LSTM-AE},
  year={2026},
  url={https://github.com/your-repo}
}
```

## Changelog

- **v1.0** (Feb 2026): 
  - Isolation Forest implementation (F1=0.676)
  - LSTM-AE implementation and analysis
  - Comprehensive performance comparison
  - NaN imputation pipeline
