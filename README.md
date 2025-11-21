# Counter-Strike Performance Forecasting

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![uv](https://img.shields.io/badge/uv-compatible-green.svg)

Time series forecasting for professional Counter-Strike player performance using deep learning.

## 📊 Overview

This project introduces a novel approach to performance forecasting in professional Counter-Strike using time series analysis. By leveraging Rating 1.0 data from HLTV.org, various forecasting models have been trained, including LSTM and Transformer networks, to predict player performance over periods of 30, 60, 120, and 150 days.

**Key Results:**
- The **Transformer** model significantly outperforms all baselines and other architectures across all horizons at very high confidence.
- Up to **40% improvement** over random walk baselines.
- Random Forest struggle with extrapolation on long horizons.
- Deep Learning approaches (specifically Attention-based) demonstrate robust performance on long-term forecasts that statistical methods

## ⚠️ Disclaimer : Data Collection & Terms of Service

**This repository includes the scraping logic solely for the sake of methodological transparency and reproducibility.**

I **strongly discourage** using this codebase to scrape HLTV.org actively. HLTV employs strict anti-bot measures (Cloudflare) to protect their infrastructure. The scraper included here is designed for low-volume, compliant data gathering, but using it may still result in IP bans or violate HLTV's Terms of Service.

**For Researchers:**
The goal of this repository is to allow the replication of the results presented in my paper. To avoid burdening HLTV's servers or dealing with anti-bot blocking, I am willing to share the dataset privately for academic replication purposes.

Please contact me at *yvann.vincent@gmail.com* to request access to the dataset used in the paper.

## 🚀 Quick Start

### Installation with uv

1. Install [uv]:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone the repository:
```bash
git clone https://github.com/IUseAMouse/csgo-performance-forecasting.git
cd csgo-performance-forecasting
```

3. Create and activate virtual environment:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

4. Install dependencies:
```bash
uv pip install -e ".[dev]"
```

### Alternative: pip installation

```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

## 📁 Project Structure

```
csgo-performance-forecasting/
├── data/                   # Data directory
│   ├── metadata/           # Metadata of players to allow for retrieval
│   ├── raw/                # Raw scraped data
│   ├── processed/          # Cleaned and preprocessed data
│   └── models/             # Trained model checkpoints
├── src/csgo_forecasting/   # Source code
│   ├── data/               # Data scraping and preprocessing
│   ├── models/             # Model architectures
│   ├── training/           # Training utilities
│   └── evaluation/         # Evaluation metrics
└── scripts/                # Executable scripts
```

## 🎯 Usage

### 1. Download Data

```bash
# Using make
make download-data

# Or directly
python scripts/download_data.py
```

### 2. Preprocess Data

```bash
# Using make
make preprocess

# Or directly
python scripts/preprocess_data.py --input data/raw/player_data_all.json \
                                   --output data/processed/player_data_all_cleaned.json
```

### 3. Train Models

```bash
# Train all models
make train-all

# Train specific model
python scripts/train_models.py --model lstm --seq-length 90 --out-length 30

# Available models: lstm, transformer, gru, ridge, random_forest
```

### 4. Evaluate Models

```bash
make evaluate

# Or
python scripts/evaluate_models.py --models-dir data/models
```

## 📊 Dataset

The dataset comprises **832 professional Counter-Strike players** meeting the following criteria:
- Minimum 200 maps played at top-tier competition
- Competed at a level where their team was ranked within the top 50 globally

**Statistics:**
- Average time series length: 1,742 days (~4.8 years)
- Total rating observations: 1,449,752
- Players range from top-rated performers (ZywOo, s1mple, sh1ro) to lower-rated players (HUNDEN, RuFire, OCEAN)

## 🧠 Models

### Implemented Models

1. **Baseline Method**
   - Random Walk
   
2. **Statistical Methods**
   - AutoARIMA
   - ETS

3. **Classical ML**
   - Random Forest Regressor
   - Ridge Regression

4. **Deep Learning**
   - LSTM 
   - GRU
   - Transformer (with RoPE & RevIN)

### Comprehensive Performance Evaluation

The table below presents the detailed performance of all models across different forecasting horizons. Statistical significance is calculated against the Random Walk baseline using a paired t-test.

| Horizon | Model | RMSE | $R^2$ | Improv. vs RW | Win Rate | Cohen's $d$ | $p < 0.05$ | $p < 0.01$ | $p < 0.001$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **30 Days** | **Transformer** | **0.0102** | **0.98** | **+40.26%** | 72.6% | 0.61 | ✔️ | ✔️ | ✔️ |
| | ETS | 0.0126 | 0.97 | +26.47% | 75.0% | 0.43 | ✔️ | ✔️ | ✔️ |
| | ARIMA | 0.0130 | 0.97 | +24.07% | 71.4% | 0.41 | ✔️ | ✔️ | ✔️ |
| | Ridge | 0.0134 | 0.97 | +21.93% | 57.1% | 0.27 | ✔️ | - | - |
| | GRU | 0.0144 | 0.97 | +16.14% | 57.1% | 0.18 | - | - | - |
| | LSTM | 0.0167 | 0.95 | +2.79% | 48.8% | 0.03 | - | - | - |
| | Random Walk | 0.0171 | 0.95 | - | - | - | - | - | - |
| | Random Forest | 0.0206 | 0.93 | -20.35% | 56.0% | -0.10 | - | - | - |
| **60 Days** | **Transformer** | **0.0270** | **0.88** | **+30.73%** | 57.1% | 0.39 | ✔️ | ✔️ | ✔️ |
| | Ridge | 0.0317 | 0.83 | +18.72% | 53.6% | 0.18 | - | - | - |
| | GRU | 0.0338 | 0.81 | +13.14% | 50.0% | 0.10 | - | - | - |
| | LSTM | 0.0363 | 0.78 | +6.93% | 44.0% | 0.00 | - | - | - |
| | ETS | 0.0364 | 0.78 | +6.68% | 65.5% | 0.16 | - | - | - |
| | ARIMA | 0.0386 | 0.75 | +0.82% | 58.3% | 0.07 | - | - | - |
| | Random Walk | 0.0390 | 0.75 | - | - | - | - | - | - |
| | Random Forest | 0.0551 | 0.50 | -41.55% | 44.0% | -0.26 | ✔️ | - | - |
| **120 Days** | **Transformer** | **0.0440** | **0.62** | **+34.48%** | 71.4% | 0.49 | ✔️ | ✔️ | ✔️ |
| | Ridge | 0.0568 | 0.37 | +15.41% | 47.6% | 0.11 | - | - | - |
| | Random Walk | 0.0671 | 0.12 | - | - | - | - | - | - |
| | LSTM | 0.0680 | 0.09 | -1.33% | 38.1% | -0.10 | - | - | - |
| | GRU | 0.0682 | 0.09 | -1.54% | 41.7% | -0.10 | - | - | - |
| | ETS | 0.0689 | 0.07 | -2.67% | 53.6% | 0.07 | - | - | - |
| | Random Forest | 0.0732 | -0.05 | -9.05% | 42.9% | -0.14 | - | - | - |
| | ARIMA | 0.0759 | -0.13 | -13.09% | 51.2% | -0.05 | - | - | - |
| **150 Days** | **Transformer** | **0.0491** | **0.50** | **+26.02%** | 66.7% | 0.42 | ✔️ | ✔️ | ✔️ |
| | Ridge | 0.0625 | 0.19 | +5.83% | 48.8% | 0.03 | - | - | - |
| | Random Walk | 0.0664 | 0.09 | - | - | - | - | - | - |
| | LSTM | 0.0726 | -0.08 | -9.27% | 40.5% | -0.20 | - | - | - |
| | GRU | 0.0751 | -0.16 | -13.13% | 33.3% | -0.22 | ✔️ | - | - |
| | Random Forest | 0.0834 | -0.43 | -25.62% | 41.7% | -0.25 | ✔️ | - | - |
| | ARIMA | 0.0913 | -0.72 | -37.46% | 36.9% | -0.26 | ✔️ | - | - |
| | ETS | 0.0951 | -0.86 | -43.21% | 44.0% | -0.22 | - | - | - |

*Note: A checkmark (✔️) indicates the p-value is below the threshold. For Random Forest, GRU, and ARIMA at 150 days (and RF at 60 days), the significance indicates they are significantly **worse** than the Random Walk baseline.*

*Metrics shown: RMSE (Root Mean Squared Error) - Lower is better*

## 🛠️ Development

### Code Quality

```bash
make format

make lint

make typecheck
```

### All Quality Checks

```bash
make check-all
```

## 📝 Citation

> 🚀 **Preprint submitted to HAL/arXiv.** 
> Paper title: "Time Series Forecasting for Professional Counter-Strike Player Performance"

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Data sourced from [HLTV.org]
- Built with PyTorch, scikit-learn, and tslearn

## 📧 Contact

Yvann VINCENT - yvann.vincent@gmail.com

Project Link: [https://github.com/IUseAMouse/CSForecast]