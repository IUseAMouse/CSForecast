# Counter-Strike Performance Forecasting

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![uv](https://img.shields.io/badge/uv-compatible-green.svg)

Time series forecasting for professional Counter-Strike player performance using deep learning.

## 📊 Overview

This project introduces a rigorous approach to performance forecasting in professional Counter-Strike using time series analysis. Leveraging Rating 1.0 data from HLTV.org, multiple forecasting models are evaluated — including classical statistical methods, machine learning baselines, and deep learning architectures — to predict player performance over horizons of 30, 60, 120, and 150 days.

**Key Results:**
- All learning-based models **significantly outperform the Random Walk baseline** across all horizons.
- **Deep learning models (GRU, LSTM, Transformer)** achieve the strongest overall performance, particularly at short and medium horizons.
- Performance differences between architectures narrow as the forecast horizon increases, indicating a shared modeling ceiling on long-term uncertainty.
- Tree-based methods (Random Forest) perform competitively at short horizons but remain less robust for long-term extrapolation.
- Improvements over Random Walk range from **~50–55% at 30–60 days** to **~33–37% at 120–150 days**, all at very high statistical confidence.

## ⚠️ Disclaimer : Data Collection & Terms of Service

**This repository includes the scraping logic solely for the sake of methodological transparency and reproducibility.**

I **strongly discourage** using this codebase to scrape HLTV.org actively. HLTV employs strict anti-bot measures (Cloudflare) to protect their infrastructure. The scraper included here is designed for low-volume, compliant data gathering, but using it may still result in IP bans or violate HLTV's Terms of Service.

**For Researchers:**  
The goal of this repository is to allow replication of the results presented in my paper. To avoid burdening HLTV's servers or dealing with anti-bot blocking, I am willing to share the dataset privately for academic replication purposes.

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
├── data/
│   ├── metadata/
│   ├── raw/
│   ├── processed/
│   └── models/
├── src/csgo_forecasting/
│   ├── data/
│   ├── models/
│   ├── training/
│   └── evaluation/
└── scripts/
```

## 🎯 Usage

### 1. Download Data

```bash
make download-data
# or
python scripts/download_data.py
```

### 2. Preprocess Data

```bash
make preprocess
# or
python scripts/preprocess_data.py --input data/raw/player_data_all.json \
                                   --output data/processed/player_data_all_cleaned.json
```

### 3. Train Models

```bash
make train-all
# or
python scripts/train_models.py --model lstm --seq-length 90 --out-length 30
```

Available models: `lstm`, `gru`, `transformer`, `ridge`, `random_forest`

### 4. Evaluate Models

```bash
make evaluate
# or
python scripts/evaluate_models.py --models-dir data/models
```

## 📊 Dataset

The dataset includes **832 professional Counter-Strike players** meeting the following criteria:
- Minimum 200 maps played at top-tier competition
- Team ranked within the global top 50 during active competition

**Statistics:**
- Average time series length: 1,742 days (~4.8 years)
- Total rating observations: 1,449,752
- Players span the full competitive spectrum, from elite stars to lower-rated professionals

## 🧠 Models

### Implemented Models

1. **Baseline**
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
   - Transformer (RoPE + RevIN)

### Comprehensive Performance Evaluation

Updated model performance across all forecasting horizons. Statistical significance is computed against the Random Walk baseline using paired tests.

| Horizon | Model | RMSE | MAE | MAPE | $R^2$ | Improv. vs RW | Win Rate | Cohen's $d$ | $p<0.05$ | $p<0.01$ | $p<0.001$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **30 Days** | GRU | 0.0096 | 0.0057 | 0.58 | 0.99 | +54.7% | 81.0% | 0.69 | ✔️ | ✔️ | ✔️ |
| | LSTM | 0.0095 | 0.0061 | 0.63 | 0.99 | +55.2% | 73.8% | 0.64 | ✔️ | ✔️ | ✔️ |
| | Transformer | 0.0102 | 0.0067 | 0.69 | 0.99 | +51.9% | 76.2% | 0.61 | ✔️ | ✔️ | ✔️ |
| | Random Forest | 0.0103 | 0.0067 | 0.70 | 0.99 | +51.6% | 73.8% | 0.62 | ✔️ | ✔️ | ✔️ |
| | Ridge | 0.0120 | 0.0077 | 0.80 | 0.98 | +43.5% | 76.2% | 0.61 | ✔️ | ✔️ | ✔️ |
| | Random Walk | 0.0212 | 0.0132 | 1.36 | 0.95 | – | – | – | – | – | – |
| **60 Days** | LSTM | 0.0206 | 0.0131 | 1.35 | 0.95 | +54.9% | 77.4% | 0.62 | ✔️ | ✔️ | ✔️ |
| | Transformer | 0.0221 | 0.0141 | 1.44 | 0.95 | +51.8% | 71.4% | 0.59 | ✔️ | ✔️ | ✔️ |
| | GRU | 0.0225 | 0.0140 | 1.44 | 0.95 | +50.9% | 75.0% | 0.57 | ✔️ | ✔️ | ✔️ |
| | Random Walk | 0.0458 | 0.0284 | 2.89 | 0.77 | – | – | – | – | – | – |
| **120 Days** | GRU | 0.0444 | 0.0306 | 3.12 | 0.77 | +42.5% | 75.0% | 0.63 | ✔️ | ✔️ | ✔️ |
| | Transformer | 0.0511 | 0.0355 | 3.60 | 0.70 | +33.8% | 70.2% | 0.48 | ✔️ | ✔️ | ✔️ |
| | Random Walk | 0.0773 | 0.0543 | 5.56 | 0.31 | – | – | – | – | – | – |
| **150 Days** | GRU | 0.0536 | 0.0367 | 3.74 | 0.66 | +37.2% | 67.9% | 0.63 | ✔️ | ✔️ | ✔️ |
| | Transformer | 0.0559 | 0.0392 | 3.97 | 0.63 | +34.5% | 66.7% | 0.59 | ✔️ | ✔️ | ✔️ |
| | Random Walk | 0.0853 | 0.0603 | 6.07 | 0.14 | – | – | – | – | – | – |

*Metrics: RMSE, MAE, MAPE (lower is better). All reported improvements are relative to the Random Walk baseline.*

## 🛠️ Development

```bash
make format
make lint
make typecheck
```

```bash
make check-all
```

## 📝 Citation

> 🚀 **Preprint submitted to HAL/arXiv**  
> *Time Series Forecasting for Professional Counter-Strike Player Performance*

## 📄 License

MIT License — see the [LICENSE](LICENSE) file.

## 📧 Contact

Yvann VINCENT — yvann.vincent@gmail.com  
Project link: https://github.com/IUseAMouse/CSForecast
