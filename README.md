# Counter-Strike Performance Forecasting

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![uv](https://img.shields.io/badge/uv-compatible-green.svg)

Time series forecasting for professional Counter-Strike player performance using deep learning.

## 📊 Overview

This project introduces a novel approach to performance forecasting in professional Counter-Strike using time series analysis. By leveraging Rating 1.0 data from HLTV.org, various forecasting models have been trained, including LSTM and Transformer networks, to predict player performance over periods of 30, 60, 120, and 150 days.

**Key Results:**
- All models significantly outperform random and naive baseline methods
- Up to 87% improvement over random walk baselines for short-term predictions
- Linear regression excels in short-term forecasts
- LSTM shows notable improvements in long-term predictions

## Disclaimer

**The goal of this GitHub repository is not to be used as a production package to scrap data from HLTV. I strongly discourage anyone from doing this.**

Rightfully, HLTV protects their website from data scraping by using Cloudflare to block bots. The scraping script here is slow to work around getting blocked, and it might still happen when running the scrapper. If it does, I advise being patient and increasing sleep times to not saturate HLTV servers.

The only goal behind this GitHub repository is to show, from the perspective of applied timeseries research, that my work and results are reproducible. I still have a copy of the dataset I used to write my paper, and will happily share it privately with HLTV's permission if you are a researcher trying to replicate the results and you do not want to wait a full day with long sleep times to gather the data.

## 🚀 Quick Start

### Installation with uv

1. Install [uv](https://github.com/astral-sh/uv):
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

1. **Baseline Methods**
   - Random Walk
   - Fixed Mean Prediction
   - Ridge Regression

2. **Classical ML**
   - Random Forest Regressor

3. **Deep Learning**
   - Bidirectional LSTM
   - GRU
   - Transformer Encoder

### Performance Comparison

| Model              | 30 days | 60 days | 120 days | 150 days |
|--------------------|---------|---------|----------|----------|
| Random Walk        | 0.0399  | 0.0649  | 0.0763   | 0.0791   |
| Ridge Regression   | **0.0091** | 0.0098  | 0.0459   | 0.0584   |
| Random Forest      | 0.0105  | **0.0086** | 0.0529   | 0.0603   |
| BiLSTM            | 0.0118  | 0.0332  | **0.0437** | **0.0418** |
| Transformer        | 0.0236  | 0.0324  | 0.0488   | 0.0469   |

*Metrics shown: RMSE (Root Mean Squared Error)*

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

If you use this work in your research, please cite (preprint in progress, will have the right bibtex ref soon):

```bibtex
@software{vincent2024csgo,
  author = {Vincent, Yvann},
  title = {Time Series Forecasting for Professional Counter-Strike Performance Prediction},
  year = {2024},
  url = {https://github.com/IUseAMouse/csgo-performance-forecasting}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Data sourced from [HLTV.org](https://www.hltv.org/)
- Built with PyTorch, scikit-learn, and tslearn

## 📧 Contact

Yvann VINCENT - yvann.vincent@gmail.com

Project Link: [https://github.com/IUseAMouse/CSForecast](https://github.com/IUseAMouse/CSForecast)