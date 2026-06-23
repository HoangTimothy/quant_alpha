# Alpha Factor Discovery & Quantitative Backtesting

> End-to-end quantitative research pipeline for alpha factor engineering,
> ML-driven signal generation, and realistic backtesting.

---

## Architecture

```
quant_alpha/
├── configs/                    # Hydra YAML configs
│   ├── config.yaml             # Master config
│   └── model/                  # Per-model hyperparameters
│       ├── logistic.yaml
│       ├── random_forest.yaml
│       ├── xgboost.yaml
│       ├── lightgbm.yaml
│       ├── mlp.yaml
│       ├── tcnn.yaml
│       ├── transformer.yaml
│       └── tft.yaml
├── data/
│   ├── raw/                    # Downloaded OHLCV data
│   └── processed/              # Train/val/test parquets
├── src/
│   ├── data_loader/            # MarketDataset (yfinance + CSV)
│   ├── features/               # FactorEngine + AdvancedFactors + Labels
│   ├── models/                 # 8 models: LR, RF, XGB, LGBM, MLP, TCNN, Transformer, TFT
│   ├── training/               # Walk-forward validation, HPO, losses
│   ├── backtesting/            # vectorbt-powered backtest engine
│   ├── portfolio/              # Position sizing: equal/vol-target/Kelly
│   ├── evaluation/             # 30+ metrics + statistical tests
│   ├── visualization/          # Dark-themed publication plots
│   └── utils/                  # Seeding, logging, IO
├── scripts/
│   ├── train.py                # Main entry point (Hydra)
│   ├── backtest.py             # Standalone backtesting
│   ├── ablation.py             # Ablation study
│   └── generate_report.py      # Report generation
├── tests/                      # pytest suite
├── outputs/                    # Generated outputs
│   ├── plots/
│   ├── checkpoints/
│   ├── metrics.json
│   └── ablation_results.csv
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Setup

```bash
cd quant_alpha

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Training (Full Pipeline)

```bash
# Default config: XGBoost, yfinance data, walk-forward validation
python scripts/train.py

# Quick smoke test (~2 min)
python scripts/train.py training.quick_mode=true

# Override model
python scripts/train.py model=lightgbm

# Multi-model sweep
python scripts/train.py --multirun model=xgboost,lightgbm,random_forest

# Custom tickers and dates
python scripts/train.py data.tickers="[AAPL,MSFT,GOOGL]" data.start_date=2018-01-01
```

### 3. Backtesting

```bash
# Standalone backtest from saved model
python scripts/backtest.py

# Long-only with volatility targeting
python scripts/backtest.py backtest.mode=long_only backtest.position_sizing=vol_targeting
```

### 4. Ablation Study

```bash
python scripts/ablation.py
```

### 5. Report Generation

```bash
python scripts/generate_report.py
```

### 6. Tests

```bash
python -m pytest tests/ -v
```

---

## Pipeline Steps

| Step | Module | Description |
|------|--------|-------------|
| 1 | `data_loader/` | Load OHLCV from yfinance or CSV; temporal train/val/test split |
| 2 | `features/factor_engine.py` | 40+ basic factors: returns, trend, momentum, volatility, volume, price, time, lag |
| 3 | `features/advanced_factors.py` | Realized vol, rolling beta, downside deviation, entropy, Hurst exponent |
| 4 | `features/label_constructor.py` | Binary, regression, ranking, position labels |
| 5 | `features/factor_selection.py` | MI, SHAP, PCA, IC ranking, RFE |
| 6 | `models/` | 8 models: Logistic, RF, XGBoost, LightGBM, MLP, Temporal CNN, Transformer, TFT |
| 7 | `training/trainer.py` | Walk-forward, expanding, rolling window CV |
| 8 | `backtesting/` | vectorbt engine with costs, sizing, risk controls |
| 9 | `evaluation/` | 30+ metrics + bootstrap CI + t-tests |
| 10 | `visualization/` | 6 publication plots (dark theme) |

---

## Models

| # | Model | Type | Key Features |
|---|-------|------|-------------|
| 1 | Logistic Regression | Baseline | L2 regularisation, StandardScaler |
| 2 | Random Forest | Baseline | 500 trees, balanced class weights |
| 3 | XGBoost | Gradient Boosting | Early stopping, GPU support |
| 4 | LightGBM | Gradient Boosting | Leaf-wise growth, early stopping |
| 5 | MLP | Deep Learning | 3 hidden layers, BatchNorm, dropout |
| 6 | Temporal CNN | Deep Learning | 1D convolutions over time windows |
| 7 | Transformer Encoder | Deep Learning | Multi-head attention, positional encoding |
| 8 | Temporal Fusion Transformer | Deep Learning | Variable selection, LSTM + attention |

---

## Evaluation Metrics

### Prediction
- Accuracy, Precision, Recall, F1, ROC-AUC
- RMSE, MAE, MAPE
- IC, Rank IC

### Trading
- Sharpe, Sortino, Calmar Ratio
- Max Drawdown, CAGR
- Profit Factor, Win Rate, Turnover

### Statistical
- Bootstrap CI for Sharpe ratio
- t-test for mean returns
- Paired t-test for model comparison

---

## Configuration

All parameters are managed via [Hydra](https://hydra.cc/):

```yaml
# configs/config.yaml (excerpt)
data:
  source: yfinance
  tickers: [AAPL, MSFT, GOOGL, NVDA, SPY, QQQ]
  train_end: "2021-12-31"
  val_end: "2023-12-31"

model:
  name: xgboost
  params:
    n_estimators: 1000
    max_depth: 6

backtest:
  mode: long_short
  commission_bps: 10
  slippage_bps: 5
```

Override any parameter from the command line:
```bash
python scripts/train.py model.params.n_estimators=500 backtest.mode=long_only
```

---

## Reproducibility

1. **Seed**: All random operations seeded via `set_seed(42)` (numpy, torch, python)
2. **Temporal splits**: No random shuffling — strict date-based train/val/test
3. **Deterministic**: `torch.backends.cudnn.deterministic = True`
4. **Config versioning**: Every run logs its full Hydra config
5. **Checkpoints**: Models saved to `outputs/checkpoints/`

---

## Experiment Tracking

Optional integration with wandb and MLflow:

```bash
# Enable wandb
python scripts/train.py experiment.use_wandb=true experiment.wandb_project=my_project

# Enable MLflow
python scripts/train.py experiment.use_mlflow=true
```

---

## CV / Resume Bullet Examples

- Engineered 50+ alpha factors (technical, statistical, cross-sectional) and trained 8 ML/DL models achieving **X% ROC-AUC** on walk-forward validation with realistic transaction cost assumptions.
- Built an end-to-end quantitative research pipeline in Python processing 10+ years of multi-asset OHLCV data through feature engineering, ML training, and backtesting, generating **Sharpe X.XX** strategy with walk-forward validation.
- Implemented Temporal Fusion Transformer and Transformer Encoder architectures for financial time series prediction with early stopping, cosine annealing, and gradient clipping.
- Designed modular backtesting engine supporting long/short strategies, 3 position sizing methods (equal weight, vol targeting, Kelly criterion), configurable transaction costs, and risk controls (stop-loss, take-profit, max drawdown).
- Conducted ablation studies demonstrating the marginal value of advanced factor engineering, hyperparameter optimization, and feature selection on out-of-sample strategy performance.

---

## License

Research / educational use. Not financial advice.
