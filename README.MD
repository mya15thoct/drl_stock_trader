# DRL Stock Trading with DDPG

A Deep Reinforcement Learning system for stock trading using the DDPG (Deep Deterministic Policy Gradient) algorithm. This project is designed as a contribution focusing on DDPG implementation with adaptive hyperparameters for different risk profiles.

## Overview

This system uses DDPG to learn optimal trading strategies from historical stock data. It automatically adjusts hyperparameters based on stock risk classification (Low/Medium/High Risk) for better performance.

## Features

- **DDPG Algorithm**: Single, focused implementation of Deep Deterministic Policy Gradient
- **Risk-Adaptive Hyperparameters**: Automatic parameter tuning based on stock volatility and drawdown metrics
- **Stock Risk Classification**: Analyzes 50+ US stocks to select 9 representative stocks across risk levels
- **Technical Indicators**: Uses SMA, RSI, MACD, and Bollinger Bands for state representation
- **Performance Analysis**: Compares against Moving Average, Signal Rolling, and Buy & Hold strategies

## Project Structure

```
drl-stock-trader/
├── agents/
│   ├── agent_base.py          # Base agent class
│   └── agent_ddpg.py          # DDPG implementation
├── classification/
│   ├── quick_us_analyzer.py   # Stock analysis & selection
│   ├── threshold.py           # Risk threshold calculation
│   └── us_9_stocks.txt        # Selected 9 stocks
├── dataset/
│   ├── fetch_us_50.py         # Crawl 50 US stocks
│   └── us_stock_fetcher.py    # Stock data fetcher
├── envs/
│   └── stock_env.py           # Trading environment
├── train/
│   ├── config.py              # Training configuration
│   ├── evaluator.py           # Model evaluation
│   └── replay_buffer.py       # Experience replay
├── us_dataset/                # 9 selected stocks + S&P 500 (10 CSV files)
├── utils/
│   └── visualize.py           # Results visualization
└── run.py                     # Main training/testing script
```

## Installation

```bash
# Clone repository
git clone https://github.com/mya15thoct/drl-stock-trader-with-forecast.git
cd drl-stock-trader-with-forecast

# Install dependencies
pip install torch numpy pandas matplotlib yfinance gymnasium
```

## Dataset

The project includes 9 carefully selected US stocks + S&P 500 index:

**Low Risk (3 stocks):**
- MSFT (Microsoft)
- LMT (Lockheed Martin)
- HD (Home Depot)

**Medium Risk (3 stocks):**
- AAPL (Apple)
- GOOGL (Alphabet)
- JPM (JPMorgan Chase)

**High Risk (3 stocks):**
- AMZN (Amazon)
- META (Meta)
- LOW (Lowe's)

**Market Index:**
- GSPC (S&P 500)

All data covers the period **2018-01-01 to 2025-01-01** (1761 trading days).

## Usage

### Train and Test on All 9 Stocks

```bash
python run.py --data_dir ./us_dataset/ --mode train_and_test --episodes 5
```

### Train Only

```bash
python run.py --data_dir ./us_dataset/ --mode train --stock_code AAPL
```

### Test Trained Model

```bash
python run.py --data_dir ./us_dataset/ --mode test --stock_code AAPL --episodes 10
```

### Command Line Arguments

- `--data_dir`: Directory containing stock CSV files (default: `./us_dataset/`)
- `--mode`: Operation mode - `train`, `test`, or `train_and_test` (default: `train_and_test`)
- `--stock_code`: Specific stock to process (optional, processes all if not specified)
- `--episodes`: Number of test episodes (default: 5)

## Risk-Adaptive Hyperparameters

The DDPG agent automatically adjusts its hyperparameters based on stock risk classification:

### High Risk Stocks (e.g., AMZN, META)
- Higher exploration noise (0.25)
- Larger network (512-384-256)
- Larger replay buffer (800k)
- More aggressive learning

### Medium Risk Stocks (e.g., AAPL, GOOGL, JPM)
- Moderate exploration (0.18)
- Medium network (384-256-128)
- Medium buffer (600k)
- Balanced learning

### Low Risk Stocks (e.g., MSFT, LMT, HD)
- Conservative exploration (0.12)
- Smaller network (256-128)
- Smaller buffer (400k)
- Stable learning

## Results

After training, the system generates:

1. **Trading Analysis Chart**: `{STOCK}StockTrading_DDPG_0/trading_analysis.png`
   - Portfolio value over time
   - Buy/Sell signals
   - Comparison with baseline strategies

2. **Performance Metrics**: Printed to console
   - Return percentage
   - Sharpe ratio
   - Sortino ratio
   - Maximum drawdown
   - Number of trades

3. **Detailed Trade Log**: `{STOCK}StockTrading_DDPG_0/all_detailed_trades.csv`
   - Date, price, type (buy/sell)
   - Shares, amount, fees
   - Portfolio state

## Stock Selection Process

1. **Crawl 50 diverse US stocks** across sectors
2. **Calculate metrics** for each stock:
   - Volatility (7-year standard deviation)
   - Maximum Drawdown
3. **Determine thresholds** using 33rd and 67th percentiles
4. **Select 9 stocks**: 3 from each risk category with best composite scores

See `classification/quick_us_analyzer.py` for implementation details.

## Key Components

### DDPG Agent (`agents/agent_ddpg.py`)
- Actor-Critic architecture with target networks
- Ornstein-Uhlenbeck exploration noise
- Soft target updates
- Experience replay

### Trading Environment (`envs/stock_env.py`)
- Gymnasium-compatible interface
- State: Technical indicators + position info
- Action: Continuous (-1 to +1) representing position change
- Reward: Portfolio return with transaction costs

### Training Loop (`run.py`)
- Experience collection with exploration
- Batch learning from replay buffer
- Periodic evaluation on test set
- Model checkpointing

## Contribution Focus

This implementation emphasizes:

1. **Single Algorithm Mastery**: Deep focus on DDPG rather than multiple algorithms
2. **Risk Adaptation**: Hyperparameter tuning based on stock characteristics
3. **Reproducibility**: Fixed dataset with 9 representative stocks
4. **Clean Architecture**: Modular design with clear separation of concerns

## Requirements

- Python 3.8+
- PyTorch 1.12+
- NumPy
- Pandas
- Matplotlib
- yfinance
- Gymnasium

## License

This project is for research and educational purposes.

## Citation

If you use this code in your research, please cite:

```
@misc{drl-stock-trader-ddpg,
  author = {Your Name},
  title = {DRL Stock Trading with DDPG},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/mya15thoct/drl-stock-trader-with-forecast}
}
```

## Contact

For questions or collaborations, please open an issue on GitHub.

