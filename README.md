# ROCm XGBoost Stock Prediction & Backtest Demo (MI300X-ready)

## Overvie## UI Inputs

| Parameter | Description | Default |
|-----------|-------------|---------|
| **Ticker** | Stock symbol (e.g., AAPL, MSFT, TSLA) | AAPL |
| **Start Date** | Historical data start (YYYY-MM-DD) | 2019-01-01 |
| **End Date** | Historical data end (YYYY-MM-DD), leave empty for today | (today) |
| **num_boost_round** | Number of boosting iterations | 1000 |
| **max_depth** | Maximum depth of trees | 6 |
| **eta** | Learning rate | 0.05 |
| **Probability Threshold** | Threshold for classifying "up" signal | 0.5 |

## UI Outputs

- **Summary / Status**: Training info, dataset size, accuracy and AUC metrics
- **Equity Curve**: Strategy performance vs Buy & Hold over time
- **ROC Curve**: Model classification performance with AUC visualization
- **Feature Importance**: Gain-based importance ranking of technical indicators
- **AUC**: Numeric ROC-AUC score for model evaluation GPU-accelerated stock prediction and backtesting demo built using XGBoost compiled for AMD ROCm. It allows users to explore technical analysis-based stock predictions, visualize backtest results, and evaluate model performance — all in a browser-based UI powered by Gradio.

The demo is designed to leverage AMD MI300X GPUs via ROCm, providing fast model training and evaluation for historical stock data with real-time interactive visualizations.

## Features

- **Dynamic stock data fetching**: Users can input any stock symbol (e.g., AAPL, MSFT, TSLA) and retrieve historical data from Yahoo Finance in real-time.

- **Comprehensive technical indicator-based features**:
  - Moving Averages: MA5, MA10, MA20, MA50, MA100
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence) with signal and histogram
  - Bollinger Bands with band width calculation
  - Price ratios vs moving averages
  - Volatility measures (5, 20, 50-day rolling)
  - Moving average crossover signals

- **Market-wide features**: Incorporates S&P 500 (^GSPC) and NASDAQ (^IXIC) data for broader market context

- **GPU-accelerated model training**: XGBoost uses ROCm HIP backend on MI300X with configurable hyperparameters

- **Binary classification**: Predicts next-day direction (up/down) with probability scores

- **Advanced backtesting**: Simulates trading strategies based on predicted signals and compares against buy & hold with customizable probability thresholds

- **Interactive visualizations**:
  - **Equity Curve**: Strategy performance vs Buy & Hold over time
  - **ROC Curve**: Model classification performance with AUC score
  - **Feature Importance**: Gain-based feature impact analysis

- **Gradio Web UI**: Modern, accessible browser interface with real-time parameter adjustment

Demo Screenshot (Example)


Equity curve visualization comparing strategy vs buy & hold

## Installation

### 1. Clone the repository
```bash
git clone <repo_url>
cd rocm_finance
```

### 2. Create Python virtual environment
```bash
python3 -m venv ~/rocm_venv
source ~/rocm_venv/bin/activate
```

### 3. Install dependencies
```bash
pip install yfinance pandas numpy scikit-learn matplotlib gradio pillow
```

### 4. Install ROCm XGBoost
Follow [XGBoost ROCm build instructions](https://xgboost.readthedocs.io/en/latest/build.html#building-with-gpu-support) to compile and install XGBoost with HIP support for your MI300X.

## Running the Demo
```bash
source ~/rocm_venv/bin/activate
python rocm_finance_xgboost.py
```

- Opens the demo on `http://0.0.0.0:7865`
- Accessible on local network if firewall allows port 7865
- Enter a ticker symbol and optional parameters (start date, end date, model hyperparameters)
- Click **Run** to fetch data, train model, backtest, and visualize results

UI Inputs
Parameter	Description	Default
Ticker	Stock symbol (e.g., AAPL)	AAPL
Start Date	Historical data start (YYYY-MM-DD)	2015-01-01
End Date	Historical data end (YYYY-MM-DD), leave empty for today	(today)
num_boost_round	Number of boosting iterations	200
max_depth	Maximum depth of trees	6
eta	Learning rate	0.1
Probability Threshold	Threshold for classifying “up” signal	0.5
UI Outputs

Summary / Status: Training info, dataset size, metrics

Equity Curve: Strategy vs Buy & Hold

ROC Curve: Model performance

Feature Importance: Gain-based importance of features

AUC: Numeric ROC-AUC value

## Technical Details

### Data Pipeline
- Fetches historical stock data via `yfinance` API
- Computes comprehensive technical indicators:
  - Daily returns and lagged returns
  - Multiple moving averages (5, 10, 20, 50, 100-day)
  - Volatility measures (rolling standard deviation)
  - RSI with 14-day period
  - MACD with signal line and histogram
  - Bollinger Bands with band width calculation
  - Moving average crossover signals
- Incorporates market-wide features from S&P 500 and NASDAQ indices
- **Target**: `direction = 1` if next-day return > 0, else 0

### Model Configuration
- **XGBoost** with ROCm HIP GPU acceleration
- **Tree method**: `hist` (optimized for GPU)
- **Device**: `gpu` (utilizes MI300X)
- **Objective**: Binary logistic regression
- **Evaluation metric**: AUC (Area Under Curve)
- **Early stopping**: 10 rounds to prevent overfitting
- **Hyperparameters**: Configurable via UI (boost rounds, max depth, learning rate)

### Backtesting Strategy
- Generates trading signals from predicted probabilities
- **Strategy**: Go long when `probability > threshold`
- Computes cumulative returns vs buy & hold baseline
- **Risk-free assumption**: No transaction costs or slippage included
- Performance visualization through equity curves

### Visualization Pipeline
- **Matplotlib** plots converted to PIL Images for Gradio compatibility
- **Equity Curve**: Time series comparison of strategy vs benchmark
- **ROC Curve**: Classification performance with AUC score
- **Feature Importance**: XGBoost gain-based feature ranking
- AMD branding with custom color scheme (AMD Red: #ED1C24)

## Notes

- **ROCm XGBoost** must be compiled for HIP to leverage MI300X GPU acceleration
- The demo is designed for **research and demonstration purposes** — not for live trading
- **Historical data only**: Predictions are based on past patterns and technical indicators
- **No financial advice**: This tool is for educational and research purposes only
- **Performance disclaimer**: Past performance does not guarantee future results

## System Requirements

- **AMD MI300X GPU** with ROCm drivers installed
- **ROCm 5.0+** with HIP support
- **Python 3.8+**
- **XGBoost compiled with ROCm support**
- Minimum 8GB system RAM (16GB recommended for larger datasets)
- Network access for Yahoo Finance API calls

## Future Improvements

- Add real-time GPU utilization monitoring and performance metrics
- Enable batch prediction for multiple symbols simultaneously
- Add downloadable PDF reports for backtest results
- Implement advanced feature engineering (momentum indicators, volume patterns)
- Add support for different asset classes (crypto, forex, commodities)
- Include risk management features (stop-loss, position sizing)
- Add model interpretability features (SHAP values, partial dependence plots)

## References

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [XGBoost ROCm GPU Support](https://xgboost.readthedocs.io/en/latest/gpu/index.html)
- [Yahoo Finance API via yfinance](https://pypi.org/project/yfinance/)
- [Gradio Documentation](https://gradio.app/docs/)
- [AMD ROCm Documentation](https://rocmdocs.amd.com/)
- [Technical Analysis Indicators Reference](https://www.investopedia.com/technical-analysis-4689657)

## License

This project is provided for educational and research purposes. Please ensure compliance with your organization's policies regarding financial data usage and model deployment.