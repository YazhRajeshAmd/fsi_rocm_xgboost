ROCm XGBoost Stock Prediction & Backtest Demo (MI300X-ready)
Overview

This project is a GPU-accelerated stock prediction and backtesting demo built using XGBoost compiled for AMD ROCm. It allows users to explore technical analysis-based stock predictions, visualize backtest results, and evaluate model performance — all in a browser-based UI.

The demo is designed to leverage AMD MI300X GPUs via ROCm, providing fast model training and evaluation for historical stock data.

Features

Dynamic stock data fetching: Users can input any stock symbol (e.g., AAPL, MSFT, TSLA) and retrieve historical data from Yahoo Finance in real-time.

Technical indicator-based features:

Moving Averages: MA5, MA10, MA20

RSI (Relative Strength Index)

MACD (Moving Average Convergence Divergence)

Bollinger Bands

Price ratios vs moving averages

GPU-accelerated model training: XGBoost uses ROCm HIP backend on MI300X.

Binary classification: Predicts next-day direction (up/down).

Backtesting: Simulates a simple trading strategy based on predicted signals and compares against buy & hold.

Interactive visualizations:

Equity Curve: Strategy vs Buy & Hold

ROC Curve: Model performance

Feature Importance: Gain-based feature impact

Gradio Web UI: Accessible from browser, interactive, user-friendly.

Demo Screenshot (Example)


Equity curve visualization comparing strategy vs buy & hold

Installation

Clone the repository

git clone <repo_url>
cd rocm_finance


Create Python virtual environment

python3 -m venv ~/rocm_venv
source ~/rocm_venv/bin/activate


Install dependencies

pip install yfinance pandas numpy scikit-learn matplotlib gradio pillow


Install ROCm XGBoost

Follow XGBoost ROCm build instructions
 to compile and install XGBoost with HIP support for your MI300X.

Running the Demo
source ~/rocm_venv/bin/activate
python stock_rocm_ui_7865_mi300x_gradio_fixed.py


Opens the demo on http://0.0.0.0:7865

Accessible on local network if firewall allows

Enter a ticker symbol and optional parameters (start date, end date, model hyperparameters)

Click Run to fetch data, train model, backtest, and visualize results

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

Technical Details

Data Pipeline

Fetches historical stock data via yfinance.

Computes daily returns, lag features, moving averages, volatility, and technical indicators.

Target: direction = 1 if next-day return > 0.

Model

XGBoost (ROC/HIP GPU)

Tree-based gradient boosting classifier

Binary logistic objective

Metric: AUC

Device: MI300X GPU (device="gpu")

Backtesting

Signals generated from predicted probabilities

Simple strategy: go long if probability > threshold

Computes cumulative returns vs buy & hold

Plots

Equity curve (strategy vs buy & hold)

ROC curve

Feature importance (gain)

Gradio Integration

Real-time interactivity

Converts Matplotlib plots to PIL Images for Gradio

Runs on port 7865

Notes

ROCm XGBoost must be compiled for HIP to leverage MI300X GPU acceleration.

The demo is designed for research and demonstration purposes — not for live trading.

Data is historical; predictions are based on past patterns.

Future Improvements

Add live GPU utilization monitoring

Allow batch prediction for multiple symbols

Add downloadable report for backtest results

Improve feature engineering (momentum, volume patterns)

References

XGBoost Documentation

Yahoo Finance API via yfinance

Gradio Documentation