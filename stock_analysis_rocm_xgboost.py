# stock_analysis_rocm_xgboost_runtime.py

import pandas as pd
import numpy as np
import yfinance as yf
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score

# ---------------------------
# 1. Fetch stock data
# ---------------------------
def fetch_stock_data(symbol, start="2015-01-01", end="2025-01-01"):
    print(f"📥 Downloading data for {symbol}...")
    df = yf.download(symbol, start=start, end=end)
    df.reset_index(inplace=True)
    return df

# ---------------------------
# 2. Feature engineering
# ---------------------------
def add_features(df):
    df['return'] = df['Close'].pct_change()
    df['lag1'] = df['return'].shift(1)
    df['ma5'] = df['Close'].rolling(5).mean()
    df['ma20'] = df['Close'].rolling(20).mean()
    df['vol5'] = df['return'].rolling(5).std()
    df['return_next'] = df['return'].shift(-1)
    df['direction'] = (df['return_next'] > 0).astype(int)  # classification target
    df = df.dropna()
    return df

# ---------------------------
# 3. Train/test split
# ---------------------------
def split_data(df):
    features = ['lag1', 'ma5', 'ma20', 'vol5', 'Open', 'High', 'Low', 'Volume']
    X = df[features]
    y = df['direction']

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    return X_train, X_test, y_train, y_test

# ---------------------------
# 4. Train ROCm XGBoost model
# ---------------------------
def train_xgboost(X_train, y_train, X_test, y_test):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        "tree_method": "hist",      # ROCm HIP build backend
        'device': 'gpu',               # Use MI300X GPU
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 6,
        "eta": 0.1,
    }

    evals = [(dtrain, "train"), (dtest, "test")]
    model = xgb.train(params, dtrain, num_boost_round=200, evals=evals, early_stopping_rounds=10)
    return model, dtest, y_test

# ---------------------------
# 5. Evaluate model
# ---------------------------
def evaluate(model, dtest, y_test):
    y_pred_prob = model.predict(dtest)
    y_pred = (y_pred_prob > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)

    print(f"\n✅ Model Results:")
    print(f"   Accuracy: {acc:.4f}")
    print(f"   AUC: {auc:.4f}\n")

    xgb.plot_importance(model)
    plt.title("Feature Importance")
    plt.show()

# ---------------------------
# 6. Run full pipeline
# ---------------------------
def run_demo():
    symbol = input("Enter stock ticker symbol (e.g. AAPL, TSLA, MSFT): ").upper().strip()
    df = fetch_stock_data(symbol)
    df = add_features(df)
    X_train, X_test, y_train, y_test = split_data(df)
    model, dtest, y_test = train_xgboost(X_train, y_train, X_test, y_test)
    evaluate(model, dtest, y_test)

# ---------------------------
# Main entry point
# ---------------------------
if __name__ == "__main__":
    run_demo()
