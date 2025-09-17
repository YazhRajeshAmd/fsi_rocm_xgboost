"""
ROCm XGBoost Stock Demo (MI300X-ready, Fixed)

Run:
  source ~/rocm_venv/bin/activate
  python stock_rocm_ui_7865_mi300x_fixed.py

Then open http://<your-host-ip>:7865

Dependencies:
  pip install yfinance pandas numpy scikit-learn matplotlib gradio

NOTE: xgboost should be the ROCm-built package installed in your venv.
"""
import io
from PIL import Image
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import xgboost as xgb
import gradio as gr
from datetime import datetime

plt.switch_backend("Agg")  # server-friendly

# -------------------------
# Feature engineering
# -------------------------
def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=period - 1, adjust=False).mean()
    ma_down = down.ewm(com=period - 1, adjust=False).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger_bands(series, window=20, n_std=2):
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = ma + n_std * std
    lower = ma - n_std * std
    width = (upper - lower) / ma
    return upper, lower, width

# -------------------------
# Data pipeline
# -------------------------
def fetch_and_featurize(symbol, start="2015-01-01", end=None):
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    df = yf.download(symbol, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError(f"No data fetched for symbol: {symbol}")
    df = df.reset_index()

    # -------------------------
    # Force Close as Series
    # -------------------------
    if isinstance(df['Close'], pd.DataFrame):
        close_series = df['Close'].iloc[:,0]
    else:
        close_series = df['Close']
    close_series = close_series.astype(float)

    # price-based features
    df['return'] = close_series.pct_change()
    df['ret_lag1'] = df['return'].shift(1)
    df['ma5'] = close_series.rolling(5).mean()
    df['ma10'] = close_series.rolling(10).mean()
    df['ma20'] = close_series.rolling(20).mean()
    df['vol5'] = df['return'].rolling(5).std()
    df['vol20'] = df['return'].rolling(20).std()

    # technical indicators
    df['rsi14'] = rsi(close_series, period=14)
    macd_line, macd_signal, macd_hist = macd(close_series)
    df['macd'] = macd_line
    df['macd_signal'] = macd_signal
    df['macd_hist'] = macd_hist
    bb_upper, bb_lower, bb_width = bollinger_bands(close_series, window=20)
    df['bb_upper'] = bb_upper
    df['bb_lower'] = bb_lower
    df['bb_width'] = bb_width

    # price ratios (Series ÷ Series ensures no DataFrame error)
    df['close_ma5_ratio'] = close_series / df['ma5'] - 1
    df['close_ma20_ratio'] = close_series / df['ma20'] - 1

    # target: next day direction
    df['return_next'] = df['return'].shift(-1)
    df['direction'] = (df['return_next'] > 0).astype(int)

    df = df.dropna().reset_index(drop=True)
    return df

# -------------------------
# Backtest
# -------------------------
def backtest_signals(df, pred_prob, threshold=0.5):
    sig = (pred_prob > threshold).astype(int)
    strat_ret = sig * df['return_next'].values
    cum = np.cumprod(1 + strat_ret) - 1
    bh_cum = np.cumprod(1 + df['return_next'].values) - 1
    return strat_ret, cum, bh_cum

# -------------------------
# Run pipeline
# -------------------------
def run_pipeline(symbol="AAPL", start="2015-01-01", end=None,
                 num_boost_round=200, max_depth=6, eta=0.1, threshold=0.5):
    try:
        df = fetch_and_featurize(symbol, start=start, end=end)
    except Exception as e:
        return {"error": f"Failed to fetch data: {e}"}

    feature_cols = [
        "ret_lag1", "ma5", "ma10", "ma20", "vol5", "vol20",
        "rsi14", "macd", "macd_signal", "macd_hist",
        "bb_width", "close_ma5_ratio", "close_ma20_ratio", "Volume"
    ]
    X = df[feature_cols]
    y = df["direction"]

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    df_test = df.iloc[split_idx:].reset_index(drop=True)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # MI300X params
    params = {
        "tree_method": "hist",
        "device": "gpu",
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": int(max_depth),
        "eta": float(eta),
    }

    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=int(num_boost_round),
        evals=[(dtrain, "train"), (dtest, "test")],
        early_stopping_rounds=10,
        verbose_eval=False
    )

    y_prob = bst.predict(dtest)
    y_pred = (y_prob > threshold).astype(int)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    strat_ret, cum, bh_cum = backtest_signals(df_test, y_prob, threshold=float(threshold))

    # plots
    plots = {}

    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(df_test['Date'], cum, label='Model strategy')
    ax.plot(df_test['Date'], bh_cum, label='Buy & Hold')
    ax.set_title(f"Equity Curve ({symbol})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.legend()
    fig.autofmt_xdate()
    buf = io.BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight'); buf.seek(0)
    plots['equity_curve'] = buf
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0,1],[0,1],'--', color='gray')
    ax.set_title("ROC Curve")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend()
    buf = io.BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight'); buf.seek(0)
    plots['roc'] = buf
    plt.close(fig)

    try:
        fig = xgb.plot_importance(bst, importance_type='gain', show_values=False).figure
        buf = io.BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight'); buf.seek(0)
        plots['feat_imp'] = buf
        plt.close(fig)
    except Exception:
        plots['feat_imp'] = None

    return {
        "symbol": symbol,
        "used_device": params["device"],
        "accuracy": float(acc),
        "auc": float(auc),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "plots": plots,
        "message": f"Trained with params: {params}"
    }
    
def buf_to_pil(buf):
    if buf is None:
        return None
    buf.seek(0)
    return Image.open(buf)
# -------------------------
# Gradio UI
# -------------------------
def run_and_return_ui(symbol, start, end, num_boost_round, max_depth, eta, threshold):
    out = run_pipeline(symbol=symbol, start=start, end=end,
                       num_boost_round=num_boost_round, max_depth=max_depth, eta=eta, threshold=threshold)
    if "error" in out:
        return out["error"], None, None, None, None

    

    eq_img = buf_to_pil(out['plots']['equity_curve'])
    roc_img = buf_to_pil(out['plots']['roc'])
    fi_img = buf_to_pil(out['plots']['feat_imp'])

    summary = (
        f"Symbol: {out['symbol']}\n"
        f"Device: {out['used_device']}\n"
        f"Train samples: {out['n_train']}, Test samples: {out['n_test']}\n"
        f"Accuracy: {out['accuracy']:.4f}, AUC: {out['auc']:.4f}\n\n"
        f"{out.get('message','')}"
    )

    return summary, eq_img, roc_img, fi_img, out['auc']

# -------------------------
# Launch UI
# -------------------------
with gr.Blocks(title="ROCm XGBoost Stock Demo") as demo:
    gr.Markdown("## ROCm XGBoost Stock Prediction & Backtest (MI300X-ready)")
    with gr.Row():
        with gr.Column(scale=2):
            ticker = gr.Textbox(value="AAPL", label="Ticker")
            start = gr.Textbox(value="2015-01-01", label="Start date")
            end = gr.Textbox(value="", label="End date (leave empty for today)")
            num_boost = gr.Number(value=200, label="num_boost_round", precision=0)
            max_depth = gr.Number(value=6, label="max_depth", precision=0)
            eta = gr.Number(value=0.1, label="eta")
            threshold = gr.Number(value=0.5, label="Probability threshold")
            run_btn = gr.Button("Run")
        with gr.Column(scale=3):
            out_text = gr.Textbox(label="Summary / Status", lines=8)
            eq_img = gr.Image(label="Equity Curve")
            roc_img = gr.Image(label="ROC Curve")
            feat_img = gr.Image(label="Feature Importance")
            auc_val = gr.Number(label="AUC")

    def _on_run(ticker, start, end, num_boost, max_depth, eta, threshold):
        summary, eq, roc, feat, aucv = run_and_return_ui(
            ticker, start, end, int(num_boost), int(max_depth), float(eta), float(threshold))
        return summary, eq, roc, feat, aucv

    run_btn.click(fn=_on_run, inputs=[ticker, start, end, num_boost, max_depth, eta, threshold],
                  outputs=[out_text, eq_img, roc_img, feat_img, auc_val])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7865, share=False)
