"""
ROCm XGBoost Credit Card Fraud Detection Demo (MI300X-ready)

Run:
  source ~/rocm_venv/bin/activate
  python fraud_rocm_ui_7865.py

Open:
  http://<your-host-ip>:7865

Dependencies:
  pip install pandas numpy scikit-learn matplotlib gradio pillow xgboost

Dataset:
  Download 'creditcard.csv' from Kaggle:
  https://www.kaggle.com/mlg-ulb/creditcardfraud

  Place it in the same directory as this script.
"""

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
import xgboost as xgb
import gradio as gr
from datetime import datetime
from PIL import Image

plt.switch_backend("Agg")  # server-friendly

# -------------------------
# AMD color palette
# -------------------------
AMD_TEAL = "#00C2DE"  # AMD Instinct teal
AMD_BLACK = "#1C1C1C"
AMD_GRAY = "#2B2B2B"
TEXT_WHITE = "#FFFFFF"

# -------------------------
# Data loading
# -------------------------
def load_creditcard_data(path="creditcard.csv", sample_frac=1.0):
    df = pd.read_csv(path)
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42)
    if "Class" not in df.columns:
        raise ValueError("Dataset must contain 'Class' column (0=normal, 1=fraud).")
    return df


# -------------------------
# Run pipeline
# -------------------------
def run_pipeline(data_path="creditcard.csv", sample_frac=1.0,
                 num_boost_round=400, max_depth=6, eta=0.1, threshold=0.5):
    try:
        df = load_creditcard_data(data_path, sample_frac)
    except Exception as e:
        return {"error": str(e)}

    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        "tree_method": "hist",
        "device": "gpu",
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": int(max_depth),
        "eta": float(eta),
        "scale_pos_weight": (len(y_train) - sum(y_train)) / sum(y_train),
    }

    bst = xgb.train(
        params, dtrain, num_boost_round=int(num_boost_round),
        evals=[(dtrain, "train"), (dtest, "test")],
        early_stopping_rounds=10, verbose_eval=False,
    )

    y_prob = bst.predict(dtest)
    y_pred = (y_prob > threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    plots = {}

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}", color=AMD_TEAL, lw=2)
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_title("ROC Curve", color=TEXT_WHITE)
    ax.set_xlabel("False Positive Rate", color=TEXT_WHITE)
    ax.set_ylabel("True Positive Rate", color=TEXT_WHITE)
    ax.legend(facecolor=AMD_GRAY, edgecolor="white", labelcolor=TEXT_WHITE)
    fig.patch.set_facecolor(AMD_BLACK)
    ax.set_facecolor(AMD_BLACK)
    for spine in ax.spines.values():
        spine.set_color(TEXT_WHITE)
    ax.tick_params(colors=TEXT_WHITE)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=AMD_BLACK)
    buf.seek(0)
    plots["roc"] = buf
    plt.close(fig)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Fraud"])
    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix", color=TEXT_WHITE)
    fig.patch.set_facecolor(AMD_BLACK)
    ax.set_facecolor(AMD_BLACK)
    for spine in ax.spines.values():
        spine.set_color(TEXT_WHITE)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=AMD_BLACK)
    buf.seek(0)
    plots["confmat"] = buf
    plt.close(fig)

    # Feature importance
    fig = xgb.plot_importance(bst, importance_type="gain", show_values=False)
    fig.figure.patch.set_facecolor(AMD_BLACK)
    plt.title("Feature Importance (Gain)", color=TEXT_WHITE)
    plt.xlabel("Importance", color=TEXT_WHITE)
    plt.ylabel("Feature", color=TEXT_WHITE)
    plt.xticks(color=TEXT_WHITE)
    plt.yticks(color=TEXT_WHITE)
    buf = io.BytesIO()
    fig.figure.savefig(buf, format="png", bbox_inches="tight", facecolor=AMD_BLACK)
    buf.seek(0)
    plots["feat_imp"] = buf
    plt.close(fig.figure)

    return {
        "accuracy": float(acc),
        "auc": float(auc),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "plots": plots,
    }


def buf_to_pil(buf):
    if buf is None:
        return None
    buf.seek(0)
    return Image.open(buf)


# -------------------------
# UI Logic
# -------------------------
def run_and_return_ui(data_path, sample_frac, num_boost_round, max_depth, eta, threshold):
    out = run_pipeline(data_path, sample_frac, num_boost_round, max_depth, eta, threshold)
    if "error" in out:
        return out["error"], None, None, None, None

    roc_img = buf_to_pil(out["plots"]["roc"])
    cm_img = buf_to_pil(out["plots"]["confmat"])
    feat_img = buf_to_pil(out["plots"]["feat_imp"])

    summary = (
        f"**Train samples:** {out['n_train']:,}  "
        f"**Test samples:** {out['n_test']:,}\n"
        f"**Accuracy:** {out['accuracy']:.4f}  **AUC:** {out['auc']:.4f}\n"
        f"**Threshold:** {threshold:.2f}"
    )

    return summary, roc_img, cm_img, feat_img, out["auc"]


# -------------------------
# Gradio UI Layout
# -------------------------
custom_css = f"""
#header-bar {{
    background-color: {AMD_TEAL};
    color: {TEXT_WHITE};
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 15px 25px;
    border-radius: 12px;
    margin-bottom: 20px;
}}
#header-bar h1 {{
    color: {TEXT_WHITE};
    font-family: 'Segoe UI', 'Roboto', sans-serif;
    font-weight: 600;
    font-size: 1.4em;
    margin: 0;
}}
footer {{
    visibility: hidden;
}}
.gr-button {{
    background-color: {AMD_TEAL} !important;
    color: {AMD_BLACK} !important;
    font-weight: bold;
}}
"""

with gr.Blocks(title="ROCm XGBoost Fraud Detection", css=custom_css) as demo:
    gr.HTML(f"""
    <div id="header-bar">
        <div style="display:flex;align-items:center;">
            <img src="https://upload.wikimedia.org/wikipedia/commons/7/7c/AMD_Logo.svg"
                 alt="AMD Logo" style="height:38px;margin-right:15px;">
            <h1>ROCm XGBoost Credit Card Fraud Detection (MI300X-Ready)</h1>
        </div>
    </div>
    """)

    gr.Markdown(f"""
    ### About this Demo
    This app showcases **ROCm-accelerated XGBoost** running on AMD GPUs (e.g., MI300X).  
    It detects fraudulent credit card transactions from anonymized numerical features.

    **Dataset:** `creditcard.csv` — 284,807 transactions with 30 features (V1–V28, Amount, Time).  
    The `Class` column indicates **1 = Fraud** and **0 = Legit**.

    ---
    ### 📊 What Each Output Means
    - **ROC Curve** → Measures how well the model separates fraud vs legit at all thresholds.  
      Higher **AUC** means better discrimination.
    - **Confusion Matrix** → Shows how many real frauds and normal transactions were  
      correctly or incorrectly predicted. Ideal: high diagonal values.
    - **Feature Importance** → Tells which hidden (PCA-transformed) features contribute most  
      to detecting fraud.  
      - Features `V1`–`V28` are anonymized transformations of original transaction variables.  
      - `Amount` represents transaction value; `Time` is seconds since the first transaction.
    """)

    with gr.Row():
        with gr.Column(scale=2):
            data_path = gr.Textbox(value="creditcard.csv", label="Dataset path")
            sample_frac = gr.Slider(minimum=0.05, maximum=1.0, value=1.0, step=0.05,
                                    label="Sample fraction (for faster runs)")
            num_boost = gr.Number(value=400, label="num_boost_round", precision=0)
            max_depth = gr.Number(value=6, label="max_depth", precision=0)
            eta = gr.Number(value=0.1, label="eta (learning rate)", precision=2)
            threshold = gr.Number(value=0.5, label="Fraud probability threshold")
            run_btn = gr.Button("Run on ROCm GPU")
        with gr.Column(scale=3):
            out_text = gr.Markdown()
            gr.Markdown("#### ROC Curve")
            roc_img = gr.Image(label="ROC Curve")
            gr.Markdown("#### Confusion Matrix")
            cm_img = gr.Image(label="Confusion Matrix")
            gr.Markdown("#### Feature Importance")
            feat_img = gr.Image(label="Feature Importance")
            auc_val = gr.Number(label="AUC")

    def _on_run(data_path, sample_frac, num_boost, max_depth, eta, threshold):
        return run_and_return_ui(
            data_path, float(sample_frac), int(num_boost),
            int(max_depth), float(eta), float(threshold)
        )

    run_btn.click(
        fn=_on_run,
        inputs=[data_path, sample_frac, num_boost, max_depth, eta, threshold],
        outputs=[out_text, roc_img, cm_img, feat_img, auc_val],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7865, share=False)
