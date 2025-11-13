"""
ROCm XGBoost Credit Card Fraud Detection (Train/Test UI with Time Comparison)
"""

import io
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import xgboost as xgb
import gradio as gr
from PIL import Image

plt.switch_backend("Agg")

# -------------------------
# AMD colors
# -------------------------
AMD_TEAL = "#00C2DE"
AMD_BLACK = "#1C1C1C"
TEXT_WHITE = "#FFFFFF"


# -------------------------
# Train/test ROCm pipeline
# -------------------------
def train_and_predict(data_path="creditcard.csv", max_depth=6, eta=0.1, num_round=400, compare_cpu=False):
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        return f"❌ Error loading dataset: {e}", None, None, None, None, None

    if "Class" not in df.columns:
        return "❌ Dataset must contain 'Class' column.", None, None, None, None, None

    # Split 80-20
    X = df.drop(columns=["Class"])
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # -------------------------
    # GPU training (ROCm)
    # -------------------------
    gpu_params = {
        "tree_method": "hist",
        "device": "gpu",
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": int(max_depth),
        "eta": float(eta),
        "scale_pos_weight": (len(y_train) - sum(y_train)) / sum(y_train),
    }

    print("🚀 Training XGBoost model on ROCm GPU...")
    start_gpu = time.time()
    bst_gpu = xgb.train(
        gpu_params,
        dtrain,
        num_boost_round=int(num_round),
        evals=[(dtrain, "train"), (dtest, "test")],
        early_stopping_rounds=10,
        verbose_eval=False,
    )
    gpu_time = time.time() - start_gpu

    y_prob = bst_gpu.predict(dtest)
    y_pred = (y_prob > 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    # -------------------------
    # Optional CPU training for comparison
    # -------------------------
    cpu_time = None
    if compare_cpu:
        cpu_params = gpu_params.copy()
        cpu_params["device"] = "cpu"
        print("🧠 Training XGBoost model on CPU for comparison...")
        start_cpu = time.time()
        bst_cpu = xgb.train(
            cpu_params,
            dtrain,
            num_boost_round=int(num_round),
            evals=[(dtrain, "train"), (dtest, "test")],
            early_stopping_rounds=10,
            verbose_eval=False,
        )
        cpu_time = time.time() - start_cpu

    # -------------------------
    # Confusion Matrix Plot
    # -------------------------
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Fraud"])
    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix", color=TEXT_WHITE)
    fig.patch.set_facecolor(AMD_BLACK)
    ax.set_facecolor(AMD_BLACK)
    for spine in ax.spines.values():
        spine.set_color(TEXT_WHITE)
    ax.tick_params(colors=TEXT_WHITE)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor=AMD_BLACK)
    buf.seek(0)
    cm_img = Image.open(buf)
    plt.close(fig)

    # -------------------------
    # Build predictions DataFrame
    # -------------------------
    pred_df = X_test.copy()
    pred_df["True_Label"] = y_test.values
    pred_df["Predicted_Label"] = y_pred
    pred_df["Fraud_Probability"] = y_prob
    top20 = pred_df.sort_values("Fraud_Probability", ascending=False).head(20).reset_index(drop=True)

    pred_csv = "fraud_predictions.csv"
    pred_df.to_csv(pred_csv, index=False)

    # -------------------------
    # Summary text with timing
    # -------------------------
    summary = (
        f"### 📊 Model Summary\n"
        f"**Train samples:** {len(X_train):,}  **Test samples:** {len(X_test):,}\n"
        f"**Accuracy:** {acc:.4f}  **AUC:** {auc:.4f}\n"
        f"**🕒 GPU Training Time:** {gpu_time:.2f} seconds\n"
    )

    if cpu_time is not None:
        summary += f"**🧠 CPU Training Time:** {cpu_time:.2f} seconds\n"
        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
        summary += f"**🚀 Speedup (GPU vs CPU):** {speedup:.2f}× faster\n"

    summary += f"\n✅ Predictions saved to `{pred_csv}`"

    return summary, cm_img, auc, acc, top20, pred_csv


# -------------------------
# Gradio UI with AMD logo
# -------------------------
custom_css = f"""
#header {{
    background-color: {AMD_TEAL};
    color: {TEXT_WHITE};
    padding: 15px 20px;
    border-radius: 10px;
    text-align: left;
    font-size: 1.2em;
    font-weight: bold;
    display: flex;
    justify-content: space-between;
    align-items: center;
}}
#header img {{
    height: 40px;
}}
footer {{visibility: hidden;}}
.gr-button {{
    background-color: {AMD_TEAL} !important;
    color: {AMD_BLACK} !important;
    font-weight: bold;
}}
"""

with gr.Blocks(css=custom_css, title="ROCm XGBoost Fraud Detection") as demo:
    gr.HTML(f"""
    <div style="position: relative; padding: 20px; background: linear-gradient(135deg, #f8f9fa 0%, #00c2de 100%); border-radius: 10px; margin-bottom: 20px;">
        <h1 style="margin:0; font-weight: 700;">AMD Instinct MI3xx ROCm-Powered XGBoost Credit Card Fraud Detection Demo</h1>
        <img src="https://upload.wikimedia.org/wikipedia/commons/7/7c/AMD_Logo.svg"
             alt="AMD Logo" style="position: absolute; top: 10px; right: 20px; height: 50px;">
    </div>
    """)

    gr.Markdown("""
    This app:
    1. Loads **creditcard.csv** from Kaggle  
    2. Splits into 80% train / 20% test  
    3. Trains XGBoost on **AMD ROCm GPU**  
    4. Optionally compares with CPU  
    5. Shows top 20 most suspicious transactions
    """)

    with gr.Row():
        with gr.Column(scale=2):
            data_path = gr.Textbox(value="creditcard.csv", label="Dataset path")
            max_depth = gr.Number(value=6, label="Max Depth")
            eta = gr.Number(value=0.1, label="Learning Rate (eta)")
            num_round = gr.Number(value=400, label="Boost Rounds")
            compare_cpu = gr.Checkbox(label="Also run on CPU for time comparison", value=False)
            run_btn = gr.Button("Train and Predict on ROCm GPU")

        with gr.Column(scale=3):
            out_summary = gr.Markdown()
            cm_image = gr.Image(label="Confusion Matrix")
            auc_val = gr.Number(label="AUC")
            acc_val = gr.Number(label="Accuracy")
            gr.Markdown("### Top 20 Transactions by Fraud Probability")
            top_table = gr.Dataframe(label="Top 20 Fraud Predictions (sorted by probability)")
            file_link = gr.File(label="Download All Predictions CSV")

    run_btn.click(
        fn=train_and_predict,
        inputs=[data_path, max_depth, eta, num_round, compare_cpu],
        outputs=[out_summary, cm_image, auc_val, acc_val, top_table, file_link],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7865, share=False)
