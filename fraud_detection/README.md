**ROCm XGBoost Credit Card Fraud Detection (Train/Test UI with Time Comparison)**

**Accelerated Credit Card Fraud Detection using AMD ROCm GPU and XGBoost**

This project provides a Gradio-based interactive demo for training and testing an XGBoost model on a credit card fraud dataset, leveraging AMD ROCm GPUs for acceleration. It optionally allows CPU comparison and visualizes model performance, including a confusion matrix and top predicted fraudulent transactions.

**Features**

1. Train XGBoost on ROCm-enabled AMD GPUs

2. Optional CPU training for speed comparison

3. 80/20 train/test split for evaluation

4. Accuracy, AUC, and confusion matrix metrics

5. Visualization of top 20 most suspicious transactions

6. Export all predictions to CSV

7. Interactive Gradio web interface

**Requirements**

Python 3.8+

ROCm-enabled AMD GPU (optional, CPU fallback available)

Key Python packages:
```bash
pip install numpy pandas matplotlib scikit-learn gradio pillow
```

**ROCm XGBoost for GPU acceleration:**

Installing ROCm XGBoost

Install ROCm (AMD GPU drivers & runtime) following the official guide:
ROCm Installation Guide

```bash
Install XGBoost with ROCm support:

cd $HOME
git clone --depth=1 --recurse-submodules https://github.com/ROCmSoftwarePlatform/xgboost
cd xgboost
mkdir build && cd build
export GFXARCH="$(rocm_agent_enumerator | tail -1)"
export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:/opt/rocm/lib/cmake:/opt/rocm/lib/cmake/AMDDeviceLibs/
cmake -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=${GFXARCH} -DUSE_RCCL=1 ../
make -j
pip3 install ../python-package/
```
**Verify GPU support:**

```bash
import xgboost as xgb
print(xgb.rabit.get_tracker_addr())  # Should not error
```

Ensure your AMD GPU is detected. The device parameter in XGBoost should be set to "gpu".

**Dataset**

The app expects a CSV dataset with the following:

1. creditcard.csv (from Kaggle Credit Card Fraud Detection)

2. Must include a Class column (0 = normal, 1 = fraud)

3. Features are all numeric transaction-related columns

**Usage**

Run the app:

```bash
python xgboost_fraud_detection.py
```

Open the interface in your browser at http://0.0.0.0:7865

**Configure options:**

1. Dataset path (default: creditcard.csv)

2. XGBoost hyperparameters: max depth, learning rate (eta), number of boosting rounds

3. Optionally enable CPU comparison

4. Click "Train and Predict on ROCm GPU"

**View results:**

1. Model summary (accuracy, AUC, training times)

2. Confusion matrix

3. Top 20 predicted fraud transactions

4. Download full predictions CSV

**Example Output**

Summary Example:

Train samples: 227,451    Test samples: 56,863
Accuracy: 0.9992    AUC: 0.9885
GPU Training Time: 12.34 seconds
CPU Training Time: 47.89 seconds
Speedup (GPU vs CPU): 3.88× faster
Predictions saved to fraud_predictions.csv

<img width="1389" height="1202" alt="image" src="https://github.com/user-attachments/assets/84507cdb-deb5-4fcb-951f-2bbfb33ebcca" />

Confusion Matrix:

A visually styled matrix with AMD-themed colors.

Top 20 Transactions:

A table sorted by predicted fraud probability.

**File Structure**
.
├── xgboost_fraud_detection.py                 # Main Gradio app
├── creditcard.csv         # Dataset (download from Kaggle)
├── fraud_predictions.csv  # Generated predictions CSV
├── README.md              # Project documentation

**Customization**

Change AMD theme colors in custom_css (currently teal/black)

Modify XGBoost parameters (max_depth, eta, num_round) for different performance/accuracy trade-offs

Optionally enable CPU comparison to benchmark ROCm GPU speedup

**License**

MIT License – free to use and modify.

Acknowledgements

XGBoost
 – Gradient boosting library

Gradio
 – Interactive web UI

Kaggle Credit Card Fraud Dataset

ROCm
 – AMD GPU acceleration platform
