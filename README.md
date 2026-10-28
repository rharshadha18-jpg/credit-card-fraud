# Credit Card Fraud Detection

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-3.4-red)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

> **Author:** Harshadha · B.Tech CSE (2029)

---

## What This Project Does

Credit card fraud is rare — only **0.17%** of all transactions are fraudulent. This makes it a hard problem:
- A model that blindly labels everything as "genuine" gets **99.8% accuracy** but catches **zero fraud**
- So accuracy is useless here. We need to actually catch fraudsters while not blocking real customers

This project builds and compares **4 machine learning models** that detect fraud, using the right metrics (Precision, Recall, PR-AUC) and proper techniques to handle the imbalanced data.

---

## Dataset

- **Source:** [Kaggle — Credit Card Fraud Detection (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions over 2 days by European cardholders (Sept 2013)
- **Fraud cases:** 492 out of 284,807 (0.172%)

| Column | Description |
|--------|-------------|
| `Time` | Seconds since first transaction |
| `V1`–`V28` | Anonymized PCA features (real features hidden for privacy) |
| `Amount` | Transaction amount in euros |
| `Class` | **Target** — `1` = Fraud, `0` = Genuine |

---

## Approach

### 1. Exploratory Data Analysis (EDA)
- Checked for missing values and duplicates
- Visualized class imbalance, Amount and Time distributions for fraud vs genuine transactions
- Analyzed feature correlations with the target

### 2. Preprocessing
- Scaled `Amount` and `Time` using StandardScaler (V1–V28 are already PCA-transformed)
- Stratified train/test split (80/20) to preserve the fraud ratio
- Resampling applied **only on training data** to prevent data leakage

### 3. Handling Class Imbalance
Three strategies were tested:

| Strategy | How it works |
|----------|-------------|
| `class_weight="balanced"` | Penalizes the model more for missing fraud |
| SMOTE | Generates synthetic fraud samples to balance training data |
| `scale_pos_weight` | XGBoost's built-in way to handle imbalance |

### 4. Models Trained
- **Logistic Regression** — simple baseline
- **Random Forest** with class weights
- **Random Forest** with SMOTE
- **XGBoost** with scale_pos_weight

### 5. Threshold Tuning
The default 0.5 decision threshold is not optimal for fraud detection. The best threshold per model was selected by **maximizing F1-score** using out-of-fold cross-validation on training data — the test set is only touched once for final evaluation.

### 6. Evaluation Metrics
- **PR-AUC** (primary) — best metric for imbalanced datasets
- Precision, Recall, F1
- ROC-AUC
- Confusion Matrix

---

## Results

| Model | Strategy | Threshold | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|-------|----------|-----------|-----------|--------|----|--------|---------|
| Logistic Regression | Class weights | 1.000 | 0.833 | 0.816 | 0.825 | 0.719 | 0.972 |
| Random Forest | Class weights | 0.390 | 0.882 | 0.837 | 0.859 | 0.863 | 0.957 |
| Random Forest | SMOTE | 0.770 | 0.962 | 0.776 | 0.859 | 0.875 | 0.973 |
| **XGBoost** | scale_pos_weight | 0.977 | 0.952 | 0.816 | **0.879** | **0.877** | 0.972 |

**Best model: XGBoost** — highest PR-AUC (0.877), catches 81.6% of all fraud with 95.2% precision.
