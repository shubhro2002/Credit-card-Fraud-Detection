# 🧠 Credit-card-Fraud-Detection

This project focuses on building and comparing machine learning models to detect fraudulent transactions using the **IEEE-CIS Fraud Detection Dataset**.  
The goal is to explore different modeling techniques, handle data preprocessing, address (or not address) the **class imbalance problem**, and evaluate model performance to choose the best approach.

## 📁 Dataset Description

This dataset has been taken from kaggle. It contains features like:

- **Time**: Number of seconds elapsed between this transaction and the first transaction in the dataset
- **V1 - V28**: may be result of a PCA Dimensionality reduction to protect user identities and sensitive features
- **Amount**: Transaction amount
- **Class**: 1 for fraudulent transactions, 0 otherwise

## 🧪 Approaches Used

- ✅ Random Forest Classifier with *class_weight* parameter to 'balanced'
  
- ✅ Oversampling
  
- ✅ Undersampling
  
- ✅ Random Forest Classifier without *class_weight* parameter

## 🔍 Project Goals

- Use different approaches for tha dataset

- Evaluate the model for each approach

- Comapare the results

## 📊 Results

| Method                |Precision|  Recall   | F1-Score | ROC-AUC | TN | FP | FN | TP | 
|-----------------------|---------|-----------|----------|---------|----|----|----|----|
| Class Weight Balanced | 0.961039 | 0.755102 | 0.845714 | 0.957969 | 56861 | 03 | 24 | 74 |
| SMOTE Oversampling    | 0.892473 | 0.846939 | 0.869110 | 0.968392 | 56854 | 10 | 15 | 83 |
| Random Undersampling  | 0.045000 | 0.918367 | 0.085796 | 0.974063 | 54954 |1910| 8 | 90 |
| Standard Random Forest| 0.952381 | 0.816327 | 0.879121 | 0.952806 | 56960 | 4 | 18 | 80 |

## 📊Observations

- **Without addressing class imbalance** wins if the main goal is **overall balance (F1) while keeping both precision and recall high**.

- **SMOTE is second-best**, especially if **ROC-AUC** is prioritized.

- **Class Weight Balanced** is third-best when precision is critical -- Great **precision**, but **recall drop pulls F1 down**.

- **Undersampling** should be avoided -- **huge false negatives**.

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/shubhro2002/Credit-card-Fraud-Detection.git

2. Install required dependencies:

   ```bash
   pip install -r requirements.txt

4. Run the Jupyter notebooks or Python scripts to start using.
