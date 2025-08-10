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

| Model                | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|----------------------|----------|-----------|--------|----------|---------|
| Logistic Regression  | xx%      | xx%       | xx%    | xx%      | xx%     |
| Random Forest        | xx%      | xx%       | xx%    | xx%      | xx%     |
| XGBoost              | xx%      | xx%       | xx%    | xx%      | xx%     |
| Neural Network (NN)  | xx%      | xx%       | xx%    | xx%      | xx%     |
