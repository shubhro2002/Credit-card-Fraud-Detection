import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

def split_data(df):
    X = df.drop(columns = ['Class', 'Time'])
    y = df['Class']
    return train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 42)

def rf(X_train, y_train):
    '''
    Use Random Forest Classifier without class_weight to address class
    imbalance problem in the dataset
    '''
    rf = RandomForestClassifier(
        n_estimators = 100,
        random_state = 42,
        n_jobs = -1
    )
    rf.fit(X_train, y_train)
    return rf

def train_rf(X_train, y_train):
    '''
    Use Random Forest Classifier and set class_weight to 'balanced' to address 
    class imbalance problem in the dataset
    '''
    rf = RandomForestClassifier(
        class_weight = 'balanced',
        n_estimators = 100,
        random_state = 42,
        n_jobs = -1
    )
    rf.fit(X_train, y_train)
    return rf

def oversample_smote(X_train, y_train):
    '''
    Use Oversampling for the under-represented class so that the model is not trained
    on most of the over-represented class
    '''
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res

def undersample(X_train, y_train):
    '''
    Use Undersampling for the over-represented class
    '''
    rus = RandomUnderSampler(random_state = 42)
    X_res, y_res = rus.fit_resample(X_train, y_train)
    return X_res, y_res

def evaluate_model(model, X_test, y_test, label=""):
    '''
    Evaluation of each approach, return the following for comparison:
    Precision,
    Recall,
    F1-score,
    ROC-AUC,
    TN,TP,FN,FP
    '''
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    result = {
        "Method": label,
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC AUC": roc_auc_score(y_test, y_proba),
        "TN": cm[0][0],
        "FP": cm[0][1],
        "FN": cm[1][0],
        "TP": cm[1][1],
    }

    return result

def run_experiments(df):
    '''
    Perform each approach and return the evaluation metrics into a 
    dataframe for comapring each approach:
    1. class_weight parameter in Random Forest Classifier
    2. Oversampling
    3. Undersampling
    4. Random Forest Classifier without class_weight parameter
    '''
    
    X_train, X_test, y_train, y_test = split_data(df)
    results = []

    # 1. class_weight='balanced'
    model_balanced = train_rf(X_train, y_train)
    res_bal = evaluate_model(model_balanced, X_test, y_test, "Class Weight Balanced")
    results.append(res_bal)

    # 2. Oversampling (SMOTE)
    X_over, y_over = oversample_smote(X_train, y_train)
    model_over = train_rf(X_over, y_over)
    res_over = evaluate_model(model_over, X_test, y_test, "SMOTE Oversampling")
    results.append(res_over)

    # 3. Undersampling
    X_under, y_under = undersample(X_train, y_train)
    model_under = train_rf(X_under, y_under)
    res_under = evaluate_model(model_under, X_test, y_test, "Random Undersampling")
    results.append(res_under)

    # 4. Random Forest without addressing class imbalance problem
    model_imbalanced = rf(X_train, y_train)
    res_imbal = evaluate_model(model_imbalanced, X_test, y_test, "Without Addressing Class Imbalance")
    results.append(res_imbal)

    # Convert to DataFrame for comparison
    results_df = pd.DataFrame(results)
    return results_df