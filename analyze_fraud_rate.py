import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_fraud_by_hour(df, time_col='Time', class_col='Class'):
    """
    Analyzes and visualizes the distribution of fraudulent transactions by hour.
    
    Parameters:
    - df: pandas DataFrame with transaction data
    - time_col: name of the column containing time in seconds since first transaction
    - class_col: name of the target column (0 = normal, 1 = fraud)
    
    Returns:
    - hourly_counts: DataFrame with hourly transaction counts by class
    - fraud_rate: Series with fraud rate (%) per hour
    """
    # 1. Derive 'Hour' column from 'Time'
    df = df.copy()
    df['Hour'] = (df[time_col] % 86400) // 3600

    # 2. Count transactions per hour per class
    hourly_counts = df.groupby(['Hour', class_col]).size().unstack(fill_value=0)

    # 3. Plot bar chart of transaction counts by class (log scale)
    plt.figure(figsize=(10, 6))
    hourly_counts.plot(kind='bar', stacked=False, logy=True, colormap='Set2')
    plt.title('Hourly Transaction Count by Class (Log Scale)')
    plt.xlabel('Hour of Day')
    plt.ylabel('Transaction Count (log scale)')
    plt.xticks(rotation=0)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # 4. Plot fraud-only hourly distribution
    if 1 in df[class_col].unique():
        fraud_hourly = df[df[class_col] == 1]['Hour'].value_counts().sort_index()
        plt.figure(figsize=(10, 4))
        sns.barplot(x=fraud_hourly.index, y=fraud_hourly.values, color='crimson')
        plt.title('Number of Fraudulent Transactions by Hour')
        plt.xlabel('Hour of Day')
        plt.ylabel('Fraud Count')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

        # 5. Calculate and plot fraud rate per hour
        hourly_total = df.groupby('Hour').size()
        hourly_fraud = df[df[class_col] == 1].groupby('Hour').size()
        fraud_rate = (hourly_fraud / hourly_total * 100).fillna(0)

        plt.figure(figsize=(10, 4))
        sns.barplot(x=fraud_rate.index, y=fraud_rate.values, color='orange')
        plt.title('Fraud Rate by Hour (%)')
        plt.xlabel('Hour of Day')
        plt.ylabel('Fraud Rate (%)')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()
    else:
        print("No fraud (Class = 1) cases found in the dataset.")

    return hourly_counts, fraud_rate if 'fraud_rate' in locals() else None
