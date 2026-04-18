import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def run_eda(df):
    # --- 1. Basic info ---
    print("Shape:", df.shape)
    print(df.info())
    print(df.describe())

    # --- 2. Missing values ---
    missing = df.isnull().sum()
    print("\nMissing values:\n", missing[missing > 0])

    # --- 3. Churn distribution ---
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Churn', data=df, palette='Set2')
    plt.title('Churn Distribution')
    plt.savefig('plots/churn_distribution.png', dpi=150)
    plt.show()

    # --- 4. Correlation heatmap ---
    num_df = df.select_dtypes(include=np.number)
    plt.figure(figsize=(10, 6))
    sns.heatmap(num_df.corr(), annot=True,
               fmt='.2f', cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.savefig('plots/correlation.png', dpi=150)
    plt.show()

    # --- 5. Tenure vs Churn (scipy stats) ---
    churned = df[df['Churn'] == 'Yes']['tenure']
    stayed  = df[df['Churn'] == 'No']['tenure']
    t_stat, p_val = stats.ttest_ind(churned, stayed)
    print(f"\nT-test p-value: {p_val:.4f}")

    # --- 6. Box plots ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col in zip(axes, ['tenure','MonthlyCharges','TotalCharges']):
        sns.boxplot(x='Churn', y=col, data=df,
                   palette='Set2', ax=ax)
        ax.set_title(col)
    plt.tight_layout()
    plt.savefig('plots/boxplots.png', dpi=150)
    plt.show()