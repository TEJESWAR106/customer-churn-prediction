import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score)
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess(df):
    df = df.copy()
    df['TotalCharges'] = pd.to_numeric(
        df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df.drop('customerID', axis=1, inplace=True)

    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])
    return df

def train_model(df):
    df = preprocess(df)
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100,
                                  random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)
    print(f"Accuracy : {acc:.2%}")
    print(f"ROC-AUC  : {roc:.2%}")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig('plots/confusion_matrix.png', dpi=150)
    plt.show()

    # Feature importance
    feat_imp = pd.Series(rf.feature_importances_,
                          index=df.drop('Churn',axis=1).columns)
    feat_imp.nlargest(10).plot(kind='barh')
    plt.title('Top 10 Features')
    plt.savefig('plots/feature_importance.png', dpi=150)
    plt.show()
    return rf, scaler