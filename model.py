import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath, parse_dates=['datetime'])

    # Drop rows with missing target signal
    df = df.dropna(subset=['signal'])

    # Encode target labels
    le = LabelEncoder()
    df['target'] = le.fit_transform(df['signal'])

    # Select numeric feature columns (exclude target and datetime)
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in feature_cols if col not in ['target']]

    X = df[feature_cols]
    y = df['target']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Clean features: replace inf/-inf with nan, then nan with column means
    X_clean = clean_features(X_scaled)

    return df, X_clean, y, le, scaler

def clean_features(X):
    X = np.array(X)

    # Replace inf/-inf with nan
    X[np.isinf(X)] = np.nan

    # Replace nan with column means
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])

    return X

def time_based_split(df, X, y, train_ratio=0.7):
    df = df.sort_values('datetime')
    n_train = int(len(df) * train_ratio)
    train_indices = df.index[:n_train]
    test_indices = df.index[n_train:]

    X_train = X[train_indices]
    y_train = y.iloc[train_indices]

    X_test = X[test_indices]
    y_test = y.iloc[test_indices]

    return X_train, X_test, y_train, y_test

def train_and_evaluate(X_train, y_train, X_test, y_test):
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'XGBClassifier': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    best_model_name = None
    best_accuracy = 0
    best_model = None

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"\nModel: {name}")
        print(f"Accuracy: {acc:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, preds))

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_model_name = name

    return best_model_name, best_accuracy, best_model

def save_model(model, label_encoder, scaler, model_path, le_path, scaler_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(label_encoder, le_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to: {model_path}")
    print(f"Label Encoder saved to: {le_path}")
    print(f"Scaler saved to: {scaler_path}")

# --------------------
# Usage Example
# --------------------

if __name__ == "__main__":
    filepath = '/content/drive/MyDrive/strategy-backtest/data/spot_with_signals_2023.csv'
    df, X_clean, y, le, scaler = load_and_prepare_data(filepath)
    X_train, X_test, y_train, y_test = time_based_split(df, X_clean, y)

    best_model_name, best_accuracy, best_model = train_and_evaluate(X_train, y_train, X_test, y_test)

    print(f"Best model: {best_model_name} with accuracy {best_accuracy:.4f}")

    save_model(
        best_model,
        le,
        scaler,
        'strategy-backtest/model/best_model.pkl',
        'strategy-backtest/model/label_encoder.pkl',
        'strategy-backtest/model/scaler.pkl'
    )
