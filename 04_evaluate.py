"""
04_evaluate.py — Confusion matrix, feature importance, algorithm comparison
"""
import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

INPUT_PATH = 'preprocessed_data.pkl'
MODEL_PATH = 'nids_model.pkl'


def load_data(path: str):
    if not os.path.exists(path):
        print(f"Error: {path} not found. Run 02_preprocess.py first.")
        return None
    return joblib.load(path)


def load_model(path: str):
    if not os.path.exists(path):
        print(f"Error: {path} not found. Run 03_train_model.py first.")
        return None
    return joblib.load(path)


def show_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("Confusion Matrix:")
    print(f"{'':>12} {'Pred Normal':>12} {'Pred Attack':>12}")
    print(f"{'Actual Normal':>12} {tn:>12} {fp:>12}")
    print(f"{'Actual Attack':>12} {fn:>12} {tp:>12}")
    print()

    total = tn + fp + fn + tp
    print(f"True Negatives:   {tn:>6}  ({tn/total*100:5.2f}%)")
    print(f"False Positives:  {fp:>6}  ({fp/total*100:5.2f}%)  ← False alarms")
    print(f"False Negatives:  {fn:>6}  ({fn/total*100:5.2f}%)  ← Missed attacks")
    print(f"True Positives:   {tp:>6}  ({tp/total*100:5.2f}%)")

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'],
                ax=ax, cbar=False)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    plt.show()


def show_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\nTop 10 Most Important Features:")
    print(f"{'Rank':<6} {'Feature':<35} {'Importance':<10}")
    print("-" * 51)
    for rank in range(10):
        idx = indices[rank]
        print(f"{rank + 1:<6} {feature_names[idx]:<35} {importances[idx]:.4f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    top_n = 10
    top_features = [feature_names[i] for i in indices[:top_n]]
    top_values = [importances[i] for i in indices[:top_n]]
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, top_n))
    ax.barh(range(top_n), top_values[::-1], color=colors[::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features[::-1])
    ax.set_title('Top 10 Feature Importances', fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance')
    plt.tight_layout()
    plt.show()


def compare_algorithms(X_train, y_train, X_test, y_test):
    print("\n" + "=" * 56)
    print("ALGORITHM COMPARISON")
    print("=" * 56)

    sample_size = min(20000, len(X_train))
    X_sample = X_train[:sample_size]
    y_sample = y_train[:sample_size]

    scaler = StandardScaler()
    X_sample_scaled = scaler.fit_transform(X_sample)
    X_test_scaled = scaler.transform(X_test)

    algorithms = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
    }

    results = []

    for name, model in algorithms.items():
        start = time.time()
        if name == 'KNN (k=5)':
            model.fit(X_sample_scaled, y_sample)
            score = model.score(X_test_scaled, y_test)
        else:
            model.fit(X_sample, y_sample)
            score = model.score(X_test, y_test)
        elapsed = time.time() - start
        results.append({'Algorithm': name, 'Accuracy': f"{score*100:.2f}%", 'Time': f"{elapsed:.1f}s"})
        print(f"  {name:20s}  Accuracy: {score*100:.2f}%  ({elapsed:.1f}s)")

    return results


if __name__ == '__main__':
    data = load_data(INPUT_PATH)
    if data is None:
        raise FileNotFoundError("Run 02_preprocess.py first.")
    X_train, X_test, y_train, y_test = data

    model = load_model(MODEL_PATH)
    if model is None:
        raise FileNotFoundError("Run 03_train_model.py first.")

    print(f"Testing samples: {X_test.shape[0]:,}")
    print()

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred,
                                    target_names=['Normal', 'Attack'],
                                    digits=4)
    print("Classification Report:")
    print(report)

    show_confusion_matrix(y_test, y_pred)
    show_feature_importance(model, X_train.columns.tolist())

    results = compare_algorithms(X_train, y_train, X_test, y_test)
    print("\nDone.")
