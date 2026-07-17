"""
03_train_model.py — Train and save a Random Forest classifier
"""
import os
import time
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

INPUT_PATH = 'preprocessed_data.pkl'
MODEL_PATH = 'nids_model.pkl'


def load_data(path: str):
    if not os.path.exists(path):
        print(f"Error: {path} not found. Run 02_preprocess.py first.")
        return None
    return joblib.load(path)


def train_random_forest(X_train, y_train, n_estimators=100):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1
    )
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    return model, elapsed


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=['Normal', 'Attack'],
        digits=4
    )
    return accuracy, report, y_pred


if __name__ == '__main__':
    loaded = load_data(INPUT_PATH)
    if loaded is None:
        raise FileNotFoundError(
            "Preprocessed data not found. "
            "Run 02_preprocess.py before this script."
        )
    X_train, X_test, y_train, y_test = loaded

    print(f"Training samples: {X_train.shape[0]:,}")
    print(f"Testing samples:  {X_test.shape[0]:,}")
    print(f"Features:         {X_train.shape[1]}")
    print()

    print("Training Random Forest (100 trees) ...")
    model, elapsed = train_random_forest(X_train, y_train, n_estimators=100)
    print(f"  Done in {elapsed:.1f} seconds")
    print()

    print("Evaluating on test set ...")
    accuracy, report, y_pred = evaluate(model, X_test, y_test)
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print()
    print("Classification Report:")
    print(report)

    print(f"Saving model to {MODEL_PATH} ...")
    joblib.dump(model, MODEL_PATH)
    print("Done.")
