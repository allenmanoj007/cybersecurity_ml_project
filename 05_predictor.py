"""
05_predictor.py — Test on unseen data and classify single records
"""
import os
import sys
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

TEXT_COLS = ['protocol_type', 'service', 'flag']
DROP_COLS = ['label', 'num_outbound_cmds', 'difficulty']
# The 40 columns the model is actually trained on (all columns except the dropped ones)
FEATURE_COLS = [c for c in COLUMNS if c not in DROP_COLS]
TRAIN_PATH = 'KDDTrain+.txt'
TEST_PATH = 'KDDTest+.txt'
MODEL_PATH = 'nids_model.pkl'

NORMAL_DEMO = "0,tcp,http,SF,181,5450,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,8,8,0,0,0,0,1,0,0,9,9,1,0,0.11,0.11,0,0,0,0"
ATTACK_DEMO = "0,tcp,private,S0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,168,168,1,1,0,0,1,0,0,255,255,1,0,0.07,0.07,0,0,1,1"


def build_preprocessor():
    """Fit LabelEncoders and StandardScaler on training data."""
    if not os.path.exists(TRAIN_PATH):
        print(f"Error: {TRAIN_PATH} not found.")
        sys.exit(1)

    train = pd.read_csv(TRAIN_PATH, names=COLUMNS)
    encoders = {}
    for col in TEXT_COLS:
        le = LabelEncoder()
        le.fit(train[col])
        encoders[col] = le

    scaler = StandardScaler()
    train_encoded = train.copy()
    for col in TEXT_COLS:
        train_encoded[col] = encoders[col].transform(train_encoded[col])
    train_encoded = train_encoded.drop(columns=DROP_COLS, errors='ignore')
    scaler.fit(train_encoded)

    return encoders, scaler


def preprocess_record(record_df, encoders, scaler):
    """Transform a raw DataFrame into model-ready features."""
    df = record_df.copy()
    for col in TEXT_COLS:
        if col in df.columns:
            df[col] = encoders[col].transform(df[col])
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors='ignore')
    expected_cols = [c for c in COLUMNS if c not in DROP_COLS]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_cols]
    return scaler.transform(df)


def test_on_unseen_data(model, encoders, scaler):
    """Evaluate the model on KDDTest+."""
    if not os.path.exists(TEST_PATH):
        print(f"  Warning: {TEST_PATH} not found. Skipping unseen data test.")
        return

    print("Loading unseen test data (KDDTest+) ...")
    test_df = pd.read_csv(TEST_PATH, names=COLUMNS)
    print(f"  Loaded {test_df.shape[0]:,} records")

    y_true = (test_df['label'] != 'normal').astype(int).values
    X_test = preprocess_record(test_df, encoders, scaler)
    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_true).mean()

    print(f"\n  Accuracy on unseen data: {accuracy*100:.2f}%")
    print(f"  Correct: {(y_pred == y_true).sum():,} / {len(y_true):,}")

    print("\n  First 10 predictions:")
    print(f"  {'Actual':<10} {'Predicted':<10} {'Result':<8}")
    print(f"  {'-'*28}")
    for i in range(min(10, len(y_true))):
        actual = "Attack" if y_true[i] else "Normal"
        predicted = "Attack" if y_pred[i] else "Normal"
        result = "OK" if y_true[i] == y_pred[i] else "MISMATCH"
        print(f"  {actual:<10} {predicted:<10} {result:<8}")


def classify_single_record(model, encoders, scaler):
    """Interactive single-record predictor."""
    print("\n" + "=" * 56)
    print("SINGLE RECORD PREDICTOR")
    print("=" * 56)
    print(f"Paste a {len(FEATURE_COLS)}-field comma-separated traffic record.")
    print("Press Enter to use the demo NORMAL record.")
    print()

    user_input = input("Record: ").strip()
    record_str = user_input if user_input else NORMAL_DEMO

    if not user_input:
        print(f"  Using demo record: {record_str[:60]}...")

    parts = record_str.split(",")
    if len(parts) != len(FEATURE_COLS):
        print(f"Error: expected {len(FEATURE_COLS)} fields, got {len(parts)}")
        return

    record_df = pd.DataFrame([parts], columns=FEATURE_COLS)
    X = preprocess_record(record_df, encoders, scaler)
    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]

    icon = "NORMAL" if pred == 0 else "ATTACK"
    print(f"\n  Prediction: {icon}")
    print(f"  Confidence: Normal = {proba[0]*100:.1f}%, Attack = {proba[1]*100:.1f}%")

    print("\n  --- Demo: Testing an ATTACK record for comparison ---")
    attack_df = pd.DataFrame([ATTACK_DEMO.split(",")], columns=FEATURE_COLS)
    X_attack = preprocess_record(attack_df, encoders, scaler)
    attack_pred = int(model.predict(X_attack)[0])
    attack_proba = model.predict_proba(X_attack)[0]
    attack_icon = "NORMAL" if attack_pred == 0 else "ATTACK"
    print(f"  Attack record prediction: {attack_icon}")
    print(f"  Confidence: Normal = {attack_proba[0]*100:.1f}%, Attack = {attack_proba[1]*100:.1f}%")


if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found. Run 03_train_model.py first.")
        sys.exit(1)

    print("Loading model ...")
    model = joblib.load(MODEL_PATH)
    print("  Model loaded")

    print("Building preprocessor from training data ...")
    encoders, scaler = build_preprocessor()
    print("  Preprocessor ready")

    print("\n" + "=" * 56)
    print("PART 1: UNSEEN DATA EVALUATION")
    test_on_unseen_data(model, encoders, scaler)

    print("\n" + "=" * 56)
    print("PART 2: SINGLE RECORD PREDICTOR")
    classify_single_record(model, encoders, scaler)
