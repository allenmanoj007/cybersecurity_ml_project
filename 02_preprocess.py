"""
02_preprocess.py — Clean, encode, split, and save preprocessed data
"""
import os
import sys
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

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
TARGET = 'binary_label'

INPUT_PATH = 'KDDTrain+.txt'
OUTPUT_PATH = 'preprocessed_data.pkl'


def load_raw_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        print("Download from: https://github.com/defcom17/NSL_KDD/raw/master/KDDTrain%2B.txt")
        sys.exit(1)
    df = pd.read_csv(path, names=COLUMNS)
    return df


def create_binary_label(df: pd.DataFrame) -> pd.DataFrame:
    df[TARGET] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)
    return df


def encode_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in TEXT_COLS:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
        n_classes = len(encoder.classes_)
        mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    return df


def remove_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=DROP_COLS, errors='ignore')


def separate_features_and_target(df: pd.DataFrame):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


def log_distribution(y_train, y_test):
    train_attack_rate = y_train.mean() * 100
    test_attack_rate = y_test.mean() * 100
    print(f"  Train — normal: {(y_train == 0).sum():>6,}  "
          f"attack: {(y_train == 1).sum():>6,}  "
          f"(attack rate: {train_attack_rate:.1f}%)")
    print(f"  Test  — normal: {(y_test == 0).sum():>6,}  "
          f"attack: {(y_test == 1).sum():>6,}  "
          f"(attack rate: {test_attack_rate:.1f}%)")


if __name__ == '__main__':
    print("Loading raw data ...")
    df = load_raw_data(INPUT_PATH)
    print(f"  Loaded {df.shape[0]:,} records with {df.shape[1]} columns")

    print("Creating binary labels ...")
    df = create_binary_label(df)
    print(f"  Normal: {(df[TARGET] == 0).sum():,}  "
          f"Attack: {(df[TARGET] == 1).sum():,}")

    print("Encoding text columns ...")
    df = encode_text_columns(df)
    for col in TEXT_COLS:
        print(f"  {col}: {df[col].nunique()} unique values encoded")

    print("Removing unused columns ...")
    df = remove_unused_columns(df)
    print(f"  Remaining columns: {list(df.columns)}")

    print("Separating features and target ...")
    X, y = separate_features_and_target(df)
    print(f"  Features (X): {X.shape[1]} columns, {X.shape[0]:,} rows")
    print(f"  Target  (y): {y.shape[0]:,} samples")

    print("Splitting into train/test (80/20) ...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    log_distribution(y_train, y_test)

    print(f"Saving to {OUTPUT_PATH} ...")
    joblib.dump((X_train, X_test, y_train, y_test), OUTPUT_PATH)
    print("Done. Ready for model training.")
