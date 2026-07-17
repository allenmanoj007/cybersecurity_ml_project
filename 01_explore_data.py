"""
01_explore_data.py — Load, inspect, and visualize the NSL-KDD dataset
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

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

DATA_PATH = 'KDDTrain+.txt'


def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        print("Download it from: https://github.com/defcom17/NSL_KDD/raw/master/KDDTrain%2B.txt")
        sys.exit(1)
    df = pd.read_csv(path, names=COLUMNS)
    return df


def explore(df: pd.DataFrame):
    print("=" * 56)
    print("DATASET OVERVIEW")
    print("=" * 56)
    print(f"Rows:       {df.shape[0]:>8,}")
    print(f"Columns:    {df.shape[1]:>8,}")
    print(f"Memory:     {df.memory_usage(deep=True).sum() / 1024**2:>8.1f} MB")
    print()

    print("=" * 56)
    print("FIRST 5 ROWS")
    print("=" * 56)
    print(df.head().to_string(index=False))
    print()

    print("=" * 56)
    print("DATA TYPES")
    print("=" * 56)
    for col, dtype in df.dtypes.items():
        print(f"  {col:30s} {str(dtype)}")
    print()

    print("=" * 56)
    print("MISSING VALUES (should all be 0)")
    print("=" * 56)
    missing = df.isnull().sum()
    print(missing.to_string())
    print()

    print("=" * 56)
    print("LABEL DISTRIBUTION — ALL 23 CLASSES")
    print("=" * 56)
    label_counts = df['label'].value_counts()
    for label, count in label_counts.items():
        print(f"  {label:25s} {count:>6,}  ({count/len(df)*100:5.2f}%)")
    print()

    print("=" * 56)
    print("BINARY CLASS DISTRIBUTION (normal vs attack)")
    print("=" * 56)
    is_attack = df['label'] != 'normal'
    normal_count = (is_attack == False).sum()
    attack_count = is_attack.sum()
    print(f"  Normal:     {normal_count:>6,}  ({normal_count/len(df)*100:5.2f}%)")
    print(f"  Attack:     {attack_count:>6,}  ({attack_count/len(df)*100:5.2f}%)")
    print(f"  Ratio:      {normal_count/attack_count:.1f}:1 normal to attack")
    return normal_count, attack_count


def visualize(df: pd.DataFrame, normal_count: int, attack_count: int) -> None:
    label_counts = df['label'].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    top_10 = label_counts.head(10)
    colors = ['#e74c3c' if idx != 'normal' else '#2ecc71' for idx in top_10.index]
    axes[0].barh(top_10.index, top_10.values, color=colors)
    axes[0].set_title('Top 10 Traffic Types', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Count')
    axes[0].invert_yaxis()

    wedges, texts, autotexts = axes[1].pie(
        [normal_count, attack_count],
        labels=['Normal', 'Attack'],
        autopct='%1.1f%%',
        colors=['#2ecc71', '#e74c3c'],
        startangle=90,
        explode=(0.02, 0.02),
        textprops={'fontsize': 12}
    )
    axes[1].set_title('Normal vs Attack', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    df = load_dataset(DATA_PATH)
    normal_count, attack_count = explore(df)
    visualize(df, normal_count, attack_count)
