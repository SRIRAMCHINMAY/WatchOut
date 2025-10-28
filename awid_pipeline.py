"""
AWID pipeline with the requested present_features and time feature handling.

Usage:
    - Train: python awid_pipeline.py train --csv AWID-CLS-R-Trn.csv --out model.joblib
    - Predict on a text batch or simulated batch: call predict_batch(batch_list, model, pipeline)
"""

import re
import hashlib
import argparse
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib


# ---------------------------
# Configuration / Features
# ---------------------------
PRESENT_FEATURES = [
    'frame.len',
    'frame.cap_len',
    'radiotap.dbm_antsignal',
    'radiotap.channel.freq',
    'wlan.da',
    'wlan.sa',
    'wlan.bssid',
    'frame.time_delta',
    'frame.time_relative',
    'radiotap.present.channel',
    'wlan_mgt.ds.current_channel'
]


# ---------------------------
# Helpers
# ---------------------------
def mac_to_int_hash(mac: Optional[str], mod=2**20):
    """
    Deterministic hash of MAC-like string to an integer feature.
    Returns None if missing.
    """
    if not mac or pd.isna(mac):
        return None
    # normalize mac
    mac_norm = mac.lower().replace('-', ':')
    h = hashlib.sha1(mac_norm.encode('utf8')).hexdigest()
    # convert hex -> int and reduce
    return int(h, 16) % mod


def extract_from_log_text(log_text: str) -> List[Dict]:
    """
    Parse packet-style text (like your input dumps) and return list of dicts
    with basic radiotap/wlan fields. Useful for quick tests.
    """
    raw_packets = re.split(r'---- PACKET ----', log_text)
    packets = []
    for pkt in raw_packets:
        pkt = pkt.strip()
        if not pkt:
            continue
        rssi = re.search(r'RSSI:\s*(-?\d+)', pkt)
        channel = re.search(r'Channel:\s*(\d+)', pkt)
        length = re.search(r'Len:\s*(\d+)', pkt)
        da = re.search(r'DA:\s*([\dA-F:]+)', pkt, re.I)
        sa = re.search(r'SA:\s*([\dA-F:]+)', pkt, re.I)
        bssid = re.search(r'BSSID:\s*([\dA-F:]+)', pkt, re.I)

        packets.append({
            "radiotap.dbm_antsignal": int(rssi.group(1)) if rssi else None,
            "radiotap.channel.freq": int(channel.group(1)) if channel else None,
            "frame.len": int(length.group(1)) if length else None,
            "frame.cap_len": int(length.group(1)) if length else None,
            "wlan.da": da.group(1) if da else None,
            "wlan.sa": sa.group(1) if sa else None,
            "wlan.bssid": bssid.group(1) if bssid else None,
        })
    return packets


# ---------------------------
# Time processing function
# ---------------------------
def process_packets(packets: List[Dict], total_duration: float = 10.0, randomize: bool = False) -> pd.DataFrame:
    """
    Given a list of packet dicts (each having at least the fields you parse),
    append frame.time_relative and frame.time_delta and return a DataFrame.

    If packets already contain 'frame.time_epoch', we derive relative + delta from that.
    Otherwise we assign synthetic timestamps across total_duration.
    """
    df = pd.DataFrame(packets).copy()

    # If there's an existing epoch-like timestamp, use it
    if 'frame.time_epoch' in df.columns and df['frame.time_epoch'].notna().any():
        # convert to numeric if needed
        df['frame.time_epoch'] = pd.to_numeric(df['frame.time_epoch'], errors='coerce')
        df = df.sort_values('frame.time_epoch').reset_index(drop=True)
        df['frame.time_relative'] = df['frame.time_epoch'] - df['frame.time_epoch'].iloc[0]
        df['frame.time_delta'] = df['frame.time_epoch'].diff().fillna(0)
    else:
        # Use synthetic or provided 'time' column if present
        if 'time' in df.columns:
            df = df.sort_values('time').reset_index(drop=True)
            df['frame.time_relative'] = df['time'] - df['time'].iloc[0]
            df['frame.time_delta'] = df['time'].diff().fillna(0)
        else:
            # create synthetic timestamps over [0, total_duration]
            n = len(df)
            if n == 0:
                return df
            if randomize:
                ts = np.sort(np.random.uniform(0, total_duration, size=n))
            else:
                ts = np.linspace(0.0, float(total_duration), num=n)
            df['frame.time_relative'] = ts
            df['frame.time_delta'] = np.concatenate(([0.0], np.diff(ts)))

    # Ensure present flags for radiotap.present.channel existence:
    if 'radiotap.present.channel' not in df.columns:
        # if radiotap.channel.freq is present, set flag True
        df['radiotap.present.channel'] = df['radiotap.channel.freq'].notna().astype(int)

    # If wlan_mgt.ds.current_channel not present, attempt to map from channel freq
    if 'wlan_mgt.ds.current_channel' not in df.columns and 'radiotap.channel.freq' in df.columns:
        # Convert freq to channel number for common mappings (2.4 GHz and 5 GHz common values)
        def freq_to_channel(freq):
            try:
                f = int(freq)
            except:
                return None
            # 2.4 GHz channels
            if 2400 <= f <= 2500:
                return (f - 2407) // 5
            # 5 GHz channels roughly
            if 5000 <= f <= 6000:
                return (f - 5000) // 5
            return None
        df['wlan_mgt.ds.current_channel'] = df['radiotap.channel.freq'].apply(freq_to_channel)

    return df


# ---------------------------
# Preprocessing / Feature builder
# ---------------------------
def build_feature_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only PRESENT_FEATURES (if available) and convert MACs to hashed ints,
    ensure numeric columns exist and impute missing values.
    """
    df2 = df.copy()

    # Ensure all requested columns exist (create if missing with NaN)
    for c in PRESENT_FEATURES:
        if c not in df2.columns:
            df2[c] = np.nan

    # Convert MAC columns to hashed ints
    for mac_col in ['wlan.da', 'wlan.sa', 'wlan.bssid']:
        if mac_col in df2.columns:
            df2[mac_col + '_hash'] = df2[mac_col].apply(lambda x: mac_to_int_hash(x))
        else:
            df2[mac_col + '_hash'] = np.nan

    # Numeric columns we want to train on (drop original MAC strings)
    numeric_cols = [
        'frame.len',
        'frame.cap_len',
        'radiotap.dbm_antsignal',
        'radiotap.channel.freq',
        'frame.time_delta',
        'frame.time_relative',
        'radiotap.present.channel',
        'wlan_mgt.ds.current_channel',
        'wlan.da_hash',
        'wlan.sa_hash',
        'wlan.bssid_hash'
    ]

    # Ensure numeric dtype
    for c in numeric_cols:
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors='coerce')
        else:
            df2[c] = np.nan

    # Build final dataframe with only numeric_cols
    feat_df = df2[numeric_cols].copy()

    return feat_df


# ---------------------------
# Training / evaluation
# ---------------------------
def train_and_evaluate(df: pd.DataFrame, target_col: str = 'class', model_out: str = 'awid_rf.joblib'):
    """
    Train/test split (80/20), preprocess, train a RandomForest, and save model + pipeline.
    Returns trained pipeline and model.
    """
    # Check target present
    if target_col not in df.columns:
        raise ValueError(f"target column '{target_col}' not found in dataframe")

    # Split into train/test
    train_df, test_df = train_test_split(df, test_size=0.20, random_state=42, stratify=df[target_col])

    # Build features
    X_train = build_feature_dataframe(train_df)
    X_test = build_feature_dataframe(test_df)
    y_train = train_df[target_col].values
    y_test = test_df[target_col].values

    # Impute + scale pipeline
    numeric_features = X_train.columns.tolist()
    preprocessor = ColumnTransformer(transformers=[
        ("num", Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features)
    ])

    clf = Pipeline(steps=[
        ('preproc', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
    ])

    print("Training classifier...")
    clf.fit(X_train, y_train)

    # Evaluate
    print("Evaluating on test split...")
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save
    joblib.dump(clf, model_out)
    print(f"Saved trained pipeline to {model_out}")

    return clf


# ---------------------------
# Prediction for incoming batch
# ---------------------------
def predict_batch(raw_packets: List[Dict], clf_pipeline, duration: float = 10.0, randomize: bool = False):
    """
    raw_packets: list of packet dicts (like parse from logs or incoming from ESP)
    clf_pipeline: trained sklearn Pipeline (loaded)
    duration: seconds over which these packets occurred (used if no timestamps exist)
    Returns list of predictions (one per frame) and the DataFrame used for prediction.
    """
    # 1) add time features / normalize
    df = process_packets(raw_packets, total_duration=duration, randomize=randomize)

    # 2) build feature matrix same as training preprocessing
    feat_df = build_feature_dataframe(df)

    # 3) predict
    # handle the case where there are missing columns due to tiny batch
    # Pipeline will impute missing values
    preds = clf_pipeline.predict(feat_df)

    # return predictions alongside the packet info
    df_out = df.reset_index(drop=True).copy()
    df_out['predicted_class'] = preds

    return df_out


# ---------------------------
# Example / CLI
# ---------------------------
def main_train(csv_path: str, model_out: str):
    print(f"Loading CSV: {csv_path} ...")
    df = pd.read_csv("AWID-CLS-R-Trn/1")
    # Optionally look for a column named 'class' or 'label'
    if 'class' not in df.columns:
        # heuristics: try 'Label' or 'attack' alternatives
        alt = None
        for c in df.columns:
            if c.lower() in ('label', 'attack', 'target'):
                alt = c
                break
        if alt:
            df = df.rename(columns={alt: 'class'})
        else:
            raise ValueError("No target column named 'class' or similar found in CSV.")

    # Train & save
    clf = train_and_evaluate(df, target_col='class', model_out=model_out)
    print("Training complete.")


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'demo'], help='train or demo prediction')
    parser.add_argument('--csv', help='Path to AWID training CSV (for train mode)')
    parser.add_argument('--model', default='awid_rf.joblib', help='Model path')
    parser.add_argument('--out', default='awid_rf.joblib', help='Output model path (train)')
    parser.add_argument('--duration', type=float, default=10.0, help='Duration for batch timestamps (seconds)')
    args = parser.parse_args()

    if args.mode == 'train':
        if not args.csv:
            raise ValueError("Please provide --csv path to AWID training CSV")
        main_train(args.csv, args.out)

    elif args.mode == 'demo':
        # Demo: load model and run predictions on example parsed packets
        clf = joblib.load(args.model)
        # Example input: a few packet dumps
        example_log = """
        ---- PACKET ----
        RSSI: -81 dBm  Channel: 84  Len: 60
        DA: 0F:00:00:00:00:00  SA: 00:00:C7:59:00:00  BSSID: B9:29:99:A5:01:00

        ---- PACKET ----
        RSSI: -83 dBm  Channel: 112  Len: 128
        DA: 76:82:D0:BE:E6:F0  SA: 4C:F6:03:00:00:00  BSSID: 64:00:31:04:00:0B
        """
        packets = extract_from_log_text(example_log)
        df_out = predict_batch(packets, clf, duration=args.duration, randomize=False)
        print(df_out.to_string(index=False))


if __name__ == "__main__":
    cli()
