import os
import random
import numpy as np
import pandas as pd

from feature_extraction import process_farseeing_file


ADL_DIR = "data/farseeing/adl"
FALL_DIR = "data/farseeing/fall"

OUTPUT_DIR = "features/w3_s1"
INDEX_FILE = "features/feature_index_w3_s1.csv"
meta = pd.read_excel("../data/farseeing/description.xlsx", engine="openpyxl")

os.makedirs(OUTPUT_DIR, exist_ok=True)

N_ADL = 30
N_FALL = 30

def parse_fall_filename(filename):
    """
    Example:
    F_00002186-01-2013-11-15-13-50-18.csv
    """
    base = filename.replace(".csv", "")
    subject_id = base.split("_")[1].split("-")[0]
    file_id = "-".join(base.split("-")[2:])  # keep session timestamp
    return subject_id, file_id

def parse_adl_filename(filename):
    """
    Example:
    subAtaxia1_1.csv
    """
    base = filename.replace(".csv", "")
    subject_id, file_num = base.split("_")
    file_id = file_num
    return subject_id, file_id

def get_sampling_rate(filename):
    """
    Example:
    F_00002186-01-2013-11-15-13-50-18.csv
    file id is 00002186-01
    """
    file_id = filename.split("_")[1].split("-")[0] + "-" + filename.split("_")[1].split("-")[1]
    fs = meta[meta["Randomnumber"] == file_id]["Sample_rate_Hz"].iloc[0]
    return fs


adl_files = [f for f in os.listdir(ADL_DIR) if f.endswith(".csv")]
fall_files = [f for f in os.listdir(FALL_DIR) if f.endswith(".csv")]

random.seed(42)

adl_sample = random.sample(adl_files, min(N_ADL, len(adl_files)))
fall_sample = random.sample(fall_files, min(N_FALL, len(fall_files)))

selected_files = []

for f in adl_sample:
    selected_files.append(("adl", f))

for f in fall_sample:
    selected_files.append(("fall", f))

print(f"Processing {len(selected_files)} files...")

if os.path.exists(INDEX_FILE):
    index_df = pd.read_csv(INDEX_FILE)
else:
    index_df = pd.DataFrame(columns=[
        "subject_id",
        "file_id",
        "label_type",
        "feature_path",
        "timestamp_path"
    ])


for label_type, filename in selected_files:

    try:
        if label_type == "adl":
            subject_id, file_id = parse_adl_filename(filename)
            csv_path = os.path.join(ADL_DIR, filename)
        else:
            subject_id, file_id = parse_fall_filename(filename)
            csv_path = os.path.join(FALL_DIR, filename)

        if filename.startswith("F_"):
            fs = get_sampling_rate(filename)
        else:
            fs = 100

        features, timestamps = process_farseeing_file(csv_path, fs)

        feature_name = f"{subject_id}__{file_id}.npy"
        timestamp_name = f"{subject_id}__{file_id}_ts.npy"

        feature_path = os.path.join(OUTPUT_DIR, feature_name)
        timestamp_path = os.path.join(OUTPUT_DIR, timestamp_name)

        np.save(feature_path, features)
        np.save(timestamp_path, timestamps)

        # update index
        index_df.loc[len(index_df)] = [
            subject_id,
            file_id,
            label_type,
            feature_path,
            timestamp_path
        ]

        print(f"Saved: {subject_id} | {file_id}")

    except Exception as e:
        print(f"Error processing {filename}: {e}")

# save updated index
index_df.to_csv(INDEX_FILE, index=False)

print("Feature extraction batch complete.")
