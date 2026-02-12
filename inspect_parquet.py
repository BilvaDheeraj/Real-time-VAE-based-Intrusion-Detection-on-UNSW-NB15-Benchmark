import pandas as pd
import os

data_dir = r"c:/Users/Bilva Dheeraj/Documents/WOXSEN/3rd year/Sem-6/Deep learning/PBL/UNSW-NB15"
train_file = os.path.join(data_dir, "UNSW_NB15_training-set.parquet")

try:
    df = pd.read_parquet(train_file)
    print("Columns:")
    for col in df.columns:
        print(col)
except Exception as e:
    print(f"Error reading parquet: {e}")
