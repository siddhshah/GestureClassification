import numpy as np
import pandas as pd
import glob
import os

csv_files = glob.glob("data/raw/*.csv")
for csv in csv_files:
    df = pd.read_csv(csv, header=0)  # assumes header exists
    df = df.apply(pd.to_numeric, errors='coerce')  # force numeric
    df.dropna(inplace=True)
    arr = df.to_numpy().astype(np.float32)  # enforce dtype
    out_path = csv.replace(".csv", ".npy")
    np.save(out_path, arr)
    print(f"Converted: {csv} → {out_path}")
