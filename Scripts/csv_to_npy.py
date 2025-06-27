import numpy as np
import pandas as pd
import glob
import os

csv_files = glob.glob("data/raw/*.csv")
for csv in csv_files:
    df = pd.read_csv(csv)
    arr = df.to_numpy()
    out_path = csv.replace(".csv", ".npy")
    np.save(out_path, arr)
    print(f"Converted: {csv} → {out_path}")
