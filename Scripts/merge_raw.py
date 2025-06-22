import numpy as np
import glob

# 1) Adjust the path if you store your .csv raws somewhere else
files = glob.glob("data/raw/*.csv")

# 2) Load each one, stack them, and save the merged array
arrays = [np.load(f, allow_pickle=True) for f in files]
merged = np.vstack(arrays)
np.save("data/raw/all_gestures.npy", merged)

print("Merged into:", merged.shape)