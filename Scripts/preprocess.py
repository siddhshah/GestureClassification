# Data preprocessing: Load raw `.npy` files, apply DC removal, filtering, and normalization
# Save processed arrays for training

import argparse
import os

import numpy as np
import pandas as pd
from scipy.signal import firwin, filtfilt


def preprocess(input_path, output_path, filter_cutoff, sample_rate):
    # Load data
    if input_path.lower().endswith('.csv'):
        df = pd.read_csv(input_path)
        labels = df.iloc[:, 0].values
        data = df.iloc[:, 1:].values.astype(float)
    elif input_path.lower().endswith('.npy'):
        arr = np.load(input_path)
        labels = arr[:, 0]
        data = arr[:, 1:]
    else:
        raise ValueError("Unsupported file type (.csv and .npy only).")

    num_samples, num_features = data.shape
    window_size = num_features // 3

    # Design FIR low-pass filter
    nyquist = 0.5 * sample_rate
    cutoff_norm = filter_cutoff / nyquist
    numtaps = 31  # must be odd
    taps = firwin(numtaps, cutoff_norm)

    processed = np.zeros_like(data)
    for i in range(num_samples):
        w = data[i].reshape(window_size, 3)
        # DC removal
        w = w - w.mean(axis=0)
        # Low-pass filter each axis
        for axis in range(3):
            w[:, axis] = filtfilt(taps, [1.0], w[:, axis])
        # Normalize to max absolute 1
        max_abs = np.max(np.abs(w))
        if max_abs > 0:
            w = w / max_abs
        processed[i] = w.flatten()

    # Combine labels and processed features
    out_arr = np.column_stack((labels, processed))

    # Ensure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    np.save(output_path, out_arr)
    print(f"Processed data saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess accelerometer gesture data")
    parser.add_argument("input", help="Input CSV or NPY file path")
    parser.add_argument("output", help="Output NPY file path for processed data")
    parser.add_argument("--cutoff", type=float, default=20.0, help="Low-pass filter cutoff frequency in Hz")
    parser.add_argument("--rate", type=float, default=100.0, help="Sampling rate in Hz")
    args = parser.parse_args()
    preprocess(args.input, args.output, args.cutoff, args.rate)


if __name__ == "__main__":
    main()