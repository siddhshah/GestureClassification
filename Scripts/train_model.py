# Define 1D-CNN in TensorFlow/Keras
# Load preprocessed data, train, and export model

import argparse
import numpy as np
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Arg parsing
def parse_args():
    p = argparse.ArgumentParser("Train a 1D-CNN gesture model")
    p.add_argument("input",        help="Processed .npy (labels+features)")
    p.add_argument("-b","--batch_size", type=int,   default=32)
    p.add_argument("-e","--epochs",     type=int,   default=50)
    p.add_argument("-v","--val_split",  type=float, default=0.2)
    p.add_argument("-o","--output",     default="gesture_model.h5")
    return p.parse_args()

# Load & prep data
def load_data(path):
    arr = np.load(path)
    labels = arr[:,0].astype(int)
    feats  = arr[:,1:].astype(np.float32)
    window_size = feats.shape[1] // 3
    X = feats.reshape(-1, window_size, 3)
    y = to_categorical(labels, labels.max()+1)
    return X, y

# Model builder
def build_model(input_shape, num_classes):
    m = Sequential([
        Conv1D(64, 5, activation='relu', input_shape=input_shape),
        MaxPooling1D(2),
        Conv1D(32, 3, activation='relu'),
        MaxPooling1D(2),
        Flatten(),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax'),
    ])
    return m

# 4. Compile, train, eval, save
def main():
    args = parse_args()
    X, y = load_data(args.input)
    model = build_model((X.shape[1], X.shape[2]), y.shape[1])
    model.compile(Adam(1e-3), 'categorical_crossentropy', ['accuracy'])
    model.fit(X, y, batch_size=args.batch_size, epochs=args.epochs, validation_split=args.val_split)
    loss, acc = model.evaluate(X, y, verbose=0)
    print(f"Final accuracy: {acc:.3f}")
    model.save(args.output)
    print(f"Model written to {args.output}")

if __name__=="__main__":
    main()