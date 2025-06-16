# Define 1D convolutional neural network (CNN) model in TensorFlow/Keras
# Load preprocessed data, train model, and export gesture_model.h5

import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Drop
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

def parse_args():
    parser = argparse.ArgumentParser(description="Train a 1D-CNN gesture recognition model")
    parser.add_argument("input", help="Path to the processed data .npy file (labels + features)")
    parser.add_argument("--batch_size", "-b", type=int, default=32, help="Batch size for training (default: 32)")
    parser.add_argument("--epochs", "-e", type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument("--val_split", "-v", type=float, default=0.2, help="Validation split fraction (default: 0.2)")
    parser.add_argument("--output", "-o", default="gesture_model.h5", help="Filename for saving the trained model (default: gesture_model.h5)")
    return parser.parse_args()

# Load and prepare data
def load_data(input_path):
    arr = np.load(input_path)
    
    # Split labels (first column) and features
    labels = arr[:, 0].astype(int)            # e.g. [0,1,2,...]
    flat_feats = arr[:, 1:].astype(np.float32) 
    
    # Reshape to [N, window_size, 3]
    window_size = flat_feats.shape[1] // 3
    X = flat_feats.reshape(-1, window_size, 3)
    
    # One-hot encode
    num_classes = labels.max() + 1
    y = to_categorical(labels, num_classes).astype(np.float32)
    
    return X, y, num_classes, window_size

def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=num_classes, activation='softmax'))

def compile_model(model):
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

def train_model(X, y, batch_size, epochs, val_split):
    input_shape = (X.shape[1], X.shape[2])
    num_classes = y.shape[1]

    model = build_model(input_shape, num_classes)
    compile_model(model)

    history = model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=val_split)
    
    return model, history

def evaluate_model(model, X, y):
    val_loss, val_accuracy = model.evaluate(X, y, verbose=0)
    print(f"Validation Loss: {val_loss:.4f}; Validation Accuracy: {val_accuracy:.4f}")

def save_model(model, output_path):
    model.save(output_path)
    print(f"Model saved to {output_path}")