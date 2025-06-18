# Load model file
# Run the TFLiteConverter with quantization
# Write out gesture_model.tflite into ../Models/

import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model as _load_keras_model

# Arg parsing
def parse_args():
    p = argparse.ArgumentParser(description="Convert Keras model to TFLite")
    p.add_argument("input", help="Path to the trained Keras .h5 model file")
    p.add_argument("repr_data", help="Path to representative data .npy file (labels, features)")
    p.add_argument("-o", "--output", default="../Models/gesture_model.tflite", help="Output path for the TFLite model (default: ../Models/gesture_model.tflite)")
    p.add_argument("--quantize", action="store_true", help="Enable full INT8 quantization (default: False)")
    return p.parse_args()

# Load Keras model
def load_keras_model(path):
    model = _load_keras_model(path)
    print(f"Loaded Keras model from {path}")
    return model

# Prepare Representative Dataset Generator
def repr_data_gen(repr_path):
    arr = np.load(repr_path)

    for sample in arr[:100]:
        feature_vector = sample[1:].astype(np.float32)
        window_size = feature_vector.size // 3
        reshaped_vector = feature_vector.reshape(1, window_size, 3)
        yield [reshaped_vector]

# Configure TFLite Converter
def convert_to_tflite(model, repr_path, quantize):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: repr_data_gen(repr_path)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        print("Configured converter for full UINT8 quantization")

        tflite_model = converter.convert()
        return tflite_model
    
def main():
    args = parse_args()
    model = load_keras_model(args.input)
    tflite_model = convert_to_tflite(model, args.repr_data, args.quantize)

    out_dir = args.output.rsplit('/', 1)[0]
    if out_dir and not tf.io.gfile.exists(out_dir):
        tf.io.gfile.makedirs(outdir)

    with open(args.output, 'wb') as f:
        f.write(tflite_model)
    print(f"Saved TFLite model to {args.output}")

if __name__ == "__main__":
    main()