
import tensorflow as tf
import numpy as np
import os

MODEL_PATH = "best_model.keras"
IMG_SIZE = (128, 128)

def verify():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}")
        return

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully.")
        
        # Print input shape
        input_shape = model.input_shape
        print(f"Model Input Shape: {input_shape}")
        
        # Test Prediction
        dummy_input = np.random.rand(1, IMG_SIZE[0], IMG_SIZE[1], 3).astype(np.float32)
        print(f"Testing with input shape: {dummy_input.shape}")
        
        prediction = model.predict(dummy_input, verbose=0)
        print(f"✅ Prediction Check: Output shape {prediction.shape}, Value: {prediction}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    verify()
