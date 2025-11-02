from tensorflow import keras
import tensorflow as tf
from config import MODEL_PATH, MODEL_DIR

def to_tflite_dynamic():
    model = keras.models.load_model(MODEL_PATH)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # 動的量子化
    tflite = converter.convert()
    (MODEL_DIR / "gymbar_classifier_dynamic.tflite").write_bytes(tflite)
    print("Saved:", MODEL_DIR / "gymbar_classifier_dynamic.tflite")

if __name__ == "__main__":
    to_tflite_dynamic()