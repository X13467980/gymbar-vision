import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
from config import DATA_DIR, IMG_SIZE, BATCH_SIZE, SEED, MODEL_PATH, LABELS_PATH

def make_val_ds():
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    return val_ds, val_ds.class_names

def main():
    model = keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH) as f:
        labels = json.load(f)

    val_ds, class_names = make_val_ds()
    y_true = []
    y_pred = []
    for x, y in val_ds:
        probs = model.predict(x, verbose=0)
        y_pred.extend(np.argmax(probs, axis=1).tolist())
        y_true.extend(y.numpy().tolist())

    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    main()