import sys, json, numpy as np
from tensorflow import keras
from PIL import Image
from config import MODEL_PATH, LABELS_PATH, IMG_SIZE

def infer(path):
    model = keras.models.load_model(MODEL_PATH)
    labels = json.loads(open(LABELS_PATH).read())
    img = Image.open(path).convert("RGB").resize(IMG_SIZE)
    x = np.expand_dims(np.array(img, dtype=np.float32), 0)
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    return labels[idx], float(probs[idx])

if __name__ == "__main__":
    p = sys.argv[1]
    label, prob = infer(p)
    print(label, prob)